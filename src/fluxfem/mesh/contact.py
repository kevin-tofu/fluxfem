from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING, TypeAlias
import warnings

import numpy as np
import numpy.typing as npt

from .contact_interface import (
    assemble_contact_interface_jacobian as _assemble_contact_interface_jacobian,
    assemble_contact_interface_residual as _assemble_contact_interface_residual,
    assemble_onesided_bilinear,
    assemble_contact_onesided_floor,
    assemble_contact_coupling_matrices as _assemble_contact_coupling_matrices,
    volume_shape_values_at_points as _volume_shape_values_at_points,
    map_surface_facets_to_tet_elements,
    map_surface_facets_to_hex_elements,
)
from .supermesh import build_surface_supermesh
from .surface import SurfaceMesh
from .base import BaseMesh

if TYPE_CHECKING:
    from .contact_interface import ContactCouplingMatrix
    from ..core.weakform import Params as WeakParams
    from .contact_interface import SurfaceMixedFormContext

ContactJacobianReturn: TypeAlias = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, int]
MixedSurfaceResidualForm: TypeAlias = Callable[
    ["SurfaceMixedFormContext", Mapping[str, npt.ArrayLike], Any],
    Mapping[str, npt.ArrayLike],
]
SurfaceHatFn: TypeAlias = Callable[[np.ndarray], npt.ArrayLike]

_CONTACT_SETUP_CACHE: dict[tuple, "ContactSurfaceSpace"] = {}


@dataclass(frozen=True)
class ContactOperators:
    """Container for assembled contact operators."""

    enforcement: str
    law: str | None = None
    formulation: str | None = None
    coupling_aa: Any | None = None
    coupling_ab: Any | None = None
    B_a: Any | None = None
    B_b: Any | None = None
    B: Any | None = None
    Kuu: Any | None = None
    residual: Any | None = None
    jacobian: Any | None = None
    facet_conn_master: np.ndarray | None = None
    rho: float | None = None
    multiplier_space: str | None = None


@dataclass(frozen=True)
class ContactKKTSolveConfig:
    """Linear solve configuration for ``solve_contact_kkt``."""

    backend: str = "numpy"
    diagonal_shift: float = 0.0
    allow_dense_fallback: bool = True
    jax_solver: str = "gmres"
    jax_tol: float = 1e-8
    jax_atol: float = 0.0
    jax_restart: int = 20
    jax_maxiter: int | None = None
    # Prefer iterative path even when a dense matrix is passed.
    jax_dense_mode: str = "iterative"  # "iterative" | "direct_custom_vjp"
    petsc_ksp_type: str = "gmres"
    petsc_pc_type: str = "none"
    petsc_preconditioner: str | None = "diag0"
    petsc_rtol: float | None = 1e-10
    petsc_atol: float | None = None
    petsc_max_it: int | None = None
    petsc_options: Mapping[str, Any] | None = None
    petsc_options_prefix: str | None = "contact_kkt_"

    def validate(self) -> "ContactKKTSolveConfig":
        backend = str(self.backend).lower()
        if backend not in {"numpy", "jax", "petsc4py"}:
            raise ValueError("backend must be 'numpy', 'petsc4py', or 'jax'.")
        if self.jax_solver not in {"gmres", "spsolve"}:
            raise ValueError("jax_solver must be 'gmres' or 'spsolve'.")
        if self.jax_dense_mode not in {"iterative", "direct_custom_vjp"}:
            raise ValueError("jax_dense_mode must be 'iterative' or 'direct_custom_vjp'.")
        if int(self.jax_restart) <= 0:
            raise ValueError("jax_restart must be positive.")
        if self.jax_solver == "spsolve" and float(self.diagonal_shift) != 0.0:
            raise ValueError("jax_solver='spsolve' currently requires diagonal_shift == 0.")
        return self


@dataclass(frozen=True)
class EmbeddingMap:
    """Sparse mapping ``u_slave = W * u_master``."""

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]
    mode: str = "nodal"
    meta: Mapping[str, Any] | None = None


def build_nodal_embedding_map(master_coords: np.ndarray, slave_coords: np.ndarray) -> EmbeddingMap:
    """
    Build nearest-neighbor nodal embedding map from slave nodes to master nodes.
    """
    xm = np.asarray(master_coords, dtype=float)
    xs = np.asarray(slave_coords, dtype=float)
    if xm.ndim != 2 or xs.ndim != 2:
        raise ValueError("master_coords and slave_coords must be rank-2 arrays.")
    if xm.shape[1] != xs.shape[1]:
        raise ValueError("master/slave coordinates must share spatial dimension.")
    if xm.shape[0] == 0 or xs.shape[0] == 0:
        return EmbeddingMap(
            rows=np.zeros((0,), dtype=int),
            cols=np.zeros((0,), dtype=int),
            data=np.zeros((0,), dtype=float),
            shape=(int(xs.shape[0]), int(xm.shape[0])),
            mode="nodal",
            meta={"mapped_count": 0, "unmapped_count": int(xs.shape[0])},
        )

    # Brute-force nearest master node per slave node.
    diffs = xs[:, None, :] - xm[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(d2, axis=1).astype(int)
    rows = np.arange(xs.shape[0], dtype=int)
    cols = nearest
    data = np.ones((xs.shape[0],), dtype=float)
    return EmbeddingMap(
        rows=rows,
        cols=cols,
        data=data,
        shape=(int(xs.shape[0]), int(xm.shape[0])),
        mode="nodal",
        meta={"mapped_count": int(xs.shape[0]), "unmapped_count": 0},
    )


def build_barycentric_embedding_map(
    master_coords: np.ndarray,
    master_conn: np.ndarray,
    slave_coords: np.ndarray,
    *,
    tol: float = 1e-8,
    allow_unmapped: str | bool = "error",
    return_unmapped_ids: bool = False,
) -> EmbeddingMap | tuple[EmbeddingMap, np.ndarray]:
    """
    Build barycentric/isoparametric embedding map from slave points to master element nodes.

    Notes:
    - Uses broad-phase AABB filtering and deterministic tie-break.
    - If multiple master elements pass inside checks (e.g. point on element boundary),
      the smallest candidate element id is selected.
    - Supports element types handled by ``volume_shape_values_at_points``.
    """
    xm = np.asarray(master_coords, dtype=float)
    conn = np.asarray(master_conn, dtype=int)
    xs = np.asarray(slave_coords, dtype=float)
    if xm.ndim != 2 or xs.ndim != 2:
        raise ValueError("master_coords and slave_coords must be rank-2 arrays.")
    if conn.ndim != 2:
        raise ValueError("master_conn must be rank-2 array.")
    if xm.shape[1] != xs.shape[1]:
        raise ValueError("master/slave coordinates must share spatial dimension.")
    if isinstance(allow_unmapped, bool):
        warnings.warn(
            "Boolean allow_unmapped is deprecated; use 'error' or 'skip'.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "skip" if allow_unmapped else "error"
    else:
        mode = str(allow_unmapped).lower()
    if mode not in {"error", "skip"}:
        raise ValueError("allow_unmapped must be 'error' or 'skip' (bool is accepted for compatibility).")

    if xs.shape[0] == 0:
        emb = EmbeddingMap(
            rows=np.zeros((0,), dtype=int),
            cols=np.zeros((0,), dtype=int),
            data=np.zeros((0,), dtype=float),
            shape=(int(xs.shape[0]), int(xm.shape[0])),
            mode="barycentric",
            meta={"mapped_count": 0, "unmapped_count": 0},
        )
        if return_unmapped_ids:
            return emb, np.zeros((0,), dtype=int)
        return emb
    if conn.shape[0] == 0:
        if mode == "error":
            raise ValueError("Failed to map slave points: master_conn has no elements.")
        emb = EmbeddingMap(
            rows=np.zeros((0,), dtype=int),
            cols=np.zeros((0,), dtype=int),
            data=np.zeros((0,), dtype=float),
            shape=(int(xs.shape[0]), int(xm.shape[0])),
            mode="barycentric",
            meta={"mapped_count": 0, "unmapped_count": int(xs.shape[0])},
        )
        if return_unmapped_ids:
            return emb, np.arange(xs.shape[0], dtype=int)
        return emb

    rows_l: list[int] = []
    cols_l: list[int] = []
    data_l: list[float] = []
    unmapped_l: list[int] = []

    # Broad-phase acceleration: precompute master element AABBs.
    elem_coords_all = xm[conn]  # (n_elem, n_loc, dim)
    elem_mins = np.min(elem_coords_all, axis=1)
    elem_maxs = np.max(elem_coords_all, axis=1)
    tol_eff = float(tol)

    for i_s, p in enumerate(xs):
        found = False
        in_min = p[None, :] >= (elem_mins - tol_eff)
        in_max = p[None, :] <= (elem_maxs + tol_eff)
        candidates = np.nonzero(np.all(in_min & in_max, axis=1))[0]
        if candidates.size:
            candidates = np.sort(candidates, kind="stable")
        for e_id in candidates:
            elem_nodes = conn[int(e_id)]
            elem_nodes_i = np.asarray(elem_nodes, dtype=int)
            elem_coords = xm[elem_nodes_i]
            try:
                N = np.asarray(_volume_shape_values_at_points(p[None, :], elem_coords, tol=tol_eff)[0], dtype=float)
            except Exception:
                continue
            if np.any(~np.isfinite(N)):
                continue
            # Robust inside check for small Newton / floating-point errors.
            if np.min(N) < -tol_eff or np.max(N) > 1.0 + tol_eff:
                continue
            if abs(float(np.sum(N)) - 1.0) > 10.0 * tol_eff:
                continue

            for j_local, w in enumerate(N):
                if abs(float(w)) <= tol_eff:
                    continue
                rows_l.append(int(i_s))
                cols_l.append(int(elem_nodes_i[j_local]))
                data_l.append(float(w))
            found = True
            break
        if not found:
            unmapped_l.append(int(i_s))
            if mode == "error":
                raise ValueError(f"Failed to map slave point index {i_s} to any master element (tol={tol}).")

    if rows_l:
        rows = np.asarray(rows_l, dtype=int)
        cols = np.asarray(cols_l, dtype=int)
        data = np.asarray(data_l, dtype=float)
    else:
        rows = np.zeros((0,), dtype=int)
        cols = np.zeros((0,), dtype=int)
        data = np.zeros((0,), dtype=float)
    mapped_ids = np.unique(rows) if rows.size else np.zeros((0,), dtype=int)
    unmapped_ids_np = np.asarray(unmapped_l, dtype=int)
    emb = EmbeddingMap(
        rows=rows,
        cols=cols,
        data=data,
        shape=(int(xs.shape[0]), int(xm.shape[0])),
        mode="barycentric",
        meta={
            "mapped_count": int(mapped_ids.shape[0]),
            "unmapped_count": int(unmapped_ids_np.shape[0]),
        },
    )
    if return_unmapped_ids:
        return emb, unmapped_ids_np
    return emb


def build_barycentric_embedding_map_from_meshes(
    master_mesh: BaseMesh,
    slave_mesh: BaseMesh,
    *,
    slave_facet_selector: Callable[[BaseMesh], np.ndarray] | None = None,
    slave_node_selector: Callable[[BaseMesh], np.ndarray] | None = None,
    master_element_selector: Callable[[BaseMesh], np.ndarray] | None = None,
    tol: float = 1e-8,
    allow_unmapped: str | bool = "error",
    return_unmapped_ids: bool = False,
) -> EmbeddingMap | tuple[EmbeddingMap, np.ndarray]:
    """
    Build barycentric embedding map directly from mesh objects and selectors.

    Typical usage is to select slave boundary facets (e.g., plane) and embed those
    slave nodes into the master volume.
    """
    if slave_facet_selector is not None and slave_node_selector is not None:
        raise ValueError("Provide only one of slave_facet_selector or slave_node_selector.")

    x_master = np.asarray(master_mesh.coords, dtype=float)
    conn_master = np.asarray(master_mesh.conn, dtype=int)
    x_slave = np.asarray(slave_mesh.coords, dtype=float)
    n_slave_total = int(x_slave.shape[0])
    n_master_total = int(x_master.shape[0])

    if master_element_selector is not None:
        master_elem_ids = np.asarray(master_element_selector(master_mesh), dtype=int)
        conn_embed = conn_master[master_elem_ids]
    else:
        conn_embed = conn_master

    if slave_node_selector is not None:
        slave_node_ids = np.asarray(slave_node_selector(slave_mesh), dtype=int).reshape(-1)
    elif slave_facet_selector is not None:
        facets = np.asarray(slave_facet_selector(slave_mesh), dtype=int)
        slave_node_ids = np.unique(facets.reshape(-1)) if facets.size else np.zeros((0,), dtype=int)
    else:
        slave_node_ids = np.arange(n_slave_total, dtype=int)

    x_slave_sel = x_slave[slave_node_ids] if slave_node_ids.size else np.zeros((0, x_slave.shape[1]), dtype=float)
    out_local = build_barycentric_embedding_map(
        x_master,
        conn_embed,
        x_slave_sel,
        tol=tol,
        allow_unmapped=allow_unmapped,
        return_unmapped_ids=return_unmapped_ids,
    )
    if return_unmapped_ids:
        emb_local, unmapped_local = out_local
    else:
        emb_local = out_local
        unmapped_local = np.zeros((0,), dtype=int)
    rows_global = slave_node_ids[np.asarray(emb_local.rows, dtype=int)] if emb_local.rows.size else np.zeros((0,), dtype=int)
    emb_global = EmbeddingMap(
        rows=np.asarray(rows_global, dtype=int),
        cols=np.asarray(emb_local.cols, dtype=int),
        data=np.asarray(emb_local.data, dtype=float),
        shape=(n_slave_total, n_master_total),
        mode="barycentric",
        meta={
            "mapped_count": int(np.unique(rows_global).shape[0]) if np.asarray(rows_global).size else 0,
            "unmapped_count": int(np.asarray(unmapped_local).shape[0]),
            "slave_selection": "node_selector"
            if slave_node_selector is not None
            else ("facet_selector" if slave_facet_selector is not None else "all_nodes"),
            "master_selection": "element_selector" if master_element_selector is not None else "all_elements",
        },
    )
    if return_unmapped_ids:
        return emb_global, slave_node_ids[np.asarray(unmapped_local, dtype=int)]
    return emb_global


def assemble_embedding_constraint_matrix(
    embedding: EmbeddingMap,
    *,
    n_master_nodes: int,
    n_slave_nodes: int,
    value_dim: int = 1,
    backend: str = "numpy",
):
    """
    Assemble ``C`` for equality constraints ``W*u_master - u_slave = 0``.

    Returns matrix with shape ``(n_slave_nodes*value_dim, (n_master_nodes+n_slave_nodes)*value_dim)``.
    """
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    n_m = int(n_master_nodes)
    n_s = int(n_slave_nodes)
    vd = int(value_dim)
    if vd <= 0:
        raise ValueError("value_dim must be positive.")
    if int(embedding.shape[0]) > n_s or int(embedding.shape[1]) != n_m:
        raise ValueError("embedding.shape must satisfy (<= n_slave_nodes, n_master_nodes).")

    emb_rows = np.asarray(embedding.rows, dtype=int)
    emb_cols = np.asarray(embedding.cols, dtype=int)
    emb_data = np.asarray(embedding.data, dtype=float)
    if emb_rows.size != emb_cols.size or emb_rows.size != emb_data.size:
        raise ValueError("embedding rows/cols/data must have same length.")
    if emb_rows.size == 0:
        n_rows = 0
        row_ids = np.zeros((0,), dtype=int)
    else:
        if np.min(emb_rows) < 0 or np.max(emb_rows) >= n_s:
            raise ValueError("embedding row ids must be within [0, n_slave_nodes).")
        if np.min(emb_cols) < 0 or np.max(emb_cols) >= n_m:
            raise ValueError("embedding col ids must be within [0, n_master_nodes).")
        row_ids = np.unique(emb_rows)
        n_rows = int(row_ids.shape[0]) * vd
    row_pos = {int(r): i for i, r in enumerate(row_ids.tolist())}
    n_cols = (n_m + n_s) * vd
    if backend == "jax":
        import jax.numpy as jnp

        C = jnp.zeros((n_rows, n_cols), dtype=float)
        for r_s, c_m, w in zip(emb_rows, emb_cols, emb_data):
            for d in range(vd):
                row = int(row_pos[int(r_s)]) * vd + d
                col_m = int(c_m) * vd + d
                col_s = n_m * vd + int(r_s) * vd + d
                C = C.at[row, col_m].add(float(w))
                C = C.at[row, col_s].add(-1.0)
        return C

    C = np.zeros((n_rows, n_cols), dtype=float)
    for r_s, c_m, w in zip(emb_rows, emb_cols, emb_data):
        for d in range(vd):
            row = int(row_pos[int(r_s)]) * vd + d
            col_m = int(c_m) * vd + d
            col_s = n_m * vd + int(r_s) * vd + d
            C[row, col_m] += float(w)
            C[row, col_s] += -1.0
    return C


def assemble_rbe2_constraint_matrix(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    backend: str = "numpy",
):
    """
    Assemble 3D RBE2-style rigid kinematic constraints.

    Unknown ordering:
      q = [u_ref(3), omega_ref(3), u_slave_0(3), ..., u_slave_{n-1}(3)]

    Constraint for each slave node i:
      u_slave_i - u_ref - (omega_ref x (x_i - x_ref)) = 0
    """
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape[0] != 3:
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")

    n_s = int(x_s.shape[0])
    n_rows = 3 * n_s
    n_cols = 6 + 3 * n_s

    if backend == "jax":
        import jax.numpy as jnp

        C = jnp.zeros((n_rows, n_cols), dtype=float)
        for i in range(n_s):
            rx, ry, rz = (x_s[i] - x_ref).tolist()
            r0 = 3 * i
            c_slave = 6 + 3 * i
            C = C.at[r0 + 0, 0].set(-1.0)
            C = C.at[r0 + 1, 1].set(-1.0)
            C = C.at[r0 + 2, 2].set(-1.0)
            C = C.at[r0 + 0, 4].set(-rz)
            C = C.at[r0 + 0, 5].set(+ry)
            C = C.at[r0 + 1, 3].set(+rz)
            C = C.at[r0 + 1, 5].set(-rx)
            C = C.at[r0 + 2, 3].set(-ry)
            C = C.at[r0 + 2, 4].set(+rx)
            C = C.at[r0 + 0, c_slave + 0].set(+1.0)
            C = C.at[r0 + 1, c_slave + 1].set(+1.0)
            C = C.at[r0 + 2, c_slave + 2].set(+1.0)
        return C

    C = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_s):
        rx, ry, rz = (x_s[i] - x_ref).tolist()
        r0 = 3 * i
        c_slave = 6 + 3 * i
        C[r0 + 0, 0] = -1.0
        C[r0 + 1, 1] = -1.0
        C[r0 + 2, 2] = -1.0
        C[r0 + 0, 4] = -rz
        C[r0 + 0, 5] = +ry
        C[r0 + 1, 3] = +rz
        C[r0 + 1, 5] = -rx
        C[r0 + 2, 3] = -ry
        C[r0 + 2, 4] = +rx
        C[r0 + 0, c_slave + 0] = +1.0
        C[r0 + 1, c_slave + 1] = +1.0
        C[r0 + 2, c_slave + 2] = +1.0
    return C


def assemble_contact_interface_residual(*args, **kwargs):
    """Assemble residual on a contact interface supermesh."""
    return _assemble_contact_interface_residual(*args, **kwargs)


def assemble_contact_interface_jacobian(*args, **kwargs):
    """Assemble Jacobian on a contact interface supermesh."""
    return _assemble_contact_interface_jacobian(*args, **kwargs)


def assemble_contact_coupling_matrices(*args, **kwargs):
    """Assemble coupling matrices for contact interface constraints."""
    return _assemble_contact_coupling_matrices(*args, **kwargs)


def _coo_to_dense(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape: tuple[int, int], *, backend: str):
    if backend == "jax":
        import jax.numpy as jnp

        out = jnp.zeros(shape, dtype=jnp.asarray(data).dtype if np.asarray(data).size else float)
        if len(rows) == 0:
            return out
        return out.at[np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)].add(np.asarray(data))
    out = np.zeros(shape, dtype=float)
    for r, c, v in zip(np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(data, dtype=float)):
        out[int(r), int(c)] += float(v)
    return out


def _p0_reduction_matrix_from_facets(facet_conn: np.ndarray, n_nodes: int):
    facets = np.asarray(facet_conn, dtype=int)
    S = np.zeros((int(facets.shape[0]), int(n_nodes)), dtype=float)
    for f, nodes in enumerate(facets):
        S[int(f), np.asarray(nodes, dtype=int)] = 1.0
    return S


def _coalesce_int_coo(rows: np.ndarray, cols: np.ndarray, data: np.ndarray):
    from ..solver.sparse import coalesce_coo

    r, c, d = coalesce_coo(rows, cols, data)
    return np.asarray(r, dtype=int), np.asarray(c, dtype=int), np.asarray(d, dtype=float)


def _kkt_coo_from_coupling(
    coupling_aa,
    coupling_ab,
    *,
    rho: float,
    multiplier_space: str,
    facet_conn_master: np.ndarray | None,
):
    rows_aa, cols_aa, data_aa = _coalesce_int_coo(coupling_aa.rows, coupling_aa.cols, coupling_aa.data)
    rows_ab, cols_ab, data_ab = _coalesce_int_coo(coupling_ab.rows, coupling_ab.cols, coupling_ab.data)
    n_a = int(coupling_aa.shape[0])
    n_b = int(coupling_ab.shape[1])
    n_u = n_a + n_b

    if multiplier_space == "nodal":
        n_l = n_a
        b_rows = np.concatenate([rows_aa, rows_ab])
        b_cols = np.concatenate([cols_aa, n_a + cols_ab])
        b_data = np.concatenate([data_aa, -data_ab])
    elif multiplier_space == "p0":
        if facet_conn_master is None:
            raise ValueError("facet_conn_master is required when multiplier_space='p0'.")
        facets = np.asarray(facet_conn_master, dtype=int)
        n_l = int(facets.shape[0])
        row_map: dict[int, list[int]] = {i: [] for i in range(n_a)}
        for k, r in enumerate(rows_aa):
            row_map[int(r)].append(int(k))
        row_map_ab: dict[int, list[int]] = {i: [] for i in range(n_a)}
        for k, r in enumerate(rows_ab):
            row_map_ab[int(r)].append(int(k))
        b_rows_l: list[int] = []
        b_cols_l: list[int] = []
        b_data_l: list[float] = []
        for lf, nodes in enumerate(facets):
            acc: dict[int, float] = {}
            for n in np.asarray(nodes, dtype=int):
                for k in row_map.get(int(n), []):
                    c = int(cols_aa[k])
                    acc[c] = acc.get(c, 0.0) + float(data_aa[k])
                for k in row_map_ab.get(int(n), []):
                    c = n_a + int(cols_ab[k])
                    acc[c] = acc.get(c, 0.0) - float(data_ab[k])
            for c, v in acc.items():
                b_rows_l.append(int(lf))
                b_cols_l.append(int(c))
                b_data_l.append(float(v))
        if b_rows_l:
            b_rows = np.asarray(b_rows_l, dtype=int)
            b_cols = np.asarray(b_cols_l, dtype=int)
            b_data = np.asarray(b_data_l, dtype=float)
            b_rows, b_cols, b_data = _coalesce_int_coo(b_rows, b_cols, b_data)
        else:
            b_rows = np.zeros((0,), dtype=int)
            b_cols = np.zeros((0,), dtype=int)
            b_data = np.zeros((0,), dtype=float)
    else:
        raise ValueError("multiplier_space must be 'nodal' or 'p0'")

    # Build Kuu = rho * B^T B from row-wise products.
    by_row: dict[int, list[int]] = {}
    for k, r in enumerate(b_rows):
        by_row.setdefault(int(r), []).append(int(k))
    kuu_acc: dict[tuple[int, int], float] = {}
    if float(rho) != 0.0:
        rr = float(rho)
        for ids in by_row.values():
            for i in ids:
                ci = int(b_cols[i])
                vi = float(b_data[i])
                for j in ids:
                    cj = int(b_cols[j])
                    vj = float(b_data[j])
                    key = (ci, cj)
                    kuu_acc[key] = kuu_acc.get(key, 0.0) + rr * vi * vj

    kuu_rows = np.fromiter((k[0] for k in kuu_acc.keys()), dtype=int, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=int)
    kuu_cols = np.fromiter((k[1] for k in kuu_acc.keys()), dtype=int, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=int)
    kuu_data = np.fromiter((v for v in kuu_acc.values()), dtype=float, count=len(kuu_acc)) if kuu_acc else np.zeros((0,), dtype=float)

    # KKT COO assembly:
    # [Kuu  B^T]
    # [ B    0 ]
    k_rows = []
    k_cols = []
    k_data = []
    if kuu_rows.size:
        k_rows.append(kuu_rows)
        k_cols.append(kuu_cols)
        k_data.append(kuu_data)
    if b_rows.size:
        # B^T block (top-right)
        k_rows.append(b_cols)
        k_cols.append(n_u + b_rows)
        k_data.append(b_data)
        # B block (bottom-left)
        k_rows.append(n_u + b_rows)
        k_cols.append(b_cols)
        k_data.append(b_data)
    if k_rows:
        rows = np.concatenate(k_rows)
        cols = np.concatenate(k_cols)
        data = np.concatenate(k_data)
        rows, cols, data = _coalesce_int_coo(rows, cols, data)
    else:
        rows = np.zeros((0,), dtype=int)
        cols = np.zeros((0,), dtype=int)
        data = np.zeros((0,), dtype=float)
    n_total = int(n_u + n_l)
    return rows, cols, data, n_total


def assemble_contact_constraint_operators(
    contact,
    *,
    law: str | None = None,
    formulation: str | None = None,
    rho: float = 0.0,
    multiplier_space: str = "nodal",
    backend: str = "numpy",
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    """Assemble constraint-family operators (coupling/B/Kuu, optionally residual/jacobian metadata)."""
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    if state is not None and u is not None and state is not u:
        raise ValueError("state and u are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form
    u_eff = state if state is not None else u
    has_eval_inputs = (res_form_eff is not None) or (u_eff is not None) or (params is not None)
    if has_eval_inputs and (res_form_eff is None or u_eff is None or params is None):
        raise ValueError(
            "weak_form/state/params (or res_form/u/params) must be provided together for constraint residual/jacobian evaluation."
        )
    f_arg = None if formulation is None else str(formulation).lower()
    if f_arg is not None and f_arg in {"penalty", "penalty_consistent", "nitsche"}:
        raise ValueError(
            "Constraint operators are multiplier-family only. Use a multiplier/augmented_lagrangian formulation."
        )
    resolved = "mortar"
    law_resolved = str(law) if law is not None else "one_sided_normal_frictionless"
    formulation_resolved = str(formulation) if formulation is not None else "multiplier"
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")

    if not hasattr(contact, "assemble_contact_coupling_matrices"):
        raise TypeError("contact must provide assemble_contact_coupling_matrices() for constraint operators.")
    coupling_aa, coupling_ab = contact.assemble_contact_coupling_matrices()

    facet_conn_master = None
    if hasattr(contact, "surface_master"):
        facet_conn_master = np.asarray(contact.surface_master.conn, dtype=int)
    elif hasattr(contact, "contacts") and len(getattr(contact, "contacts")) > 0:
        first = contact.contacts[0]
        if hasattr(first, "surface_master"):
            facet_conn_master = np.asarray(first.surface_master.conn, dtype=int)

    M_aa = _coo_to_dense(coupling_aa.rows, coupling_aa.cols, coupling_aa.data, coupling_aa.shape, backend=backend)
    M_ab = _coo_to_dense(coupling_ab.rows, coupling_ab.cols, coupling_ab.data, coupling_ab.shape, backend=backend)

    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if multiplier_space == "nodal":
        B_a = M_aa
        B_b = M_ab
    elif multiplier_space == "p0":
        if facet_conn_master is None:
            raise ValueError("facet_conn_master is required when multiplier_space='p0'.")
        n_master_nodes = int(coupling_aa.shape[0])
        S_np = _p0_reduction_matrix_from_facets(facet_conn_master, n_master_nodes)
        S = xp.asarray(S_np)
        B_a = S @ M_aa
        B_b = S @ M_ab
    else:
        raise ValueError("multiplier_space must be 'nodal' or 'p0'")

    B = xp.concatenate([B_a, -B_b], axis=1)
    Kuu = xp.asarray(rho) * (B.T @ B)
    residual = None
    jacobian = None
    if has_eval_inputs:
        if not hasattr(contact, "assemble_residual") or not hasattr(contact, "assemble_jacobian"):
            raise TypeError("contact must provide assemble_residual() and assemble_jacobian() for weak-form evaluation.")
        residual = contact.assemble_residual(res_form_eff, u_eff, params, normal_source=normal_source)
        jacobian = contact.assemble_jacobian(
            res_form_eff,
            u_eff,
            params,
            normal_source=normal_source,
            sparse=sparse,
            backend=backend,
            batch_jac=batch_jac,
        )
    return ContactOperators(
        enforcement=resolved,
        law=law_resolved,
        formulation=formulation_resolved,
        coupling_aa=coupling_aa,
        coupling_ab=coupling_ab,
        B_a=B_a,
        B_b=B_b,
        B=B,
        Kuu=Kuu,
        residual=residual,
        jacobian=jacobian,
        facet_conn_master=facet_conn_master,
        rho=rho,
        multiplier_space=str(multiplier_space),
    )


def assemble_contact_penalty_operators(
    contact,
    *,
    law: str | None = None,
    formulation: str | None = None,
    backend: str = "numpy",
    weak_form: MixedSurfaceResidualForm | None = None,
    state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    res_form: MixedSurfaceResidualForm | None = None,
    u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
    params: "WeakParams" | None = None,
    normal_source: str = "master",
    sparse: bool = False,
    batch_jac: bool | None = None,
) -> ContactOperators:
    """Assemble penalty-family operators (residual/jacobian)."""
    f_arg = None if formulation is None else str(formulation).lower()
    if f_arg is not None and f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
        raise ValueError(
            "Penalty operators are penalty-family only. Use penalty/penalty_consistent formulation."
        )
    resolved = "nitsche"
    if weak_form is not None and res_form is not None and weak_form is not res_form:
        raise ValueError("weak_form and res_form are aliases; provide only one.")
    if state is not None and u is not None and state is not u:
        raise ValueError("state and u are aliases; provide only one.")
    res_form_eff = weak_form if weak_form is not None else res_form
    u_eff = state if state is not None else u

    law_resolved = str(law) if law is not None else "one_sided_normal_frictionless"
    formulation_resolved = str(formulation) if formulation is not None else "penalty_consistent"
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    if res_form_eff is None or u_eff is None or params is None:
        raise ValueError("weak_form/state/params (or res_form/u/params) are required for penalty operators.")
    if not hasattr(contact, "assemble_residual") or not hasattr(contact, "assemble_jacobian"):
        raise TypeError("contact must provide assemble_residual() and assemble_jacobian() for penalty operators.")
    residual = contact.assemble_residual(res_form_eff, u_eff, params, normal_source=normal_source)
    jacobian = contact.assemble_jacobian(
        res_form_eff,
        u_eff,
        params,
        normal_source=normal_source,
        sparse=sparse,
        backend=backend,
        batch_jac=batch_jac,
    )
    return ContactOperators(
        enforcement=resolved,
        law=law_resolved,
        formulation=formulation_resolved,
        residual=residual,
        jacobian=jacobian,
    )


def assemble_contact_kkt(
    coupling_aa,
    coupling_ab,
    *,
    rho: float = 0.0,
    multiplier_space: str = "nodal",
    facet_conn_master: np.ndarray | None = None,
    backend: str = "numpy",
    format: str = "fluxsparse",
    return_blocks: bool = False,
):
    """
    Assemble contact KKT block from coupling matrices.

    KKT is assembled as:
      B = [B_a, -B_b]
      Kuu = rho * (B^T B)
      KKT = [[Kuu, B^T], [B, 0]]

    multiplier_space:
    - "nodal": lambda lives on interface nodal basis (B_a=M_aa, B_b=M_ab)
    - "p0": lambda is facet-wise constant on master side (B_* = S * M_*)
    """
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be 'numpy' or 'jax'")
    if multiplier_space not in {"nodal", "p0"}:
        raise ValueError("multiplier_space must be 'nodal' or 'p0'")
    if format not in {"dense", "fluxsparse", "bcoo"}:
        raise ValueError("format must be 'dense', 'fluxsparse', or 'bcoo'")
    if return_blocks and format != "dense":
        raise ValueError("return_blocks=True is supported only with format='dense'.")

    if format != "dense":
        import jax

        if isinstance(rho, jax.core.Tracer):
            raise ValueError("format='fluxsparse'/'bcoo' currently requires rho to be a concrete scalar.")
        rows, cols, data, n_total = _kkt_coo_from_coupling(
            coupling_aa,
            coupling_ab,
            rho=float(rho),
            multiplier_space=multiplier_space,
            facet_conn_master=facet_conn_master,
        )
        if format == "fluxsparse":
            from ..solver import FluxSparseMatrix

            return FluxSparseMatrix(rows, cols, data, n_dofs=n_total)
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        idx = jnp.stack([jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(cols, dtype=jnp.int32)], axis=-1)
        return jsparse.BCOO((jnp.asarray(data), idx), shape=(n_total, n_total))

    M_aa = _coo_to_dense(coupling_aa.rows, coupling_aa.cols, coupling_aa.data, coupling_aa.shape, backend=backend)
    M_ab = _coo_to_dense(coupling_ab.rows, coupling_ab.cols, coupling_ab.data, coupling_ab.shape, backend=backend)

    if backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if multiplier_space == "nodal":
        B_a = M_aa
        B_b = M_ab
    else:
        if facet_conn_master is None:
            raise ValueError("facet_conn_master is required when multiplier_space='p0'.")
        n_master_nodes = int(coupling_aa.shape[0])
        S_np = _p0_reduction_matrix_from_facets(facet_conn_master, n_master_nodes)
        S = xp.asarray(S_np)
        B_a = S @ M_aa
        B_b = S @ M_ab

    B = xp.concatenate([B_a, -B_b], axis=1)
    Kuu = xp.asarray(rho) * (B.T @ B)
    n_lambda = int(B.shape[0])
    Zll = xp.zeros((n_lambda, n_lambda), dtype=Kuu.dtype)
    KKT = xp.block([[Kuu, B.T], [B, Zll]])

    if return_blocks:
        return KKT, B_a, B_b
    return KKT


def _resolve_kkt_solve_config(
    *,
    backend: str,
    diagonal_shift: float,
    config: ContactKKTSolveConfig | None,
) -> ContactKKTSolveConfig:
    if config is None:
        return ContactKKTSolveConfig(backend=backend, diagonal_shift=diagonal_shift).validate()
    return config.validate()


def _as_numpy_dense(kkt_matrix) -> np.ndarray:
    return np.asarray(kkt_matrix.to_dense(), dtype=float) if hasattr(kkt_matrix, "to_dense") else np.asarray(kkt_matrix, dtype=float)


def _as_numpy_csr(kkt_matrix):
    try:
        import scipy.sparse as sp
    except Exception:
        return None
    if hasattr(kkt_matrix, "to_csr"):
        return kkt_matrix.to_csr()
    if sp.issparse(kkt_matrix):
        return kkt_matrix.tocsr()
    return sp.csr_matrix(_as_numpy_dense(kkt_matrix))


def _as_jax_linear_op(kkt_matrix):
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse  # type: ignore

    is_fluxsparse = hasattr(kkt_matrix, "matvec") and hasattr(kkt_matrix, "n_dofs")
    is_bcoo = isinstance(kkt_matrix, jsparse.BCOO)
    if is_fluxsparse:
        return (lambda x: kkt_matrix.matvec(x)), True
    if is_bcoo:
        return (lambda x: kkt_matrix @ x), True
    A = jnp.asarray(kkt_matrix.to_dense()) if hasattr(kkt_matrix, "to_dense") else jnp.asarray(kkt_matrix)
    return (lambda x: A @ x), False


def _solve_kkt_petsc(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    from ..solver.petsc import petsc_shell_solve

    A_petsc = _as_numpy_csr(kkt_matrix)
    if A_petsc is None:
        A_petsc = _as_numpy_dense(kkt_matrix)
    if float(cfg.diagonal_shift) != 0.0:
        try:
            import scipy.sparse as sp
        except Exception:
            sp = None
        if sp is not None and hasattr(A_petsc, "tocsr"):
            A_petsc = A_petsc.tocsr() + float(cfg.diagonal_shift) * sp.eye(A_petsc.shape[0], format="csr")
        else:
            A_np = np.asarray(A_petsc, dtype=float)
            A_petsc = A_np + float(cfg.diagonal_shift) * np.eye(A_np.shape[0], dtype=A_np.dtype)

    rhs_np = np.asarray(rhs, dtype=float)
    n = int(rhs_np.shape[0])
    return petsc_shell_solve(
        A_petsc,
        rhs_np,
        n_dofs=n,
        ksp_type=str(cfg.petsc_ksp_type),
        pc_type=str(cfg.petsc_pc_type),
        preconditioner=cfg.petsc_preconditioner,
        pmat=A_petsc,
        rtol=cfg.petsc_rtol,
        atol=cfg.petsc_atol,
        max_it=cfg.petsc_max_it if cfg.petsc_max_it is not None else max(10 * n, 200),
        options=None if cfg.petsc_options is None else dict(cfg.petsc_options),
        options_prefix=cfg.petsc_options_prefix,
    )


def _solve_kkt_numpy(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    A_csr = _as_numpy_csr(kkt_matrix)
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception:
        sp = None
        spla = None

    if A_csr is not None and spla is not None:
        if float(cfg.diagonal_shift) != 0.0:
            A_csr = A_csr + float(cfg.diagonal_shift) * sp.eye(A_csr.shape[0], format="csr")
        return np.asarray(spla.spsolve(A_csr, np.asarray(rhs, dtype=float)))

    if not bool(cfg.allow_dense_fallback):
        raise ValueError("Dense fallback is disabled by ContactKKTSolveConfig.allow_dense_fallback.")
    A = _as_numpy_dense(kkt_matrix)
    if float(cfg.diagonal_shift) != 0.0:
        A = A + float(cfg.diagonal_shift) * np.eye(A.shape[0], dtype=A.dtype)
    return np.linalg.solve(A, np.asarray(rhs, dtype=float))


def _solve_kkt_jax(kkt_matrix, rhs, cfg: ContactKKTSolveConfig):
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax.experimental import sparse as jsparse  # type: ignore

    mv_base, is_sparse_like = _as_jax_linear_op(kkt_matrix)

    def _gmres_solve(mv, bvec):
        maxiter = cfg.jax_maxiter if cfg.jax_maxiter is not None else max(10 * int(bvec.shape[0]), 100)
        x, _ = jsp.sparse.linalg.gmres(
            mv,
            bvec,
            tol=float(cfg.jax_tol),
            atol=float(cfg.jax_atol),
            restart=int(cfg.jax_restart),
            maxiter=int(maxiter),
        )
        return x

    if cfg.jax_solver == "spsolve":
        from jax.experimental.sparse.linalg import spsolve as jspsolve

        if hasattr(kkt_matrix, "to_bcoo"):
            bcoo = kkt_matrix.to_bcoo()
        elif isinstance(kkt_matrix, jsparse.BCOO):
            bcoo = kkt_matrix
        else:
            raise ValueError("jax_solver='spsolve' requires sparse input (FluxSparseMatrix or BCOO).")

        bcsr = jsparse.BCSR.from_bcoo(bcoo)
        b = jnp.asarray(rhs)
        if b.ndim == 1:
            return jspsolve(bcsr.data, bcsr.indices, bcsr.indptr, b)
        if b.ndim == 2:
            return jnp.stack([jspsolve(bcsr.data, bcsr.indices, bcsr.indptr, b[:, i]) for i in range(b.shape[1])], axis=1)
        raise ValueError("rhs must be rank-1 or rank-2.")

    shift = jnp.asarray(cfg.diagonal_shift, dtype=jnp.asarray(rhs).dtype)
    mv = (lambda x: mv_base(x) + shift * x)
    b = jnp.asarray(rhs)
    if is_sparse_like or cfg.jax_dense_mode == "iterative":
        if b.ndim == 1:
            return _gmres_solve(mv, b)
        if b.ndim == 2:
            return jnp.stack([_gmres_solve(mv, b[:, i]) for i in range(b.shape[1])], axis=1)
        raise ValueError("rhs must be rank-1 or rank-2.")

    @jax.custom_vjp
    def _solve_jax(A, bvec):
        return jnp.linalg.solve(A, bvec)

    def _solve_jax_fwd(A, bvec):
        x = jnp.linalg.solve(A, bvec)
        return x, (A, x)

    def _solve_jax_bwd(res, g):
        A, x = res
        lam = jnp.linalg.solve(A.T, g)
        gA = -jnp.outer(lam, x)
        gb = lam
        return gA, gb

    _solve_jax.defvjp(_solve_jax_fwd, _solve_jax_bwd)
    if not bool(cfg.allow_dense_fallback):
        raise ValueError("Dense fallback is disabled by ContactKKTSolveConfig.allow_dense_fallback.")
    A = jnp.asarray(kkt_matrix.to_dense()) if hasattr(kkt_matrix, "to_dense") else jnp.asarray(kkt_matrix)
    A = A + jnp.asarray(cfg.diagonal_shift, dtype=A.dtype) * jnp.eye(A.shape[0], dtype=A.dtype)
    return _solve_jax(A, b)


def solve_contact_kkt(
    kkt_matrix,
    rhs,
    *,
    backend: str = "numpy",
    diagonal_shift: float = 0.0,
    config: ContactKKTSolveConfig | None = None,
):
    """
    Solve KKT linear system ``KKT * x = rhs``.

    `config` is the preferred control surface. `backend`/`diagonal_shift` are kept for compatibility.
    """
    cfg = _resolve_kkt_solve_config(backend=backend, diagonal_shift=diagonal_shift, config=config)
    if cfg.backend == "petsc4py":
        return _solve_kkt_petsc(kkt_matrix, rhs, cfg)
    if cfg.backend == "numpy":
        return _solve_kkt_numpy(kkt_matrix, rhs, cfg)
    return _solve_kkt_jax(kkt_matrix, rhs, cfg)


@dataclass(frozen=True)
class ContactSide:
    surface: SurfaceMesh
    elem_conn: np.ndarray | None
    value_dim: int
    space: object | None = None

    @classmethod
    def from_facets(
        cls,
        mesh: BaseMesh,
        facets: np.ndarray,
        space=None,
        *,
        value_dim: int | None = None,
        mode: str = "touching",
    ):
        side = mesh.surface_with_elem_conn_from_facets(facets, mode=mode)
        if value_dim is None:
            if space is None:
                raise ValueError("space or value_dim is required for ContactSide.from_facets")
            value_dim = int(getattr(space, "value_dim", 1))
        return cls(surface=side.surface, elem_conn=side.elem_conn, value_dim=int(value_dim), space=space)

    @classmethod
    def from_surfaces(
        cls,
        surface: SurfaceMesh,
        *,
        elem_conn: np.ndarray | None = None,
        value_dim: int = 1,
        space: object | None = None,
    ):
        return cls(surface=surface, elem_conn=elem_conn, value_dim=int(value_dim), space=space)


def _facet_map_for_elem_conn(surface: SurfaceMesh, elem_conn: np.ndarray | None) -> np.ndarray:
    if elem_conn is None:
        raise ValueError("elem_conn is required to build facet_to_elem mapping.")
    if elem_conn.shape[1] in {4, 10}:
        return map_surface_facets_to_tet_elements(surface, elem_conn)
    if elem_conn.shape[1] in {8, 20, 27}:
        return map_surface_facets_to_hex_elements(surface, elem_conn)
    raise NotImplementedError("elem_conn must be tet4/tet10/hex8/hex20/hex27")


def facet_gap_values(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
) -> tuple[np.ndarray, float]:
    """
    Compute per-facet gap values for a one-sided contact plane.

    Returns (g_f, min_g_all) where g_f is reduced per facet and min_g_all is
    the global minimum node gap.
    """
    coords_np = np.asarray(coords, dtype=float)
    if value_dim is None:
        value_dim = int(coords_np.shape[1])
    u_nodes = np.asarray(u, dtype=float).reshape(-1, value_dim)
    x_cur = coords_np + u_nodes
    g_all = np.dot(x_cur, np.asarray(n, dtype=float)) - float(c)
    min_g_all = float(np.min(g_all)) if g_all.size else 0.0
    if facets is None or len(facets) == 0:
        return np.zeros((0,), dtype=float), min_g_all
    if reduce == "min":
        g_f = np.array([np.min(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    elif reduce == "mean":
        g_f = np.array([np.mean(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    else:
        raise ValueError("reduce must be 'min' or 'mean'")
    return g_f, min_g_all


def active_contact_facets(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
    threshold: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return active facet indices and global minimum gap for one-sided contact."""
    g_f, min_g_all = facet_gap_values(
        coords,
        facets,
        u,
        n,
        c,
        value_dim=value_dim,
        reduce=reduce,
    )
    active_ids = np.nonzero(g_f < threshold)[0]
    return active_ids, min_g_all


@dataclass(frozen=True)
class OneSidedContact:
    side: ContactSide
    n: np.ndarray | None
    c: float
    k: float
    beta: float
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    facet_map: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        n: np.ndarray | None,
        c: float,
        k: float,
        beta: float,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        facet_map: np.ndarray | None = None,
    ) -> "OneSidedContact":
        if facet_map is None:
            facet_map = _facet_map_for_elem_conn(side.surface, side.elem_conn)
        return cls(
            side=side,
            n=n,
            c=float(c),
            k=float(k),
            beta=float(beta),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            facet_map=facet_map,
        )

    def assemble(self, u, *, return_metrics: bool = False):
        return assemble_contact_onesided_floor(
            self.side.surface,
            np.asarray(u, dtype=float),
            n=None if self.n is None else np.asarray(self.n, dtype=float),
            c=self.c,
            k=self.k,
            beta=self.beta,
            value_dim=self.side.value_dim,
            elem_conn=np.asarray(self.side.elem_conn) if self.side.elem_conn is not None else None,
            facet_to_elem=self.facet_map,
            quad_order=self.quad_order,
            normal_sign=self.normal_sign,
            tol=self.tol,
            return_metrics=return_metrics,
        )


@dataclass(eq=False)
class OneSidedContactSurfaceSpace:
    """Surface wrapper for one-sided (Dirichlet) contact assembly."""

    surface_slave: SurfaceMesh
    elem_conn_slave: np.ndarray
    facet_to_elem_slave: np.ndarray
    value_dim: int = 1
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    surface_master: SurfaceMesh | None = None
    elem_conn_master: np.ndarray | None = None
    facet_to_elem_master: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        surface_master: SurfaceMesh | None = None,
        elem_conn_master: np.ndarray | None = None,
        facet_to_elem_master: np.ndarray | None = None,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ) -> "OneSidedContactSurfaceSpace":
        if side.elem_conn is None:
            raise ValueError("side.elem_conn is required for one-sided assembly")
        facet_map_slave = _facet_map_for_elem_conn(side.surface, side.elem_conn)
        facet_map_master = facet_to_elem_master
        if surface_master is not None and elem_conn_master is not None and facet_map_master is None:
            facet_map_master = _facet_map_for_elem_conn(surface_master, elem_conn_master)
        return cls(
            surface_slave=side.surface,
            elem_conn_slave=np.asarray(side.elem_conn, dtype=int),
            facet_to_elem_slave=np.asarray(facet_map_slave, dtype=int),
            value_dim=int(side.value_dim),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            surface_master=surface_master,
            elem_conn_master=None if elem_conn_master is None else np.asarray(elem_conn_master, dtype=int),
            facet_to_elem_master=None if facet_map_master is None else np.asarray(facet_map_master, dtype=int),
        )

    @classmethod
    def from_facets(
        cls,
        mesh: BaseMesh,
        facets: np.ndarray,
        space=None,
        *,
        surface_master: SurfaceMesh | None = None,
        elem_conn_master: np.ndarray | None = None,
        facet_to_elem_master: np.ndarray | None = None,
        value_dim: int | None = None,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        mode: str = "touching",
    ) -> "OneSidedContactSurfaceSpace":
        side = ContactSide.from_facets(mesh, facets, space, value_dim=value_dim, mode=mode)
        return cls.from_side(
            side,
            surface_master=surface_master,
            elem_conn_master=elem_conn_master,
            facet_to_elem_master=facet_to_elem_master,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
        )

    def assemble_bilinear(
        self,
        u_hat_fn: SurfaceHatFn | None,
        params: "WeakParams",
        *,
        u_master: np.ndarray | None = None,
        grad_source: str = "volume",
        dof_source: str = "volume",
        quad_order: int | None = None,
        normal_sign: float | None = None,
        tol: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return assemble_onesided_bilinear(
            self.surface_slave,
            u_hat_fn,
            params,
            surface_master=self.surface_master,
            u_master=u_master,
            value_dim=self.value_dim,
            elem_conn=self.elem_conn_slave,
            facet_to_elem=self.facet_to_elem_slave,
            elem_conn_master=self.elem_conn_master,
            facet_to_elem_master=self.facet_to_elem_master,
            grad_source=grad_source,
            dof_source=dof_source,
            quad_order=self.quad_order if quad_order is None else int(quad_order),
            normal_sign=self.normal_sign if normal_sign is None else float(normal_sign),
            tol=self.tol if tol is None else float(tol),
        )


@dataclass(eq=False)
class ContactSurfaceSpace:
    """Surface interface wrapper for contact assembly on a supermesh."""

    surface_master: SurfaceMesh
    surface_slave: SurfaceMesh
    supermesh_coords: np.ndarray
    supermesh_conn: np.ndarray
    source_facets_master: np.ndarray
    source_facets_slave: np.ndarray
    elem_conn_master: np.ndarray | None
    elem_conn_slave: np.ndarray | None
    facet_to_elem_master: np.ndarray | None
    facet_to_elem_slave: np.ndarray | None
    field_master: str = "a"
    field_slave: str = "b"
    value_dim_master: int = 1
    value_dim_slave: int = 1
    space_mode_master: str = "nodal"
    space_mode_slave: str = "nodal"
    facet_dofs_master: np.ndarray | None = None
    facet_dofs_slave: np.ndarray | None = None
    quad_order: int = 1
    normal_sign: float | None = None
    tol: float = 1e-8
    backend: str = "jax"
    fd_eps: float = 1e-6
    fd_mode: str = "central"
    fd_block_size: int = 1
    batch_jac: bool | None = None

    @classmethod
    def from_surfaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slave: SurfaceMesh,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        import hashlib
        import os

        if setup_cache_enabled is None:
            setup_cache_enabled = os.getenv("FLUXFEM_CONTACT_SETUP_CACHE", "0") not in ("0", "", "false", "False")
        if setup_cache_trace is None:
            setup_cache_trace = os.getenv("FLUXFEM_CONTACT_SETUP_CACHE_TRACE", "0") not in ("0", "", "false", "False")

        def _array_sig(arr: np.ndarray) -> tuple:
            arr_c = np.ascontiguousarray(arr)
            h = hashlib.blake2b(arr_c.view(np.uint8), digest_size=8).hexdigest()
            return (arr_c.shape, str(arr_c.dtype), h)

        if setup_cache_enabled:
            global _CONTACT_SETUP_CACHE
            try:
                _CONTACT_SETUP_CACHE
            except NameError:
                _CONTACT_SETUP_CACHE = {}
            key = (
                _array_sig(np.asarray(surface_master.coords)),
                _array_sig(np.asarray(surface_master.conn)),
                _array_sig(np.asarray(surface_slave.coords)),
                _array_sig(np.asarray(surface_slave.conn)),
                None if elem_conn_master is None else _array_sig(np.asarray(elem_conn_master)),
                None if elem_conn_slave is None else _array_sig(np.asarray(elem_conn_slave)),
                field_master,
                field_slave,
                int(value_dim_master),
                int(value_dim_slave),
                str(space_mode_master),
                str(space_mode_slave),
                None if facet_dofs_master is None else _array_sig(np.asarray(facet_dofs_master)),
                None if facet_dofs_slave is None else _array_sig(np.asarray(facet_dofs_slave)),
                int(quad_order),
                float(normal_sign) if normal_sign is not None else None,
                float(tol),
                backend,
                float(fd_eps),
                fd_mode,
                int(fd_block_size),
                bool(batch_jac) if batch_jac is not None else None,
            )
            cached = _CONTACT_SETUP_CACHE.get(key)
            if cached is not None:
                if setup_cache_trace:
                    print(
                        f"[contact] setup cache hit n_tris={int(cached.supermesh_conn.shape[0])}",
                        flush=True,
                    )
                return cached

        sm = build_surface_supermesh(surface_master, surface_slave, tol=tol)
        facet_map_master = None
        facet_map_slave = None
        if elem_conn_master is not None:
            if elem_conn_master.shape[1] in {4, 10}:
                facet_map_master = map_surface_facets_to_tet_elements(surface_master, elem_conn_master)
            elif elem_conn_master.shape[1] in {8, 20, 27}:
                facet_map_master = map_surface_facets_to_hex_elements(surface_master, elem_conn_master)
            else:
                raise NotImplementedError("elem_conn_master must be tet4/tet10/hex8/hex20/hex27")
        if elem_conn_slave is not None:
            if elem_conn_slave.shape[1] in {4, 10}:
                facet_map_slave = map_surface_facets_to_tet_elements(surface_slave, elem_conn_slave)
            elif elem_conn_slave.shape[1] in {8, 20, 27}:
                facet_map_slave = map_surface_facets_to_hex_elements(surface_slave, elem_conn_slave)
            else:
                raise NotImplementedError("elem_conn_slave must be tet4/tet10/hex8/hex20/hex27")
        obj = cls(
            surface_master=surface_master,
            surface_slave=surface_slave,
            supermesh_coords=sm.coords,
            supermesh_conn=sm.conn,
            source_facets_master=sm.source_facets_a,
            source_facets_slave=sm.source_facets_b,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            facet_to_elem_master=facet_map_master,
            facet_to_elem_slave=facet_map_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=None if facet_dofs_master is None else np.asarray(facet_dofs_master, dtype=int),
            facet_dofs_slave=None if facet_dofs_slave is None else np.asarray(facet_dofs_slave, dtype=int),
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            batch_jac=batch_jac,
        )
        if setup_cache_enabled:
            _CONTACT_SETUP_CACHE[key] = obj
            if setup_cache_trace:
                print(
                    f"[contact] setup cache store n_tris={int(obj.supermesh_conn.shape[0])}",
                    flush=True,
                )
        return obj

    @classmethod
    def from_facets(
        cls,
        coords: np.ndarray,
        facets: np.ndarray,
        *,
        elem_conn: np.ndarray | None = None,
        value_dim: int = 1,
        space_mode: str = "nodal",
        facet_dofs: np.ndarray | None = None,
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        surface = SurfaceMesh.from_facets(coords, facets)
        return cls.from_surfaces(
            surface,
            surface,
            elem_conn_master=elem_conn,
            elem_conn_slave=elem_conn,
            value_dim_master=value_dim,
            value_dim_slave=value_dim,
            space_mode_master=space_mode,
            space_mode_slave=space_mode,
            facet_dofs_master=facet_dofs,
            facet_dofs_slave=facet_dofs,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod
    def from_surfaces_and_spaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slave: SurfaceMesh,
        space_master,
        space_slave,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int | None = None,
        value_dim_slave: int | None = None,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
    ) -> "ContactSurfaceSpace":
        if value_dim_master is None:
            value_dim_master = int(getattr(space_master, "value_dim", 1))
        if value_dim_slave is None:
            value_dim_slave = int(getattr(space_slave, "value_dim", 1))
        return cls.from_surfaces(
            surface_master,
            surface_slave,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            batch_jac=batch_jac,
        )

    @classmethod
    def from_sides(
        cls,
        master: ContactSide,
        slave: ContactSide,
        *,
        field_master: str = "a",
        field_slave: str = "b",
        quad_order: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        return cls.from_surfaces(
            master.surface,
            slave.surface,
            elem_conn_master=master.elem_conn,
            elem_conn_slave=slave.elem_conn,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=master.value_dim,
            value_dim_slave=slave.value_dim,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod  # type: ignore[no-redef]
    def from_facets(
        cls,
        coords_master: np.ndarray,
        facets_master: np.ndarray,
        coords_slave: np.ndarray,
        facets_slave: np.ndarray,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        surface_master = SurfaceMesh.from_facets(coords_master, facets_master)
        surface_slave = SurfaceMesh.from_facets(coords_slave, facets_slave)
        return cls.from_surfaces(
            surface_master,
            surface_slave,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    def _split_fields(self, u: Mapping[str, np.ndarray] | Sequence[np.ndarray]):
        if isinstance(u, Mapping):
            return u[self.field_master], u[self.field_slave]
        if len(u) != 2:
            raise ValueError("u must be a mapping or a length-2 sequence")
        return u[0], u[1]

    def _auto_normal_sign(self) -> float:
        if not hasattr(self.surface_master, "facet_normals"):
            return 1.0
        normals = self.surface_master.facet_normals()
        coords = np.asarray(self.surface_master.coords)
        coords_slave = np.asarray(self.surface_slave.coords)
        facets_m = np.asarray(self.surface_master.conn, dtype=int)
        facets_s = np.asarray(self.surface_slave.conn, dtype=int)
        dots = []
        for fa, fb in zip(self.source_facets_master, self.source_facets_slave):
            n = normals[int(fa)]
            cm = np.mean(coords[facets_m[int(fa)]], axis=0)
            cs = np.mean(coords_slave[facets_s[int(fb)]], axis=0)
            dots.append(float(np.dot(n, cs - cm)))
        if not dots:
            return 1.0
        return 1.0 if np.sum(dots) >= 0.0 else -1.0

    def _resolve_backend(self, backend: str | None) -> str:
        use_backend = self.backend if backend is None else backend
        if use_backend not in {"jax", "numpy"}:
            raise ValueError("backend must be 'jax' or 'numpy'")
        return use_backend

    def assemble_contact_coupling_matrices(self) -> tuple["ContactCouplingMatrix", "ContactCouplingMatrix"]:
        """Return (M_aa, M_ab) coupling matrices on this contact interface."""
        return _assemble_contact_coupling_matrices(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
        )

    def assemble_contact_kkt(
        self,
        *,
        rho: float = 0.0,
        multiplier_space: str = "nodal",
        backend: str = "numpy",
        format: str = "fluxsparse",
        return_blocks: bool = False,
    ):
        m_aa, m_ab = self.assemble_contact_coupling_matrices()
        return assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=rho,
            multiplier_space=multiplier_space,
            facet_conn_master=np.asarray(self.surface_master.conn, dtype=int),
            backend=backend,
            format=format,
            return_blocks=return_blocks,
        )

    def assemble_contact_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier_space: str = "nodal",
        backend: str = "numpy",
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_constraint_operators(
            self,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier_space=multiplier_space,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str = "numpy",
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_penalty_operators(
            self,
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_residual(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
        params: "WeakParams",
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
    ) -> np.ndarray:
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        return _assemble_contact_interface_residual(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            res_form,
            u_master,
            u_slave,
            params,
            value_dim_a=self.value_dim_master,
            value_dim_b=self.value_dim_slave,
            space_mode_a=self.space_mode_master,
            space_mode_b=self.space_mode_slave,
            facet_dofs_a=self.facet_dofs_master,
            facet_dofs_b=self.facet_dofs_slave,
            field_a=self.field_master,
            field_b=self.field_slave,
            elem_conn_a=self.elem_conn_master,
            elem_conn_b=self.elem_conn_slave,
            facet_to_elem_a=self.facet_to_elem_master,
            facet_to_elem_b=self.facet_to_elem_slave,
            normal_source=normal_source,
            normal_from="master",
            master_field=self.field_master,
            normal_sign=normal_sign,
            grad_source="volume",
            dof_source="volume",
            quad_order=self.quad_order,
            tol=self.tol,
        )

    def assemble_jacobian(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
        params: "WeakParams",
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> ContactJacobianReturn:
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        use_backend = self._resolve_backend(backend)
        use_batch_jac = self.batch_jac if batch_jac is None else batch_jac
        return _assemble_contact_interface_jacobian(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            res_form,
            u_master,
            u_slave,
            params,
            value_dim_a=self.value_dim_master,
            value_dim_b=self.value_dim_slave,
            space_mode_a=self.space_mode_master,
            space_mode_b=self.space_mode_slave,
            facet_dofs_a=self.facet_dofs_master,
            facet_dofs_b=self.facet_dofs_slave,
            field_a=self.field_master,
            field_b=self.field_slave,
            elem_conn_a=self.elem_conn_master,
            elem_conn_b=self.elem_conn_slave,
            facet_to_elem_a=self.facet_to_elem_master,
            facet_to_elem_b=self.facet_to_elem_slave,
            normal_source=normal_source,
            normal_from="master",
            master_field=self.field_master,
            normal_sign=normal_sign,
            grad_source="volume",
            dof_source="volume",
            quad_order=self.quad_order,
            tol=self.tol,
            sparse=sparse,
            backend=use_backend,
            batch_jac=use_batch_jac,
            fd_eps=self.fd_eps,
            fd_mode=self.fd_mode,
            fd_block_size=self.fd_block_size,
        )

    def assemble_bilinear(
        self,
        bilin: Callable[..., Any],
        u_master: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | npt.ArrayLike,
        u_slave: npt.ArrayLike | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = False,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        """
        Assemble a mixed surface bilinear form with signature (v1, v2, u1, u2, params).

        Notes:
        - v1/v2/u1/u2 are symbolic field refs; use .val/.grad/.sym_grad in the expression.
        - The bilinear must be linear in v1 and v2 and include ds() in its expression.
        - When building dot products, prefer dot(v1, ...) and dot(v2, ...) to keep shapes consistent.
        - Normal orientation, grad_source, and dof_source are fixed internally for simplicity.
        - u_master/u_slave can be passed as a single mapping/length-2 sequence; in that case,
          pass params as the next positional arg or a keyword.
        """
        from ..core.weakform import (
            compile_mixed_surface_residual,
            compile_mixed_surface_residual_numpy,
            unknown_ref,
            test_ref,
            param_ref,
            zero_ref,
        )

        def _is_field_pair(obj) -> bool:
            if isinstance(obj, Mapping):
                return True
            return isinstance(obj, Sequence) and not hasattr(obj, "shape")

        if params is None:
            if u_slave is None:
                raise TypeError("params is required")
            if _is_field_pair(u_master):
                params = u_slave
                u_master, u_slave = self._split_fields(u_master)
            else:
                raise TypeError("params is required")
        elif u_slave is None:
            u_master, u_slave = self._split_fields(u_master)

        v1 = test_ref(self.field_master)
        v2 = test_ref(self.field_slave)
        u1 = unknown_ref(self.field_master)
        u2 = unknown_ref(self.field_slave)
        z1 = zero_ref(self.field_master)
        z2 = zero_ref(self.field_slave)
        p = param_ref()

        expr_a = bilin(v1, z2, u1, u2, p)
        expr_b = bilin(z1, v2, u1, u2, p)
        use_backend = self._resolve_backend(None)
        if use_backend == "numpy":
            res_form = compile_mixed_surface_residual_numpy({self.field_master: expr_a, self.field_slave: expr_b})
        else:
            res_form = compile_mixed_surface_residual({self.field_master: expr_a, self.field_slave: expr_b})
        return self.assemble_jacobian(
            res_form,
            {self.field_master: u_master, self.field_slave: u_slave},
            params,
            normal_sign=None,
            normal_source=normal_source,
            sparse=sparse,
            backend=use_backend,
        )


def _field_n_dofs(
    *,
    n_nodes: int,
    n_facets: int,
    value_dim: int,
    space_mode: str,
    facet_dofs: np.ndarray | None,
) -> int:
    if space_mode == "p0":
        if facet_dofs is not None:
            arr = np.asarray(facet_dofs, dtype=int)
            if arr.size == 0:
                return 0
            if np.any(arr < 0):
                raise ValueError("facet_dofs must be non-negative.")
            return int(arr.max()) + 1
        return int(n_facets) * int(value_dim)
    return int(n_nodes) * int(value_dim)


def _contact_space_side_n_dofs(space: "ContactSurfaceSpace", *, side: str) -> int:
    if side == "master":
        return _field_n_dofs(
            n_nodes=int(np.asarray(space.surface_master.coords).shape[0]),
            n_facets=int(np.asarray(space.surface_master.conn).shape[0]),
            value_dim=int(space.value_dim_master),
            space_mode=space.space_mode_master,
            facet_dofs=space.facet_dofs_master,
        )
    if side == "slave":
        return _field_n_dofs(
            n_nodes=int(np.asarray(space.surface_slave.coords).shape[0]),
            n_facets=int(np.asarray(space.surface_slave.conn).shape[0]),
            value_dim=int(space.value_dim_slave),
            space_mode=space.space_mode_slave,
            facet_dofs=space.facet_dofs_slave,
        )
    raise ValueError("side must be 'master' or 'slave'")


@dataclass(eq=False)
class OneToManyContactSurfaceSpace:
    """One-master/multi-slave wrapper built from pairwise ContactSurfaceSpace objects."""

    contacts: tuple[ContactSurfaceSpace, ...]
    field_master: str = "master"
    field_slave: str = "slave"

    @classmethod
    def from_meshes(
        cls,
        master_mesh: BaseMesh,
        slave_meshes: Sequence[BaseMesh],
        *,
        master_facets: np.ndarray | None = None,
        slave_facets_list: Sequence[np.ndarray] | None = None,
        master_facet_selector: Callable[[BaseMesh], np.ndarray] | None = None,
        slave_facet_selectors: Sequence[Callable[[BaseMesh], np.ndarray] | None] | Callable[[BaseMesh], np.ndarray] | None = None,
        master_space: object | None = None,
        slave_spaces: Sequence[object | None] | object | None = None,
        value_dim_master: int | None = None,
        value_dim_slaves: Sequence[int | None] | int | None = None,
        mode_master: str = "touching",
        mode_slave: str = "touching",
        field_master: str = "master",
        field_slave: str = "slave",
        quad_order: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        if len(slave_meshes) == 0:
            raise ValueError("slave_meshes must contain at least one mesh.")
        n_slaves = len(slave_meshes)

        if master_facets is None:
            if master_facet_selector is None:
                raise ValueError("Provide either master_facets or master_facet_selector.")
            master_facets = np.asarray(master_facet_selector(master_mesh), dtype=int)
        else:
            master_facets = np.asarray(master_facets, dtype=int)

        if slave_facets_list is None:
            if slave_facet_selectors is None:
                raise ValueError("Provide either slave_facets_list or slave_facet_selectors.")
            if callable(slave_facet_selectors):
                slave_facets_list = [np.asarray(slave_facet_selectors(mesh), dtype=int) for mesh in slave_meshes]
            else:
                if len(slave_facet_selectors) != n_slaves:
                    raise ValueError("slave_facet_selectors length must match slave_meshes length.")
                out_facets: list[np.ndarray] = []
                for mesh, sel in zip(slave_meshes, slave_facet_selectors):
                    if sel is None:
                        raise ValueError("slave_facet_selectors contains None; provide a selector for each slave.")
                    out_facets.append(np.asarray(sel(mesh), dtype=int))
                slave_facets_list = out_facets
        else:
            if len(slave_facets_list) != n_slaves:
                raise ValueError("slave_facets_list length must match slave_meshes length.")
            slave_facets_list = [np.asarray(facets, dtype=int) for facets in slave_facets_list]

        if slave_spaces is None:
            slave_spaces = [None] * n_slaves
        elif isinstance(slave_spaces, Sequence) and not isinstance(slave_spaces, (str, bytes)):
            if len(slave_spaces) != n_slaves:
                raise ValueError("slave_spaces length must match slave_meshes length.")
            slave_spaces = list(slave_spaces)
        else:
            slave_spaces = [slave_spaces] * n_slaves

        if value_dim_slaves is None:
            value_dim_slaves = [None] * n_slaves
        elif isinstance(value_dim_slaves, Sequence) and not isinstance(value_dim_slaves, (str, bytes)):
            if len(value_dim_slaves) != n_slaves:
                raise ValueError("value_dim_slaves length must match slave_meshes length.")
            value_dim_slaves = list(value_dim_slaves)
        else:
            value_dim_slaves = [int(value_dim_slaves)] * n_slaves

        master_side = ContactSide.from_facets(
            master_mesh,
            master_facets,
            master_space,
            value_dim=value_dim_master,
            mode=mode_master,
        )
        slave_sides = [
            ContactSide.from_facets(
                mesh,
                np.asarray(facets, dtype=int),
                space,
                value_dim=value_dim,
                mode=mode_slave,
            )
            for mesh, facets, space, value_dim in zip(slave_meshes, slave_facets_list, slave_spaces, value_dim_slaves)
        ]
        return cls.from_sides(
            master_side,
            slave_sides,
            field_master=field_master,
            field_slave=field_slave,
            quad_order=quad_order,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod
    def from_sides(
        cls,
        master: ContactSide,
        slaves: Sequence[ContactSide],
        *,
        field_master: str = "master",
        field_slave: str = "slave",
        quad_order: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        if len(slaves) == 0:
            raise ValueError("slaves must contain at least one ContactSide.")
        contacts = tuple(
            ContactSurfaceSpace.from_sides(
                master,
                slave,
                field_master=field_master,
                field_slave=field_slave,
                quad_order=quad_order,
                space_mode_master=space_mode_master,
                space_mode_slave=space_mode_slave,
                facet_dofs_master=facet_dofs_master,
                facet_dofs_slave=facet_dofs_slave,
                normal_sign=normal_sign,
                tol=tol,
                backend=backend,
                fd_eps=fd_eps,
                fd_mode=fd_mode,
                fd_block_size=fd_block_size,
                batch_jac=batch_jac,
                setup_cache_enabled=setup_cache_enabled,
                setup_cache_trace=setup_cache_trace,
            )
            for slave in slaves
        )
        return cls(contacts=contacts, field_master=field_master, field_slave=field_slave)

    @classmethod
    def from_surfaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slaves: Sequence[SurfaceMesh],
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slaves: Sequence[np.ndarray | None] | None = None,
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        field_master: str = "master",
        field_slave: str = "slave",
        quad_order: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "OneToManyContactSurfaceSpace":
        if len(surface_slaves) == 0:
            raise ValueError("surface_slaves must contain at least one surface.")
        if elem_conn_slaves is None:
            elem_conn_slaves = [None] * len(surface_slaves)
        if len(elem_conn_slaves) != len(surface_slaves):
            raise ValueError("elem_conn_slaves length must match surface_slaves length.")
        contacts = tuple(
            ContactSurfaceSpace.from_surfaces(
                surface_master,
                surface_slave,
                elem_conn_master=elem_conn_master,
                elem_conn_slave=elem_conn_slave,
                field_master=field_master,
                field_slave=field_slave,
                value_dim_master=value_dim_master,
                value_dim_slave=value_dim_slave,
                space_mode_master=space_mode_master,
                space_mode_slave=space_mode_slave,
                facet_dofs_master=facet_dofs_master,
                facet_dofs_slave=facet_dofs_slave,
                quad_order=quad_order,
                normal_sign=normal_sign,
                tol=tol,
                backend=backend,
                fd_eps=fd_eps,
                fd_mode=fd_mode,
                fd_block_size=fd_block_size,
                batch_jac=batch_jac,
                setup_cache_enabled=setup_cache_enabled,
                setup_cache_trace=setup_cache_trace,
            )
            for surface_slave, elem_conn_slave in zip(surface_slaves, elem_conn_slaves)
        )
        return cls(contacts=contacts, field_master=field_master, field_slave=field_slave)

    def _split_fields(
        self, u: Mapping[str, npt.ArrayLike] | Sequence[Any]
    ) -> tuple[npt.ArrayLike, list[npt.ArrayLike]]:
        if isinstance(u, Mapping):
            if self.field_master not in u:
                raise KeyError(f"u mapping must contain master field '{self.field_master}'.")
            if "slaves" not in u:
                raise KeyError("u mapping must contain key 'slaves' with per-slave states.")
            u_master = u[self.field_master]
            u_slaves = list(u["slaves"])
        else:
            if len(u) != 2:
                raise ValueError("u must be a mapping or a sequence like (u_master, u_slaves).")
            u_master = u[0]
            u_slaves = list(u[1])
        if len(u_slaves) != len(self.contacts):
            raise ValueError(
                f"u_slaves length mismatch: got {len(u_slaves)}, expected {len(self.contacts)}."
            )
        return u_master, u_slaves

    def _dof_layout(self) -> tuple[int, list[int], int]:
        if len(self.contacts) == 0:
            return 0, [], 0
        n_master = _contact_space_side_n_dofs(self.contacts[0], side="master")
        slave_sizes = [_contact_space_side_n_dofs(contact, side="slave") for contact in self.contacts]
        total = int(n_master + sum(slave_sizes))
        return n_master, slave_sizes, total

    @staticmethod
    def _scatter_pair_indices(local_idx: np.ndarray, *, n_master: int, slave_offset: int) -> np.ndarray:
        idx = np.asarray(local_idx, dtype=int)
        out = np.empty_like(idx)
        master_mask = idx < int(n_master)
        out[master_mask] = idx[master_mask]
        out[~master_mask] = int(n_master) + int(slave_offset) + (idx[~master_mask] - int(n_master))
        return out

    def assemble_residual(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any],
        params: "WeakParams",
        *,
        normal_source: str = "master",
    ) -> np.ndarray:
        u_master, u_slaves = self._split_fields(u)
        n_master, slave_sizes, n_total = self._dof_layout()
        R = np.zeros((n_total,), dtype=float)
        slave_offset = 0
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            r_local = np.asarray(
                contact.assemble_residual(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                ),
                dtype=float,
            )
            if r_local.shape[0] != n_master + n_slave:
                raise ValueError("Pair residual size mismatch while assembling one-to-many residual.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            R[idx] += r_local
            slave_offset += n_slave
        return R

    def assemble_jacobian(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any],
        params: "WeakParams",
        *,
        normal_source: str = "master",
        sparse: bool = False,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> ContactJacobianReturn:
        u_master, u_slaves = self._split_fields(u)
        n_master, slave_sizes, n_total = self._dof_layout()
        slave_offset = 0
        if sparse:
            rows_all: list[np.ndarray] = []
            cols_all: list[np.ndarray] = []
            data_all: list[np.ndarray] = []
            for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
                j_local = contact.assemble_jacobian(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                    sparse=True,
                    backend=backend,
                    batch_jac=batch_jac,
                )
                rows, cols, data, n_pair = j_local
                if int(n_pair) != n_master + n_slave:
                    raise ValueError("Pair Jacobian size mismatch while assembling sparse one-to-many Jacobian.")
                rows_all.append(
                    self._scatter_pair_indices(np.asarray(rows, dtype=int), n_master=n_master, slave_offset=slave_offset)
                )
                cols_all.append(
                    self._scatter_pair_indices(np.asarray(cols, dtype=int), n_master=n_master, slave_offset=slave_offset)
                )
                data_all.append(np.asarray(data, dtype=float))
                slave_offset += n_slave
            if rows_all:
                rows_out = np.concatenate(rows_all)
                cols_out = np.concatenate(cols_all)
                data_out = np.concatenate(data_all)
            else:
                rows_out = np.zeros((0,), dtype=int)
                cols_out = np.zeros((0,), dtype=int)
                data_out = np.zeros((0,), dtype=float)
            return rows_out, cols_out, data_out, n_total

        K = np.zeros((n_total, n_total), dtype=float)
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            j_local = np.asarray(
                contact.assemble_jacobian(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                    sparse=False,
                    backend=backend,
                    batch_jac=batch_jac,
                ),
                dtype=float,
            )
            if j_local.shape != (n_master + n_slave, n_master + n_slave):
                raise ValueError("Pair Jacobian shape mismatch while assembling dense one-to-many Jacobian.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            K[np.ix_(idx, idx)] += j_local
            slave_offset += n_slave
        return K

    def assemble_bilinear(
        self,
        bilin: Callable[..., Any],
        u_master: Mapping[str, npt.ArrayLike] | Sequence[Any] | npt.ArrayLike,
        u_slaves: Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = False,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        if params is None:
            if u_slaves is None:
                raise TypeError("params is required")
            if isinstance(u_master, Mapping) or (isinstance(u_master, Sequence) and not hasattr(u_master, "shape")):
                params = u_slaves  # type: ignore[assignment]
                u_master, u_slaves = self._split_fields(u_master)  # type: ignore[arg-type]
            else:
                raise TypeError("params is required")
        elif u_slaves is None:
            u_master, u_slaves = self._split_fields(u_master)  # type: ignore[arg-type]
        assert params is not None
        assert u_slaves is not None

        n_master, slave_sizes, n_total = self._dof_layout()
        slave_offset = 0
        if sparse:
            rows_all: list[np.ndarray] = []
            cols_all: list[np.ndarray] = []
            data_all: list[np.ndarray] = []
            for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
                j_local = contact.assemble_bilinear(
                    bilin,
                    u_master,
                    u_slave,
                    params,
                    sparse=True,
                    normal_source=normal_source,
                )
                rows, cols, data, n_pair = j_local
                if int(n_pair) != n_master + n_slave:
                    raise ValueError("Pair Jacobian size mismatch while assembling sparse one-to-many bilinear.")
                rows_all.append(
                    self._scatter_pair_indices(np.asarray(rows, dtype=int), n_master=n_master, slave_offset=slave_offset)
                )
                cols_all.append(
                    self._scatter_pair_indices(np.asarray(cols, dtype=int), n_master=n_master, slave_offset=slave_offset)
                )
                data_all.append(np.asarray(data, dtype=float))
                slave_offset += n_slave
            if rows_all:
                rows_out = np.concatenate(rows_all)
                cols_out = np.concatenate(cols_all)
                data_out = np.concatenate(data_all)
            else:
                rows_out = np.zeros((0,), dtype=int)
                cols_out = np.zeros((0,), dtype=int)
                data_out = np.zeros((0,), dtype=float)
            return rows_out, cols_out, data_out, n_total

        K = np.zeros((n_total, n_total), dtype=float)
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            j_local = np.asarray(
                contact.assemble_bilinear(
                    bilin,
                    u_master,
                    u_slave,
                    params,
                    sparse=False,
                    normal_source=normal_source,
                ),
                dtype=float,
            )
            if j_local.shape != (n_master + n_slave, n_master + n_slave):
                raise ValueError("Pair Jacobian shape mismatch while assembling dense one-to-many bilinear.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            K[np.ix_(idx, idx)] += j_local
            slave_offset += n_slave
        return K

    def assemble_contact_coupling_matrices(self):
        from .contact_interface import ContactCouplingMatrix

        n_master, slave_sizes, _ = self._dof_layout()
        n_slaves_total = int(sum(slave_sizes))
        rows_mm: list[np.ndarray] = []
        cols_mm: list[np.ndarray] = []
        data_mm: list[np.ndarray] = []
        rows_ms: list[np.ndarray] = []
        cols_ms: list[np.ndarray] = []
        data_ms: list[np.ndarray] = []

        slave_offset = 0
        for contact, n_slave in zip(self.contacts, slave_sizes):
            m_mm, m_ms_local = contact.assemble_contact_coupling_matrices()
            rows_mm.append(np.asarray(m_mm.rows, dtype=int))
            cols_mm.append(np.asarray(m_mm.cols, dtype=int))
            data_mm.append(np.asarray(m_mm.data, dtype=float))
            rows_ms.append(np.asarray(m_ms_local.rows, dtype=int))
            cols_ms.append(np.asarray(m_ms_local.cols, dtype=int) + slave_offset)
            data_ms.append(np.asarray(m_ms_local.data, dtype=float))
            if m_mm.shape != (n_master, n_master):
                raise ValueError("Pair M_aa shape mismatch while assembling one-to-many coupling matrices.")
            if m_ms_local.shape != (n_master, n_slave):
                raise ValueError("Pair M_ab shape mismatch while assembling one-to-many coupling matrices.")
            slave_offset += n_slave

        mm = ContactCouplingMatrix(
            rows=np.concatenate(rows_mm) if rows_mm else np.zeros((0,), dtype=int),
            cols=np.concatenate(cols_mm) if cols_mm else np.zeros((0,), dtype=int),
            data=np.concatenate(data_mm) if data_mm else np.zeros((0,), dtype=float),
            shape=(n_master, n_master),
        )
        ms = ContactCouplingMatrix(
            rows=np.concatenate(rows_ms) if rows_ms else np.zeros((0,), dtype=int),
            cols=np.concatenate(cols_ms) if cols_ms else np.zeros((0,), dtype=int),
            data=np.concatenate(data_ms) if data_ms else np.zeros((0,), dtype=float),
            shape=(n_master, n_slaves_total),
        )
        return mm, ms

    def assemble_contact_kkt(
        self,
        *,
        rho: float = 0.0,
        multiplier_space: str = "nodal",
        backend: str = "numpy",
        format: str = "fluxsparse",
        return_blocks: bool = False,
    ):
        m_aa, m_ab = self.assemble_contact_coupling_matrices()
        master_facets = np.asarray(self.contacts[0].surface_master.conn, dtype=int)
        return assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=rho,
            multiplier_space=multiplier_space,
            facet_conn_master=master_facets,
            backend=backend,
            format=format,
            return_blocks=return_blocks,
        )

    def assemble_contact_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier_space: str = "nodal",
        backend: str = "numpy",
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_constraint_operators(
            self,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier_space=multiplier_space,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str = "numpy",
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        return assemble_contact_penalty_operators(
            self,
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )


__all__ = [
    "ContactSide",
    "OneSidedContact",
    "OneSidedContactSurfaceSpace",
    "ContactSurfaceSpace",
    "OneToManyContactSurfaceSpace",
    "ContactOperators",
    "ContactKKTSolveConfig",
    "EmbeddingMap",
    "build_nodal_embedding_map",
    "build_barycentric_embedding_map",
    "build_barycentric_embedding_map_from_meshes",
    "assemble_embedding_constraint_matrix",
    "assemble_rbe2_constraint_matrix",
    "assemble_contact_constraint_operators",
    "assemble_contact_penalty_operators",
    "assemble_contact_interface_residual",
    "assemble_contact_interface_jacobian",
    "assemble_contact_coupling_matrices",
    "assemble_contact_kkt",
    "solve_contact_kkt",
    "facet_gap_values",
    "active_contact_facets",
]
