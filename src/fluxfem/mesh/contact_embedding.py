from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import numpy.typing as npt

from .base import BaseMesh
from .contact_interface import volume_shape_values_at_points as _volume_shape_values_at_points
from .surface import SurfaceMesh


def _is_jax_like(x: Any) -> bool:
    try:
        import jax
    except Exception:
        return False
    return isinstance(x, jax.Array) or isinstance(x, jax.core.Tracer)


def _contains_jax_value(obj: Any) -> bool:
    if _is_jax_like(obj):
        return True
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(obj, Mapping):
        return any(_contains_jax_value(v) for v in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return any(_contains_jax_value(v) for v in obj)
    data = getattr(obj, "data", None)
    if data is not None and not isinstance(obj, np.ndarray) and data is not obj and _contains_jax_value(data):
        return True
    return False


def _infer_backend(*values: Any, default: str) -> str:
    return "jax" if any(_contains_jax_value(v) for v in values) else default


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
    backend: str | None = None,
):
    """
    Assemble ``C`` for equality constraints ``W*u_master - u_slave = 0``.

    Returns matrix with shape ``(n_slave_nodes*value_dim, (n_master_nodes+n_slave_nodes)*value_dim)``.
    """
    backend = _infer_backend(embedding, default="numpy") if backend is None else str(backend).lower()
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
    slave_components: Sequence[int] | None = None,
    backend: str | None = None,
):
    """
    Assemble 3D RBE2-style rigid kinematic constraints.

    Unknown ordering:
      q = [u_ref(3), omega_ref(3), u_slave_0(3), ..., u_slave_{n-1}(3)]

    Constraint for each slave node i:
      u_slave_i - u_ref - (omega_ref x (x_i - x_ref)) = 0

    ``slave_components`` selects constrained dependent slave translation
    components from ``[Tx, Ty, Tz]``. The default constrains all three
    components at every slave node.
    """
    backend = "numpy" if backend is None else str(backend).lower()
    if backend != "numpy":
        raise ValueError("RBE2 constraint assembly currently supports backend='numpy' only.")
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape[0] != 3:
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")

    n_s = int(x_s.shape[0])
    n_rows = 3 * n_s
    n_cols = 6 + 3 * n_s
    if slave_components is None:
        comps = np.arange(3, dtype=np.int64)
    else:
        comps = np.asarray(tuple(slave_components), dtype=np.int64).reshape(-1)
    if comps.size == 0:
        raise ValueError("slave_components must contain at least one component.")
    if np.any(comps < 0) or np.any(comps >= 3) or np.unique(comps).size != comps.size:
        raise ValueError("slave_components must be unique integers in [0, 2].")

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
    rows = np.asarray([3 * i + int(c) for i in range(n_s) for c in comps], dtype=np.int64)
    return C[rows, :]


def assemble_fixed_rigid_hub_constraint_matrix(
    ref_point: np.ndarray,
    hub_coords: np.ndarray,
    hub_dofs: np.ndarray | None = None,
    *,
    n_structural_dofs: int | None = None,
    backend: str | None = None,
):
    """
    Assemble structural-only constraints for a rigid hub fixed in 6 DOFs.

    This is the fixed-reference specialization of an RBE2 coupling:

      u_i - u_ref - omega_ref x (x_i - x_ref) = 0

    with ``u_ref = 0`` and ``omega_ref = 0``.  The returned matrix therefore
    has only structural DOF columns and can be used directly as ``C u = 0``.

    Parameters
    ----------
    ref_point:
        Hub reference point, shape ``(3,)``.
    hub_coords:
        Coordinates of bore/hub nodes rigidly attached to the fixed hub,
        shape ``(n_hub_nodes, 3)``.
    hub_dofs:
        Optional global structural DOF ids for each hub node, shape
        ``(n_hub_nodes, 3)``.  If omitted, local node-major ordering is used.
    n_structural_dofs:
        Total structural DOF count.  Required when ``hub_dofs`` is passed and
        larger than the inferred maximum is desired.

    Returns
    -------
    numpy.ndarray
        Constraint matrix with shape ``(3*n_hub_nodes, n_structural_dofs)``.
        For a fixed rigid hub this matrix enforces zero displacement at every
        attached hub node, which is equivalent to an RBE2 hub whose 6 reference
        DOFs are fixed.
    """
    backend = "numpy" if backend is None else str(backend).lower()
    if backend != "numpy":
        raise ValueError("fixed rigid hub constraint assembly currently supports backend='numpy' only.")
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_hub = np.asarray(hub_coords, dtype=float)
    if x_ref.shape != (3,):
        raise ValueError("ref_point must have shape (3,).")
    if x_hub.ndim != 2 or x_hub.shape[1] != 3:
        raise ValueError("hub_coords must have shape (n_hub_nodes, 3).")
    if x_hub.shape[0] == 0:
        raise ValueError("hub_coords must contain at least one node.")

    if hub_dofs is None:
        dofs = np.arange(3 * x_hub.shape[0], dtype=np.int64).reshape(-1, 3)
    else:
        dofs = np.asarray(hub_dofs, dtype=np.int64)
        if dofs.shape != (x_hub.shape[0], 3):
            raise ValueError("hub_dofs must have shape (n_hub_nodes, 3).")
        if np.any(dofs < 0):
            raise ValueError("hub_dofs must be non-negative.")

    inferred = int(dofs.max()) + 1
    n_cols = inferred if n_structural_dofs is None else int(n_structural_dofs)
    if n_cols < inferred:
        raise ValueError("n_structural_dofs is smaller than the largest hub_dofs entry.")

    c = np.zeros((3 * x_hub.shape[0], n_cols), dtype=float)
    for i in range(x_hub.shape[0]):
        rows = slice(3 * i, 3 * i + 3)
        c[rows, dofs[i]] = np.eye(3)
    return c


def assemble_rigid_hub_constraint_matrix(
    ref_point: np.ndarray,
    hub_coords: np.ndarray,
    hub_dofs: np.ndarray,
    *,
    hub_reference_dofs: np.ndarray | None = None,
    n_total_dofs: int | None = None,
    backend: str | None = None,
):
    """
    Assemble RBE2 constraints between structural hub nodes and a 6-DOF hub.

    The global unknown ordering is arbitrary and specified by ``hub_dofs`` and
    ``hub_reference_dofs``.  The constraints enforce

      u_i - u_ref - omega_ref x (x_i - x_ref) = 0

    for each attached hub node.
    """
    backend = "numpy" if backend is None else str(backend).lower()
    if backend != "numpy":
        raise ValueError("rigid hub constraint assembly currently supports backend='numpy' only.")
    x_hub = np.asarray(hub_coords, dtype=float)
    dofs = np.asarray(hub_dofs, dtype=np.int64)
    if x_hub.ndim != 2 or x_hub.shape[1] != 3:
        raise ValueError("hub_coords must have shape (n_hub_nodes, 3).")
    if dofs.shape != (x_hub.shape[0], 3):
        raise ValueError("hub_dofs must have shape (n_hub_nodes, 3).")
    if np.any(dofs < 0):
        raise ValueError("hub_dofs must be non-negative.")
    if hub_reference_dofs is None:
        inferred_structural = int(dofs.max()) + 1
        ref_dofs = np.arange(inferred_structural, inferred_structural + 6, dtype=np.int64)
    else:
        ref_dofs = np.asarray(hub_reference_dofs, dtype=np.int64).reshape(-1)
        if ref_dofs.shape != (6,):
            raise ValueError("hub_reference_dofs must have shape (6,).")
        if np.any(ref_dofs < 0):
            raise ValueError("hub_reference_dofs must be non-negative.")

    inferred = int(max(int(dofs.max()), int(ref_dofs.max()))) + 1
    n_cols = inferred if n_total_dofs is None else int(n_total_dofs)
    if n_cols < inferred:
        raise ValueError("n_total_dofs is smaller than the largest referenced DOF.")

    local = assemble_rbe2_constraint_matrix(ref_point, x_hub, backend=backend)
    c = np.zeros((local.shape[0], n_cols), dtype=float)
    c[:, ref_dofs] = local[:, :6]
    for i in range(x_hub.shape[0]):
        c[:, dofs[i]] += local[:, 6 + 3 * i : 6 + 3 * i + 3]
    return c


def assemble_rbe3_constraint_matrix(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    normalize_weights: bool = True,
    dependent_components: Sequence[int] | None = None,
    slave_components: Sequence[int] | None = None,
    backend: str | None = None,
):
    """
    Assemble a weighted 3D RBE3-style distributed-coupling constraint.

    Unknown ordering:
      q = [u_ref(3), omega_ref(3), u_slave_0(3), ..., u_slave_{n-1}(3)]

    The constraints are formed from weighted rigid-body reconstruction in normal-
    equation form:

      (sum_i w_i B_i^T B_i) q_ref - sum_i w_i B_i^T u_i = 0

    where ``B_i = [I, -[r_i]_x]`` and ``r_i = x_i - x_ref``.

    ``dependent_components`` selects the reference-point components to constrain
    from ``[Tx, Ty, Tz, Rx, Ry, Rz]``. ``slave_components`` selects the
    independent nodal translation components from ``[Tx, Ty, Tz]`` that
    contribute to the weighted fit. This is closer to Nastran-style RBE3
    component selection, but remains a weighted least-squares remote
    reconstruction rather than a full RBE3 card parser. It yields
    ``len(dependent_components) x (6 + 3*n_slave)`` rows.
    """
    backend = "numpy" if backend is None else str(backend).lower()
    if backend != "numpy":
        raise ValueError("RBE3 constraint assembly currently supports backend='numpy' only.")
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape[0] != 3:
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")

    n_s = int(x_s.shape[0])
    if n_s == 0:
        raise ValueError("slave_coords must contain at least one node.")

    if weights is None:
        w = np.ones((n_s,), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (n_s,):
            raise ValueError("weights must have shape (n_slave,).")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite.")
    if normalize_weights:
        w_sum = float(np.sum(w))
        if abs(w_sum) <= 1e-15:
            raise ValueError("weights sum must be non-zero when normalize_weights=True.")
        w = w / w_sum

    if dependent_components is None:
        dep = np.arange(6, dtype=np.int64)
    else:
        dep = np.asarray(tuple(dependent_components), dtype=np.int64).reshape(-1)
    if slave_components is None:
        comps = np.arange(3, dtype=np.int64)
    else:
        comps = np.asarray(tuple(slave_components), dtype=np.int64).reshape(-1)
    if dep.size == 0:
        raise ValueError("dependent_components must contain at least one component.")
    if comps.size == 0:
        raise ValueError("slave_components must contain at least one component.")
    if np.any(dep < 0) or np.any(dep >= 6) or np.unique(dep).size != dep.size:
        raise ValueError("dependent_components must be unique integers in [0, 5].")
    if np.any(comps < 0) or np.any(comps >= 3) or np.unique(comps).size != comps.size:
        raise ValueError("slave_components must be unique integers in [0, 2].")

    def _bmat(point: np.ndarray) -> np.ndarray:
        rx, ry, rz = (point - x_ref).tolist()
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0, rz, -ry],
                [0.0, 1.0, 0.0, -rz, 0.0, rx],
                [0.0, 0.0, 1.0, ry, -rx, 0.0],
            ],
            dtype=float,
        )

    M = np.zeros((dep.size, dep.size), dtype=float)
    slave_blocks = []
    for wi, xi in zip(w.tolist(), x_s):
        Bi = _bmat(xi)
        Bic = Bi[comps, :]
        Bid = Bic[:, dep]
        M += float(wi) * (Bid.T @ Bid)
        slave_block = np.zeros((dep.size, 3), dtype=float)
        slave_block[:, comps] = -float(wi) * Bid.T
        slave_blocks.append(slave_block)

    n_cols = 6 + 3 * n_s
    C = np.zeros((dep.size, n_cols), dtype=float)
    C[:, dep] = M
    for i, blk in enumerate(slave_blocks):
        c0 = 6 + 3 * i
        C[:, c0 : c0 + 3] = blk
    return C


def build_rbe3_weights(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    method: str = "equal",
    surface: SurfaceMesh | None = None,
    power: float = 2.0,
    eps: float = 1e-12,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build convenience weights for RBE3-style interpolation.

    Supported methods
    -----------------
    ``equal``:
        Uniform node weights.
    ``distance``:
        Inverse-distance^power weights from the remote point.
    ``facet_area``:
        Lump each facet area equally to its nodes, then normalize per node.
        Requires ``surface`` whose node numbering matches ``slave_coords`` order.
    """
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape != (3,):
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")
    n_s = int(x_s.shape[0])
    if n_s == 0:
        raise ValueError("slave_coords must contain at least one node.")

    method_key = str(method).lower()
    if method_key == "equal":
        w = np.ones((n_s,), dtype=float)
    elif method_key == "distance":
        d = np.linalg.norm(x_s - x_ref[None, :], axis=1)
        w = 1.0 / np.maximum(d, float(eps)) ** float(power)
    elif method_key == "facet_area":
        if surface is None:
            raise ValueError("surface is required for method='facet_area'.")
        facets = np.asarray(surface.conn, dtype=int)
        areas = np.asarray(surface.facet_areas(), dtype=float)
        if facets.shape[0] != areas.shape[0]:
            raise ValueError("surface facet count and facet areas mismatch.")
        w = np.zeros((n_s,), dtype=float)
        for nodes, area in zip(facets, areas):
            if np.any(nodes < 0) or np.any(nodes >= n_s):
                raise ValueError("surface facets must index slave_coords in local node numbering.")
            share = float(area) / float(len(nodes))
            for node in nodes:
                w[int(node)] += share
    else:
        raise ValueError("method must be one of: equal, distance, facet_area.")

    if normalize:
        s = float(np.sum(w))
        if abs(s) <= float(eps):
            raise ValueError("weight sum is zero; cannot normalize.")
        w = w / s
    return w


def build_rbe3_remote_resultant(
    ref_point: np.ndarray,
    slave_coords: np.ndarray,
    *,
    surface: SurfaceMesh,
    load: npt.ArrayLike | None = None,
    pressure: float | npt.ArrayLike | None = None,
    outward_from: npt.ArrayLike | None = None,
) -> np.ndarray:
    """
    Build the equivalent remote-point resultant for an RBE3-supported surface.

    The returned 6-vector is ordered as ``[force(3), moment(3)]`` and is
    compatible with a 6-DOF remote-point field ordered as
    ``[u_ref(3), omega_ref(3)]``.

    Exactly one of ``load`` or ``pressure`` must be provided:

    - ``load``: constant vector load per unit area with shape ``(3,)`` or
      ``(n_facets, 3)``
    - ``pressure``: scalar normal traction with shape ``()`` or ``(n_facets,)``
    """
    x_ref = np.asarray(ref_point, dtype=float).reshape(-1)
    x_s = np.asarray(slave_coords, dtype=float)
    if x_ref.shape != (3,):
        raise ValueError("ref_point must be 3D.")
    if x_s.ndim != 2 or x_s.shape[1] != 3:
        raise ValueError("slave_coords must have shape (n_slave, 3).")
    if surface is None:
        raise ValueError("surface is required.")
    if (load is None) == (pressure is None):
        raise ValueError("Specify exactly one of load or pressure.")

    n_s = int(x_s.shape[0])
    facets = np.asarray(surface.conn, dtype=int)
    if np.any(facets < 0) or np.any(facets >= n_s):
        raise ValueError("surface facets must index slave_coords in local node numbering.")

    if load is not None:
        nodal_load = surface.assemble_load(load, dim=3, n_total_nodes=n_s)
    else:
        from ..solver.bc import assemble_surface_traction

        nodal_load = assemble_surface_traction(
            surface,
            pressure,
            dim=3,
            n_total_nodes=n_s,
            outward_from=outward_from,
        )

    nodal_load = np.asarray(nodal_load, dtype=float).reshape(n_s, 3)
    force = np.sum(nodal_load, axis=0)
    arm = x_s - x_ref[None, :]
    moment = np.sum(np.cross(arm, nodal_load), axis=0)
    return np.concatenate([force, moment], axis=0)

