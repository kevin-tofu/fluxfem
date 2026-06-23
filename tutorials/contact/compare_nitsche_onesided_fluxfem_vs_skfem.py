import os
import importlib.util
import sys
from pathlib import Path

import numpy as np

import fluxfem as ff

TUTORIALS_ROOT = Path(__file__).resolve().parents[1]
if str(TUTORIALS_ROOT) not in sys.path:
    sys.path.insert(0, str(TUTORIALS_ROOT))

from common.contact_compare_utils import (
    build_fluxfem_onesided_contact_space,
    fluxfem_mesh_for,
    tet4_coords,
)


def _u_hat_fn(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.stack(
        [
            0.1 + 0.2 * x[:, 0],
            -0.05 + 0.1 * x[:, 1],
            0.02 + 0.15 * x[:, 2],
        ],
        axis=1,
    )


def build_fluxfem_onesided(
    elem: str,
    *,
    alpha: float,
    quad_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords, conn, contact_space = build_fluxfem_onesided_contact_space(
        elem,
        quad_order=quad_order,
        with_master=True,
    )
    E, nu = 210e9, 0.3
    lam, mu = ff.lame_parameters(E, nu)
    params = ff.Params(lam=float(lam), mu=float(mu), alpha=float(alpha))
    u_master = _u_hat_fn(coords).reshape(-1)

    class _OneSidedAdapter:
        def __init__(self, cs, uhat, u_master_vec):
            self._cs = cs
            self._uhat = uhat
            self._u_master = u_master_vec
            self._cached = None

        def _assemble(self, params_in):
            if self._cached is None:
                self._cached = self._cs.assemble_bilinear(
                    self._uhat,
                    params_in,
                    u_master=self._u_master,
                )
            return self._cached

        def assemble_residual(self, _res_form, _u, params_in, *, normal_source="master"):
            _ = normal_source
            _K, f = self._assemble(params_in)
            return np.asarray(f)

        def assemble_jacobian(
            self,
            _res_form,
            _u,
            params_in,
            *,
            normal_source="master",
            sparse=False,
            backend="numpy",
            batch_jac=None,
        ):
            _ = (normal_source, sparse, backend, batch_jac)
            K, _f = self._assemble(params_in)
            return np.asarray(K)

    def _dummy_res_form(ctx, u, p):
        _ = (ctx, u, p)
        return {"a": np.array([0.0])}

    ops = ff.assemble_contact_operators(
        _OneSidedAdapter(contact_space, _u_hat_fn, u_master),
        enforcement="penalty",
        weak_form=_dummy_res_form,
        state={"a": np.zeros(int(contact_space.surface_slave.n_nodes * contact_space.value_dim), dtype=float)},
        params=params,
        backend="jax",
    )
    return np.asarray(ops.jacobian), np.asarray(ops.residual), coords


def _vector_perm_for_skfem(coords_ff: np.ndarray, doflocs_sf: np.ndarray, value_dim: int) -> np.ndarray:
    coords_ff = np.asarray(coords_ff)
    doflocs_sf = np.asarray(doflocs_sf)
    if doflocs_sf.shape[0] == 3 and doflocs_sf.shape[1] != 3:
        doflocs_sf = doflocs_sf.T
    perm_nodes = np.empty(coords_ff.shape[0], dtype=int)
    for i, c in enumerate(coords_ff):
        matches = np.nonzero(np.all(np.isclose(doflocs_sf, c, atol=1e-8), axis=1))[0]
        if len(matches) != 1:
            raise ValueError("could not match dofloc to fluxfem coord")
        perm_nodes[i] = matches[0]
    perm_vec = np.array([perm_nodes[node] * value_dim + comp for node in range(len(perm_nodes)) for comp in range(value_dim)], dtype=int)
    return perm_vec


def _vector_perm_from_mesh(coords_ff: np.ndarray, mesh_coords: np.ndarray, nodal_dofs: np.ndarray, value_dim: int) -> np.ndarray:
    coords_ff = np.asarray(coords_ff)
    mesh_coords = np.asarray(mesh_coords)
    perm_nodes = np.empty(coords_ff.shape[0], dtype=int)
    for i, c in enumerate(coords_ff):
        matches = np.nonzero(np.all(np.isclose(mesh_coords, c, atol=1e-8), axis=1))[0]
        if len(matches) != 1:
            raise ValueError("could not match node coord to skfem mesh")
        perm_nodes[i] = matches[0]
    perm_vec = np.array(
        [nodal_dofs[comp, perm_nodes[node]] for node in range(len(perm_nodes)) for comp in range(value_dim)],
        dtype=int,
    )
    return perm_vec


def _node_perm_from_doflocs(coords_ff: np.ndarray, doflocs_sf: np.ndarray) -> np.ndarray:
    coords_ff = np.asarray(coords_ff)
    doflocs_sf = np.asarray(doflocs_sf)
    if doflocs_sf.shape[0] == 3 and doflocs_sf.shape[1] != 3:
        doflocs_sf = doflocs_sf.T
    perm_nodes = np.empty(coords_ff.shape[0], dtype=int)
    for i, c in enumerate(coords_ff):
        matches = np.nonzero(np.all(np.isclose(doflocs_sf, c, atol=1e-8), axis=1))[0]
        if len(matches) != 1:
            raise ValueError("could not match node coord to scalar doflocs")
        perm_nodes[i] = matches[0]
    return perm_nodes


def _vector_perm_from_doflocs(coords_ff: np.ndarray, doflocs_vec: np.ndarray, value_dim: int) -> np.ndarray:
    coords_ff = np.asarray(coords_ff)
    doflocs_vec = np.asarray(doflocs_vec)
    if doflocs_vec.shape[0] == 3 and doflocs_vec.shape[1] != 3:
        doflocs_vec = doflocs_vec.T
    comp = np.arange(doflocs_vec.shape[0]) % value_dim
    mapping = {}
    for idx, (pt, c) in enumerate(zip(doflocs_vec, comp)):
        key = tuple(np.round(pt, 12)) + (int(c),)
        mapping[key] = idx
    perm_vec = []
    for node in range(coords_ff.shape[0]):
        key_base = tuple(np.round(coords_ff[node], 12))
        for c in range(value_dim):
            key = key_base + (c,)
            if key not in mapping:
                raise ValueError("could not match vector dofloc to fluxfem coord")
            perm_vec.append(mapping[key])
    return np.asarray(perm_vec, dtype=int)


def build_skfem_onesided(elem: str, *, alpha: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if importlib.util.find_spec("skfem") is None:
        return None, None

    import skfem
    from skfem import MeshHex, MeshTet
    from skfem import FacetBasis, ElementVectorH1
    from skfem.helpers import dot, sym_grad, mul
    from skfem.supermeshing import intersect, elementwise_quadrature
    from skfem.models.elasticity import lame_parameters, linear_stress
    try:
        from skfem import ElementHex1, ElementHex2, ElementTetP1, ElementTetP2
    except Exception:
        from skfem.element import ElementHex1, ElementHex2, ElementTetP1, ElementTetP2

    if elem == "hex8":
        xs = np.linspace(0.0, 1.0, 2)
        ys = np.linspace(0.0, 1.0, 2)
        zs = np.linspace(0.0, 1.0, 2)
        mesh_a = MeshHex().init_tensor(xs, ys, zs)
        mesh_b = MeshHex().init_tensor(xs, ys, zs)
        elem_s = ElementHex1()
        trace_type = skfem.MeshQuad
    elif elem == "hex27":
        xs = np.linspace(0.0, 1.0, 2)
        ys = np.linspace(0.0, 1.0, 2)
        zs = np.linspace(0.0, 1.0, 2)
        mesh_a = MeshHex().init_tensor(xs, ys, zs)
        mesh_b = MeshHex().init_tensor(xs, ys, zs)
        elem_s = ElementHex2()
        trace_type = skfem.MeshQuad
    elif elem == "tet4":
        coords = tet4_coords()
        conn = np.array([[0, 1, 2, 3]], dtype=int)
        mesh_a = MeshTet(coords.T, conn.T)
        mesh_b = MeshTet(coords.T, conn.T)
        elem_s = ElementTetP1()
        trace_type = skfem.MeshTri
    elif elem == "tet10":
        coords = tet4_coords()
        conn = np.array([[0, 1, 2, 3]], dtype=int)
        mesh_a = MeshTet(coords.T, conn.T)
        mesh_b = MeshTet(coords.T, conn.T)
        elem_s = ElementTetP2()
        trace_type = skfem.MeshTri
    else:
        raise ValueError(f"unsupported element: {elem}")

    def is_contact_surface(x):
        return np.isclose(x[2], 0.0)

    mesh_a = mesh_a.with_boundaries({"contact": is_contact_surface})
    mesh_b = mesh_b.with_boundaries({"contact": is_contact_surface})
    m1t, orig1 = mesh_a.trace("contact", mtype=trace_type, project=lambda p: p[[0, 1]])
    m2t, orig2 = mesh_b.trace("contact", mtype=trace_type, project=lambda p: p[[0, 1]])
    m12, t1, t2 = intersect(m1t, m2t)
    quad_order = int(os.getenv("QUAD_ORDER", "5"))
    try:
        quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
        quad2 = elementwise_quadrature(m2t, m12, t2, intorder=quad_order)
    except TypeError:
        quad1 = elementwise_quadrature(m1t, m12, t1)
        quad2 = elementwise_quadrature(m2t, m12, t2)

    elem_v = ElementVectorH1(elem_s)
    basis_scalar_a = skfem.Basis(mesh_a, elem_s)
    basis_scalar_b = skfem.Basis(mesh_b, elem_s)
    basis_vec_a = skfem.Basis(mesh_a, elem_v)
    basis_vec_b = skfem.Basis(mesh_b, elem_v)
    fb_u_top = FacetBasis(mesh_a, elem_v, facets=orig1[t1], quadrature=quad1)
    fb_u_bot = FacetBasis(mesh_b, elem_v, facets=orig2[t2], quadrature=quad2)

    E, nu = 210e9, 0.3
    lam, mu = lame_parameters(E, nu)
    C = linear_stress(lam, mu)
    disable_consistency = os.getenv("FF_ONESIDED_DISABLE_CONSISTENCY", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_symmetry = os.getenv("FF_ONESIDED_DISABLE_SYMMETRY", "").strip().lower() in {"1", "true", "yes", "on"}
    disable_penalty = os.getenv("FF_ONESIDED_DISABLE_PENALTY", "").strip().lower() in {"1", "true", "yes", "on"}

    @skfem.BilinearForm
    def bilin(u, v, w):
        t_u = mul(C(sym_grad(u)), w.n)
        t_v = mul(C(sym_grad(v)), w.n)
        expr = 0.0
        if not disable_consistency:
            expr = expr - dot(v, t_u)
        if not disable_symmetry:
            expr = expr - dot(t_v, u)
        if not disable_penalty:
            expr = expr + (alpha / w.h) * dot(v, u)
        return expr

    @skfem.LinearForm
    def lin(v, w):
        t_v = mul(C(sym_grad(v)), w.n)
        expr = 0.0
        if not disable_symmetry:
            expr = expr + dot(t_v, w.u_hat)
        if not disable_penalty:
            expr = expr - (alpha / w.h) * dot(v, w.u_hat)
        return expr

    coords_ff, _conn_ff, _order = fluxfem_mesh_for(elem)
    doflocs_vec = np.asarray(basis_vec_a.doflocs).T
    u_vals = _u_hat_fn(doflocs_vec)
    u_master = u_vals[np.arange(doflocs_vec.shape[0]), np.arange(doflocs_vec.shape[0]) % 3]
    u_hat_vals = fb_u_top.interpolate(u_master)

    params = {"h": fb_u_bot.mesh_parameters(), "u_hat": u_hat_vals}
    K = skfem.asm(bilin, fb_u_bot, **params)
    f = skfem.asm(lin, fb_u_bot, **params)

    perm_vec = _vector_perm_from_doflocs(coords_ff, np.asarray(basis_vec_b.doflocs), 3)
    K_np = K.toarray()
    K_np = K_np[np.ix_(perm_vec, perm_vec)]
    f_np = np.asarray(f)[perm_vec]
    return K_np, f_np


def _rel_norm_gap(a: np.ndarray, b: np.ndarray) -> float:
    return abs(float(np.linalg.norm(a)) - float(np.linalg.norm(b))) / max(1.0, abs(float(np.linalg.norm(b))))


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


if __name__ == "__main__":
    elem = os.getenv("ELEM", "hex27").strip().lower()
    quad_order = int(os.getenv("QUAD_ORDER", "4"))
    alpha = float(os.getenv("ALPHA", "10.0"))

    K_ff, f_ff, _coords = build_fluxfem_onesided(
        elem,
        alpha=alpha,
        quad_order=quad_order,
    )
    K_sf, f_sf = build_skfem_onesided(elem, alpha=alpha)
    if K_sf is None:
        print("[compare] skfem not installed; skipping")
        raise SystemExit(0)

    norm_gap_K = _rel_norm_gap(K_ff, K_sf)
    norm_gap_f = _rel_norm_gap(f_ff, f_sf)
    rel_diff_K = _rel_diff(K_ff, K_sf)
    rel_diff_f = _rel_diff(f_ff, f_sf)
    rel_diff_K_flipped = _rel_diff(K_ff, -K_sf)
    rel_diff_f_flipped = _rel_diff(f_ff, -f_sf)
    diff_K = float(np.linalg.norm(K_ff - K_sf))
    diff_f = float(np.linalg.norm(f_ff - f_sf))
    diff_K_flipped = float(np.linalg.norm(K_ff + K_sf))
    diff_f_flipped = float(np.linalg.norm(f_ff + f_sf))

    print(
        f"[onesided][{elem}] raw_rel_diff_K={rel_diff_K:.3e} raw_rel_diff_f={rel_diff_f:.3e} "
        f"sign_aligned_rel_diff_K={rel_diff_K_flipped:.3e} sign_aligned_rel_diff_f={rel_diff_f_flipped:.3e} "
        f"norm_gap_K={norm_gap_K:.3e} norm_gap_f={norm_gap_f:.3e} "
        f"raw_diff_K={diff_K:.3e} raw_diff_f={diff_f:.3e} "
        f"sign_aligned_diff_K={diff_K_flipped:.3e} sign_aligned_diff_f={diff_f_flipped:.3e}"
    )
    if rel_diff_K_flipped < 1e-8 and rel_diff_f_flipped < 1e-8:
        print(
            f"[onesided][{elem}] note: FluxFEM and scikit-fem match after sign alignment. "
            "The remaining raw sign difference is due to residual/jacobian vs bilinear/linear-form conventions."
        )
    elif rel_diff_f_flipped < 1e-8 and rel_diff_K_flipped > 1e-2:
        print(
            f"[onesided][{elem}] note: RHS matches after sign flip, but stiffness still differs after sign alignment; "
            "remaining mismatch is in the bilinear terms, not in the prescribed-displacement loading."
        )
