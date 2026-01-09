import os
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum as wf_einsum

jax.config.update("jax_enable_x64", True)


def _build_hex_facets(conn: np.ndarray, order: int) -> np.ndarray:
    elem = conn[0]
    if order == 1:
        pattern = (0, 1, 2, 3)
    elif order == 2:
        pattern = (0, 8, 1, 9, 2, 10, 3, 11)
    elif order == 3:
        pattern = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    else:
        raise ValueError("order must be 1, 2, or 3")
    return np.array([[int(elem[i]) for i in pattern]], dtype=int)


def _build_tet_facets(conn: np.ndarray, order: int) -> np.ndarray:
    elem = conn[0]
    if order == 1:
        pattern = (0, 1, 2)
    elif order == 2:
        pattern = (0, 1, 2)
    else:
        raise ValueError("order must be 1 or 2")
    return np.array([[int(elem[i]) for i in pattern]], dtype=int)


def _tet4_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _tet10_coords() -> np.ndarray:
    p = _tet4_coords()
    n0, n1, n2, n3 = p
    n01 = 0.5 * (n0 + n1)
    n12 = 0.5 * (n1 + n2)
    n02 = 0.5 * (n0 + n2)
    n03 = 0.5 * (n0 + n3)
    n13 = 0.5 * (n1 + n3)
    n23 = 0.5 * (n2 + n3)
    return np.array([n0, n1, n2, n3, n01, n12, n02, n03, n13, n23], dtype=float)


def _fluxfem_mesh_for(elem: str) -> tuple[np.ndarray, np.ndarray, int]:
    if elem == "hex8":
        mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=1).build()
        return np.asarray(mesh.coords, dtype=float), np.asarray(mesh.conn, dtype=int), 1
    if elem == "hex27":
        mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=3).build()
        return np.asarray(mesh.coords, dtype=float), np.asarray(mesh.conn, dtype=int), 3
    if elem == "tet4":
        coords = _tet4_coords()
        conn = np.array([[0, 1, 2, 3]], dtype=int)
        return coords, conn, 1
    if elem == "tet10":
        coords = _tet10_coords()
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=int)
        return coords, conn, 2
    raise ValueError(f"unsupported element: {elem}")


def _perm_by_coords(coords_ff: np.ndarray, doflocs_sf: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    coords_ff = np.asarray(coords_ff)
    doflocs_sf = np.asarray(doflocs_sf)
    if doflocs_sf.shape[0] == 3 and doflocs_sf.shape[1] != 3:
        doflocs_sf = doflocs_sf.T
    perm = np.empty(coords_ff.shape[0], dtype=int)
    for i, c in enumerate(coords_ff):
        matches = np.nonzero(np.all(np.isclose(doflocs_sf, c, atol=atol), axis=1))[0]
        if len(matches) != 1:
            raise RuntimeError("dof mapping ambiguous")
        perm[i] = matches[0]
    return perm


def _vector_perm_for_skfem(
    coords_ff: np.ndarray,
    scalar_doflocs: np.ndarray,
    vector_doflocs: np.ndarray,
    value_dim: int,
    *,
    atol: float = 1e-8,
) -> np.ndarray:
    scalar_doflocs = np.asarray(scalar_doflocs)
    if scalar_doflocs.shape[0] == 3 and scalar_doflocs.shape[1] != 3:
        scalar_doflocs = scalar_doflocs.T
    vector_doflocs = np.asarray(vector_doflocs)
    if vector_doflocs.shape[0] == 3 and vector_doflocs.shape[1] != 3:
        vector_doflocs = vector_doflocs.T

    coords_ff = np.asarray(coords_ff, dtype=float)
    perm_nodes = _perm_by_coords(coords_ff, scalar_doflocs, atol=atol)
    n_nodes = coords_ff.shape[0]
    if vector_doflocs.shape[0] != n_nodes * value_dim:
        raise RuntimeError("vector doflocs size mismatch")

    node_major = np.repeat(scalar_doflocs, value_dim, axis=0)
    comp_major = np.tile(scalar_doflocs, (value_dim, 1))
    if np.allclose(node_major, vector_doflocs, atol=atol):
        order = "node"
    elif np.allclose(comp_major, vector_doflocs, atol=atol):
        order = "component"
    else:
        order = "unknown"

    if order == "component":
        perm_vec = np.array(
            [comp * n_nodes + perm_nodes[node] for node in range(n_nodes) for comp in range(value_dim)],
            dtype=int,
        )
    else:
        perm_vec = np.array(
            [perm_nodes[node] * value_dim + comp for node in range(n_nodes) for comp in range(value_dim)],
            dtype=int,
        )
    return perm_vec


def build_fluxfem_contact(
    elem: str,
    *,
    alpha: float,
    h: float,
    use_penalty: bool,
    use_traction: bool,
    normal_sign: float | None,
    quad_order: int,
) -> np.ndarray:
    coords, conn, order = _fluxfem_mesh_for(elem)
    if elem.startswith("hex"):
        facets = _build_hex_facets(conn, order)
    else:
        facets = _build_tet_facets(conn, order)

    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)
    side_a = ff.ContactSide.from_surfaces(surf_a, elem_conn=conn, value_dim=3)
    side_b = ff.ContactSide.from_surfaces(surf_b, elem_conn=conn, value_dim=3)
    contact = ff.ContactSurfaceSpace.from_sides(
        side_a,
        side_b,
        quad_order=quad_order,
        normal_sign=normal_sign,
    )

    E, nu = 210e9, 0.3
    lam, mu = ff.lame_parameters(E, nu)

    def bilin(v1, v2, u1, u2, p):
        n = h_wf.normal()
        ju = u1.val - u2.val
        t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
        t_v1 = h_wf.traction(v1, n, p)
        t_v2 = h_wf.traction(v2, n, p)
        penalty = (p.alpha * p.inv_h) * (h_wf.dot(v1, ju) - h_wf.dot(v2, ju))
        traction = -h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
        traction -= 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
        if not use_penalty:
            penalty = 0.0
        if not use_traction:
            traction = 0.0
        return (penalty + traction) * h_wf.ds()

    u_a = jnp.zeros(coords.shape[0] * 3)
    u_b = jnp.zeros(coords.shape[0] * 3)
    params = ff.Params(alpha=float(alpha), inv_h=float(1.0 / h), lam=float(lam), mu=float(mu))
    K = contact.assemble_bilinear(
        bilin,
        u_a,
        u_b,
        params=params,
        sparse=False,
    )
    return np.asarray(K)


def build_skfem_contact(
    elem: str,
    *,
    alpha: float,
    use_penalty: bool,
    use_traction: bool,
) -> tuple[np.ndarray | None, float | None]:
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
        coords = _tet4_coords()
        conn = np.array([[0, 1, 2, 3]], dtype=int)
        mesh_a = MeshTet(coords.T, conn.T)
        mesh_b = MeshTet(coords.T, conn.T)
        elem_s = ElementTetP1()
        trace_type = skfem.MeshTri
    elif elem == "tet10":
        coords = _tet4_coords()
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
    fbasis = fb_u_top * fb_u_bot

    E, nu = 210e9, 0.3
    lam, mu = lame_parameters(E, nu)
    C = linear_stress(lam, mu)

    @skfem.BilinearForm
    def bilin(u1, u2, v1, v2, w):
        ju = u1 - u2
        t_u = 0.5 * (mul(C(sym_grad(u1)), w.n) + mul(C(sym_grad(u2)), w.n))
        t_v1 = mul(C(sym_grad(v1)), w.n)
        t_v2 = mul(C(sym_grad(v2)), w.n)
        penalty = (alpha / w.h) * dot(v1 - v2, ju)
        traction = -dot(v1, t_u) + dot(v2, t_u)
        traction -= 0.5 * dot(t_v1, ju)
        traction -= 0.5 * dot(t_v2, ju)
        if not use_penalty:
            penalty = 0.0
        if not use_traction:
            traction = 0.0
        return penalty + traction

    K = skfem.asm(bilin, fbasis, h=fb_u_top.mesh_parameters())
    mesh_params = fb_u_top.mesh_parameters()
    h_ref = None
    if isinstance(mesh_params, dict):
        h_val = mesh_params.get("h", None)
        if h_val is not None:
            h_ref = float(np.asarray(h_val).mean())
    else:
        h_ref = float(np.asarray(mesh_params).mean())

    coords_ff, _conn_ff, _order = _fluxfem_mesh_for(elem)
    perm_vec_a = _vector_perm_for_skfem(
        coords_ff,
        np.asarray(basis_scalar_a.doflocs),
        np.asarray(basis_vec_a.doflocs),
        3,
    )
    offset_vec = int(fb_u_top.N)
    perm_vec_b_local = _vector_perm_for_skfem(
        coords_ff,
        np.asarray(basis_scalar_b.doflocs),
        np.asarray(basis_vec_b.doflocs),
        3,
    )
    perm_vec_b = perm_vec_b_local + offset_vec
    perm_vec = np.concatenate([perm_vec_a, perm_vec_b])
    K_np = K.toarray()
    K_np = K_np[np.ix_(perm_vec, perm_vec)]
    return K_np, h_ref


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


if __name__ == "__main__":
    alpha = float(os.getenv("ALPHA", "10.0"))
    h = float(os.getenv("H_REF", "1.0"))
    quad_order = int(os.getenv("QUAD_ORDER", "5"))
    normal_sign = float(os.getenv("NORMAL_SIGN", "-1.0"))

    only_elems = [s.strip().lower() for s in os.getenv("COMPARE_ELEMS", "").split(",") if s.strip()]
    cases_env = [s.strip().lower() for s in os.getenv("CASES", "penalty,traction,full").split(",") if s.strip()]
    case_map = {"penalty": (True, False), "traction": (False, True), "full": (True, True)}
    cases = [(name, case_map[name][0], case_map[name][1]) for name in cases_env if name in case_map]
    elems = ["hex8", "hex27", "tet4", "tet10"]

    def _enabled(name: str) -> bool:
        return not only_elems or name.lower() in only_elems

    print(f"[simple] quad_order={quad_order}")
    for elem in elems:
        if not _enabled(elem):
            continue
        for name, use_penalty, use_traction in cases:
            K_sf, h_ref = build_skfem_contact(
                elem,
                alpha=alpha,
                use_penalty=use_penalty,
                use_traction=use_traction,
            )
            if K_sf is None:
                print(f"[{elem}/{name}] skfem not installed; skipping")
                continue
            h_use = h_ref if h_ref is not None else h
            K_ff = build_fluxfem_contact(
                elem,
                alpha=alpha,
                h=h_use,
                use_penalty=use_penalty,
                use_traction=use_traction,
                normal_sign=normal_sign,
                quad_order=quad_order,
            )
            n_inf_ff = float(np.linalg.norm(K_ff, ord=np.inf))
            n_inf_sf = float(np.linalg.norm(K_sf, ord=np.inf))
            n_2_ff = float(np.linalg.norm(K_ff))
            n_2_sf = float(np.linalg.norm(K_sf))
            max_ff = float(np.max(np.abs(K_ff))) if K_ff.size else 0.0
            max_sf = float(np.max(np.abs(K_sf))) if K_sf.size else 0.0
            rel_inf = _rel_err(n_inf_ff, n_inf_sf)
            rel_2 = _rel_err(n_2_ff, n_2_sf)
            rel_max = _rel_err(max_ff, max_sf)
            diff = K_ff - K_sf
            diff_inf = float(np.linalg.norm(diff, ord=np.inf))
            diff_2 = float(np.linalg.norm(diff))
            diff_max = float(np.max(np.abs(diff))) if diff.size else 0.0
            rel_diff_inf = diff_inf / max(1.0, n_inf_sf)
            rel_diff_2 = diff_2 / max(1.0, n_2_sf)
            rel_diff_max = diff_max / max(1.0, max_sf)
            h_note = f"h={h_use:.6g}" if h_ref is not None else f"h={h_use:.6g} (default)"
            print(
                f"[{elem}/{name}/n={normal_sign:+.0f}] {h_note} rel_inf={rel_inf:.3e} "
                f"rel_2={rel_2:.3e} rel_max={rel_max:.3e}"
            )
            print(
                f"[{elem}/{name}/n={normal_sign:+.0f}] rel_diff_inf={rel_diff_inf:.3e} "
                f"rel_diff_2={rel_diff_2:.3e} rel_diff_max={rel_diff_max:.3e}"
            )
