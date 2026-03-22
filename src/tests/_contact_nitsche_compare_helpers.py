from __future__ import annotations

import importlib.util
import os

import numpy as np

import fluxfem as ff


def build_hex_facets(conn: np.ndarray, order: int) -> np.ndarray:
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


def build_tet_facets(conn: np.ndarray, order: int) -> np.ndarray:
    elem = conn[0]
    if order in (1, 2):
        pattern = (0, 1, 2)
    else:
        raise ValueError("order must be 1 or 2")
    return np.array([[int(elem[i]) for i in pattern]], dtype=int)


def tet4_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def tet10_coords() -> np.ndarray:
    p = tet4_coords()
    n0, n1, n2, n3 = p
    n01 = 0.5 * (n0 + n1)
    n12 = 0.5 * (n1 + n2)
    n02 = 0.5 * (n0 + n2)
    n03 = 0.5 * (n0 + n3)
    n13 = 0.5 * (n1 + n3)
    n23 = 0.5 * (n2 + n3)
    return np.array([n0, n1, n2, n3, n01, n12, n02, n03, n13, n23], dtype=float)


def fluxfem_mesh_for(elem: str) -> tuple[np.ndarray, np.ndarray, int]:
    if elem == "hex8":
        mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=1).build()
        return np.asarray(mesh.coords, dtype=float), np.asarray(mesh.conn, dtype=int), 1
    if elem == "hex27":
        mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=3).build()
        return np.asarray(mesh.coords, dtype=float), np.asarray(mesh.conn, dtype=int), 3
    if elem == "tet4":
        coords = tet4_coords()
        conn = np.array([[0, 1, 2, 3]], dtype=int)
        return coords, conn, 1
    if elem == "tet10":
        coords = tet10_coords()
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=int)
        return coords, conn, 2
    raise ValueError(f"unsupported element: {elem}")


def build_fluxfem_onesided_contact_space(
    elem: str,
    *,
    quad_order: int,
    with_master: bool = False,
):
    coords, conn, order = fluxfem_mesh_for(elem)
    facets = build_hex_facets(conn, order) if elem.startswith("hex") else build_tet_facets(conn, order)
    surface = ff.SurfaceMesh.from_facets(coords, facets)
    side = ff.ContactSideSpec.from_surfaces(surface, elem_conn=conn, value_dim=3)
    if with_master:
        contact_space = ff.OneSidedContactSpec(
            side=side,
            surface_master=surface,
            elem_conn_master=conn,
        ).prepare(quad_order=quad_order)
    else:
        contact_space = ff.OneSidedContactSpec(side=side).prepare(quad_order=quad_order)
    return coords, conn, contact_space


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
    _ = conn
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

    ops = ff.assemble_penalty(
        _OneSidedAdapter(contact_space, _u_hat_fn, u_master),
        weak_form=_dummy_res_form,
        state={"a": np.zeros(int(contact_space.surface_slave.n_nodes * contact_space.value_dim), dtype=float)},
        params=params,
        backend="jax",
    )
    return np.asarray(ops.jacobian), np.asarray(ops.residual), coords


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
    from skfem import FacetBasis, ElementVectorH1, MeshHex, MeshTet
    from skfem.helpers import dot, mul, sym_grad
    from skfem.models.elasticity import lame_parameters, linear_stress
    from skfem.supermeshing import elementwise_quadrature, intersect

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
