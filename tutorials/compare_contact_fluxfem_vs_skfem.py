import os
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp

import fluxfem as ff
import fluxfem.helpers_wf as h_wf
from fluxfem.core.weakform import einsum as wf_einsum
from fluxfem.mesh import mortar as mortar_mod
from fluxfem.solver.bc import facet_normals

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


def _build_fluxfem_surface_mesh(elem: str):
    coords, conn, order = _fluxfem_mesh_for(elem)
    if elem.startswith("hex"):
        facets = _build_hex_facets(conn, order)
    else:
        facets = _build_tet_facets(conn, order)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)
    sm = ff.build_surface_supermesh(surf_a, surf_b)
    return coords, surf_a, surf_b, sm


def _diag_contact_surface(elem: str, quad_order: int, *, verbose: bool) -> None:
    coords, surf_a, surf_b, sm = _build_fluxfem_surface_mesh(elem)
    if sm.conn.shape[0] == 0:
        print(f"[diag][{elem}] supermesh empty; check facet intersection")
        return
    if quad_order > 5:
        quad_order = 5

    quad_pts, quad_w = mortar_mod.tri_quadrature(quad_order)
    coords_a = np.asarray(surf_a.coords, dtype=float)
    coords_b = np.asarray(surf_b.coords, dtype=float)
    facets_a = np.asarray(surf_a.conn, dtype=int)
    facets_b = np.asarray(surf_b.conn, dtype=int)
    normals_a = facet_normals(surf_a, outward_from=np.mean(coords_a, axis=0), normalize=True)
    normals_b = facet_normals(surf_b, outward_from=np.mean(coords_b, axis=0), normalize=True)

    def _facet_area(nodes: np.ndarray, coords_local: np.ndarray) -> float:
        n = int(len(nodes))
        if n == 3:
            pts = coords_local[nodes]
            return mortar_mod.tri_area(pts[0], pts[1], pts[2])
        if n == 4:
            pts = coords_local[nodes]
            return mortar_mod.tri_area(pts[0], pts[1], pts[2]) + mortar_mod.tri_area(pts[0], pts[2], pts[3])
        if n == 8:
            corner = nodes[:4]
            pts = coords_local[corner]
            return mortar_mod.tri_area(pts[0], pts[1], pts[2]) + mortar_mod.tri_area(pts[0], pts[2], pts[3])
        if n == 9:
            corner = nodes[[0, 2, 8, 6]]
            pts = coords_local[corner]
            return mortar_mod.tri_area(pts[0], pts[1], pts[2]) + mortar_mod.tri_area(pts[0], pts[2], pts[3])
        pts = coords_local[nodes]
        area = 0.0
        p0 = pts[0]
        for i in range(1, len(pts) - 1):
            area += mortar_mod.tri_area(p0, pts[i], pts[i + 1])
        return float(area)

    area_scale = float(os.getenv("FLUXFEM_SMALL_TRI_EPS_SCALE", "1e-14"))
    facet_area_a = np.array([_facet_area(fa, coords_a) for fa in facets_a], dtype=float)
    facet_area_b = np.array([_facet_area(fb, coords_b) for fb in facets_b], dtype=float)

    area_total = 0.0
    wsum_total = 0.0
    small_tri = 0
    neg_dot_a = 0
    neg_dot_b = 0
    dot_a_vals = []
    dot_b_vals = []

    t0 = np.array([1.0, 0.0, 0.0], dtype=float)
    force_a = np.zeros(3, dtype=float)
    force_b = np.zeros(3, dtype=float)

    nsum_bad = 0
    for tri_id, (tri, fa, fb) in enumerate(zip(sm.conn, sm.source_facets_a, sm.source_facets_b)):
        a, b, c = sm.coords[tri]
        area = mortar_mod.tri_area(a, b, c)
        area_ref = max(float(facet_area_a[int(fa)]), float(facet_area_b[int(fb)]))
        eps_area = area_scale * area_ref if area_ref > 0.0 else area_scale
        if area <= eps_area:
            small_tri += 1
            if verbose:
                pts = sm.coords[tri]
                edges = [float(np.linalg.norm(pts[(i + 1) % 3] - pts[i])) for i in range(3)]
                dup = np.min(edges) <= 1e-12
                print(
                    f"[diag][{elem}] small_tri id={tri_id} fa={int(fa)} fb={int(fb)} "
                    f"area={area:.3e} eps={eps_area:.3e} edges={[f'{e:.3e}' for e in edges]} dup={dup}"
                )
            continue
        detJ = 2.0 * area
        area_total += area
        wsum_total += float(np.sum(quad_w) * detJ)

        n_tri = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n_tri)
        if n_norm > 0.0:
            n_tri /= n_norm
            dot_a = float(np.dot(n_tri, normals_a[int(fa)]))
            dot_b = float(np.dot(n_tri, normals_b[int(fb)]))
            dot_a_vals.append(dot_a)
            dot_b_vals.append(dot_b)
            if dot_a < 0.0:
                neg_dot_a += 1
            if dot_b < 0.0:
                neg_dot_b += 1

        x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
        facet_a = facets_a[int(fa)]
        facet_b = facets_b[int(fb)]
        Na = np.array([mortar_mod.facet_shape_values(pt, facet_a, coords_a, tol=1e-8) for pt in x_q], dtype=float)
        Nb = np.array([mortar_mod.facet_shape_values(pt, facet_b, coords_b, tol=1e-8) for pt in x_q], dtype=float)
        wJ = quad_w * detJ
        if verbose and (np.max(np.abs(Na.sum(axis=1) - 1.0)) > 1e-10 or np.max(np.abs(Nb.sum(axis=1) - 1.0)) > 1e-10):
            nsum_bad += 1

        fe_a = Na.T @ wJ
        fe_b = Nb.T @ wJ
        force_a += t0 * float(fe_a.sum())
        force_b += t0 * float(fe_b.sum())

    if area_total <= 0.0:
        print(f"[diag][{elem}] total area is non-positive")
        return
    area_rel = abs(wsum_total - area_total) / area_total
    expected = t0 * area_total
    rel_force_a = np.linalg.norm(force_a - expected) / max(1.0, np.linalg.norm(expected))
    rel_force_b = np.linalg.norm(force_b - expected) / max(1.0, np.linalg.norm(expected))
    dot_a_min = min(dot_a_vals) if dot_a_vals else 0.0
    dot_b_min = min(dot_b_vals) if dot_b_vals else 0.0
    dot_a_mean = float(np.mean(dot_a_vals)) if dot_a_vals else 0.0
    dot_b_mean = float(np.mean(dot_b_vals)) if dot_b_vals else 0.0

    print(
        f"[diag][{elem}] area_rel={area_rel:.3e} small_tri={small_tri} "
        f"force_rel_a={rel_force_a:.3e} force_rel_b={rel_force_b:.3e}"
    )
    print(
        f"[diag][{elem}] dot_a_min={dot_a_min:.3e} dot_a_mean={dot_a_mean:.3e} neg_a={neg_dot_a} "
        f"dot_b_min={dot_b_min:.3e} dot_b_mean={dot_b_mean:.3e} neg_b={neg_dot_b}"
    )
    if verbose:
        print(f"[diag][{elem}] nsum_bad={nsum_bad} / {len(sm.conn)}")


def _hex27_param_data():
    coords_ff, surf_a, _surf_b, _sm = _build_fluxfem_surface_mesh("hex27")
    facet_nodes = np.asarray(surf_a.conn[0], dtype=int)
    facet_coords = coords_ff[facet_nodes]
    corner_coords = facet_coords[[0, 2, 8, 6]]
    quad_nodes = np.array([0, 1, 2, 3], dtype=int)

    def _quad_local(point: np.ndarray) -> tuple[float, float]:
        _values, xi, eta = mortar_mod.quad_shape_and_local(point, quad_nodes, corner_coords, tol=1e-10)
        return float(xi), float(eta)

    flux_xi_eta = [(_quad_local(pt), int(node)) for pt, node in zip(facet_coords, facet_nodes)]

    try:
        from skfem import MeshHex, ElementHex2, Basis, MeshQuad, ElementQuad2
    except Exception as exc:
        return {
            "error": str(exc),
            "flux_xi_eta": flux_xi_eta,
            "facet_nodes": facet_nodes,
            "corner_coords": corner_coords,
        }

    xs = np.linspace(0.0, 1.0, 2)
    ys = np.linspace(0.0, 1.0, 2)
    zs = np.linspace(0.0, 1.0, 2)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    basis_sf = Basis(mesh_sf, ElementHex2())
    doflocs = np.asarray(basis_sf.doflocs)
    if doflocs.shape[0] == 3:
        doflocs = doflocs.T
    dof_ids = np.nonzero(np.isclose(doflocs[:, 2], 0.0, atol=1e-8))[0]
    dof_coords = doflocs[dof_ids]
    skfem_xi_eta = [(_quad_local(pt), int(dof_id)) for pt, dof_id in zip(dof_coords, dof_ids)]

    mesh_quad = MeshQuad().init_tensor(xs, ys)
    basis_quad = Basis(mesh_quad, ElementQuad2())
    quad_doflocs = np.asarray(basis_quad.doflocs)
    if quad_doflocs.shape[0] == 2:
        quad_doflocs = quad_doflocs.T

    return {
        "flux_xi_eta": flux_xi_eta,
        "skfem_xi_eta": skfem_xi_eta,
        "facet_nodes": facet_nodes,
        "corner_coords": corner_coords,
        "quad_doflocs": quad_doflocs,
        "element_quad": basis_quad.elem,
    }


def _diag_hex27_param(*, verbose: bool) -> None:
    data = _hex27_param_data()
    if "error" in data:
        print(f"[diag][hex27][param] skfem not available: {data['error']}")
        return

    flux_xi_eta = data["flux_xi_eta"]
    skfem_xi_eta = data["skfem_xi_eta"]

    canonical = [
        (-1.0, -1.0, "corner(-1,-1)"),
        (1.0, -1.0, "corner(1,-1)"),
        (1.0, 1.0, "corner(1,1)"),
        (-1.0, 1.0, "corner(-1,1)"),
        (0.0, -1.0, "edge(0,-1)"),
        (1.0, 0.0, "edge(1,0)"),
        (0.0, 1.0, "edge(0,1)"),
        (-1.0, 0.0, "edge(-1,0)"),
        (0.0, 0.0, "center(0,0)"),
    ]

    def _match_label(xi: float, eta: float, tol: float = 1e-6) -> str:
        best = None
        best_d = 1e9
        for cx, cy, label in canonical:
            d = abs(xi - cx) + abs(eta - cy)
            if d < best_d:
                best_d = d
                best = label
        if best_d > tol:
            return f"unmatched(d={best_d:.2e})"
        return str(best)

    def _max_dev(xi_eta_list):
        dev = 0.0
        for (xi, eta), _ in xi_eta_list:
            d = min(abs(xi - cx) + abs(eta - cy) for cx, cy, _ in canonical)
            dev = max(dev, d)
        return dev

    flux_dev = _max_dev(flux_xi_eta)
    sk_dev = _max_dev(skfem_xi_eta)
    print(f"[diag][hex27][param] flux_dev={flux_dev:.3e} skfem_dev={sk_dev:.3e} n_flux={len(flux_xi_eta)} n_sk={len(skfem_xi_eta)}")

    if verbose:
        print("[diag][hex27][param] fluxfem facet nodes:")
        for (xi, eta), node in sorted(flux_xi_eta, key=lambda v: (v[0][1], v[0][0])):
            label = _match_label(xi, eta)
            print(f"  node={node:2d} xi={xi:+.6f} eta={eta:+.6f} {label}")
        print("[diag][hex27][param] skfem facet dofs:")
        for (xi, eta), dof in sorted(skfem_xi_eta, key=lambda v: (v[0][1], v[0][0])):
            label = _match_label(xi, eta)
            print(f"  dof={dof:2d} xi={xi:+.6f} eta={eta:+.6f} {label}")

        def _find_match(xi_eta_list, target, tol=1e-6):
            for (xi, eta), idx in xi_eta_list:
                if abs(xi - target[0]) + abs(eta - target[1]) <= tol:
                    return idx
            return None

        print("[diag][hex27][param] flux->skfem mapping by (xi,eta):")
        for (xi, eta), node in sorted(flux_xi_eta, key=lambda v: (v[0][1], v[0][0])):
            match = _find_match(skfem_xi_eta, (xi, eta))
            print(f"  node={node:2d} -> dof={match}")


def _diag_hex27_shape_compare(*, verbose: bool) -> None:
    data = _hex27_param_data()
    if "error" in data:
        print(f"[diag][hex27][shape] skfem not available: {data['error']}")
        return

    flux_xi_eta = data["flux_xi_eta"]
    quad_doflocs = data["quad_doflocs"]
    elem_quad = data["element_quad"]
    flux_node_order = [node for _pt, node in flux_xi_eta]
    flux_node_index = {node: i for i, node in enumerate(flux_node_order)}

    def _map_flux_to_quad_dof(xi: float, eta: float, tol: float = 1e-8) -> int | None:
        r = 0.5 * (xi + 1.0)
        s = 0.5 * (eta + 1.0)
        for i, (x, y) in enumerate(quad_doflocs):
            if abs(x - r) <= tol and abs(y - s) <= tol:
                return int(i)
        return None

    flux_to_quad = []
    for (xi, eta), node in flux_xi_eta:
        qdof = _map_flux_to_quad_dof(xi, eta)
        flux_to_quad.append((node, qdof, xi, eta))

    if any(qdof is None for _node, qdof, _xi, _eta in flux_to_quad):
        print("[diag][hex27][shape] quad dof mapping failed")
        return

    def _skfem_quadN(xi: float, eta: float) -> np.ndarray:
        r = 0.5 * (xi + 1.0)
        s = 0.5 * (eta + 1.0)
        pts = np.array([[r], [s]], dtype=float)
        vals = []
        for i in range(len(quad_doflocs)):
            v, _ = elem_quad.lbasis(pts, i)
            vals.append(float(v[0]))
        return np.array(vals, dtype=float)

    def _flux_quadN(xi: float, eta: float) -> np.ndarray:
        return mortar_mod.quad9_shape_values(xi, eta)

    def _basis_to_dofloc_map() -> list[int]:
        mapping = []
        for i in range(len(quad_doflocs)):
            vals = []
            for r, s in quad_doflocs:
                pts = np.array([[r], [s]], dtype=float)
                v, _ = elem_quad.lbasis(pts, i)
                vals.append(float(v[0]))
            mapping.append(int(np.argmax(vals)))
        return mapping

    basis_to_dofloc = _basis_to_dofloc_map()
    dofloc_to_basis = np.zeros(len(basis_to_dofloc), dtype=int)
    for bidx, didx in enumerate(basis_to_dofloc):
        dofloc_to_basis[didx] = bidx

    sample = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, 0.0),
        (-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
        (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
        (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
    ]

    max_diff = 0.0
    for xi, eta in sample:
        flux_vals = _flux_quadN(xi, eta)
        sk_vals_basis = _skfem_quadN(xi, eta)
        sk_vals = np.zeros_like(sk_vals_basis)
        for didx, bidx in enumerate(dofloc_to_basis):
            sk_vals[didx] = sk_vals_basis[bidx]
        sk_reordered = np.zeros_like(flux_vals)
        for node, qdof, _xi, _eta in flux_to_quad:
            sk_reordered[flux_node_index[node]] = sk_vals[qdof]
        diff = np.max(np.abs(flux_vals - sk_reordered))
        max_diff = max(max_diff, diff)
        if verbose:
            print(f"[diag][hex27][shape] pt=({xi:+.6f},{eta:+.6f}) max_diff={diff:.3e}")
    print(f"[diag][hex27][shape] max_diff={max_diff:.3e} samples={len(sample)}")


def _diag_hex27_dof_groups(K_ff: np.ndarray, K_sf: np.ndarray, coords: np.ndarray) -> None:
    n_nodes = coords.shape[0]
    n_dofs = n_nodes * 3
    if K_ff.shape != K_sf.shape:
        print("[diag][hex27][dof] shape mismatch")
        return
    if K_ff.shape[0] < 2 * n_dofs:
        print("[diag][hex27][dof] unexpected matrix size")
        return

    diff = K_ff - K_sf
    node_kind = []
    for i in range(n_nodes):
        x, y, z = coords[i]
        mids = int(abs(x - 0.5) < 1e-8) + int(abs(y - 0.5) < 1e-8) + int(abs(z - 0.5) < 1e-8)
        if mids == 0:
            node_kind.append("corner")
        elif mids == 1:
            node_kind.append("edge")
        elif mids == 2:
            node_kind.append("face")
        else:
            node_kind.append("center")

    def _block_stats(offset: int, label: str) -> None:
        rows = diff[offset:offset + n_dofs]
        stats = {"corner": [], "edge": [], "face": [], "center": []}
        for i in range(n_nodes):
            dofs = slice(3 * i, 3 * i + 3)
            val = float(np.max(np.abs(rows[dofs, :])))
            stats[node_kind[i]].append(val)
        print(f"[diag][hex27][dof] {label} max/mean:")
        for key in ("corner", "edge", "face", "center"):
            arr = np.asarray(stats[key], dtype=float)
            if arr.size == 0:
                continue
            print(f"  {key}: max={arr.max():.3e} mean={arr.mean():.3e}")

    _block_stats(0, "master")
    _block_stats(n_dofs, "slave")


def _diag_hex27_block_compare(K_ff: np.ndarray, K_sf: np.ndarray, coords: np.ndarray) -> None:
    n_nodes = coords.shape[0]
    n_dofs = n_nodes * 3
    if K_ff.shape[0] < 2 * n_dofs or K_sf.shape[0] < 2 * n_dofs:
        print("[diag][hex27][block] unexpected matrix size")
        return

    def _max_abs(a: np.ndarray) -> float:
        return float(np.max(np.abs(a))) if a.size else 0.0

    ff_aa = K_ff[:n_dofs, :n_dofs]
    ff_ab = K_ff[:n_dofs, n_dofs:2 * n_dofs]
    ff_ba = K_ff[n_dofs:2 * n_dofs, :n_dofs]
    ff_bb = K_ff[n_dofs:2 * n_dofs, n_dofs:2 * n_dofs]

    sf_aa = K_sf[:n_dofs, :n_dofs]
    sf_ab = K_sf[:n_dofs, n_dofs:2 * n_dofs]
    sf_ba = K_sf[n_dofs:2 * n_dofs, :n_dofs]
    sf_bb = K_sf[n_dofs:2 * n_dofs, n_dofs:2 * n_dofs]

    print(
        "[diag][hex27][block] max_diff "
        f"AA={_max_abs(ff_aa - sf_aa):.3e} "
        f"AB={_max_abs(ff_ab - sf_ab):.3e} "
        f"BA={_max_abs(ff_ba - sf_ba):.3e} "
        f"BB={_max_abs(ff_bb - sf_bb):.3e}"
    )

    cand = [
        ("AB", sf_ab),
        ("BA", sf_ba),
        ("-AB", -sf_ab),
        ("-BA", -sf_ba),
    ]
    best_ab = min(((name, _max_abs(ff_ab - ref)) for name, ref in cand), key=lambda v: v[1])
    best_ba = min(((name, _max_abs(ff_ba - ref)) for name, ref in cand), key=lambda v: v[1])
    print(
        "[diag][hex27][block] best_match "
        f"ff_ab~{best_ab[0]} diff={best_ab[1]:.3e} "
        f"ff_ba~{best_ba[0]} diff={best_ba[1]:.3e}"
    )


def _diag_hex27_component_perm(K_ff: np.ndarray, K_sf: np.ndarray, coords: np.ndarray) -> None:
    n_nodes = coords.shape[0]
    n_dofs = n_nodes * 3
    if K_ff.shape[0] < 2 * n_dofs or K_sf.shape[0] < 2 * n_dofs:
        print("[diag][hex27][compperm] unexpected matrix size")
        return

    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]

    def _perm_vec(order):
        perm = []
        for node in range(n_nodes):
            base = node * 3
            for comp in order:
                perm.append(base + comp)
        perm = np.array(perm, dtype=int)
        return np.concatenate([perm, perm + n_dofs])

    best = None
    for order in perms:
        p = _perm_vec(order)
        K_perm = K_sf[np.ix_(p, p)]
        diff = float(np.max(np.abs(K_ff - K_perm)))
        if best is None or diff < best[1]:
            best = (order, diff)
    if best is None:
        return
    print(f"[diag][hex27][compperm] best_order={best[0]} max_diff={best[1]:.3e}")


def _diag_hex27_face_mask(K_ff: np.ndarray, K_sf: np.ndarray, coords: np.ndarray) -> None:
    n_nodes = coords.shape[0]
    n_dofs = n_nodes * 3
    if K_ff.shape[0] < 2 * n_dofs or K_sf.shape[0] < 2 * n_dofs:
        print("[diag][hex27][facemask] unexpected matrix size")
        return

    on_face = np.isclose(coords[:, 2], 0.0, atol=1e-12)
    off_face = ~on_face
    if not np.any(on_face):
        print("[diag][hex27][facemask] no face nodes detected")
        return

    def _dofs(mask):
        nodes = np.nonzero(mask)[0]
        return np.array([n * 3 + c for n in nodes for c in range(3)], dtype=int)

    dofs_on = _dofs(on_face)
    dofs_off = _dofs(off_face)

    def _max_abs(a: np.ndarray) -> float:
        return float(np.max(np.abs(a))) if a.size else 0.0

    ff = K_ff[:2 * n_dofs, :2 * n_dofs]
    sf = K_sf[:2 * n_dofs, :2 * n_dofs]
    diff = ff - sf

    def _sub(mask_rows, mask_cols):
        return diff[np.ix_(mask_rows, mask_cols)]

    on_on = _sub(dofs_on, dofs_on)
    on_off = _sub(dofs_on, dofs_off)
    off_on = _sub(dofs_off, dofs_on)
    off_off = _sub(dofs_off, dofs_off)
    print(
        "[diag][hex27][facemask] max_diff "
        f"on_on={_max_abs(on_on):.3e} "
        f"on_off={_max_abs(on_off):.3e} "
        f"off_on={_max_abs(off_on):.3e} "
        f"off_off={_max_abs(off_off):.3e}"
    )


def _diag_hex27_quad_compare(quad_order: int) -> None:
    try:
        import skfem
        from skfem import MeshHex, ElementHex2, ElementVectorH1, FacetBasis
        from skfem.supermeshing import intersect, elementwise_quadrature
    except Exception as exc:
        print(f"[diag][hex27][quad] skfem not available: {exc}")
        return

    coords, surf_a, _surf_b, sm = _build_fluxfem_surface_mesh("hex27")
    if sm.conn.shape[0] == 0:
        print("[diag][hex27][quad] supermesh empty")
        return
    if quad_order > 5:
        quad_order = 5

    quad_pts, quad_w = mortar_mod.tri_quadrature(quad_order)
    q_flux = []
    w_flux = []
    for tri, fa in zip(sm.conn, sm.source_facets_a):
        if int(fa) != 0:
            continue
        a, b, c = sm.coords[tri]
        area = mortar_mod.tri_area(a, b, c)
        if area <= 1e-12:
            continue
        detJ = 2.0 * area
        for (r, s), w in zip(quad_pts, quad_w):
            xq = a + r * (b - a) + s * (c - a)
            q_flux.append(xq)
            w_flux.append(float(w * detJ))
    q_flux = np.array(q_flux, dtype=float)
    w_flux = np.array(w_flux, dtype=float)

    xs = np.linspace(0.0, 1.0, 2)
    ys = np.linspace(0.0, 1.0, 2)
    zs = np.linspace(0.0, 1.0, 2)
    mesh = MeshHex().init_tensor(xs, ys, zs)
    mesh = mesh.with_boundaries({"contact": lambda x: np.isclose(x[2], 0.0)})
    m1t, orig1 = mesh.trace("contact", mtype=skfem.MeshQuad, project=lambda p: p[[0, 1]])
    m2t, orig2 = mesh.trace("contact", mtype=skfem.MeshQuad, project=lambda p: p[[0, 1]])
    m12, t1, t2 = intersect(m1t, m2t)
    try:
        quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
    except TypeError:
        quad1 = elementwise_quadrature(m1t, m12, t1)
    fb = FacetBasis(mesh, ElementVectorH1(ElementHex2()), facets=orig1[t1], quadrature=quad1)

    if fb.X.shape[1] == 0:
        print("[diag][hex27][quad] skfem facets empty")
        return
    q_sf = []
    w_sf = []
    for fi in range(fb.X.shape[1]):
        x_ref = fb.X[:, fi, :]
        w_ref = fb.W[fi, :]
        pts = np.vstack([x_ref[0], x_ref[1], np.zeros_like(x_ref[0])]).T
        q_sf.append(pts)
        w_sf.append(w_ref)
    q_sf = np.vstack(q_sf) if q_sf else np.zeros((0, 3), dtype=float)
    w_sf = np.concatenate(w_sf) if w_sf else np.zeros((0,), dtype=float)

    def _max_min_dist(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return float(np.max(np.min(d, axis=1)))

    d_flux_sf = _max_min_dist(q_flux, q_sf)
    d_sf_flux = _max_min_dist(q_sf, q_flux)
    wsum_flux = float(np.sum(w_flux)) if w_flux.size else 0.0
    wsum_sf = float(np.sum(w_sf)) if w_sf.size else 0.0
    print(
        "[diag][hex27][quad] "
        f"n_flux={q_flux.shape[0]} n_sf={q_sf.shape[0]} "
        f"w_sum_flux={wsum_flux:.6e} w_sum_sf={wsum_sf:.6e} "
        f"max_min_flux_to_sf={d_flux_sf:.3e} max_min_sf_to_flux={d_sf_flux:.3e}"
    )


def _diag_hex27_volumeN_compare(quad_order: int) -> None:
    try:
        from skfem import MeshHex, ElementHex2, Basis
    except Exception as exc:
        print(f"[diag][hex27][volN] skfem not available: {exc}")
        return

    coords_ff, surf_a, _surf_b, sm = _build_fluxfem_surface_mesh("hex27")
    elem_conn = np.asarray(_fluxfem_mesh_for("hex27")[1], dtype=int)[0]
    elem_coords = coords_ff[elem_conn]
    if sm.conn.shape[0] == 0:
        print("[diag][hex27][volN] supermesh empty")
        return
    if quad_order > 5:
        quad_order = 5
    quad_pts, _quad_w = mortar_mod.tri_quadrature(quad_order)
    tri = sm.conn[0]
    a, b, c = sm.coords[tri]
    x_q = np.array([a + r * (b - a) + s * (c - a) for r, s in quad_pts], dtype=float)
    Na_flux = mortar_mod.volume_shape_values_at_points(x_q, elem_coords, tol=1e-10)

    xs = np.linspace(0.0, 1.0, 2)
    ys = np.linspace(0.0, 1.0, 2)
    zs = np.linspace(0.0, 1.0, 2)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    basis_sf = Basis(mesh_sf, ElementHex2())
    probes = basis_sf.probes(x_q.T)
    if hasattr(probes, "toarray"):
        N_sf = probes.toarray()
    else:
        N_sf = np.asarray(probes)
    perm_nodes = _perm_by_coords(elem_coords, np.asarray(basis_sf.doflocs))
    N_sf = N_sf[:, perm_nodes]

    max_diff = float(np.max(np.abs(Na_flux - N_sf)))
    print(f"[diag][hex27][volN] max_diff={max_diff:.3e} nq={x_q.shape[0]}")


def _diag_hex27_sigma(*, verbose: bool) -> None:
    try:
        from skfem import MeshHex, ElementHex2, Basis
    except Exception as exc:
        print(f"[diag][hex27][sigma] skfem not available: {exc}")
        return

    coords_ff, _conn_ff, _order = _fluxfem_mesh_for("hex27")
    mesh_ff = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0, order=3).build()
    elem_conn = np.asarray(mesh_ff.conn)[0]
    elem_coords = np.asarray(mesh_ff.coords)[elem_conn]

    # linear displacement u = A x + b
    A = np.array([[0.2, -0.1, 0.3], [0.05, 0.4, -0.2], [-0.1, 0.25, 0.15]], dtype=float)
    b = np.array([0.01, -0.02, 0.03], dtype=float)

    u_nodes = elem_coords @ A.T + b[None, :]

    xs = np.linspace(0.0, 1.0, 2)
    ys = np.linspace(0.0, 1.0, 2)
    zs = np.linspace(0.0, 1.0, 2)
    mesh_sf = MeshHex().init_tensor(xs, ys, zs)
    basis_sf = Basis(mesh_sf, ElementHex2())
    doflocs = np.asarray(basis_sf.doflocs)
    if doflocs.shape[0] == 3:
        doflocs = doflocs.T
    u_sf = doflocs @ A.T + b[None, :]

    lam, mu = ff.lame_parameters(210e9, 0.3)
    bounds_min = mesh_sf.p.min(axis=1)
    bounds_max = mesh_sf.p.max(axis=1)

    def _grad_ff(point: np.ndarray) -> np.ndarray:
        gradN = mortar_mod.hex27_gradN(point, elem_coords, tol=1e-12)
        return u_nodes.T @ gradN

    interp_fns = []
    for d in range(3):
        interp_fns.append(basis_sf.interpolator(u_sf[:, d]))

    def _grad_sf(point: np.ndarray) -> np.ndarray:
        eps = 1e-6
        grad = np.zeros((3, 3), dtype=float)
        for a in range(3):
            x_fwd = point.copy()
            x_bwd = point.copy()
            x_fwd[a] += eps
            x_bwd[a] -= eps
            for comp in range(3):
                f0 = float(interp_fns[comp](point[:, None]))
                if x_bwd[a] < bounds_min[a]:
                    f_fwd = float(interp_fns[comp](x_fwd[:, None]))
                    grad[comp, a] = (f_fwd - f0) / eps
                elif x_fwd[a] > bounds_max[a]:
                    f_bwd = float(interp_fns[comp](x_bwd[:, None]))
                    grad[comp, a] = (f0 - f_bwd) / eps
                else:
                    f_fwd = float(interp_fns[comp](x_fwd[:, None]))
                    f_bwd = float(interp_fns[comp](x_bwd[:, None]))
                    grad[comp, a] = (f_fwd - f_bwd) / (2.0 * eps)
        return grad

    def _sigma(grad: np.ndarray) -> np.ndarray:
        eps = 0.5 * (grad + grad.T)
        tr = float(np.trace(eps))
        return lam * tr * np.eye(3) + 2.0 * mu * eps

    sample = [
        np.array([0.2, 0.3, 0.4], dtype=float),
        np.array([0.4, 0.2, 0.7], dtype=float),
        np.array([0.3, 0.5, 0.1], dtype=float),
        np.array([0.1, 0.1, 0.9], dtype=float),
    ]
    face_pts = [
        np.array([0.2, 0.3, 0.0], dtype=float),
        np.array([0.7, 0.2, 0.0], dtype=float),
        np.array([0.6, 0.8, 0.0], dtype=float),
    ]

    max_grad = 0.0
    max_sigma = 0.0
    max_trac = 0.0
    max_grad_ff = 0.0
    max_grad_sf = 0.0
    max_sum_grad = 0.0
    max_moment_grad = 0.0
    n = np.array([0.0, 0.0, -1.0], dtype=float)
    for pt in sample:
        g_ff = _grad_ff(pt)
        g_sf = _grad_sf(pt)
        max_grad = max(max_grad, float(np.max(np.abs(g_ff - g_sf))))
        max_grad_ff = max(max_grad_ff, float(np.max(np.abs(g_ff - A))))
        max_grad_sf = max(max_grad_sf, float(np.max(np.abs(g_sf - A))))
        s_ff = _sigma(g_ff)
        s_sf = _sigma(g_sf)
        max_sigma = max(max_sigma, float(np.max(np.abs(s_ff - s_sf))))

    for pt in face_pts:
        g_ff = _grad_ff(pt)
        g_sf = _grad_sf(pt)
        s_ff = _sigma(g_ff)
        s_sf = _sigma(g_sf)
        t_ff = s_ff @ n
        t_sf = s_sf @ n
        max_trac = max(max_trac, float(np.max(np.abs(t_ff - t_sf))))

    # gradN consistency checks at a representative point
    check_pt = np.array([0.3, 0.2, 0.4], dtype=float)
    gradN = mortar_mod.hex27_gradN(check_pt, elem_coords, tol=1e-12)
    sum_grad = np.sum(gradN, axis=0)
    max_sum_grad = float(np.max(np.abs(sum_grad)))
    moment = elem_coords.T @ gradN
    max_moment_grad = float(np.max(np.abs(moment - np.eye(3))))

    print(
        f"[diag][hex27][sigma] max_grad_diff={max_grad:.3e} "
        f"max_sigma_diff={max_sigma:.3e} max_trac_diff={max_trac:.3e}"
    )
    print(
        f"[diag][hex27][sigma] max_grad_ff_vs_A={max_grad_ff:.3e} "
        f"max_grad_sf_vs_A={max_grad_sf:.3e}"
    )
    print(
        f"[diag][hex27][gradN] max_sum_grad={max_sum_grad:.3e} "
        f"max_moment_grad={max_moment_grad:.3e}"
    )
    if verbose:
        print(f"[diag][hex27][sigma] A=\n{A}")


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

    if os.getenv("DIAG_DUMP_SKFEM_QP", "0") == "1":
        quad_pts_top = np.asarray(fb_u_top.X[:, 0, :]).T
        quad_w_top = np.asarray(fb_u_top.W[0, :])
        np.savez("quad_top.npz", quad_pts=quad_pts_top, quad_w=quad_w_top)
        print(
            "[diag] dumped skfem quad -> quad_top.npz",
            f"pts={quad_pts_top.shape}",
            f"w={quad_w_top.shape}",
        )

        quad_pts_bot = np.asarray(fb_u_bot.X[:, 0, :]).T
        quad_w_bot = np.asarray(fb_u_bot.W[0, :])
        np.savez("quad_bot.npz", quad_pts=quad_pts_bot, quad_w=quad_w_bot)
        print(
            "[diag] dumped skfem quad -> quad_bot.npz",
            f"pts={quad_pts_bot.shape}",
            f"w={quad_w_bot.shape}",
        )
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

    params = {"h": fb_u_top.mesh_parameters()}
    K = skfem.asm(bilin, fbasis, **params)
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

    def _is_node_major() -> bool:
        node_major = np.repeat(scalar_doflocs, value_dim, axis=0)
        return np.allclose(node_major, vector_doflocs, atol=atol)

    def _is_comp_major() -> bool:
        comp_major = np.tile(scalar_doflocs, (value_dim, 1))
        return np.allclose(comp_major, vector_doflocs, atol=atol)

    if _is_node_major():
        order = "node"
    elif _is_comp_major():
        order = "component"
    else:
        order = "unknown"

    if order == "component":
        perm_vec = np.array([comp * n_nodes + perm_nodes[node] for node in range(n_nodes) for comp in range(value_dim)], dtype=int)
    else:
        perm_vec = np.array([perm_nodes[node] * value_dim + comp for node in range(n_nodes) for comp in range(value_dim)], dtype=int)

    if os.getenv("DIAG_VEC_ORDER", "0") == "1":
        print(f"[diag][skfem][vec-order] order={order} n_nodes={n_nodes} value_dim={value_dim}")
    return perm_vec


if __name__ == "__main__":
    alpha = 10.0
    h = 1.0
    quad_order = int(os.getenv("QUAD_ORDER", "5"))
    if quad_order > 5:
        print(f"[compare] quad_order={quad_order} not supported; using quad_order=5")
        quad_order = 5
    diag_verbose = os.getenv("DIAG_VERBOSE", "0") == "1"
    diag_voln = os.getenv("DIAG_VOLN", "0") == "1"
    diag_quad = os.getenv("DIAG_QUAD", "0") == "1"

    only_elems = [s.strip().lower() for s in os.getenv("COMPARE_ELEMS", "").split(",") if s.strip()]

    def _enabled(name: str) -> bool:
        return not only_elems or name.lower() in only_elems

    cases = [
        ("penalty", True, False),
        ("traction", False, True),
        ("full", True, True),
    ]
    elems = ["hex8", "hex27", "tet4", "tet10"]

    for elem in elems:
        if not _enabled(elem):
            continue
        if elem == "hex27":
            _diag_hex27_param(verbose=diag_verbose)
            _diag_hex27_shape_compare(verbose=diag_verbose)
            _diag_hex27_sigma(verbose=diag_verbose)
            if diag_voln:
                _diag_hex27_volumeN_compare(quad_order)
            if diag_quad:
                _diag_hex27_quad_compare(quad_order)
            coords_hex27, _conn_hex27, _order_hex27 = _fluxfem_mesh_for("hex27")
        _diag_contact_surface(elem, quad_order, verbose=diag_verbose)
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
            sign = -1.0
            K_ff = build_fluxfem_contact(
                elem,
                alpha=alpha,
                h=h_use,
                use_penalty=use_penalty,
                use_traction=use_traction,
                normal_sign=sign,
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
                f"[{elem}/{name}/n={sign:+.0f}] {h_note} rel_inf={rel_inf:.3e} "
                f"rel_2={rel_2:.3e} rel_max={rel_max:.3e}"
            )
            print(
                f"[{elem}/{name}/n={sign:+.0f}] rel_diff_inf={rel_diff_inf:.3e} "
                f"rel_diff_2={rel_diff_2:.3e} rel_diff_max={rel_diff_max:.3e}"
            )
            if elem == "hex27" and name == "full":
                _diag_hex27_dof_groups(K_ff, K_sf, coords_hex27)
                _diag_hex27_block_compare(K_ff, K_sf, coords_hex27)
                _diag_hex27_component_perm(K_ff, K_sf, coords_hex27)
                _diag_hex27_face_mask(K_ff, K_sf, coords_hex27)
