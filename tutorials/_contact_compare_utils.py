from __future__ import annotations

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
    if order == 1:
        pattern = (0, 1, 2)
    elif order == 2:
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


def build_fluxfem_surface_mesh(elem: str):
    coords, conn, order = fluxfem_mesh_for(elem)
    if elem.startswith("hex"):
        facets = build_hex_facets(conn, order)
    else:
        facets = build_tet_facets(conn, order)
    surf_a = ff.SurfaceMesh.from_facets(coords, facets)
    surf_b = ff.SurfaceMesh.from_facets(coords, facets)
    sm = ff.build_surface_supermesh(surf_a, surf_b)
    return coords, surf_a, surf_b, sm


def build_fluxfem_contact_space(
    elem: str,
    *,
    quad_order: int,
    normal_sign: float | None,
):
    coords, conn, order = fluxfem_mesh_for(elem)
    if elem.startswith("hex"):
        facets = build_hex_facets(conn, order)
    else:
        facets = build_tet_facets(conn, order)
    if normal_sign is None:
        surface = ff.SurfaceMesh.from_facets(coords, facets)
        master = ff.ContactSideSpec.from_surfaces(surface, elem_conn=conn, value_dim=3)
        slave = ff.ContactSideSpec.from_surfaces(surface, elem_conn=conn, value_dim=3)
        contact = ff.ContactPairSpec(master=master, slave=slave).prepare(
            quad_order=quad_order,
        )
    else:
        contact = ff.ContactSurfaceSpace.from_facets(
            coords,
            facets,
            coords,
            facets,
            elem_conn_master=conn,
            elem_conn_slave=conn,
            value_dim_master=3,
            value_dim_slave=3,
            quad_order=quad_order,
            normal_sign=normal_sign,
        )
    return coords, conn, contact


def build_fluxfem_onesided_contact_space(
    elem: str,
    *,
    quad_order: int,
    with_master: bool = False,
):
    coords, conn, order = fluxfem_mesh_for(elem)
    if elem.startswith("hex"):
        facets = build_hex_facets(conn, order)
    else:
        facets = build_tet_facets(conn, order)
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


__all__ = [
    "build_hex_facets",
    "build_tet_facets",
    "tet4_coords",
    "tet10_coords",
    "fluxfem_mesh_for",
    "build_fluxfem_surface_mesh",
    "build_fluxfem_contact_space",
    "build_fluxfem_onesided_contact_space",
]
