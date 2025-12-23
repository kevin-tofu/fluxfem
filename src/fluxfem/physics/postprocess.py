from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

# from ..core.assembly import build_form_contexts
from ..tools.visualizer import write_vtu
from ..core.interp import interpolate_field_at_element_points


def make_point_data_displacement(mesh, space, u, *, compute_j: bool = True, deformed_scale: float = 1.0):
    """
    Common postprocess helper to build point data dictionaries:
      - displacement
      - deformed_coords = X + scale * u
      - optional J (nodal average of det(F))

    Assumes 3 dof/node ordering [u0,v0,w0, u1,v1,w1, ...].
    """
    coords = np.asarray(mesh.coords)
    u_np = np.asarray(u)
    n_nodes = coords.shape[0]
    if u_np.shape[0] != 3 * n_nodes:
        raise ValueError(f"Expected 3 dof/node vector; got {u_np.shape[0]} entries for {n_nodes} nodes")

    u_nodes = u_np.reshape(n_nodes, 3)
    disp = u_nodes
    deformed_coords = coords + deformed_scale * disp

    point_data = {
        "displacement": disp.astype(np.float64),
        "deformed_coords": deformed_coords.astype(np.float64),
    }

    if compute_j:
        elem_conns = np.asarray(space.elem_dofs)
        # ctxs = build_form_contexts(space)
        ctxs = space.build_form_contexts()
        u_arr = jnp.asarray(u)

        from .elasticity.hyperelastic import deformation_gradient  # local import to avoid circular

        def elem_J(ctx, conn):
            F = deformation_gradient(ctx, u_arr[conn])
            return jnp.mean(jnp.linalg.det(F))

        J_elem = np.asarray(jax.vmap(elem_J)(ctxs, elem_conns), dtype=float)
        J_sum = np.zeros(n_nodes, dtype=float)
        J_cnt = np.zeros(n_nodes, dtype=float)
        for conn, J_mean in zip(elem_conns, J_elem):
            node_ids = np.unique(np.asarray(conn) // 3)
            J_sum[node_ids] += J_mean
            J_cnt[node_ids] += 1.0
        J_nodal = J_sum / np.maximum(J_cnt, 1.0)
        point_data["J"] = J_nodal.astype(np.float64)

    return point_data


def write_point_data_vtu(mesh, space, u, filepath: str, *, compute_j: bool = True, deformed_scale: float = 1.0):
    """Write VTU with displacement/deformed_coords and optional J."""
    pdata = make_point_data_displacement(mesh, space, u, compute_j=compute_j, deformed_scale=deformed_scale)
    write_vtu(mesh, filepath, point_data=pdata)


__all__ = ["make_point_data_displacement", "write_point_data_vtu", "interpolate_at_points"]
def interpolate_at_points(space, u, points: np.ndarray):
    """
    Interpolate displacement field at given physical points (Hex8 only, structured search).
    - points: (m,3) array of physical coordinates.
    Returns: (m,3) interpolated displacement.
    """
    pts = np.asarray(points, dtype=float)
    mesh = space.mesh
    # Only support StructuredHexBox-backed HexMesh (regular grid)
    if not hasattr(mesh, "origin") and not hasattr(mesh, "lx"):
        raise NotImplementedError("interpolate_at_points currently supports StructuredHexBox meshes.")
    coords = np.asarray(mesh.coords)
    # grid dimensions from origin/extents and nx,ny,nz inferred from spacing
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    zs = np.unique(coords[:, 2])
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    ox, oy, oz = xs.min(), ys.min(), zs.min()
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

    # map physical point to element indices and local coords
    def phys_to_elem_local(p):
        x, y, z = p
        i = min(max(int(np.floor((x - ox) / dx)), 0), nx - 1)
        j = min(max(int(np.floor((y - oy) / dy)), 0), ny - 1)
        k = min(max(int(np.floor((z - oz) / dz)), 0), nz - 1)
        # local coords in [-1,1]
        xi = 2 * ((x - (ox + i * dx)) / dx) - 1
        eta = 2 * ((y - (oy + j * dy)) / dy) - 1
        zeta = 2 * ((z - (oz + k * dz)) / dz) - 1
        elem_idx = k * (nx * ny) + j * nx + i
        return elem_idx, np.array([xi, eta, zeta], dtype=float)

    elem_indices = []
    locals = []
    for p in pts:
        e, loc = phys_to_elem_local(p)
        elem_indices.append(e)
        locals.append(loc)
    locals = np.stack(locals, axis=0)  # (m,3)

    vals_per_elem = interpolate_field_at_element_points(space, u, locals)  # (n_elem,m,3) but m is same locals; pick per elem
    vals = np.zeros((len(pts), 3), dtype=float)
    for idx, e in enumerate(elem_indices):
        vals[idx] = vals_per_elem[e, idx]
    return vals
