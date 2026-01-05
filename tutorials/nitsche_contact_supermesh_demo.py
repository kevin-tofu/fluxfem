import os
import numpy as np
from skfem.helpers import dot, ddot, sym_grad, mul
from scipy.sparse.linalg import minres
import scipy
from scipy.sparse import bmat, csr_matrix, coo_matrix, lil_matrix

import skfem
from skfem.helpers import dot, ddot, sym_grad
from skfem import InteriorBasis, FacetBasis, ElementTetP1, ElementVectorH1

from skfem.supermeshing import intersect, elementwise_quadrature

# Demo: Nitsche contact assembled on a supermesh (not projection-based mortar).


import numpy as np
from skfem import *
from skfem.supermeshing import intersect, elementwise_quadrature
from skfem.models.elasticity import (linear_elasticity, lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, sym_grad, jump, mul
from skfem.io.json import from_file
from pathlib import Path


mesh_top = skfem.MeshTet.init_tensor(
    np.linspace(0, 2, 20),
    np.linspace(0, 2, 20),
    np.linspace(0, 1, 10)
)
mesh_bot = skfem.MeshTet.init_tensor(
    np.linspace(0.5, 1.5, 20),
    np.linspace(0.5, 1.5, 20),
    np.linspace(-0.5, 0.0, 5)
)


def is_contact_surface(x):
    return np.isclose(x[2], 0.0)


def is_force_surface(x):
    return np.isclose(x[2], 1.0)


def is_dirichlet_surface(x):
    return np.isclose(x[2], -0.5)


mesh_top = mesh_top.with_boundaries({
    "contact": is_contact_surface,
    "force": is_force_surface
})
mesh_bot = mesh_bot.with_boundaries({
    "contact": is_contact_surface,
    "dirichlet": is_dirichlet_surface
})


elem = skfem.ElementTetP1()
u_elem = skfem.ElementVectorH1(elem)
basis_top = skfem.InteriorBasis(mesh_top, u_elem)
basis_bot = skfem.InteriorBasis(mesh_bot, u_elem)

E, nu = 210e9, 0.3
mu = E/(2*(1+nu))
lam = E*nu/((1+nu)*(1-2*nu))
C = linear_stress(*lame_parameters(E, nu))


# @skfem.BilinearForm
# def a_elastic(u, v, w):
#     eps_u = sym_grad(u)      # symmetric gradient of u
#     eps_v = sym_grad(v)
#     return lam * np.trace(eps_u) * np.trace(eps_v) + 2*mu * ddot(eps_u, eps_v)


@skfem.BilinearForm
def a_elastic(u, v, w):
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    I = np.eye(3).reshape(3, 3, 1)
    return lam * ddot(eps_u, I) * ddot(eps_v, I) + 2 * mu * ddot(eps_u, eps_v)


K1 = a_elastic.assemble(basis_top)
K2 = a_elastic.assemble(basis_bot)


basis_top = skfem.Basis(mesh_top, ElementVectorH1(skfem.ElementTetP1()))
basis_bot = skfem.Basis(mesh_bot, ElementVectorH1(skfem.ElementTetP1()))

elem_s = ElementTetP1()
elem_v = ElementVectorH1(elem_s)
m1t, orig1 = mesh_top.trace('contact', mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
m2t, orig2 = mesh_bot.trace('contact', mtype=skfem.MeshTri, project=lambda p: p[[0, 1]])
m12, t1, t2 = intersect(m1t, m2t)
quad_order = int(os.getenv("QUAD_ORDER", "5"))
try:
    quad1 = elementwise_quadrature(m1t, m12, t1, intorder=quad_order)
    quad2 = elementwise_quadrature(m2t, m12, t2, intorder=quad_order)
except TypeError:
    quad1 = elementwise_quadrature(m1t, m12, t1)
    quad2 = elementwise_quadrature(m2t, m12, t2)


fb_u_top = FacetBasis(
    mesh_top, elem_v, facets=orig1[t1], quadrature=quad1
)
fb_u_bot = FacetBasis(
    mesh_bot, elem_v, facets=orig2[t2], quadrature=quad2
)
fbasis = fb_u_top * fb_u_bot


aplha_w = 100.0  # 10-100?
aplha_w = 10000
alpha = 20 * (aplha_w * mu + lam)


@skfem.BilinearForm
def bilin_mortar(u1, u2, v1, v2, w):

    ju = u1 - u2
    jv = v1 - v2
    t_u = 0.5 * (mul(C(sym_grad(u1)), w.n) + mul(C(sym_grad(u2)), w.n))
    t_v = 0.5 * (mul(C(sym_grad(v1)), w.n) + mul(C(sym_grad(v2)), w.n))
    return (
        (alpha / w.h) * dot(ju, jv)
        - dot(t_u, jv)
        - dot(t_v, ju)
    )


params = {
    'h': fb_u_top.mesh_parameters(),
    # 'n': fb_u_top.normals,
}

B = skfem.asm(bilin_mortar, fbasis, **params)

K = bmat([[K1, None],
          [None, K2]], 'csr') + B

# f_mortar = skfem.asm(lin_mortar, fbasis, **params)

print("K1:", K1.shape)
print("K2:", K2.shape)
print("B1:", B.shape)
F_facet_ids = mesh_top.facets_satisfying(is_force_surface)
# F_facet_ids = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))

fbasis = skfem.FacetBasis(mesh_top, elem_v, facets=F_facet_ids)
total_force = 100.0  # [N]


@skfem.Functional
def surface_measure(w):
    return 1.0


A_proj_z = surface_measure.assemble(fbasis)
pressure = total_force / A_proj_z


@skfem.LinearForm
def l_comp(v, w):
    return pressure * v[2]


top_F = l_comp.assemble(fbasis)
bot_F = np.zeros(mesh_bot.p.shape[1] * mesh_bot.dim())
F = np.hstack([top_F, bot_F])

D_facets = mesh_bot.facets_satisfying(is_dirichlet_surface)
D_dofs = basis_bot.get_dofs(facets=D_facets).all() + K1.shape[0]

Kc, Fc, uc, I = skfem.condense(K, F, D=D_dofs)
# u = skfem.solve(Kc, Fc, uc, I)
uc, info = minres(Kc, Fc, rtol=1e-10, maxiter=20000)
uc = scipy.sparse.linalg.spsolve(Kc, Fc)

u = np.zeros(K.shape[0])
u[I] = uc

F_from_U = Kc.dot(u[I])

print("len(t1):", len(t1), "len(t2):", len(t2))
print(u)
print(np.sum(np.abs(F_from_U - Fc)))

F_pred_full = K @ u   # full system prediction
print("K.shape, u.shape, F_pred_full.shape", K.shape, u.shape, F_pred_full.shape)
print("F_pred_full.shape, F.shape", F_pred_full.shape, F.shape)
residual_full = F_pred_full - F

print("len(F_pred_full) =", len(F_pred_full))
print("len(F) =", len(F))
print("‖residual‖₂ =", np.linalg.norm(residual_full))

res_norm = np.linalg.norm(Fc - Kc @ uc)
rhs_norm = np.linalg.norm(Fc)
print("relative residual =", res_norm / rhs_norm)

log_path = os.getenv("NITSCHE_SKFEM_LOG")
if log_path:
    summary_lines = [
        f"K1_shape={K1.shape}",
        f"K2_shape={K2.shape}",
        f"B_shape={B.shape}",
        f"len_t1={len(t1)}",
        f"len_t2={len(t2)}",
        f"sum_abs_Fdiff={np.sum(np.abs(F_from_U - Fc))}",
        f"residual_norm_2={np.linalg.norm(residual_full)}",
        f"relative_residual={res_norm / rhs_norm}",
    ]
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"wrote summary log: {log_path}")


def export_combined_vtu(
    mesh_top, mesh_bot,
    u, file_path="combined.vtu"
):
    import meshio

    # ---------------------------
    # 1. coords and cells
    # ---------------------------
    if isinstance(mesh_top, skfem.MeshTet):
        cell_type = "tetra"
    elif isinstance(mesh_top, skfem.MeshHex):
        cell_type = "hexahedron"
    else:
        raise ValueError("Unsupported mesh type")

    n_top_nodes = mesh_top.p.shape[1]
    n_bot_nodes = mesh_bot.p.shape[1]

    # points
    points = np.vstack([mesh_top.p.T, mesh_bot.p.T])

    # cells (offset bottom mesh node indices)
    cells = [(
        cell_type,
        np.vstack([
            mesh_top.t.T,
            mesh_bot.t.T + n_top_nodes
        ])
    )]

    # u_top_bot = u[:(n_top_nodes+n_bot_nodes)*3]
    n_nodes_total = n_top_nodes + n_bot_nodes
    u_top_bot = u[: n_nodes_total * 3].reshape(n_nodes_total, 3)

    # ---------------------------
    # 3. node_tag: Dirichlet / Force / Contact
    # ---------------------------
    node_tag = np.zeros(n_top_nodes + n_bot_nodes, dtype=int)
    top_nodes_contact = np.unique(mesh_top.facets[:, mesh_top.boundaries["contact"]])
    bot_nodes_contact = np.unique(mesh_bot.facets[:, mesh_bot.boundaries["contact"]]) + n_top_nodes
    nodes_contact = np.hstack([top_nodes_contact, bot_nodes_contact])
    # dofs_dirichlet = basis_top.get_dofs("dirichlet").all()
    bot_nodes_dirichlet = np.unique(
        mesh_bot.facets[:, mesh_bot.boundaries["dirichlet"]]
    ) + n_top_nodes
    nodes_dirichlet = bot_nodes_dirichlet
    top_nodes_force = np.unique(mesh_top.facets[:, mesh_top.boundaries["force"]])
    nodes_force = top_nodes_force
    node_tag[nodes_contact] = 1
    node_tag[nodes_dirichlet] = 2
    node_tag[nodes_force] = 3

    # ---------------------------
    # 4. cells_tag
    # ---------------------------
    # cells_tag = np.zeros(mesh_top.nelements + mesh_bot.nelements, dtype=int)

    meshio_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={
            "u": u_top_bot,
            "node_tag": node_tag,
        },
        # cell_data={
        #     "cells_tag": [cells_tag]
        # }
    )

    meshio_mesh.write(file_path)
    print(f"✅ Exported combined VTU: {file_path}")


export_combined_vtu(
    mesh_top,
    mesh_bot,
    u,
    "combined-nitsche.vtu"
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(F_from_U)
ax2.plot(Fc)

fig.savefig("mortar.png")

np.savez("init_from_nitsche.npz", u=u)
