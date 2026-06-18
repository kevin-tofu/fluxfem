"""Compare dual/coarse mortar APIs on a small contact interface."""

from __future__ import annotations

import logging
import warnings

import numpy as np

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Running in float32 mode.*", category=RuntimeWarning)

import fluxfem as ff


def make_contact():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2, 3]], dtype=int)
    facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return ff.ContactSurfaceSpace.from_facets(
        coords,
        facets,
        coords,
        facets,
        elem_conn_master=conn,
        elem_conn_slave=conn,
        value_dim_master=1,
        value_dim_slave=1,
        quad_order=1,
    )


def make_builder():
    builder = ff.NumpyCoupledSystemBuilder.from_structural(np.eye(8), np.zeros(8))
    builder.register_field("master", n_dofs=4, value_dim=1, n_nodes=4, offset=0)
    builder.register_field("slave", n_dofs=4, value_dim=1, n_nodes=4, offset=4)
    return builder


def build_from_explicit_ops(contact, multiplier, *, rho: float = 2.0):
    ops = contact.assemble_multiplier(rho=rho, multiplier=multiplier, backend="numpy")
    builder = make_builder()
    builder.add_contact_mortar(ops, master="master", slave="slave", value_dim=1)
    return builder.build(), ops


def build_from_builder_sugar(contact, *, mortar: str, rho: float = 2.0, **kwargs):
    builder = make_builder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        builder.add_contact(
            contact,
            master="master",
            slave="slave",
            family="constraint",
            mortar=mortar,
            rho=rho,
            value_dim=1,
            **kwargs,
        )
    return builder.build()


def dense(system):
    matrix = system.K_contact_lifted if system.K_contact_lifted is not None else system.K_u
    return np.asarray(matrix.toarray())


def main() -> None:
    contact = make_contact()
    n_structural = 8

    dual_explicit, dual_ops = build_from_explicit_ops(contact, ff.MultiplierSpec.dual_mortar())
    dual_sugar = build_from_builder_sugar(contact, mortar="dual")
    dual_error = np.linalg.norm(dense(dual_explicit) - dense(dual_sugar))

    coarse_explicit, coarse_ops = build_from_explicit_ops(
        contact,
        ff.MultiplierSpec.coarse_dual_mortar(rank=1),
    )
    coarse_sugar = build_from_builder_sugar(contact, mortar="coarse_dual", mortar_rank=1)
    coarse_error = np.linalg.norm(dense(coarse_explicit) - dense(coarse_sugar))

    coarse_auto = build_from_builder_sugar(contact, mortar="coarse_dual", mortar_max_rank=2)
    p0_explicit, p0_ops = build_from_explicit_ops(contact, ff.MultiplierSpec.p0_mortar(contact))
    coarse_p0_explicit, coarse_p0_ops = build_from_explicit_ops(
        contact,
        ff.MultiplierSpec.coarse_p0_mortar(contact, patch_ids=np.array([0, 0], dtype=int)),
    )
    coarse_p1_basis = ff.coarse_p1_basis_from_surface_grid(contact.surface_master, shape=(2, 2), axes=(0, 1))
    coarse_p1_explicit, coarse_p1_ops = build_from_explicit_ops(
        contact,
        ff.MultiplierSpec.coarse_p1_mortar(basis=coarse_p1_basis),
    )

    print("dual explicit lambda dofs:", dense(dual_explicit).shape[0] - n_structural)
    print("coarse fixed-rank lambda dofs:", dense(coarse_explicit).shape[0] - n_structural)
    print("coarse auto lambda dofs:", dense(coarse_auto).shape[0] - n_structural)
    print("p0 lambda dofs:", dense(p0_explicit).shape[0] - n_structural)
    print("integrated coarse p0 lambda dofs:", dense(coarse_p0_explicit).shape[0] - n_structural)
    print("integrated coarse p1 lambda dofs:", dense(coarse_p1_explicit).shape[0] - n_structural)
    print("dual explicit-vs-sugar K error:", f"{dual_error:.3e}")
    print("coarse explicit-vs-sugar K error:", f"{coarse_error:.3e}")
    print("dual B rows:", dual_ops.B.shape[0])
    print("coarse B rows:", coarse_ops.B.shape[0])
    print("p0 B rows:", p0_ops.B.shape[0])
    print("integrated coarse p0 B rows:", coarse_p0_ops.B.shape[0])
    print("integrated coarse p1 B rows:", coarse_p1_ops.B.shape[0])


if __name__ == "__main__":
    main()
