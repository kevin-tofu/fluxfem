"""Space variants (closure vs pytree) consistency checks."""
import numpy as np
import jax

import fluxfem as ff


def _build_mesh():
    return ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()


def test_space_data_from_closure_and_pytree():
    mesh = _build_mesh()
    space_closure = ff.make_hex_space(mesh, dim=1, intorder=2)
    space_pytree = ff.make_hex_space_pytree(mesh, dim=1, intorder=2)

    data_closure = ff.SpaceData.from_space(space_closure)
    data_pytree = ff.SpaceData.from_space(space_pytree)

    assert data_closure.n_dofs == data_pytree.n_dofs
    assert data_closure.n_ldofs == data_pytree.n_ldofs
    assert np.array_equal(np.asarray(data_closure.elem_dofs), np.asarray(data_pytree.elem_dofs))


def test_closure_space_jit_via_closure():
    mesh = _build_mesh()
    space = ff.make_hex_space(mesh, dim=1, intorder=2)

    def assemble(kappa):
        K = space.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense()
        return K

    assemble_jit = jax.jit(assemble)
    K = assemble_jit(1.0)
    jax.block_until_ready(K)
    assert K.shape == (space.n_dofs, space.n_dofs)


def test_pytree_space_jit_with_space_argument():
    mesh = _build_mesh()
    space = ff.make_hex_space_pytree(mesh, dim=1, intorder=2)

    def assemble(space_arg, kappa):
        K = space_arg.assemble_bilinear_form(ff.diffusion_form, params=kappa).to_dense()
        return K

    assemble_jit = jax.jit(assemble)
    K = assemble_jit(space, 1.0)
    jax.block_until_ready(K)
    assert K.shape == (space.n_dofs, space.n_dofs)
