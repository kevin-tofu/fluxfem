from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def main():
    nx, ny, nz = 2, 1, 1
    lx, ly, lz = 1.0, 0.2, 0.2
    E, nu = 1_000.0, 0.3
    n_modes = 3

    mesh = ff.StructuredHexBox(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz).build()
    space = ff.make_hex_space(mesh, dim=3, intorder=2)
    D = ff.isotropic_3d_D(E, nu)

    K = space.assemble_bilinear_form(ff.linear_elasticity_form, params=D).to_dense()
    # Use lumped mass for this compact ROM setup demo. It keeps the generalized
    # eigenproblem well-conditioned while the tutorial focuses on surface-derived
    # retained/contact DOFs.
    M = jnp.diag(space.assemble_mass_matrix(lumped=True))

    coords = np.asarray(mesh.coords)
    xmax = float(coords[:, 0].max())
    contact_facets = np.asarray(
        mesh.boundary_facets_where(lambda pts: np.allclose(pts[:, 0], xmax, atol=1e-8))
    )
    contact_surface = ff.make_surface_from_facets(coords, contact_facets)

    retained = ff.retained_dofs_from_surface(contact_surface, dim=3)
    cb = ff.make_craig_bampton_basis(K, M, retained_dofs=retained, n_modes=n_modes)
    Kr = cb.project_matrix(K)
    Mr = cb.project_matrix(M)

    # Plane is slightly inside the +x face, so a leftward displacement activates contact.
    contact_kin = ff.plane_contact_kinematics_from_surface(
        contact_surface,
        dim=3,
        n_total_nodes=mesh.n_nodes,
        normal=jnp.array([1.0, 0.0, 0.0]),
        plane_offset=xmax - 0.01,
    )
    contact = ff.PlanePenaltyContact(contact_kin, penalty=2_000.0, smoothing=1e-3)

    structural_full = lambda u: K @ u
    full_residual = ff.compose_residuals(structural_full, contact.residual)
    reduced_internal = ff.reduced_residual_from_full(cb, full_residual)

    q0 = jnp.zeros((cb.n_reduced,), dtype=K.dtype)
    u0 = cb.expand(q0)
    initial_state = contact.state_from_displacement(u0)

    # Drive retained x-translation coordinates leftward to demonstrate contact diagnostics.
    q_probe = q0.at[: cb.n_retained : 3].set(-0.02)
    u_probe = cb.expand(q_probe)
    probe_state = contact.state_from_displacement(u_probe)
    Rr_probe = reduced_internal(q_probe)
    Jr_probe = jnp.asarray(ff.reduced_jacobian_from_full(cb, full_residual)(q_probe))

    print("FE surface Craig-Bampton contact setup")
    print(f"full DOFs: {space.n_dofs}, retained DOFs: {cb.n_retained}, reduced DOFs: {cb.n_reduced}")
    print(f"contact nodes: {contact_kin.dofs.shape[0]}, initial active: {int(jnp.count_nonzero(initial_state.active))}")
    print(f"probe active: {int(jnp.count_nonzero(probe_state.active))}")
    print(f"probe min gap: {float(jnp.min(probe_state.gaps)):.6e}")
    print(f"||Kr||_F={float(jnp.linalg.norm(Kr)):.6e}, ||Mr||_F={float(jnp.linalg.norm(Mr)):.6e}")
    print(f"||Rr_probe||_2={float(jnp.linalg.norm(Rr_probe)):.6e}, Jr shape={tuple(Jr_probe.shape)}")


if __name__ == "__main__":
    main()
