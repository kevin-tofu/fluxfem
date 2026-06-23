"""Craig-Bampton ROM with an explicit reference DOF and linear MPC.

This mirrors the core pattern used by RBE3/preload fixture models:

    u_ref - B u_patch = 0

The structural workpiece is reduced by a Craig-Bampton basis.  The explicit
reference DOF is appended after the reduced structural coordinates and is kept
physical so preload springs can be activated by changing only a small block.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import fluxfem as ff


def main() -> None:
    stiffness = jnp.array(
        [
            [8.0, -2.0, 0.0, 0.0],
            [-2.0, 7.0, -1.0, 0.0],
            [0.0, -1.0, 6.0, -1.5],
            [0.0, 0.0, -1.5, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.eye(4, dtype=jnp.float32)

    fixture = ff.ReferencePointFixture(
        "clamp",
        ff.RBE3Patch(
            dofs=jnp.array([[1], [2]], dtype=jnp.int32),
            weights=jnp.ones((2,), dtype=jnp.float32),
        ),
        reference_dofs=jnp.array([4], dtype=jnp.int32),
        direction=jnp.array([1.0], dtype=jnp.float32),
        stiffness=12.0,
        target_displacement=0.03,
    )

    # Full augmented unknown is [u_workpiece(4), u_ref(1)].
    preload_k, preload_f = ff.assemble_reference_fixture_preload([fixture], total_dofs=5)
    k_full = jnp.zeros((5, 5), dtype=jnp.float32).at[:4, :4].set(stiffness) + preload_k
    f_full = jnp.array([0.0, 0.0, 0.0, -0.4, 0.0], dtype=jnp.float32) + preload_f

    constraints = ff.linear_constraint_system_from_reference_fixtures(
        [fixture],
        n_structural_dofs=4,
        total_dofs=5,
    )
    fixed_full = jnp.array([0])
    u_full = constraints.solve(k_full, f_full, fixed_dofs=fixed_full)

    cb = ff.make_craig_bampton_basis(
        stiffness,
        mass,
        retained_dofs=jnp.unique(jnp.concatenate([jnp.array([0, 3], dtype=jnp.int32), fixture.retained_dofs])),
        n_modes=1,
    )

    # Reduced unknown is [q_cb, u_ref].  The reference DOF is appended unchanged.
    reduced_constraints = constraints.project(cb, n_extra_dofs=1)
    k_rom = jnp.zeros((cb.n_reduced + 1, cb.n_reduced + 1), dtype=jnp.float32)
    k_rom = k_rom.at[: cb.n_reduced, : cb.n_reduced].set(cb.project_matrix(stiffness))
    k_rom = k_rom.at[cb.n_reduced, cb.n_reduced].set(preload_k[4, 4])
    f_rom = jnp.concatenate([cb.project_vector(f_full[:4]), preload_f[4:]])

    # DOF 0 is retained, so the full fixed workpiece DOF maps to reduced DOF 0.
    q_rom = reduced_constraints.solve(k_rom, f_rom, fixed_dofs=jnp.array([0]))
    u_rom = reduced_constraints.expand(q_rom)

    print("full displacement:", np.asarray(u_full))
    print("rom displacement: ", np.asarray(u_rom))
    print("max abs error:    ", float(jnp.max(jnp.abs(u_rom - u_full))))
    print("constraint norm:  ", float(jnp.linalg.norm(reduced_constraints.residual(q_rom))))


if __name__ == "__main__":
    main()
