from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.sparse as sp

import fluxfem as ff
from tests.cb_test_utils import (
    assert_static_cb_projection_with_extra_matches_full,
    constrained_omegas,
    csr,
    projected_cb_system,
)


def _build_solid_shell_beam_model():
    length = 1.0
    width = 0.4
    height = 0.2
    solid_mesh = ff.StructuredHexBox(nx=2, ny=1, nz=1, lx=length, ly=width, lz=height).build()
    solid_space = ff.make_hex_space(solid_mesh, dim=3, intorder=2)
    solid_K = csr(solid_space.assemble(ff.linear_elasticity_form, params=ff.isotropic_3d_D(2.0e5, 0.30)))
    solid_F = np.zeros((solid_space.n_dofs,), dtype=float)
    solid_coords = np.asarray(solid_mesh.coords, dtype=float)

    shell_xy, shell_conn = ff.structured_plate_grid(nx=2, ny=1, length_x=length, length_y=width)
    shell_coords = np.column_stack([shell_xy[:, 0], shell_xy[:, 1], np.full(shell_xy.shape[0], height)])
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, shear_mode="mitc4")
    shell_K = ff.assemble_shell_stiffness(shell_coords, shell_conn, shell_section, format="csr")
    shell_F = ff.assemble_shell_uniform_load(shell_coords, shell_conn, (0.0, 0.0, -1.0))

    x_max = float(np.max(solid_coords[:, 0]))
    face_nodes = np.flatnonzero(np.isclose(solid_coords[:, 0], x_max, atol=1.0e-10)).astype(int)
    face_coords = solid_coords[face_nodes]
    face_dofs = ff.vector_dofs_from_nodes(face_nodes, dim=3)
    x_ref = face_coords.mean(axis=0)

    beam_coords, beam_conn = ff.structured_beam_chain(
        n_elems=2,
        length=0.8,
        origin=x_ref,
        direction=(1.0, 0.0, 0.0),
    )
    beam_section = ff.BeamSection(E=2.0e5, G=7.7e4, A=1.0e-2, Iy=1.5e-5, Iz=1.5e-5, J=3.0e-5)
    beam_K = ff.assemble_beam_stiffness(beam_coords, beam_conn, beam_section, format="csr")
    beam_tip = beam_coords.shape[0] - 1
    beam_F = ff.assemble_beam_point_load(beam_coords.shape[0], beam_tip, force=(0.0, 0.0, -5.0))

    structural_K = sp.block_diag((solid_K, shell_K, beam_K), format="csr")
    structural_F = np.concatenate([solid_F, np.asarray(shell_F, dtype=float), np.asarray(beam_F, dtype=float)])
    shell_offset = solid_space.n_dofs
    beam_offset = shell_offset + shell_K.shape[0]

    top_nodes = np.flatnonzero(np.isclose(solid_coords[:, 2], height, atol=1.0e-10))
    matched_shell, matched_solid, shell_local_dofs, solid_tie_dofs = ff.shell_solid_translational_tie_dofs(
        shell_coords,
        solid_coords,
        solid_nodes=top_nodes,
        tol=1.0e-7,
    )

    builder = ff.NumpyCoupledSystemBuilder.from_structural(structural_K, structural_F)
    builder.register_field("solid", n_dofs=solid_space.n_dofs, value_dim=1, offset=0)
    builder.register_field("shell", n_dofs=shell_K.shape[0], value_dim=1, offset=shell_offset)
    builder.register_field("beam", n_dofs=beam_K.shape[0], value_dim=1, offset=beam_offset)
    builder.add_dof_tie_constraint(
        master="solid",
        slave="shell",
        master_dofs=solid_tie_dofs,
        slave_dofs=shell_local_dofs,
    )

    weights = ff.build_rbe3_weights(x_ref, face_coords, method="equal")
    builder.add_distributed_coupling(
        source="solid",
        source_dofs=face_dofs,
        remote="beam_interface_remote",
        point=x_ref,
        slave_coords=face_coords,
        weights=weights,
        backend="numpy",
    )
    builder.add_dof_tie_constraint(
        master="beam_interface_remote",
        slave="beam",
        master_dofs=np.arange(6),
        slave_dofs=ff.beam_node_dofs([0]),
    )

    fixed_solid = solid_mesh.boundary_dofs_where(
        lambda pts: np.isclose(pts[:, 0], 0.0, atol=1.0e-10),
        components="xyz",
        dof_per_node=3,
    )
    fixed_shell_rot = shell_offset + ff.shell_node_dofs(np.flatnonzero(np.isclose(shell_coords[:, 0], 0.0)), "rxryrz")
    fixed_dofs = np.unique(np.concatenate([fixed_solid, fixed_shell_rot]))

    return {
        "system": builder.build(),
        "solid_space": solid_space,
        "shell_offset": shell_offset,
        "shell_n_dofs": shell_K.shape[0],
        "shell_coords": shell_coords,
        "shell_conn": shell_conn,
        "beam_coords": beam_coords,
        "beam_conn": beam_conn,
        "face_dofs": face_dofs,
        "solid_tie_dofs": solid_tie_dofs,
        "shell_tie_dofs": shell_offset + shell_local_dofs,
        "matched_shell_nodes": matched_shell,
        "matched_solid_nodes": matched_solid,
        "remote_dofs": builder.resolve_block_dofs("beam_interface_remote", local_dofs=np.arange(6)),
        "beam_root_dofs": beam_offset + ff.beam_node_dofs([0]),
        "beam_tip_dofs": beam_offset + ff.beam_node_dofs([beam_tip]),
        "fixed_dofs": fixed_dofs,
    }


def test_solid_shell_beam_mixed_model_projects_through_cb_rom():
    model = _build_solid_shell_beam_model()
    system = model["system"]
    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    K_lifted, F_lifted = system.assemble(format="csr")

    n_structural = model["solid_space"].n_dofs + model["shell_n_dofs"] + model["beam_coords"].shape[0] * 6
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    F = np.asarray(F_lifted[:n_primary], dtype=float)
    C = K_lifted[n_primary:, :n_primary].tocsr()

    retained_structural = np.unique(
        np.concatenate(
            [
                model["solid_tie_dofs"],
                model["shell_tie_dofs"],
                model["face_dofs"],
                model["beam_root_dofs"],
                model["beam_tip_dofs"],
            ]
        )
    )
    assert_static_cb_projection_with_extra_matches_full(K, F, C, fixed, retained_structural, n_structural, n_extra)

    u_all = np.asarray(system.solve(format="csr", dirichlet_dofs=fixed, dirichlet_vals=np.zeros((fixed.size,), dtype=float)), dtype=float)
    solid_tie_u = u_all[model["solid_tie_dofs"]].reshape(-1, 3)
    shell_tie_u = u_all[model["shell_tie_dofs"]].reshape(-1, 3)
    np.testing.assert_allclose(solid_tie_u, shell_tie_u, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(u_all[model["remote_dofs"]], u_all[model["beam_root_dofs"]], rtol=1.0e-9, atol=1.0e-9)
    assert float(u_all[model["beam_tip_dofs"]][2]) < float(u_all[model["beam_root_dofs"]][2])


def test_solid_shell_beam_mixed_model_cb_matches_constrained_frequencies():
    model = _build_solid_shell_beam_model()
    fixed = np.asarray(model["fixed_dofs"], dtype=int)
    K_lifted, _F_lifted = model["system"].assemble(format="csr")

    n_structural = model["solid_space"].n_dofs + model["shell_n_dofs"] + model["beam_coords"].shape[0] * 6
    remote_dofs = np.asarray(model["remote_dofs"], dtype=int)
    n_primary = int(remote_dofs.max()) + 1
    n_extra = n_primary - n_structural
    K = K_lifted[:n_primary, :n_primary].tocsr()
    C = K_lifted[n_primary:, :n_primary].tocsr()

    M_solid = csr(model["solid_space"].assemble_mass_matrix(backend="numpy"))
    shell_section = ff.ShellSection(E=2.0e5, nu=0.30, thickness=0.02, rho=1.0, shear_mode="mitc4")
    M_shell = ff.assemble_shell_mass(model["shell_coords"], model["shell_conn"], shell_section, format="csr")
    beam_section = ff.BeamSection(E=2.0e5, G=7.7e4, A=1.0e-2, Iy=1.5e-5, Iz=1.5e-5, J=3.0e-5, rho=1.0)
    M_beam = ff.assemble_beam_mass(model["beam_coords"], model["beam_conn"], beam_section, format="csr")
    M = sp.block_diag((M_solid, M_shell, M_beam, sp.csr_matrix((n_extra, n_extra), dtype=float)), format="csr")

    retained_structural = np.unique(
        np.concatenate(
            [
                model["solid_tie_dofs"],
                model["shell_tie_dofs"],
                model["face_dofs"],
                model["beam_root_dofs"],
                model["beam_tip_dofs"],
            ]
        )
    )
    _structural_free, _cb, K_rom, M_rom, C_rom = projected_cb_system(K, M, C, fixed, retained_structural, n_structural, n_extra)

    full = constrained_omegas(K, M, C, fixed, n_modes=6)
    rom = constrained_omegas(K_rom, M_rom, C_rom, np.array([], dtype=int), n_modes=6)

    np.testing.assert_allclose(rom, full, rtol=1.0e-6, atol=1.0e-5)
