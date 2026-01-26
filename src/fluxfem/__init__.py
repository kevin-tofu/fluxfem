from __future__ import annotations

import importlib

__all__ = [
    "FESpaceBase",
    "FESpace",
    "FESpacePytree",
    "Expr",
    "FieldRef",
    "ParamRef",
    "trial_ref",
    "test_ref",
    "unknown_ref",
    "zero_ref",
    "Params",
    "LinearForm",
    "BilinearForm",
    "ResidualForm",
    "MixedWeakForm",
    "make_mixed_residuals",
    "kernel",
    "MixedFESpace",
    "MixedProblem",
    "MixedDirichletBC",
    "MixedBlockSystem",
    "compile_bilinear",
    "compile_linear",
    "compile_residual",
    "compile_surface_linear",
    "compile_surface_bilinear",
    "compile_mixed_surface_residual",
    "compile_mixed_residual",
    "FormContext",
    "MixedFormContext",
    "VolumeContext",
    "SurfaceContext",
    "FieldPair",
    "ElementVector",
    "vector_load_form",
    "make_space",
    "make_space_pytree",
    "make_hex_basis",
    "make_hex_basis_pytree",
    "make_hex_space",
    "make_hex_space_pytree",
    "make_hex20_basis",
    "make_hex20_basis_pytree",
    "make_hex20_space",
    "make_hex20_space_pytree",
    "make_hex27_basis",
    "make_hex27_basis_pytree",
    "make_hex27_space",
    "make_hex27_space_pytree",
    "make_tet_basis",
    "make_tet_basis_pytree",
    "make_tet_space",
    "make_tet_space_pytree",
    "make_tet10_basis",
    "make_tet10_basis_pytree",
    "make_tet10_space",
    "make_tet10_space_pytree",
    "MeshData",
    "BasisData",
    "SpaceData",
    "make_element_residual_kernel",
    "make_element_bilinear_kernel",
    "make_element_linear_kernel",
    "make_element_jacobian_kernel",
    "make_element_kernel",
    "element_residual",
    "element_jacobian",
    "make_sparsity_pattern",
    "chunk_pad_stats",
    "BatchedAssembler",
    "assemble_functional",
    "assemble_mass_matrix",
    "scalar_body_force_form",
    "make_scalar_body_force_form",
    "vector_body_force_form",
    "linear_elasticity_form",
    "constant_body_force_form",
    "constant_body_force_vector_form",
    "diffusion_form",
    "right_cauchy_green",
    "green_lagrange_strain",
    "deformation_gradient",
    "pk2_neo_hookean",
    "neo_hookean_residual_form",
    "StokesSpaces",
    "make_stokes_spaces",
    "HexMesh",
    "HexMeshPytree",
    "StructuredHexBox",
    "SurfaceMesh",
    "SurfaceMeshPytree",
    "SurfaceWithElemConn",
    "surface_with_elem_conn",
    "SurfaceFormField",
    "SurfaceFormContext",
    "SurfaceBilinearContext",
    "SurfaceSupermesh",
    "vector_surface_load_form",
    "make_vector_surface_load_form",
    "assemble_surface_linear_form",
    "assemble_surface_bilinear_form",
    "MortarMatrix",
    "assemble_mortar_matrices",
    "assemble_contact_onesided_floor",
    "assemble_mixed_surface_residual",
    "assemble_mixed_surface_jacobian",
    "map_surface_facets_to_tet_elements",
    "map_surface_facets_to_hex_elements",
    "tri_area",
    "tri_quadrature",
    "facet_triangles",
    "facet_shape_values",
    "volume_shape_values_at_points",
    "quad_shape_and_local",
    "quad9_shape_values",
    "hex27_gradN",
    "ContactSurfaceSpace",
    "ContactSide",
    "OneSidedContact",
    "OneSidedContactSurfaceSpace",
    "facet_gap_values",
    "active_contact_facets",
    "tag_axis_minmax_facets",
    "load_gmsh_mesh",
    "load_gmsh_hex_mesh",
    "load_gmsh_tet_mesh",
    "make_surface_from_facets",
    "build_surface_supermesh",
    "MortarMatrix",
    "assemble_mortar_matrices",
    "assemble_mixed_surface_residual",
    "assemble_mixed_surface_jacobian",
    "map_surface_facets_to_tet_elements",
    "TetMesh",
    "TetMeshPytree",
    "StructuredTetBox",
    "StructuredTetTensorBox",
    "BaseMeshPytree",
    "SurfaceWithFacetMap",
    "bbox_predicate",
    "plane_predicate",
    "axis_plane_predicate",
    "slab_predicate",
    "HexTriLinearBasis",
    "HexTriLinearBasisPytree",
    "HexSerendipityBasis20",
    "HexSerendipityBasis20Pytree",
    "HexTriQuadraticBasis27",
    "HexTriQuadraticBasis27Pytree",
    "TetLinearBasis",
    "TetLinearBasisPytree",
    "TetQuadraticBasis10",
    "TetQuadraticBasis10Pytree",
    "lame_parameters",
    "isotropic_3d_D",
    "spdirect_solve_cpu",
    "spdirect_solve_gpu",
    "spdirect_solve_jax",
    "coo_to_csr",
    "SparsityPattern",
    "FluxSparseMatrix",
    "FluxBlockMatrix",
    "coalesce_coo",
    "concat_flux",
    "block_diag_flux",
    "build_block_system",
    "split_block_matrix",
    "BlockSystem",
    "LinearSolver",
    "NonlinearSolver",
    "petsc_solve",
    "petsc_shell_solve",
    "petsc_is_available",
    "DirichletBC",
    "enforce_dirichlet_dense",
    "enforce_dirichlet_dense_jax",
    "enforce_dirichlet_fluxsparse",
    "enforce_dirichlet_sparse",
    "enforce_dirichlet_system",
    "split_dirichlet_matrix",
    "free_dofs",
    "condense_dirichlet_fluxsparse",
    "condense_dirichlet_fluxsparse_coo",
    "condense_dirichlet_dense",
    "restrict_flux_to_free",
    "expand_dirichlet_solution",
    "condense_dirichlet_system",
    "cg_solve",
    "make_block_jacobi_preconditioner",
    "build_cg_operator",
    "CGOperator",
    "NonlinearAnalysis",
    "NewtonLoopConfig",
    "LoadStepResult",
    "NewtonSolveRunner",
    "solve_nonlinear",
    "LinearAnalysis",
    "LinearSolveConfig",
    "LinearStepResult",
    "LinearSolveRunner",
    "newton_solve",
    "write_vtu",
    "write_displacement_vtu",
    "make_jitted_residual",
    "make_jitted_jacobian",
    "block",
    "make_elastic_point_data",
    "write_elastic_vtu",
    "make_point_data_displacement",
    "write_point_data_vtu",
    "principal_stresses",
    "principal_sum",
    "max_shear_stress",
    "von_mises_stress",
]

_PHYSICS_EXPORTS = {
    "lame_parameters",
    "isotropic_3d_D",
    "linear_elasticity_form",
    "vector_body_force_form",
    "constant_body_force_vector_form",
    "diffusion_form",
    "right_cauchy_green",
    "green_lagrange_strain",
    "deformation_gradient",
    "pk2_neo_hookean",
    "neo_hookean_residual_form",
    "make_elastic_point_data",
    "write_elastic_vtu",
    "make_point_data_displacement",
    "write_point_data_vtu",
    "interpolate_at_points",
    "principal_stresses",
    "principal_sum",
    "max_shear_stress",
    "von_mises_stress",
}

_TOOLS_VIS_EXPORTS = {
    "write_vtu",
    "write_displacement_vtu",
}

_TOOLS_JIT_EXPORTS = {
    "make_jitted_residual",
    "make_jitted_jacobian",
}

_CORE_EXPORTS = set(__all__) - _PHYSICS_EXPORTS - _TOOLS_VIS_EXPORTS - _TOOLS_JIT_EXPORTS


def __getattr__(name: str):
    if name in _PHYSICS_EXPORTS:
        module = importlib.import_module(".physics", __name__)
    elif name in _TOOLS_VIS_EXPORTS:
        module = importlib.import_module(".tools.visualizer", __name__)
    elif name in _TOOLS_JIT_EXPORTS:
        module = importlib.import_module(".tools.jit", __name__)
    elif name in _CORE_EXPORTS:
        module = importlib.import_module(".core", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError


def read_version_from_pyproject():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        match = re.search(r'version\s*=\s*"(.*?)"', content)
        if match:
            return match.group(1)
    return "0.0.0"


def get_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return read_version_from_pyproject()


__version__ = get_version("fluxfem")
