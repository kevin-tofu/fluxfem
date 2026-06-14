"""Linear solver, Dirichlet, and multi-RHS behavior checks."""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

import fluxfem as ff
from fluxfem import solver as ff_solver
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from fluxfem.solver.bc import add_neumann_load, add_robin, facet_area


def test_enforce_dirichlet_dense_simple():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    F = np.array([1.0, 0.0], dtype=float)
    dofs = [0]
    vals = [1.0]

    Kc, Fc = ff.enforce_dirichlet_dense(K, F, dofs, vals)
    np.testing.assert_allclose(Kc[0, :], [1.0, 0.0])
    np.testing.assert_allclose(Kc[:, 0], [1.0, 0.0])
    assert Fc[0] == 1.0


def test_condense_and_expand_dirichlet():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    F = np.array([1.0, 0.0], dtype=float)
    dofs = [0]
    vals = [1.0]

    Kc, Fc, free, dir_dofs, dir_vals = ff.condense_dirichlet_dense(K, F, dofs, vals)
    # Reduced system is scalar: (2)*u1 = 1  => u1 = 0.5
    u_free = np.linalg.solve(Kc, Fc)
    u_full = ff.expand_dirichlet_solution(u_free, free, dir_dofs, dir_vals, n_total=2)
    np.testing.assert_allclose(u_full, [1.0, 0.5])


def test_dirichlet_bc_scalar_values_and_free_dofs():
    bc = ff.DirichletBC([0, 2], 1.5)
    np.testing.assert_allclose(bc.vals, [1.5, 1.5])
    np.testing.assert_array_equal(bc.free_dofs(3), [1])


def test_dirichlet_bc_from_boundary_dofs():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=1)
    bc = ff.DirichletBC.from_boundary_dofs(
        mesh,
        lambda pts: np.isclose(pts[:, 0], 0.0, atol=1e-8),
        components=[0],
        dof_per_node=1,
    )
    assert bc.dofs.size > 0
    free = bc.free_dofs(space.n_dofs)
    assert free.size == space.n_dofs - bc.dofs.size


def test_dirichlet_bc_from_bbox():
    mesh = ff.StructuredHexBox(nx=2, ny=2, nz=2, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=1)
    bc = ff.DirichletBC.from_bbox(mesh, components=[0], dof_per_node=1, values=0.0)
    assert bc.dofs.size > 0
    assert bc.dofs.size < space.n_dofs
    np.testing.assert_allclose(bc.vals, 0.0)


def test_mixed_dirichlet_check_equal():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=1)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("U", space),
            "v": ff.NamedSpace("V", space),
        }
    ).to_fe_space()

    bc_u = ff.DirichletBC([0], [1.0])
    bc_v = ff.DirichletBC([0], [1.0])
    mixed_bc = mixed.make_dirichlet(u=bc_u, v=bc_v, merge="check_equal")
    assert mixed_bc.dir_dofs.size == 2

    bc_u_bad = ff.DirichletBC([0, 0], [1.0, 2.0])
    with pytest.raises(ValueError):
        mixed.make_dirichlet(u=bc_u_bad, merge="check_equal")


def test_mixed_problem_solve_condense():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=1)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("U", space),
            "v": ff.NamedSpace("V", space),
        }
    ).to_fe_space()
    K = np.eye(mixed.n_dofs, dtype=float)
    b = np.arange(mixed.n_dofs, dtype=float)
    bc = mixed.make_dirichlet(u=([0], [0.0]))

    import fluxfem.helpers_wf as wf

    residuals = ff.make_mixed_residuals(
        u=lambda v, u, p: v * wf.dOmega(),
        v=lambda v, u, p: v * wf.dOmega(),
    )
    prob = ff.MixedProblem(mixed, residuals)
    u, _info = prob.solve(K, b, dirichlet=bc, dirichlet_mode="condense", n_total=mixed.n_dofs)
    assert u.shape[0] == mixed.n_dofs


def test_mixed_build_block_system_zero_blocks():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=1)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("U", space),
            "v": ff.NamedSpace("V", space),
        }
    ).to_fe_space()
    n = int(space.n_dofs)

    Kuu = np.eye(n, dtype=float)
    Kvv = 2.0 * np.eye(n, dtype=float)
    rhs = {"u": np.ones(n, dtype=float), "v": np.zeros(n, dtype=float)}
    system = mixed.build_block_system(
        diag={"u": Kuu, "v": Kvv},
        rhs=rhs,
    )
    u_slice = mixed.field_slices["u"]
    v_slice = mixed.field_slices["v"]

    np.testing.assert_allclose(system.K[u_slice, v_slice], 0.0)
    np.testing.assert_allclose(system.K[v_slice, u_slice], 0.0)
    np.testing.assert_allclose(system.K[u_slice, u_slice], Kuu)
    np.testing.assert_allclose(system.K[v_slice, v_slice], Kvv)
    np.testing.assert_allclose(system.F[u_slice], 1.0)
    np.testing.assert_allclose(system.F[v_slice], 0.0)


def test_mixed_build_block_system_constraints_check_equal():
    mesh = ff.StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()
    space = ff.make_hex_space(mesh, dim=1, intorder=1)
    mixed = ff.MixedSpaces(
        {
            "u": ff.NamedSpace("U", space),
            "v": ff.NamedSpace("V", space),
        }
    ).to_fe_space()
    n = int(space.n_dofs)

    rhs = {"u": np.zeros(n, dtype=float), "v": np.zeros(n, dtype=float)}
    system = mixed.build_block_system(
        diag={"u": np.eye(n, dtype=float), "v": np.eye(n, dtype=float)},
        rhs=rhs,
        constraints={"u": ([0], [0.0])},
    )
    assert system.free_dofs.size == mixed.n_dofs - 1

    with pytest.raises(ValueError):
        mixed.build_block_system(
            diag={"u": np.eye(n, dtype=float), "v": np.eye(n, dtype=float)},
            rhs=rhs,
            constraints={"u": ([0, 0], [1.0, 2.0])},
            merge="check_equal",
        )


def test_block_system_flux_and_split():
    rhs = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0, 5.0])}
    system = ff.build_block_system(
        diag=[np.eye(2, dtype=float), np.eye(3, dtype=float)],
        sizes={"a": 2, "b": 3},
        rhs=[rhs["a"], rhs["b"]],
    )
    assert system.K.shape == (5, 5)
    np.testing.assert_allclose(system.F, [1.0, 2.0, 3.0, 4.0, 5.0])
    parts = system.split(np.arange(5, dtype=float))
    np.testing.assert_allclose(parts["a"], [0.0, 1.0])
    np.testing.assert_allclose(parts["b"], [2.0, 3.0, 4.0])


def test_block_system_constraints_mapping():
    rhs = {"a": np.ones(2), "b": np.ones(2)}
    system = ff.build_block_system(
        diag=[np.eye(2, dtype=float), np.eye(2, dtype=float)],
        sizes={"a": 2, "b": 2},
        rhs=[rhs["a"], rhs["b"]],
        constraints=[None, ([0], [0.0])],
        format="dense",
    )
    assert system.K.shape == (3, 3)
    assert system.free_dofs.size == 3


def test_split_block_matrix_dense():
    mat = np.arange(16, dtype=float).reshape(4, 4)
    blocks = ff.split_block_matrix(mat, sizes={"a": 2, "b": 2})
    np.testing.assert_allclose(blocks["a"]["a"], mat[:2, :2])
    np.testing.assert_allclose(blocks["a"]["b"], mat[:2, 2:])
    np.testing.assert_allclose(blocks["b"]["a"], mat[2:, :2])
    np.testing.assert_allclose(blocks["b"]["b"], mat[2:, 2:])


def test_block_make_add_contiguous():
    mat = np.arange(16, dtype=float).reshape(4, 4)
    diag = ff_solver.block_diag(a=np.eye(2), b=2.0 * np.eye(2))
    blocks = ff_solver.make_block_matrix(
        diag=diag,
        add_contiguous=mat,
        sizes={"a": 2, "b": 2},
    )
    assert isinstance(blocks, ff_solver.FluxBlockMatrix)
    np.testing.assert_allclose(blocks["a"]["a"], mat[:2, :2] + np.eye(2))
    np.testing.assert_allclose(blocks["b"]["b"], mat[2:, 2:] + 2.0 * np.eye(2))


def test_block_make_rel_symmetric():
    rel = {("a", "b"): np.arange(6, dtype=float).reshape(2, 3)}
    blocks = ff_solver.make_block_matrix(
        diag=ff_solver.block_diag(a=np.eye(2), b=np.eye(3)),
        rel=rel,
        sizes={"a": 2, "b": 3},
        symmetric=True,
        transpose_rule="T",
    )
    np.testing.assert_allclose(blocks["a"]["b"], rel[("a", "b")])
    np.testing.assert_allclose(blocks["b"]["a"], rel[("a", "b")].T)


def test_block_make_diag_sequence():
    blocks = ff_solver.make_block_matrix(
        diag=[np.eye(2), 2.0 * np.eye(3)],
        sizes={"a": 2, "b": 3},
    )
    np.testing.assert_allclose(blocks["a"]["a"], np.eye(2))
    np.testing.assert_allclose(blocks["b"]["b"], 2.0 * np.eye(3))


def test_block_diag_order():
    diag = ff_solver.block_diag(order=("b", "a"), a=1.0, b=2.0)
    assert list(diag.keys()) == ["b", "a"]


def test_enforce_dirichlet_sparse():
    # same system but via FluxSparseMatrix
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    data = np.array([2.0, -1.0, -1.0, 2.0])
    pattern = ff.SparsityPattern(rows=rows, cols=cols, n_dofs=2)
    A = ff.FluxSparseMatrix(pattern, data)
    F = np.array([1.0, 0.0], dtype=float)

    Kc, Fc = ff.enforce_dirichlet_sparse(A, F, [0], [1.0])
    Kc = Kc.toarray()
    np.testing.assert_allclose(Kc[0, :], [1.0, 0.0])
    np.testing.assert_allclose(Kc[:, 0], [1.0, 0.0])
    assert Fc[0] == 1.0


def test_cg_solve_matches_dense():
    # solve 2x2 SPD system with CG via FluxSparseMatrix
    K = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    F = np.array([1.0, 0.0], dtype=np.float32)
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    pattern = ff.SparsityPattern(rows=rows, cols=cols, n_dofs=2)
    A = ff.FluxSparseMatrix(pattern, data)

    u_dense = np.linalg.solve(K, F)
    u_cg, info = ff.cg_solve(A, F, tol=1e-10, maxiter=50)
    np.testing.assert_allclose(np.asarray(u_cg), u_dense, rtol=1e-6, atol=1e-6)
    assert float(info["residual_norm"]) < 1e-6


def test_neumann_load_added():
    # Quad face with unit area, traction (1,2,3) in dim=3 → total force shared by 4 nodes
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facet = np.array([0, 1, 2, 3], dtype=int)
    area = facet_area(coords, facet)
    assert np.isclose(area, 1.0)
    F = np.zeros(4 * 3, dtype=float)
    F_new = add_neumann_load(F, facet, traction=[1.0, 2.0, 3.0], dim=3, coords=coords)
    # each node gets area/4 * traction
    expected = np.tile(np.array([0.25, 0.5, 0.75]), 4)
    np.testing.assert_allclose(F_new, expected)


def test_robin_adds_diag_and_rhs():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    facet = np.array([0, 1, 2, 3], dtype=int)
    K = np.zeros((4, 4), dtype=float)
    F = np.zeros(4, dtype=float)
    F_new, K_new = add_robin(
        F, K, facet, alpha=2.0, g=3.0, dim=1, coords=coords
    )
    # weight = alpha * area / m = 2 * 1 / 4 = 0.5
    assert np.allclose(np.diag(K_new), 0.5)
    assert np.allclose(F_new, 0.5 * 3.0)


def test_linear_solver_multi_rhs_enforce_dirichlet():
    K = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=float)
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_bc, F_bc = ff.enforce_dirichlet_dense(K, F, dofs, vals)
    u_expected = spsolve(sp.csr_matrix(K_bc), F_bc)

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(K, F, dirichlet=(dofs, vals), dirichlet_mode="enforce")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def test_linear_solver_multi_rhs_condense_dirichlet():
    K = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=float)
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_ff, F_f, free, dir_dofs, dir_vals = ff.condense_dirichlet_dense(K, F, dofs, vals)
    u_free = spsolve(sp.csr_matrix(K_ff), F_f)
    u_expected = ff.expand_dirichlet_solution(u_free, free, dir_dofs, dir_vals, n_total=K.shape[0])

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(K, F, dirichlet=(dofs, vals), dirichlet_mode="condense")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def _flux_matrix_2x2():
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    data = np.array([4.0, -1.0, -1.0, 3.0])
    pattern = ff.SparsityPattern(rows=rows, cols=cols, n_dofs=2)
    return ff.FluxSparseMatrix(pattern, data)


def test_linear_solver_multi_rhs_enforce_dirichlet_fluxsparse():
    A = _flux_matrix_2x2()
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_bc, F_bc = ff.enforce_dirichlet_sparse(A, F, dofs, vals)
    u_expected = spsolve(K_bc, F_bc)

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(A, F, dirichlet=(dofs, vals), dirichlet_mode="enforce")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def test_linear_solver_multi_rhs_condense_dirichlet_fluxsparse():
    A = _flux_matrix_2x2()
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    dofs = [0]
    vals = [1.0]

    K_ff, F_f, free, dir_dofs, dir_vals = ff.condense_dirichlet_fluxsparse(A, F, dofs, vals)
    u_free = spsolve(K_ff, F_f)
    u_expected = ff.expand_dirichlet_solution(u_free, free, dir_dofs, dir_vals, n_total=2)

    solver = ff.LinearSolver(method="spsolve")
    u_solve, _ = solver.solve(A, F, dirichlet=(dofs, vals), dirichlet_mode="condense")
    np.testing.assert_allclose(u_solve, u_expected, rtol=1e-8, atol=1e-8)


def test_cg_multi_rhs_matches_direct():
    A = _flux_matrix_2x2()
    F = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    u_direct = spsolve(A.to_csr(), F)

    u_cg, info = ff.cg_solve(A, F, tol=1e-12, maxiter=200)
    np.testing.assert_allclose(np.asarray(u_cg), u_direct, rtol=1e-6, atol=1e-6)
    assert len(info.get("iters", [])) == F.shape[1]
