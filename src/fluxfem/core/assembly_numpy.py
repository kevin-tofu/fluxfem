from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from .space_numpy import _numpy_basis_tables, _numpy_elem_coords, _numpy_elem_coords_chunk

if TYPE_CHECKING:
    from .assembly import PairReturn
    from .space import FESpaceBase
    from ..solver import SparsityPattern


def is_numpy_scalar_diffusion_body_force_pair_fast_path(
    space: "FESpaceBase",
    bilinear_form: Any,
    linear_form: Any,
) -> bool:
    basis = getattr(space, "basis", None)
    if basis is None:
        return False
    if int(getattr(space, "value_dim", 1)) != 1:
        return False
    n_nodes = int(getattr(basis, "n_nodes", -1))
    if n_nodes <= 0:
        return False
    if int(getattr(space, "n_ldofs", -1)) != n_nodes:
        return False
    if getattr(bilinear_form, "__name__", None) != "diffusion_form":
        return False
    if getattr(linear_form, "__name__", None) != "scalar_body_force_form":
        return False
    if getattr(bilinear_form, "__module__", "") != "fluxfem.physics.diffusion":
        return False
    if getattr(linear_form, "__module__", "") != "fluxfem.core.assembly":
        return False
    return (
        hasattr(basis, "shape_functions")
        and hasattr(basis, "shape_grads_ref")
        and hasattr(basis, "quad_weights")
    )


def is_numpy_scalar_diffusion_fast_path(space: "FESpaceBase", form: Any) -> bool:
    basis = getattr(space, "basis", None)
    if basis is None:
        return False
    n_nodes = int(getattr(basis, "n_nodes", -1))
    return (
        int(getattr(space, "value_dim", 1)) == 1
        and n_nodes > 0
        and int(getattr(space, "n_ldofs", -1)) == n_nodes
        and getattr(form, "__name__", None) == "diffusion_form"
        and getattr(form, "__module__", "") == "fluxfem.physics.diffusion"
        and hasattr(basis, "shape_functions")
        and hasattr(basis, "shape_grads_ref")
        and hasattr(basis, "quad_weights")
    )


def is_numpy_scalar_body_force_fast_path(space: "FESpaceBase", form: Any) -> bool:
    basis = getattr(space, "basis", None)
    if basis is None:
        return False
    n_nodes = int(getattr(basis, "n_nodes", -1))
    return (
        int(getattr(space, "value_dim", 1)) == 1
        and n_nodes > 0
        and int(getattr(space, "n_ldofs", -1)) == n_nodes
        and getattr(form, "__name__", None) == "scalar_body_force_form"
        and getattr(form, "__module__", "") == "fluxfem.core.assembly"
        and hasattr(basis, "shape_functions")
        and hasattr(basis, "shape_grads_ref")
        and hasattr(basis, "quad_weights")
    )


def is_numpy_scalar_diffusion_pair_fast_path(
    test_space: "FESpaceBase",
    trial_space: "FESpaceBase",
    form: Any,
) -> bool:
    test_basis = getattr(test_space, "basis", None)
    trial_basis = getattr(trial_space, "basis", None)
    if test_basis is None or trial_basis is None:
        return False
    if getattr(form, "__name__", None) != "diffusion_form":
        return False
    if getattr(form, "__module__", "") != "fluxfem.physics.diffusion":
        return False
    if int(getattr(test_space, "value_dim", 1)) != 1 or int(getattr(trial_space, "value_dim", 1)) != 1:
        return False
    if int(getattr(test_space, "elem_dofs").shape[0]) != int(getattr(trial_space, "elem_dofs").shape[0]):
        return False
    if test_space.mesh is not trial_space.mesh:
        return False
    return (
        hasattr(test_basis, "shape_grads_ref")
        and hasattr(test_basis, "quad_weights")
        and hasattr(trial_basis, "shape_grads_ref")
        and hasattr(trial_basis, "quad_weights")
        and np.allclose(np.asarray(test_basis.quad_weights), np.asarray(trial_basis.quad_weights))
        and np.allclose(np.asarray(test_basis.quad_points), np.asarray(trial_basis.quad_points))
    )


def assemble_numpy_scalar_diffusion_body_force_data(
    space: "FESpaceBase",
    *,
    bilinear_params: float,
    linear_params: float,
    n_chunks: int | None,
    pad_trace: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del pad_trace
    use_full_elem_cache = n_chunks is None
    elem_coords = _numpy_elem_coords(space) if use_full_elem_cache else None
    n_elems = int(space.mesh.conn.shape[0]) if elem_coords is None else int(elem_coords.shape[0])
    n_ldofs = int(space.n_ldofs)
    rows = np.asarray(space.get_elem_rows(), dtype=int)
    N_ref, dN_dxi = _numpy_basis_tables(space)
    w_ref = np.asarray(space.basis.quad_weights, dtype=float)

    if n_chunks is None:
        chunk_size = n_elems
        n_chunks_eff = 1
    else:
        chunk_size = max(1, int(np.ceil(n_elems / int(n_chunks))))
        n_chunks_eff = int(np.ceil(n_elems / chunk_size))

    K_parts: list[np.ndarray] = []
    F_parts: list[np.ndarray] = []
    F = np.zeros((int(space.n_dofs),), dtype=float)
    for chunk_idx in range(int(n_chunks_eff)):
        start = chunk_idx * int(chunk_size)
        stop = min(start + int(chunk_size), n_elems)
        if stop <= start:
            continue
        Xe = elem_coords[start:stop] if elem_coords is not None else _numpy_elem_coords_chunk(space, start, stop)
        J = np.einsum("eia,qik->eqak", Xe, dN_dxi)
        detJ = np.linalg.det(J)
        J_inv = np.linalg.inv(J)
        dN_dx = np.einsum("qik,eqka->eqia", dN_dxi, J_inv)
        Ke = float(bilinear_params) * np.einsum(
            "q,eq,eqia,eqja->eij",
            w_ref,
            detJ,
            dN_dx,
            dN_dx,
        )
        fe = float(linear_params) * np.einsum(
            "q,eq,qi->ei",
            w_ref,
            detJ,
            N_ref,
        )
        K_parts.append(Ke.reshape(-1))
        F_parts.append(fe.reshape(-1))
        chunk_rows = rows[start * n_ldofs : stop * n_ldofs]
        np.add.at(F, chunk_rows, fe.reshape(-1))

    K_data = np.concatenate(K_parts, axis=0) if K_parts else np.zeros((0,), dtype=float)
    F_data = np.concatenate(F_parts, axis=0) if F_parts else np.zeros((0,), dtype=float)
    return K_data, F, F_data


def assemble_numpy_scalar_diffusion_pair_fast(
    space: "FESpaceBase",
    bilinear_params: float,
    linear_params: float,
    *,
    pattern: "SparsityPattern",
    n_chunks: int | None,
    pad_trace: bool,
) -> "PairReturn":
    from ..solver import FluxSparseMatrix

    K_data, F, _F_data = assemble_numpy_scalar_diffusion_body_force_data(
        space,
        bilinear_params=bilinear_params,
        linear_params=linear_params,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )
    return FluxSparseMatrix(pattern, K_data), F


def assemble_numpy_scalar_diffusion_rectangular_operator(
    test_space: "FESpaceBase",
    trial_space: "FESpaceBase",
    *,
    bilinear_params: float,
):
    from ..solver import FluxSparseOperator

    elem_coords = _numpy_elem_coords(test_space)
    n_elems = int(test_space.elem_dofs.shape[0])
    test_elem_dofs = np.asarray(test_space.elem_dofs, dtype=int)
    trial_elem_dofs = np.asarray(trial_space.elem_dofs, dtype=int)
    N_test, dN_dxi_test = _numpy_basis_tables(test_space)
    _N_trial, dN_dxi_trial = _numpy_basis_tables(trial_space)
    w_ref = np.asarray(test_space.basis.quad_weights, dtype=float)

    J = np.einsum("eia,qik->eqak", elem_coords, dN_dxi_test)
    detJ = np.linalg.det(J)
    J_inv = np.linalg.inv(J)
    dN_dx_test = np.einsum("qik,eqka->eqia", dN_dxi_test, J_inv)
    dN_dx_trial = np.einsum("qik,eqka->eqia", dN_dxi_trial, J_inv)
    Ke = float(bilinear_params) * np.einsum(
        "q,eq,eqia,eqja->eij",
        w_ref,
        detJ,
        dN_dx_test,
        dN_dx_trial,
    )

    n_test_ldofs = int(test_elem_dofs.shape[1])
    n_trial_ldofs = int(trial_elem_dofs.shape[1])
    rows = np.repeat(test_elem_dofs, n_trial_ldofs, axis=1).reshape(-1)
    cols = np.tile(trial_elem_dofs, (1, n_test_ldofs)).reshape(-1)
    data = Ke.reshape(-1)
    return FluxSparseOperator(
        rows,
        cols,
        data,
        shape=(int(test_space.n_dofs), int(trial_space.n_dofs)),
    )
