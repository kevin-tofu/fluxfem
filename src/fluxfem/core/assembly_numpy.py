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
