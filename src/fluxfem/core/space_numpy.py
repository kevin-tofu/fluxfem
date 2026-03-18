from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from .forms import (
    FieldPair,
    NumpyFormContext,
    NumpyPrecomputedScalarFormField,
    NumpyPrecomputedVectorFormField,
)

if TYPE_CHECKING:
    from .space import FESpaceClosure


def _numpy_elem_coords(space: "FESpaceClosure") -> np.ndarray:
    mesh = space.mesh
    cached = getattr(mesh, "_ff_elem_coords_numpy", None)
    if cached is None:
        coords = np.asarray(mesh.coords, dtype=float)
        conn = np.asarray(mesh.conn, dtype=np.int64)
        cached = coords[conn]
        setattr(mesh, "_ff_elem_coords_numpy", cached)
    return cached


def _numpy_elem_coords_chunk(space: "FESpaceClosure", start: int, stop: int) -> np.ndarray:
    coords = np.asarray(space.mesh.coords, dtype=float)
    conn = np.asarray(space.mesh.conn[start:stop], dtype=np.int64)
    return coords[conn]


def _numpy_basis_tables(space: "FESpaceClosure") -> tuple[np.ndarray, np.ndarray]:
    basis = space.basis
    N_ref = getattr(basis, "_ff_numpy_shape_functions", None)
    dN_dxi = getattr(basis, "_ff_numpy_shape_grads_ref", None)
    if N_ref is None:
        N_ref = np.asarray(basis.shape_functions(), dtype=float)
        setattr(basis, "_ff_numpy_shape_functions", N_ref)
    if dN_dxi is None:
        dN_dxi = np.asarray(basis.shape_grads_ref(), dtype=float)
        setattr(basis, "_ff_numpy_shape_grads_ref", dN_dxi)
    return N_ref, dN_dxi


def build_form_contexts_numpy(
    space: "FESpaceClosure",
    dep=None,
    *,
    include_x_q: bool = True,
    lightweight: bool = True,
) -> list[NumpyFormContext]:
    if dep is not None:
        raise NotImplementedError("build_form_contexts_numpy currently supports dep=None only.")
    if not lightweight:
        raise NotImplementedError("build_form_contexts_numpy currently supports lightweight=True only.")
    basis = space.basis
    vd = int(space.value_dim)
    elem_coords = _numpy_elem_coords(space)
    N_ref, _dN_dxi = _numpy_basis_tables(space)
    w_ref = np.asarray(basis.quad_weights, dtype=float)
    n_elems = int(elem_coords.shape[0])
    if include_x_q:
        x_q_all = np.einsum("qa,eai->eqi", N_ref, elem_coords)
    else:
        x_q_all = np.broadcast_to(
            np.zeros((1, N_ref.shape[0], elem_coords.shape[2]), dtype=elem_coords.dtype),
            (n_elems, N_ref.shape[0], elem_coords.shape[2]),
        )
    w_all = np.broadcast_to(w_ref[None, :], (n_elems, w_ref.shape[0]))
    ctxs: list[NumpyFormContext] = []
    for e, Xe in enumerate(elem_coords):
        gradN_e, detJ_e = basis.spatial_grads_and_detJ(jax.numpy.asarray(Xe))
        gradN_np = np.asarray(gradN_e, dtype=float)
        detJ_np = np.asarray(detJ_e, dtype=float)
        if vd == 1:
            field = NumpyPrecomputedScalarFormField(
                basis=basis,
                _N=N_ref,
                _gradN=gradN_np,
                _detJ=detJ_np,
            )
        else:
            field = NumpyPrecomputedVectorFormField(
                basis=basis,
                value_dim=vd,
                _N=N_ref,
                _gradN=gradN_np,
                _detJ=detJ_np,
            )
        ctxs.append(
            NumpyFormContext(
                test=field,
                trial=field,
                x_q=x_q_all[e],
                w=w_all[e],
                elem_id=e,
            )
        )
    return ctxs


def build_form_contexts_numpy_chunked(
    space: "FESpaceClosure",
    *,
    chunk_size: int,
    dep=None,
    include_x_q: bool = True,
    lightweight: bool = True,
):
    if dep is not None:
        raise NotImplementedError("build_form_contexts_numpy_chunked currently supports dep=None only.")
    if not lightweight:
        raise NotImplementedError("build_form_contexts_numpy_chunked currently supports lightweight=True only.")
    basis = space.basis
    vd = int(space.value_dim)
    N_ref, _dN_dxi = _numpy_basis_tables(space)
    w_ref = np.asarray(basis.quad_weights, dtype=float)
    n_elems = int(space.mesh.conn.shape[0])
    chunk = max(1, int(chunk_size))
    for start in range(0, n_elems, chunk):
        stop = min(start + chunk, n_elems)
        Xe_chunk = _numpy_elem_coords_chunk(space, start, stop)
        if include_x_q:
            x_q_chunk = np.einsum("qa,eai->eqi", N_ref, Xe_chunk)
        else:
            x_q_chunk = np.broadcast_to(
                np.zeros((1, N_ref.shape[0], Xe_chunk.shape[2]), dtype=Xe_chunk.dtype),
                (Xe_chunk.shape[0], N_ref.shape[0], Xe_chunk.shape[2]),
            )
        w_chunk = np.broadcast_to(w_ref[None, :], (Xe_chunk.shape[0], w_ref.shape[0]))
        ctxs: list[NumpyFormContext] = []
        for local_e, Xe in enumerate(Xe_chunk):
            gradN_e, detJ_e = basis.spatial_grads_and_detJ(jax.numpy.asarray(Xe))
            gradN_np = np.asarray(gradN_e, dtype=float)
            detJ_np = np.asarray(detJ_e, dtype=float)
            if vd == 1:
                field = NumpyPrecomputedScalarFormField(
                    basis=basis,
                    _N=N_ref,
                    _gradN=gradN_np,
                    _detJ=detJ_np,
                )
            else:
                field = NumpyPrecomputedVectorFormField(
                    basis=basis,
                    value_dim=vd,
                    _N=N_ref,
                    _gradN=gradN_np,
                    _detJ=detJ_np,
                )
            ctxs.append(
                NumpyFormContext(
                    test=field,
                    trial=field,
                    x_q=x_q_chunk[local_e],
                    w=w_chunk[local_e],
                    elem_id=start + local_e,
                )
            )
        yield ctxs


def build_form_contexts_pair_numpy(
    test_space: "FESpaceClosure",
    trial_space: "FESpaceClosure",
    *,
    include_x_q: bool = True,
    lightweight: bool = True,
    test_name: str = "V",
    trial_name: str = "U",
) -> list[NumpyFormContext]:
    ctx_test = build_form_contexts_numpy(
        test_space,
        dep=None,
        include_x_q=include_x_q,
        lightweight=lightweight,
    )
    ctx_trial = build_form_contexts_numpy(
        trial_space,
        dep=None,
        include_x_q=include_x_q,
        lightweight=lightweight,
    )
    if len(ctx_test) != len(ctx_trial):
        raise ValueError("build_form_contexts_pair_numpy requires the same number of elements in test and trial spaces.")

    out: list[NumpyFormContext] = []
    for c_test, c_trial in zip(ctx_test, ctx_trial, strict=True):
        if c_test.x_q.shape != c_trial.x_q.shape:
            raise ValueError(
                "build_form_contexts_pair_numpy requires matching quadrature point shapes for test and trial spaces."
            )
        if c_test.w.shape != c_trial.w.shape:
            raise ValueError(
                "build_form_contexts_pair_numpy requires matching quadrature weight shapes for test and trial spaces."
            )
        if not np.allclose(c_test.x_q, c_trial.x_q):
            raise ValueError("build_form_contexts_pair_numpy requires matching physical quadrature points.")
        if not np.allclose(c_test.w, c_trial.w):
            raise ValueError("build_form_contexts_pair_numpy requires matching quadrature weights.")
        spaces = {
            str(test_name): FieldPair(test=c_test.test, trial=c_test.trial, unknown=c_test.trial),
            str(trial_name): FieldPair(test=c_trial.test, trial=c_trial.trial, unknown=c_trial.trial),
            "default": FieldPair(test=c_test.test, trial=c_trial.trial, unknown=c_trial.trial),
        }
        out.append(
            NumpyFormContext(
                test=c_test.test,
                trial=c_trial.trial,
                x_q=c_trial.x_q,
                w=c_trial.w,
                elem_id=c_trial.elem_id,
                spaces=spaces,
                default_space="default",
            )
        )
    return out
