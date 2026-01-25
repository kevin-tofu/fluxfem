from __future__ import annotations
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, TYPE_CHECKING, TypeAlias, TypeVar, Union, cast
import numpy as np
import jax
import jax.numpy as jnp

from ..mesh import HexMesh, StructuredHexBox
from .dtypes import INDEX_DTYPE
from .forms import FormContext
from .space import FESpaceBase

# Shared call signatures for kernels/forms
Array: TypeAlias = jnp.ndarray
P = TypeVar("P")

FormKernel: TypeAlias = Callable[[FormContext, P], Array]
# Form kernels return integrands; element kernels return integrated element arrays.
Kernel: TypeAlias = Callable[[FormContext, P], Array]
ResidualInput: TypeAlias = Array | Mapping[str, Array]
ResidualValue: TypeAlias = Array | Mapping[str, Array]
ResidualForm = Callable[[FormContext, Array, P], Array]
ResidualFormLike = Callable[[FormContext, ResidualInput, P], ResidualValue]
ElementDofMapper = Callable[[Array], Array]

if TYPE_CHECKING:
    from ..solver import FluxSparseMatrix, SparsityPattern
else:
    FluxSparseMatrix = Any
    SparsityPattern = Any

SparseCOO: TypeAlias = tuple[Array, Array, Array, int]
LinearCOO: TypeAlias = tuple[Array, Array, int]
JacobianReturn: TypeAlias = Union[Array, FluxSparseMatrix, SparseCOO]
BilinearReturn: TypeAlias = Union[Array, FluxSparseMatrix, SparseCOO]
LinearReturn: TypeAlias = Union[Array, LinearCOO]
MassReturn: TypeAlias = Union[FluxSparseMatrix, Array]


class ElementBilinearKernel(Protocol):
    def __call__(self, ctx: FormContext) -> Array: ...


class ElementLinearKernel(Protocol):
    def __call__(self, ctx: FormContext) -> Array: ...


class ElementResidualKernel(Protocol):
    def __call__(self, ctx: FormContext, u_elem: Array) -> Array: ...


class ElementJacobianKernel(Protocol):
    def __call__(self, u_elem: Array, ctx: FormContext) -> Array: ...

ElementKernel: TypeAlias = (
    ElementBilinearKernel
    | ElementLinearKernel
    | ElementResidualKernel
    | ElementJacobianKernel
)
def _get_pattern(space: SpaceLike, *, with_idx: bool) -> SparsityPattern | None:
    if hasattr(space, "get_sparsity_pattern"):
        return space.get_sparsity_pattern(with_idx=with_idx)
    return None


def _get_elem_rows(space: SpaceLike) -> Array:
    if hasattr(space, "get_elem_rows"):
        return space.get_elem_rows()
    return space.elem_dofs.reshape(-1)


def chunk_pad_stats(n_elems: int, n_chunks: Optional[int]) -> dict[str, int | float | None]:
    """
    Compute padding overhead for chunked assembly.
    Returns dict with chunk_size, pad, n_pad, and pad_ratio.
    """
    n_elems = int(n_elems)
    if n_chunks is None or n_elems <= 0:
        return {"chunk_size": None, "pad": 0, "n_pad": n_elems, "pad_ratio": 0.0}
    n_chunks = min(int(n_chunks), n_elems)
    chunk_size = (n_elems + n_chunks - 1) // n_chunks
    pad = (-n_elems) % chunk_size
    n_pad = n_elems + pad
    pad_ratio = float(pad) / float(n_elems) if n_elems else 0.0
    return {"chunk_size": int(chunk_size), "pad": int(pad), "n_pad": int(n_pad), "pad_ratio": pad_ratio}


def _maybe_trace_pad(
    stats: dict[str, int | float | None], *, n_chunks: Optional[int], pad_trace: bool
) -> None:
    if not pad_trace or not jax.core.trace_ctx.is_top_level():
        return
    if n_chunks is None:
        return
    print(
        "[pad]",
        f"n_chunks={int(n_chunks)}",
        f"chunk_size={stats['chunk_size']}",
        f"pad={stats['pad']}",
        f"pad_ratio={stats['pad_ratio']:.4f}",
        flush=True,
    )


class BatchedAssembler:
    """
    Assemble on a fixed space with optional masking to keep shapes static.

    Use `mask` to zero padded elements while keeping input shapes fixed.
    """

    def __init__(
        self,
        space: SpaceLike,
        elem_data: FormContext,
        elem_dofs: Array,
        *,
        pattern: SparsityPattern | None = None,
    ) -> None:
        self.space = space
        self.elem_data = elem_data
        self.elem_dofs = elem_dofs
        self.n_elems = int(elem_dofs.shape[0])
        self.n_ldofs = int(space.n_ldofs)
        self.n_dofs = int(space.n_dofs)
        self.pattern = pattern
        self._rows: Array | None = None
        self._cols: Array | None = None

    @classmethod
    def from_space(
        cls,
        space: SpaceLike,
        *,
        dep: jnp.ndarray | None = None,
        pattern: SparsityPattern | None = None,
    ) -> "BatchedAssembler":
        elem_data = space.build_form_contexts(dep=dep)
        return cls(space, elem_data, space.elem_dofs, pattern=pattern)

    def make_mask(self, n_active: int) -> Array:
        n_active = max(0, min(int(n_active), self.n_elems))
        mask: np.ndarray = np.zeros((self.n_elems,), dtype=float)
        if n_active:
            mask[:n_active] = 1.0
        return jnp.asarray(mask)

    def slice(self, n_active: int) -> "BatchedAssembler":
        n_active = max(0, min(int(n_active), self.n_elems))
        elem_data = jax.tree_util.tree_map(lambda x: x[:n_active], self.elem_data)
        elem_dofs = self.elem_dofs[:n_active]
        return BatchedAssembler(self.space, elem_data, elem_dofs, pattern=None)

    def _rows_cols(self) -> tuple[Array, Array]:
        if self.pattern is not None:
            return self.pattern.rows, self.pattern.cols
        if self._rows is None or self._cols is None:
            elem_dofs = self.elem_dofs
            n_ldofs = int(elem_dofs.shape[1])
            rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
            cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
            self._rows = rows
            self._cols = cols
        return self._rows, self._cols

    def assemble_bilinear_with_kernel(
        self, kernel: ElementBilinearKernel, *, mask: Array | None = None
    ) -> FluxSparseMatrix:
        """
        kernel(ctx) -> (n_ldofs, n_ldofs)
        """
        from ..solver import FluxSparseMatrix

        Ke = jax.vmap(kernel)(self.elem_data)
        if mask is not None:
            Ke = Ke * jnp.asarray(mask)[:, None, None]
        data = Ke.reshape(-1)
        if self.pattern is not None:
            return FluxSparseMatrix(self.pattern, data)
        rows, cols = self._rows_cols()
        return FluxSparseMatrix(rows, cols, data, n_dofs=self.n_dofs)

    def assemble_bilinear(
        self,
        form: FormKernel[P],
        params: P,
        *,
        mask: Array | None = None,
        kernel: ElementBilinearKernel | None = None,
        jit: bool = True,
    ) -> FluxSparseMatrix:
        if kernel is None:
            kernel = make_element_bilinear_kernel(form, params, jit=jit)
        return self.assemble_bilinear_with_kernel(kernel, mask=mask)

    def assemble_linear_with_kernel(
        self,
        kernel: ElementLinearKernel,
        *,
        mask: Array | None = None,
        dep: jnp.ndarray | None = None,
    ) -> Array:
        """
        kernel(ctx) -> (n_ldofs,)
        """
        elem_data = self.elem_data if dep is None else self.space.build_form_contexts(dep=dep)
        Fe = jax.vmap(kernel)(elem_data)
        if mask is not None:
            Fe = Fe * jnp.asarray(mask)[:, None]
        rows = self.elem_dofs.reshape(-1)
        data = Fe.reshape(-1)
        return jax.ops.segment_sum(data, rows, self.n_dofs)

    def assemble_linear(
        self,
        form: FormKernel[P],
        params: P,
        *,
        mask: Array | None = None,
        dep: jnp.ndarray | None = None,
        kernel: ElementLinearKernel | None = None,
    ) -> Array:
        if kernel is not None:
            return self.assemble_linear_with_kernel(kernel, mask=mask, dep=dep)
        elem_data = self.elem_data if dep is None else self.space.build_form_contexts(dep=dep)
        includes_measure = getattr(form, "_includes_measure", False)

        def per_element(ctx: FormContext):
            integrand = form(ctx, params)
            if includes_measure:
                return integrand.sum(axis=0)
            wJ = ctx.w * ctx.test.detJ
            return (integrand * wJ[:, None]).sum(axis=0)

        Fe = jax.vmap(per_element)(elem_data)
        if mask is not None:
            Fe = Fe * jnp.asarray(mask)[:, None]
        rows = self.elem_dofs.reshape(-1)
        data = Fe.reshape(-1)
        return jax.ops.segment_sum(data, rows, self.n_dofs)

    def assemble_mass_matrix(
        self, *, mask: Array | None = None, lumped: bool = False
    ) -> MassReturn:
        from ..solver import FluxSparseMatrix

        n_ldofs = self.n_ldofs

        def per_element(ctx: FormContext):
            N = ctx.test.N
            base = jnp.einsum("qa,qb->qab", N, N)
            if hasattr(ctx.test, "value_dim"):
                vd = int(ctx.test.value_dim)
                I = jnp.eye(vd, dtype=N.dtype)
                base = base[:, :, :, None, None] * I[None, None, None, :, :]
                base = base.reshape(base.shape[0], n_ldofs, n_ldofs)
            wJ = ctx.w * ctx.test.detJ
            return jnp.einsum("qab,q->ab", base, wJ)

        Me = jax.vmap(per_element)(self.elem_data)
        if mask is not None:
            Me = Me * jnp.asarray(mask)[:, None, None]
        data = Me.reshape(-1)
        rows, cols = self._rows_cols()

        if lumped:
            M = jnp.zeros((self.n_dofs,), dtype=data.dtype)
            M = M.at[rows].add(data)
            return M

        return FluxSparseMatrix(rows, cols, data, n_dofs=self.n_dofs)

    def assemble_residual_with_kernel(
        self, kernel: ElementResidualKernel, u: Array, *, mask: Array | None = None
    ) -> Array:
        """
        kernel(ctx, u_elem) -> (n_ldofs,)
        """
        u_elems = jnp.asarray(u)[self.elem_dofs]
        elem_res = jax.vmap(kernel)(self.elem_data, u_elems)
        if mask is not None:
            elem_res = elem_res * jnp.asarray(mask)[:, None]
        rows = self.elem_dofs.reshape(-1)
        data = elem_res.reshape(-1)
        return jax.ops.segment_sum(data, rows, self.n_dofs)

    def assemble_residual(
        self,
        res_form: ResidualForm[P],
        u: Array,
        params: P,
        *,
        mask: Array | None = None,
        kernel: ElementResidualKernel | None = None,
    ) -> Array:
        if kernel is None:
            kernel = make_element_residual_kernel(res_form, params)
        return self.assemble_residual_with_kernel(kernel, u, mask=mask)

    def assemble_jacobian_with_kernel(
        self,
        kernel: ElementJacobianKernel,
        u: Array,
        *,
        mask: Array | None = None,
        sparse: bool = True,
        return_flux_matrix: bool = False,
    ) -> JacobianReturn:
        """
        kernel(u_elem, ctx) -> (n_ldofs, n_ldofs)
        """
        from ..solver import FluxSparseMatrix  # local import to avoid circular

        u_elems = jnp.asarray(u)[self.elem_dofs]
        J_e = jax.vmap(kernel)(u_elems, self.elem_data)
        if mask is not None:
            J_e = J_e * jnp.asarray(mask)[:, None, None]
        data = J_e.reshape(-1)
        if sparse:
            if self.pattern is not None:
                if return_flux_matrix:
                    return FluxSparseMatrix(self.pattern, data)
                return self.pattern.rows, self.pattern.cols, data, self.n_dofs
            rows, cols = self._rows_cols()
            if return_flux_matrix:
                return FluxSparseMatrix(rows, cols, data, n_dofs=self.n_dofs)
            return rows, cols, data, self.n_dofs
        rows, cols = self._rows_cols()
        idx = (rows.astype(jnp.int64) * int(self.n_dofs) + cols.astype(jnp.int64)).astype(INDEX_DTYPE)
        n_entries = self.n_dofs * self.n_dofs
        sdn = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        K_flat = jnp.zeros(n_entries, dtype=data.dtype)
        K_flat = jax.lax.scatter_add(K_flat, idx[:, None], data, sdn)
        return K_flat.reshape(self.n_dofs, self.n_dofs)

    def assemble_jacobian(
        self,
        res_form: ResidualForm[P],
        u: Array,
        params: P,
        *,
        mask: Array | None = None,
        kernel: ElementJacobianKernel | None = None,
        sparse: bool = True,
        return_flux_matrix: bool = False,
    ) -> JacobianReturn:
        if kernel is None:
            kernel = make_element_jacobian_kernel(res_form, params)
        return self.assemble_jacobian_with_kernel(
            kernel,
            u,
            mask=mask,
            sparse=sparse,
            return_flux_matrix=return_flux_matrix,
        )

class SpaceLike(FESpaceBase, Protocol):
    pass


def assemble_bilinear_dense(
    space: SpaceLike,
    kernel: FormKernel[P],
    params: P,
    *,
    sparse: bool = False,
    return_flux_matrix: bool = False,
) -> BilinearReturn:
    """
    Similar to scikit-fem's asm(biform, basis).
    kernel: FormContext, params -> (n_ldofs, n_ldofs)
    """
    elem_dofs = space.elem_dofs   # (n_elems, n_ldofs)
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    elem_data = space.build_form_contexts()  # Pytree with leading n_elems in each field

    # apply kernel per element
    def ke_fun(ctx: FormContext):
        return kernel(ctx, params)

    K_e_all = jax.vmap(ke_fun)(elem_data)  # (n_elems, n_ldofs, n_ldofs)

    # ---- scatter into COO format ----
    # row/col indices (n_elems, n_ldofs, n_ldofs)
    pat = _get_pattern(space, with_idx=False)
    if pat is None:
        rows = jnp.repeat(elem_dofs, n_ldofs, axis=1)        # (n_elems, n_ldofs*n_ldofs)
        cols = jnp.tile(elem_dofs, (1, n_ldofs))             # (n_elems, n_ldofs*n_ldofs)
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
    else:
        rows = pat.rows
        cols = pat.cols
    data = K_e_all.reshape(-1)

    # Flatten indices for segment_sum via (row * n_dofs + col)
    idx = rows * n_dofs + cols  # (n_entries,)

    if sparse:
        if return_flux_matrix:
            from ..solver import FluxSparseMatrix  # local import to avoid circular
            return FluxSparseMatrix(rows, cols, data, n_dofs)
        return rows, cols, data, n_dofs

    n_entries = n_dofs * n_dofs
    out = jnp.zeros((n_entries,), dtype=data.dtype)
    out = out.at[idx].add(data)
    K = out.reshape(n_dofs, n_dofs)
    return K


def assemble_bilinear_form(
    space: SpaceLike,
    form: FormKernel[P],
    params: P,
    *,
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,   # None -> no chunking
    dep: jnp.ndarray | None = None,
    kernel: ElementBilinearKernel | None = None,
    jit: bool = True,
    pad_trace: bool = False,
) -> FluxSparseMatrix:
    """
    Assemble a sparse bilinear form into a FluxSparseMatrix.

    Expects form(ctx, params) -> (n_q, n_ldofs, n_ldofs).
    If kernel is provided: kernel(ctx) -> (n_ldofs, n_ldofs).
    """
    from ..solver import FluxSparseMatrix

    if pattern is None:
        if hasattr(space, "get_sparsity_pattern"):
            pat = space.get_sparsity_pattern(with_idx=True)
        else:
            pat = make_sparsity_pattern(space, with_idx=True)
    else:
        pat = pattern
    elem_data = space.build_form_contexts(dep=dep)

    includes_measure = getattr(form, "_includes_measure", False)

    def per_element(ctx):
        integrand = form(ctx, params)                      # (n_q, m, m)
        if includes_measure:
            return integrand.sum(axis=0)
        wJ = ctx.w * ctx.test.detJ                         # (n_q,)
        return (integrand * wJ[:, None, None]).sum(axis=0)  # (m, m)

    if kernel is None:
        kernel = make_element_bilinear_kernel(form, params, jit=jit)

    # --- no-chunk path (your current implementation) ---
    if n_chunks is None:
        K_e_all = jax.vmap(kernel)(elem_data)  # (n_elems, m, m)
        data = K_e_all.reshape(-1)
        return FluxSparseMatrix(pat, data)

    # --- chunked path ---
    n_elems = space.elem_dofs.shape[0]
    if n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer.")
    n_chunks = min(int(n_chunks), int(n_elems))
    chunk_size = (n_elems + n_chunks - 1) // n_chunks
    stats = chunk_pad_stats(n_elems, n_chunks)
    _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
    # Ideally get m from pat (otherwise infer from one element).
    m = getattr(pat, "n_ldofs", None)
    if m is None:
        m = kernel(jax.tree_util.tree_map(lambda x: x[0], elem_data)).shape[0]

    # Pad to fixed-size chunks for JIT stability.
    pad = (-n_elems) % chunk_size
    if pad:
        elem_data_pad = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
            elem_data,
        )
    else:
        elem_data_pad = elem_data

    n_pad = n_elems + pad
    n_chunks = n_pad // chunk_size

    def _slice_first_dim(x, start, size):
        start_idx = (start,) + (0,) * (x.ndim - 1)
        slice_sizes = (size,) + x.shape[1:]
        return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

    def chunk_fn(i):
        start = i * chunk_size
        ctx_chunk = jax.tree_util.tree_map(
            lambda x: _slice_first_dim(x, start, chunk_size),
            elem_data_pad,
        )
        Ke = jax.vmap(kernel)(ctx_chunk)             # (chunk, m, m)
        return Ke.reshape(-1)                             # (chunk*m*m,)

    data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
    data = data_chunks.reshape(-1)[: n_elems * m * m]
    return FluxSparseMatrix(pat, data)


def assemble_mass_matrix(
    space: SpaceLike,
    *,
    lumped: bool = False,
    n_chunks: Optional[int] = None,
    pad_trace: bool = False,
) -> MassReturn:
    """
    Assemble mass matrix M_ij = ∫ N_i N_j dΩ.
    Supports scalar and vector spaces. If lumped=True, rows are summed to diagonal.
    """
    from ..solver import FluxSparseMatrix  # local import to avoid circular

    ctxs = space.build_form_contexts()
    n_ldofs = space.n_ldofs

    def per_element(ctx: FormContext):
        N = ctx.test.N  # (n_q, n_nodes)
        base = jnp.einsum("qa,qb->qab", N, N)  # (n_q, n_nodes, n_nodes)
        if hasattr(ctx.test, "value_dim"):
            vd = int(ctx.test.value_dim)
            I = jnp.eye(vd, dtype=N.dtype)
            base = base[:, :, :, None, None] * I[None, None, None, :, :]
            base = base.reshape(base.shape[0], n_ldofs, n_ldofs)
        wJ = ctx.w * ctx.test.detJ
        return jnp.einsum("qab,q->ab", base, wJ)

    if n_chunks is None:
        M_e_all = jax.vmap(per_element)(ctxs)  # (n_elems, n_ldofs, n_ldofs)
        data = M_e_all.reshape(-1)
    else:
        n_elems = space.elem_dofs.shape[0]
        if n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        n_chunks = min(int(n_chunks), int(n_elems))
        chunk_size = (n_elems + n_chunks - 1) // n_chunks
        stats = chunk_pad_stats(n_elems, n_chunks)
        _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
        pad = (-n_elems) % chunk_size
        if pad:
            ctxs_pad = jax.tree_util.tree_map(
                lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
                ctxs,
            )
        else:
            ctxs_pad = ctxs

        n_pad = n_elems + pad
        n_chunks = n_pad // chunk_size

        def _slice_first_dim(x, start, size):
            start_idx = (start,) + (0,) * (x.ndim - 1)
            slice_sizes = (size,) + x.shape[1:]
            return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

        def chunk_fn(i):
            start = i * chunk_size
            ctx_chunk = jax.tree_util.tree_map(
                lambda x: _slice_first_dim(x, start, chunk_size),
                ctxs_pad,
            )
            Me = jax.vmap(per_element)(ctx_chunk)  # (chunk, n_ldofs, n_ldofs)
            return Me.reshape(-1)

        data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
        data = data_chunks.reshape(-1)[: n_elems * n_ldofs * n_ldofs]

    elem_dofs = space.elem_dofs
    pat = _get_pattern(space, with_idx=False)
    if pat is None:
        rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
        cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
    else:
        rows = pat.rows
        cols = pat.cols

    if lumped:
        n_dofs = space.n_dofs
        M = jnp.zeros((n_dofs,), dtype=data.dtype)
        M = M.at[rows].add(data)
        return M

    return FluxSparseMatrix(rows, cols, data, n_dofs=space.n_dofs)


def assemble_linear_form(
    space: SpaceLike,
    form: FormKernel[P],
    params: P,
    *,
    kernel: ElementLinearKernel | None = None,
    sparse: bool = False,
    n_chunks: Optional[int] = None,
    dep: jnp.ndarray | None = None,
    pad_trace: bool = False,
) -> LinearReturn:
    """
    Expects form(ctx, params) -> (n_q, n_ldofs) and integrates Σ_q form * wJ for RHS.
    If kernel is provided: kernel(ctx) -> (n_ldofs,).
    """
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    elem_data = space.build_form_contexts(dep=dep)

    includes_measure = getattr(form, "_includes_measure", False)

    if kernel is None:
        def per_element(ctx: FormContext):
            integrand = form(ctx, params)  # (n_q, m)
            if includes_measure:
                return integrand.sum(axis=0)
            wJ = ctx.w * ctx.test.detJ     # (n_q,)
            return (integrand * wJ[:, None]).sum(axis=0) # (m,)
    else:
        per_element = kernel

    if n_chunks is None:
        F_e_all = jax.vmap(per_element)(elem_data)            # (n_elems, m)
        data = F_e_all.reshape(-1)
    else:
        n_elems = space.elem_dofs.shape[0]
        m = n_ldofs
        if n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        n_chunks = min(int(n_chunks), int(n_elems))
        chunk_size = (n_elems + n_chunks - 1) // n_chunks
        stats = chunk_pad_stats(n_elems, n_chunks)
        _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
        pad = (-n_elems) % chunk_size
        if pad:
            elem_data_pad = jax.tree_util.tree_map(
                lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
                elem_data,
            )
        else:
            elem_data_pad = elem_data

        n_pad = n_elems + pad
        n_chunks = n_pad // chunk_size

        def _slice_first_dim(x, start, size):
            start_idx = (start,) + (0,) * (x.ndim - 1)
            slice_sizes = (size,) + x.shape[1:]
            return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

        def chunk_fn(i):
            start = i * chunk_size
            ctx_chunk = jax.tree_util.tree_map(
                lambda x: _slice_first_dim(x, start, chunk_size),
                elem_data_pad,
            )
            fe = jax.vmap(per_element)(ctx_chunk)  # (chunk, m)
            return fe.reshape(-1)

        data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
        data = data_chunks.reshape(-1)[: n_elems * m]

    rows = _get_elem_rows(space)

    if sparse:
        return rows, data, n_dofs

    F = jax.ops.segment_sum(data, rows, n_dofs)
    return F


def assemble_functional(space: SpaceLike, form: FormKernel[P], params: P) -> jnp.ndarray:
    """
    Assemble scalar functional J = ∫ form(ctx, params) dΩ.
    Expects form(ctx, params) -> (n_q,) or (n_q, 1).
    """
    elem_data = space.build_form_contexts()

    includes_measure = getattr(form, "_includes_measure", False)

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        if integrand.ndim == 2 and integrand.shape[1] == 1:
            integrand = integrand[:, 0]
        if includes_measure:
            return jnp.sum(integrand)
        wJ = ctx.w * ctx.test.detJ
        return jnp.sum(integrand * wJ)

    vals = jax.vmap(per_element)(elem_data)
    return jnp.sum(vals)


def assemble_jacobian_global(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    sparse: bool = False,
    return_flux_matrix: bool = False,
) -> JacobianReturn:
    """
    Assemble Jacobian (dR/du) from element residual res_form.
    res_form(ctx, u_elem, params) -> (n_q, n_ldofs)
    """
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    elem_data = space.build_form_contexts()

    def fe_fun(u_elem, ctx: FormContext, elem_id):
        ctx_with_id = FormContext(ctx.test, ctx.trial, ctx.x_q, ctx.w, elem_id)
        integrand = res_form(ctx_with_id, u_elem, params)  # (n_q, m)
        wJ = ctx.w * ctx.test.detJ
        fe = (integrand * wJ[:, None]).sum(axis=0)  # (m,)
        return fe

    jac_fun = jax.jacrev(fe_fun, argnums=0)

    u_elems = u[elem_dofs]  # (n_elems, n_ldofs)
    elem_ids = jnp.arange(elem_dofs.shape[0], dtype=INDEX_DTYPE)
    J_e_all = jax.vmap(jac_fun)(u_elems, elem_data, elem_ids)  # (n_elems, m, m)

    pat = _get_pattern(space, with_idx=False)
    if pat is None:
        rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
        cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
    else:
        rows = pat.rows
        cols = pat.cols
    data = J_e_all.reshape(-1)

    if sparse:
        if return_flux_matrix:
            from ..solver import FluxSparseMatrix  # local import to avoid circular
            return FluxSparseMatrix(rows, cols, data, n_dofs)
        return rows, cols, data, n_dofs

    n_entries = n_dofs * n_dofs
    idx = rows * n_dofs + cols
    K_flat = jax.ops.segment_sum(data, idx, n_entries)
    return K_flat.reshape(n_dofs, n_dofs)


def assemble_jacobian_elementwise(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    sparse: bool = False,
    return_flux_matrix: bool = False,
) -> JacobianReturn:
    """
    Assemble Jacobian with element kernels via vmap + scatter_add.
    Recompiles if n_dofs changes, but independent of element count.
    """
    from ..solver import FluxSparseMatrix  # local import to avoid circular

    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    ctxs = space.build_form_contexts()

    def fe_fun(u_elem, ctx: FormContext):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    jac_fun = jax.jacrev(fe_fun, argnums=0)
    u_elems = u[elem_dofs]
    J_e_all = jax.vmap(jac_fun)(u_elems, ctxs)  # (n_elems, m, m)

    pat = _get_pattern(space, with_idx=False)
    if pat is None:
        rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
        cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
    else:
        rows = pat.rows
        cols = pat.cols
    data = J_e_all.reshape(-1)

    if sparse:
        if return_flux_matrix:
            return FluxSparseMatrix(rows, cols, data, n_dofs)
        return rows, cols, data, n_dofs

    n_entries = n_dofs * n_dofs
    idx = rows * n_dofs + cols
    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    K_flat = jnp.zeros(n_entries, dtype=data.dtype)
    K_flat = jax.lax.scatter_add(K_flat, idx[:, None], data, sdn)
    return K_flat.reshape(n_dofs, n_dofs)


def assemble_residual_global(
    space: SpaceLike,
    form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    sparse: bool = False
) -> LinearReturn:
    """
    Assemble residual vector that depends on u.
    form(ctx, u_elem, params) -> (n_q, n_ldofs)
    """
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs

    elem_data = space.build_form_contexts()

    def per_element(ctx: FormContext, conn: jnp.ndarray, elem_id: jnp.ndarray):
        u_elem = u[conn]
        ctx_with_id = FormContext(ctx.test, ctx.trial, ctx.x_q, ctx.w, elem_id)
        integrand = form(ctx_with_id, u_elem, params)  # (n_q, m)
        wJ = ctx.w * ctx.test.detJ
        fe = (integrand * wJ[:, None]).sum(axis=0)
        return fe

    elem_ids = jnp.arange(elem_dofs.shape[0], dtype=INDEX_DTYPE)
    F_e_all = jax.vmap(per_element)(elem_data, elem_dofs, elem_ids)  # (n_elems, m)

    rows = _get_elem_rows(space)
    data = F_e_all.reshape(-1)

    if sparse:
        return rows, data, n_dofs

    F = jax.ops.segment_sum(data, rows, n_dofs)
    return F


def assemble_residual_elementwise(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    sparse: bool = False,
) -> LinearReturn:
    """
    Assemble residual using element kernels via vmap + scatter_add.
    Recompiles if n_dofs changes, but independent of element count.
    """
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    ctxs = space.build_form_contexts()

    def per_element(ctx: FormContext, u_elem: jnp.ndarray):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    u_elems = u[elem_dofs]
    F_e_all = jax.vmap(per_element)(ctxs, u_elems)  # (n_elems, m)
    rows = _get_elem_rows(space)
    data = F_e_all.reshape(-1)

    if sparse:
        return rows, data, n_dofs

    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F = jnp.zeros(n_dofs, dtype=data.dtype)
    F = jax.lax.scatter_add(F, rows[:, None], data, sdn)
    return F


# Backward compatibility aliases (prefer assemble_*_elementwise).
assemble_jacobian_elementwise_xla = assemble_jacobian_elementwise
assemble_residual_elementwise_xla = assemble_residual_elementwise


def make_element_bilinear_kernel(
    form: FormKernel[P], params: P, *, jit: bool = True
) -> ElementBilinearKernel:
    """Element kernel: (ctx) -> Ke."""

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        if getattr(form, "_includes_measure", False):
            return integrand.sum(axis=0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None, None]).sum(axis=0)

    return jax.jit(per_element) if jit else per_element


def make_element_linear_kernel(
    form: FormKernel[P], params: P, *, jit: bool = True
) -> ElementLinearKernel:
    """Element kernel: (ctx) -> fe."""

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        if getattr(form, "_includes_measure", False):
            return integrand.sum(axis=0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    return jax.jit(per_element) if jit else per_element


def make_element_residual_kernel(
    res_form: ResidualForm[P], params: P
) -> ElementResidualKernel:
    """Jitted element residual kernel: (ctx, u_elem) -> fe."""

    def per_element(ctx: FormContext, u_elem: jnp.ndarray):
        integrand = res_form(ctx, u_elem, params)
        if getattr(res_form, "_includes_measure", False):
            return integrand.sum(axis=0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    return jax.jit(per_element)


def make_element_jacobian_kernel(
    res_form: ResidualForm[P], params: P
) -> ElementJacobianKernel:
    """Jitted element Jacobian kernel: (ctx, u_elem) -> Ke."""

    def fe_fun(u_elem, ctx: FormContext):
        integrand = res_form(ctx, u_elem, params)
        if getattr(res_form, "_includes_measure", False):
            return integrand.sum(axis=0)
        wJ = ctx.w * ctx.test.detJ
        return (integrand * wJ[:, None]).sum(axis=0)

    return jax.jit(jax.jacrev(fe_fun, argnums=0))


def element_residual(
    res_form: ResidualFormLike[P], ctx: FormContext, u_elem: ResidualInput, params: P
) -> ResidualValue:
    """
    Element residual vector r_e(u_e) = sum_q w_q * detJ_q * res_form(ctx, u_e, params).
    Returns shape (n_ldofs,).
    """
    integrand = res_form(ctx, u_elem, params)  # (n_q, n_ldofs) or pytree
    includes_measure = getattr(res_form, "_includes_measure", False)
    if isinstance(integrand, jnp.ndarray):
        if includes_measure:
            return jnp.einsum("qa->a", integrand)
        wJ = ctx.w * ctx.test.detJ             # (n_q,)
        return jnp.einsum("qa,q->a", integrand, wJ)
    if hasattr(ctx, "fields") and ctx.fields is not None:
        def _reduce(name, val):
            if isinstance(includes_measure, dict) and includes_measure.get(name, False):
                return jnp.einsum("qa->a", val)
            wJ = ctx.w * ctx.fields[name].test.detJ
            return jnp.einsum("qa,q->a", val, wJ)

        return {name: _reduce(name, val) for name, val in integrand.items()}
    if includes_measure:
        return jax.tree_util.tree_map(lambda x: jnp.einsum("qa->a", x), integrand)
    return jax.tree_util.tree_map(lambda x: jnp.einsum("qa,q->a", x, ctx.w * ctx.test.detJ), integrand)


def element_jacobian(
    res_form: ResidualFormLike[P], ctx: FormContext, u_elem: ResidualInput, params: P
) -> ResidualValue:
    """
    Element Jacobian K_e = d r_e / d u_e (AD via jacfwd), shape (n_ldofs, n_ldofs).
    """
    def _r_elem(u_local):
        return element_residual(res_form, ctx, u_local, params)

    return jax.jacfwd(_r_elem)(u_elem)


def make_element_kernel(
    form: FormKernel[P] | ResidualForm[P],
    params: P,
    *,
    kind: Literal["bilinear", "linear", "residual", "jacobian"],
    jit: bool = True,
) -> ElementKernel:
    """
    Unified entry point for element kernels.

    kind:
      - "bilinear": kernel(ctx) -> (n_ldofs, n_ldofs)
      - "linear": kernel(ctx) -> (n_ldofs,)
      - "residual": kernel(ctx, u_elem) -> (n_ldofs,)
      - "jacobian": kernel(u_elem, ctx) -> (n_ldofs, n_ldofs)
    """
    kind = cast(Literal["bilinear", "linear", "residual", "jacobian"], kind.lower())
    if kind == "bilinear":
        form_bilinear = cast(FormKernel[P], form)
        return make_element_bilinear_kernel(form_bilinear, params, jit=jit)
    if kind == "linear":
        form_linear = cast(FormKernel[P], form)
        def per_element(ctx: FormContext):
            integrand = form_linear(ctx, params)
            if getattr(form_linear, "_includes_measure", False):
                return integrand.sum(axis=0)
            wJ = ctx.w * ctx.test.detJ
            return (integrand * wJ[:, None]).sum(axis=0)

        return jax.jit(per_element) if jit else per_element
    if kind == "residual":
        form_residual = cast(ResidualForm[P], form)
        return make_element_residual_kernel(form_residual, params)
    if kind == "jacobian":
        form_residual = cast(ResidualForm[P], form)
        return make_element_jacobian_kernel(form_residual, params)
    raise ValueError(f"Unknown kernel kind: {kind}")


def make_sparsity_pattern(space: SpaceLike, *, with_idx: bool = True) -> SparsityPattern:
    """
    Build a SparsityPattern (rows/cols[/idx]) that is independent of the solution.
    NOTE: rows/cols ordering matches assemble_jacobian_values(...).reshape(-1)
    so that pattern and data are aligned 1:1. If you change the flattening/
    compression strategy, keep this ordering contract in sync.
    """
    from ..solver import SparsityPattern  # local import to avoid circular

    elem_dofs = jnp.asarray(space.elem_dofs, dtype=INDEX_DTYPE)
    n_dofs = int(space.n_dofs)
    n_ldofs = int(space.n_ldofs)

    rows = jnp.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1).astype(INDEX_DTYPE)
    cols = jnp.tile(elem_dofs, (1, n_ldofs)).reshape(-1).astype(INDEX_DTYPE)

    key = rows.astype(jnp.int64) * jnp.int64(n_dofs) + cols.astype(jnp.int64)
    order = jnp.argsort(key).astype(INDEX_DTYPE)
    rows_sorted = rows[order]
    cols_sorted = cols[order]
    counts = jnp.bincount(rows_sorted, length=n_dofs).astype(INDEX_DTYPE)
    indptr_j = jnp.concatenate([jnp.array([0], dtype=INDEX_DTYPE), jnp.cumsum(counts)])
    indices_j = cols_sorted.astype(INDEX_DTYPE)
    perm = order

    if with_idx:
        idx = (rows.astype(jnp.int64) * jnp.int64(n_dofs) + cols.astype(jnp.int64)).astype(INDEX_DTYPE)
        return SparsityPattern(
            rows=rows,
            cols=cols,
            n_dofs=n_dofs,
            idx=idx,
            perm=perm,
            indptr=indptr_j,
            indices=indices_j,
        )
    return SparsityPattern(
        rows=rows,
        cols=cols,
        n_dofs=n_dofs,
        idx=None,
        perm=perm,
        indptr=indptr_j,
        indices=indices_j,
    )


def assemble_jacobian_values(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementJacobianKernel | None = None,
    n_chunks: Optional[int] = None,
    pad_trace: bool = False,
) -> Array:
    """
    Assemble only the numeric values for the Jacobian (pattern-free).
    """
    ctxs = space.build_form_contexts()
    ker = kernel if kernel is not None else make_element_jacobian_kernel(res_form, params)

    u_elems = u[space.elem_dofs]
    if n_chunks is None:
        J_e_all = jax.vmap(ker)(u_elems, ctxs)  # (n_elem, m, m)
        return J_e_all.reshape(-1)

    n_elems = int(u_elems.shape[0])
    if n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer.")
    n_chunks = min(int(n_chunks), int(n_elems))
    chunk_size = (n_elems + n_chunks - 1) // n_chunks
    stats = chunk_pad_stats(n_elems, n_chunks)
    _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
    pad = (-n_elems) % chunk_size
    if pad:
        ctxs_pad = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
            ctxs,
        )
        u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
    else:
        ctxs_pad = ctxs
        u_elems_pad = u_elems

    n_pad = n_elems + pad
    n_chunks = n_pad // chunk_size
    m = int(space.n_ldofs)

    def _slice_first_dim(x, start, size):
        start_idx = (start,) + (0,) * (x.ndim - 1)
        slice_sizes = (size,) + x.shape[1:]
        return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

    def chunk_fn(i):
        start = i * chunk_size
        ctx_chunk = jax.tree_util.tree_map(
            lambda x: _slice_first_dim(x, start, chunk_size),
            ctxs_pad,
        )
        u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
        J_e = jax.vmap(ker)(u_chunk, ctx_chunk)
        return J_e.reshape(-1)

    data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
    return data_chunks.reshape(-1)[: n_elems * m * m]


def assemble_residual_scatter(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementResidualKernel | None = None,
    sparse: bool = False,
    n_chunks: Optional[int] = None,
    pad_trace: bool = False,
) -> LinearReturn:
    """
    Assemble residual using jitted element kernel + vmap + scatter_add.
    Avoids Python loops; good for JIT stability.

    Note: `res_form` should return the integrand only; quadrature weights and detJ
    are applied in the element kernel (make_element_residual_kernel). Do not multiply
    by w or detJ inside `res_form`.
    """
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    if jax.core.trace_ctx.is_top_level():
        if np.max(elem_dofs) >= n_dofs:
            raise ValueError("elem_dofs contains index outside n_dofs")
        if np.min(elem_dofs) < 0:
            raise ValueError("elem_dofs contains negative index")
    ctxs = space.build_form_contexts()
    ker = kernel if kernel is not None else make_element_residual_kernel(res_form, params)

    u_elems = u[elem_dofs]
    if n_chunks is None:
        elem_res = jax.vmap(ker)(ctxs, u_elems)  # (n_elem, n_ldofs)
    else:
        n_elems = int(u_elems.shape[0])
        if n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        n_chunks = min(int(n_chunks), int(n_elems))
        chunk_size = (n_elems + n_chunks - 1) // n_chunks
        stats = chunk_pad_stats(n_elems, n_chunks)
        _maybe_trace_pad(stats, n_chunks=n_chunks, pad_trace=pad_trace)
        pad = (-n_elems) % chunk_size
        if pad:
            ctxs_pad = jax.tree_util.tree_map(
                lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
                ctxs,
            )
            u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
        else:
            ctxs_pad = ctxs
            u_elems_pad = u_elems

        n_pad = n_elems + pad
        n_chunks = n_pad // chunk_size

        def _slice_first_dim(x, start, size):
            start_idx = (start,) + (0,) * (x.ndim - 1)
            slice_sizes = (size,) + x.shape[1:]
            return jax.lax.dynamic_slice(x, start_idx, slice_sizes)

        def chunk_fn(i):
            start = i * chunk_size
            ctx_chunk = jax.tree_util.tree_map(
                lambda x: _slice_first_dim(x, start, chunk_size),
                ctxs_pad,
            )
            u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
            res_chunk = jax.vmap(ker)(ctx_chunk, u_chunk)
            return res_chunk.reshape(-1)

        data_chunks = jax.vmap(chunk_fn)(jnp.arange(n_chunks))
        elem_res = data_chunks.reshape(-1)[: n_elems * int(space.n_ldofs)].reshape(n_elems, -1)
    if jax.core.trace_ctx.is_top_level():
        if not bool(jax.block_until_ready(jnp.all(jnp.isfinite(elem_res)))):
            bad = int(jnp.count_nonzero(~jnp.isfinite(elem_res)))
            raise RuntimeError(f"[assemble_residual_scatter] elem_res nonfinite: {bad}")

    rows = _get_elem_rows(space)
    data = elem_res.reshape(-1)

    if sparse:
        return rows, data, n_dofs

    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    F = jnp.zeros((n_dofs,), dtype=data.dtype)
    F = jax.lax.scatter_add(F, rows[:, None], data, sdn)
    return F


def assemble_jacobian_scatter(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementJacobianKernel | None = None,
    sparse: bool = False,
    return_flux_matrix: bool = False,
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,
    pad_trace: bool = False,
) -> JacobianReturn:
    """
    Assemble Jacobian using jitted element kernel + vmap + scatter_add.
    If a SparsityPattern is provided, rows/cols are reused without regeneration.
    CONTRACT: The returned `data` ordering matches `pattern.rows/cols` exactly.
    Any change to pattern generation or data flattening must preserve this.
    """
    from ..solver import FluxSparseMatrix  # local import to avoid circular

    pat = pattern if pattern is not None else make_sparsity_pattern(space, with_idx=not sparse)
    data = assemble_jacobian_values(
        space, res_form, u, params, kernel=kernel, n_chunks=n_chunks, pad_trace=pad_trace
    )

    if sparse:
        if return_flux_matrix:
            return FluxSparseMatrix(pat, data)
        return pat.rows, pat.cols, data, pat.n_dofs

    idx = pat.idx
    if idx is None:
        idx = (pat.rows.astype(jnp.int64) * int(pat.n_dofs) + pat.cols.astype(jnp.int64)).astype(INDEX_DTYPE)

    n_entries = pat.n_dofs * pat.n_dofs
    sdn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    K_flat = jnp.zeros(n_entries, dtype=data.dtype)
    K_flat = jax.lax.scatter_add(K_flat, idx[:, None], data, sdn)
    return K_flat.reshape(pat.n_dofs, pat.n_dofs)


# Alias scatter-based assembly as the default public API
def assemble_residual(
    space: SpaceLike,
    form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementResidualKernel | None = None,
    sparse: bool = False,
    n_chunks: Optional[int] = None,
    pad_trace: bool = False,
) -> LinearReturn:
    """
    Assemble the global residual vector (scatter-based).
    If kernel is provided: kernel(ctx, u_elem) -> (n_ldofs,).
    """
    return assemble_residual_scatter(
        space, form, u, params, kernel=kernel, sparse=sparse, n_chunks=n_chunks, pad_trace=pad_trace
    )


def assemble_jacobian(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementJacobianKernel | None = None,
    sparse: bool = True,
    return_flux_matrix: bool = False,
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,
    pad_trace: bool = False,
) -> JacobianReturn:
    """
    Assemble the global Jacobian (scatter-based).
    If kernel is provided: kernel(u_elem, ctx) -> (n_ldofs, n_ldofs).
    """
    return assemble_jacobian_scatter(
        space,
        res_form,
        u,
        params,
        kernel=kernel,
        sparse=sparse,
        return_flux_matrix=return_flux_matrix,
        pattern=pattern,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )


def _make_unit_cube_mesh() -> HexMesh:
    """Single hex element on [0, 1]^3."""
    return StructuredHexBox(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0).build()


def scalar_body_force_form(ctx: FormContext, load: float) -> jnp.ndarray:
    """Linear form for constant scalar body force: f * N."""
    return load * ctx.test.N  # (n_q, n_ldofs)


scalar_body_force_form._ff_kind = "linear"  # type: ignore[attr-defined]
scalar_body_force_form._ff_domain = "volume"  # type: ignore[attr-defined]


def make_scalar_body_force_form(body_force: Callable[[Array], Array]) -> FormKernel[Any]:
    """
    Build a scalar linear form from a callable f(x_q) -> (n_q,).
    """
    def _form(ctx: FormContext, _params):
        f_q = body_force(ctx.x_q)
        return f_q[..., None] * ctx.test.N
    _form._ff_kind = "linear"  # type: ignore[attr-defined]
    _form._ff_domain = "volume"  # type: ignore[attr-defined]
    return _form


# Backward compatibility alias
constant_body_force_form = scalar_body_force_form


def _check_structured_box_connectivity() -> None:
    """Quick connectivity check for nx=2, ny=1, nz=1 (non-structured order)."""
    box = StructuredHexBox(nx=2, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0)
    mesh = box.build()

    assert mesh.coords.shape == (12, 3)
    assert mesh.conn.shape == (2, 8)

    expected_conn = jnp.array(
        [
            [0, 1, 4, 3, 6, 7, 10, 9],   # element at i=0
            [1, 2, 5, 4, 7, 8, 11, 10],  # element at i=1
        ],
        dtype=INDEX_DTYPE,
    )
    max_diff = int(jnp.max(jnp.abs(mesh.conn - expected_conn)))
    print("StructuredHexBox nx=2,ny=1,nz=1 conn matches expected:", max_diff == 0)
    if max_diff != 0:
        print("expected conn:\n", expected_conn)
        print("got conn:\n", mesh.conn)


if __name__ == "__main__":
    _check_structured_box_connectivity()
