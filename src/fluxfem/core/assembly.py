from __future__ import annotations
from dataclasses import dataclass
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
PairReturn: TypeAlias = tuple[FluxSparseMatrix, Array]


@dataclass(frozen=True)
class AssemblyPolicy:
    """
    Shared execution policy for volume assembly.

    Use this instead of passing many low-level tuning knobs to each assemble call.
    Explicit function arguments still take precedence over policy values.
    """

    n_chunks: int | None = None
    include_x_q: bool = True
    lightweight_context: bool = True
    chunk_build_context: bool = False
    pad_trace: bool = False

    @classmethod
    def chunked(
        cls,
        n_chunks: int,
        *,
        include_x_q: bool = False,
        lightweight_context: bool = True,
        chunk_build_context: bool = False,
        pad_trace: bool = False,
    ) -> "AssemblyPolicy":
        return cls(
            n_chunks=int(n_chunks),
            include_x_q=bool(include_x_q),
            lightweight_context=bool(lightweight_context),
            chunk_build_context=bool(chunk_build_context),
            pad_trace=bool(pad_trace),
        )


def _resolve_assembly_policy(
    *,
    policy: AssemblyPolicy | None,
    n_chunks: int | None,
    include_x_q: bool | None,
    lightweight_context: bool | None,
    chunk_build_context: bool | None,
    pad_trace: bool | None,
) -> tuple[int | None, bool, bool, bool, bool]:
    p = policy or AssemblyPolicy()
    return (
        n_chunks if n_chunks is not None else p.n_chunks,
        include_x_q if include_x_q is not None else p.include_x_q,
        lightweight_context if lightweight_context is not None else p.lightweight_context,
        chunk_build_context if chunk_build_context is not None else p.chunk_build_context,
        pad_trace if pad_trace is not None else p.pad_trace,
    )


def _resolve_chunk_policy(
    *,
    policy: AssemblyPolicy | None,
    n_chunks: int | None,
    pad_trace: bool | None,
) -> tuple[int | None, bool]:
    p = policy or AssemblyPolicy()
    return (
        n_chunks if n_chunks is not None else p.n_chunks,
        pad_trace if pad_trace is not None else p.pad_trace,
    )


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


def _element_mass_local(N: Array, wJ: Array, value_dim: int) -> Array:
    """
    Element mass from shape values without allocating q-by-ldofs-by-ldofs intermediates.
    """
    ms = jnp.einsum("qa,qb,q->ab", N, N, wJ)
    if int(value_dim) <= 1:
        return ms
    eye_v = jnp.eye(int(value_dim), dtype=ms.dtype)
    return jnp.einsum("ab,ij->aibj", ms, eye_v).reshape(
        ms.shape[0] * int(value_dim),
        ms.shape[1] * int(value_dim),
    )


def _integrate_q_linear(integrand: Array, wJ: Array, *, includes_measure: bool) -> Array:
    if includes_measure:
        return jnp.einsum("qa->a", integrand)
    return jnp.einsum("qa,q->a", integrand, wJ)


def _integrate_q_bilinear(integrand: Array, wJ: Array, *, includes_measure: bool) -> Array:
    if includes_measure:
        return jnp.einsum("qab->ab", integrand)
    return jnp.einsum("qab,q->ab", integrand, wJ)


def _integrate_q_tree(integrand: Any, wJ: Array, *, includes_measure: bool) -> Any:
    if includes_measure:
        return jax.tree_util.tree_map(lambda x: jnp.einsum("qa->a", x), integrand)
    return jax.tree_util.tree_map(lambda x: jnp.einsum("qa,q->a", x, wJ), integrand)


def _integrate_q_named_fields(
    integrand: Mapping[str, Array],
    ctx: FormContext,
    includes_measure: Any,
) -> dict[str, Array]:
    out: dict[str, Array] = {}
    for name, val in integrand.items():
        use_measure = bool(isinstance(includes_measure, dict) and includes_measure.get(name, False))
        if use_measure:
            out[name] = jnp.einsum("qa->a", val)
        else:
            wJ = ctx.w * ctx.fields[name].test.detJ
            out[name] = jnp.einsum("qa,q->a", val, wJ)
    return out


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


def _slice_first_dim(x: Array, start: int, size: int) -> Array:
    start_idx = (start,) + (0,) * (x.ndim - 1)
    slice_sizes = (size,) + x.shape[1:]
    return jax.lax.dynamic_slice(x, start_idx, slice_sizes)


def _prepare_chunk_iteration(
    *,
    n_elems: int,
    n_chunks: int | None,
    pad_trace: bool,
) -> tuple[int, int, int, int, Array]:
    if n_chunks is None or n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer.")
    n_chunks_eff = min(int(n_chunks), int(n_elems))
    chunk_size = (int(n_elems) + n_chunks_eff - 1) // n_chunks_eff
    stats = chunk_pad_stats(n_elems, n_chunks_eff)
    _maybe_trace_pad(stats, n_chunks=n_chunks_eff, pad_trace=pad_trace)
    pad = (-int(n_elems)) % chunk_size
    n_pad = int(n_elems) + pad
    n_chunks_eff = n_pad // chunk_size
    valid_mask = jnp.arange(int(n_pad), dtype=INDEX_DTYPE) < int(n_elems)
    return int(n_chunks_eff), int(chunk_size), int(pad), int(n_pad), valid_mask


def _prepare_chunk_context_source(
    space: "SpaceLike",
    *,
    n_pad: int,
    pad: int,
    dep: jnp.ndarray | None,
    include_x_q: bool | None,
    lightweight_context: bool | None,
    chunk_build_context: bool,
    elem_data: FormContext | None = None,
) -> tuple[bool, Array | None, Array | None, FormContext | None]:
    use_chunk_context = bool(
        chunk_build_context
        and hasattr(space, "build_form_contexts_from_elem_coords")
        and hasattr(space, "mesh")
    )
    if use_chunk_context:
        conn = space.mesh.conn
        if pad:
            conn_pad = jnp.concatenate([conn, jnp.repeat(conn[-1:], pad, axis=0)], axis=0)
        else:
            conn_pad = conn
        elem_ids = jnp.arange(int(n_pad), dtype=INDEX_DTYPE)
        return True, conn_pad, elem_ids, None

    ctxs = elem_data if elem_data is not None else space.build_form_contexts(
        dep=dep,
        include_x_q=include_x_q,
        lightweight=lightweight_context,
    )
    if pad:
        ctxs_pad = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
            ctxs,
        )
    else:
        ctxs_pad = ctxs
    return False, None, None, ctxs_pad


def _chunk_context_from_source(
    space: "SpaceLike",
    *,
    start: int,
    chunk_size: int,
    use_chunk_context: bool,
    conn_pad: Array | None,
    elem_ids: Array | None,
    ctxs_pad: FormContext | None,
    include_x_q: bool | None,
    lightweight_context: bool | None,
) -> FormContext:
    if use_chunk_context:
        assert conn_pad is not None
        assert elem_ids is not None
        conn_chunk = _slice_first_dim(conn_pad, start, chunk_size)
        elem_coords_chunk = space.mesh.coords[conn_chunk]
        elem_id_chunk = _slice_first_dim(elem_ids, start, chunk_size)
        return space.build_form_contexts_from_elem_coords(
            elem_coords_chunk,
            include_x_q=include_x_q,
            lightweight=lightweight_context,
            elem_id=elem_id_chunk,
        )
    assert ctxs_pad is not None
    return jax.tree_util.tree_map(
        lambda x: _slice_first_dim(x, start, chunk_size),
        ctxs_pad,
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
            wJ = ctx.w * ctx.test.detJ
            return _integrate_q_linear(
                integrand,
                wJ,
                includes_measure=bool(includes_measure),
            )

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

        def per_element(ctx: FormContext):
            N = ctx.test.N
            wJ = ctx.w * ctx.test.detJ
            vd = int(getattr(ctx.test, "value_dim", 1))
            return _element_mass_local(N, wJ, vd)

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
    elem_data: FormContext | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    chunk_build_context: bool | None = None,
    kernel: ElementBilinearKernel | None = None,
    jit: bool = True,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> FluxSparseMatrix:
    """
    Assemble a sparse bilinear form into a FluxSparseMatrix.

    Expects form(ctx, params) -> (n_q, n_ldofs, n_ldofs).
    If kernel is provided: kernel(ctx) -> (n_ldofs, n_ldofs).
    """
    from ..solver import FluxSparseMatrix
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=chunk_build_context,
        pad_trace=pad_trace,
    )

    if pattern is None:
        if hasattr(space, "get_sparsity_pattern"):
            pat = space.get_sparsity_pattern(with_idx=True)
        else:
            pat = make_sparsity_pattern(space, with_idx=True)
    else:
        pat = pattern

    if kernel is None:
        kernel = make_element_bilinear_kernel(form, params, jit=jit)

    if n_chunks is None or (jax.core.trace_ctx.is_top_level() and not chunk_build_context):
        if elem_data is None:
            elem_data = space.build_form_contexts(
                dep=dep,
                include_x_q=include_x_q,
                lightweight=lightweight_context,
            )
        K_e_all = jax.vmap(kernel)(elem_data)  # (n_elems, m, m)
        data = K_e_all.reshape(-1)
        return FluxSparseMatrix(pat, data)

    # --- chunked path ---
    n_elems = space.elem_dofs.shape[0]
    n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
        n_elems=int(n_elems),
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )
    # Ideally get m from pat (otherwise infer from one element).
    m = getattr(pat, "n_ldofs", None)
    if m is None and elem_data is not None:
        m = kernel(jax.tree_util.tree_map(lambda x: x[0], elem_data)).shape[0]

    # In chunked mode, default to chunk-local context generation to avoid
    # allocating all-element contexts at once.
    use_chunk_context = bool(
        dep is None
        and (chunk_build_context or elem_data is None)
    )
    use_chunk_context, conn_pad, elem_ids, elem_data_pad = _prepare_chunk_context_source(
        space,
        n_pad=int(n_pad),
        pad=int(pad),
        dep=dep,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=use_chunk_context,
        elem_data=elem_data,
    )
    sample_ctx_b = _chunk_context_from_source(
        space,
        start=0,
        chunk_size=1,
        use_chunk_context=use_chunk_context,
        conn_pad=conn_pad,
        elem_ids=elem_ids,
        ctxs_pad=elem_data_pad,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
    )
    sample_ctx = jax.tree_util.tree_map(lambda x: x[0], sample_ctx_b)

    sample_ke = kernel(sample_ctx)
    if m is None:
        m = int(sample_ke.shape[0])
    data = jnp.zeros((n_pad * m * m,), dtype=sample_ke.dtype)

    def loop_body(i, data_flat):
        start = i * chunk_size
        ctx_chunk = _chunk_context_from_source(
            space,
            start=start,
            chunk_size=chunk_size,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=elem_data_pad,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
        )
        Ke = jax.vmap(kernel)(ctx_chunk)  # (chunk, m, m)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(Ke.dtype)
        Ke = Ke * chunk_valid[:, None, None]
        ke_flat = Ke.reshape(chunk_size * m * m)
        return jax.lax.dynamic_update_slice(data_flat, ke_flat, (start * m * m,))

    data = jax.lax.fori_loop(0, n_chunks, loop_body, data)
    data = data[: n_elems * m * m]
    return FluxSparseMatrix(pat, data)


def assemble_mass_matrix(
    space: SpaceLike,
    *,
    lumped: bool = False,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> MassReturn:
    """
    Assemble mass matrix M_ij = ∫ N_i N_j dΩ.
    Supports scalar and vector spaces. If lumped=True, rows are summed to diagonal.
    """
    from ..solver import FluxSparseMatrix  # local import to avoid circular
    n_chunks, _include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=False,  # mass matrix does not require x_q
        lightweight_context=None,
        chunk_build_context=None,
        pad_trace=pad_trace,
    )
    def per_element(ctx: FormContext):
        N = ctx.test.N  # (n_q, n_nodes)
        wJ = ctx.w * ctx.test.detJ
        vd = int(getattr(ctx.test, "value_dim", 1))
        return _element_mass_local(N, wJ, vd)

    if n_chunks is None:
        ctxs = space.build_form_contexts(include_x_q=False, lightweight=lightweight_context)
        M_e_all = jax.vmap(per_element)(ctxs)  # (n_elems, n_ldofs, n_ldofs)
        data = M_e_all.reshape(-1)
    else:
        n_elems = space.elem_dofs.shape[0]
        n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
            n_elems=int(n_elems),
            n_chunks=n_chunks,
            pad_trace=pad_trace,
        )

        use_chunk_context = bool(
            chunk_build_context
            and hasattr(space, "build_form_contexts_from_elem_coords")
            and hasattr(space, "mesh")
        )

        if use_chunk_context:
            conn = space.mesh.conn
            if pad:
                conn_pad = jnp.concatenate([conn, jnp.repeat(conn[-1:], pad, axis=0)], axis=0)
            else:
                conn_pad = conn
            elem_ids = jnp.arange(int(n_pad), dtype=INDEX_DTYPE)
        else:
            ctxs = space.build_form_contexts(include_x_q=False, lightweight=lightweight_context)
            if pad:
                ctxs_pad = jax.tree_util.tree_map(
                    lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0),
                    ctxs,
                )
            else:
                ctxs_pad = ctxs

        data = jnp.zeros((n_pad * space.n_ldofs * space.n_ldofs,), dtype=space.mesh.coords.dtype)

        def loop_body(i, data_flat):
            start = i * chunk_size
            if use_chunk_context:
                conn_chunk = _slice_first_dim(conn_pad, start, chunk_size)
                elem_coords_chunk = space.mesh.coords[conn_chunk]
                elem_id_chunk = _slice_first_dim(elem_ids, start, chunk_size)
                ctx_chunk = space.build_form_contexts_from_elem_coords(
                    elem_coords_chunk,
                    include_x_q=False,
                    lightweight=lightweight_context,
                    elem_id=elem_id_chunk,
                )
            else:
                ctx_chunk = jax.tree_util.tree_map(
                    lambda x: _slice_first_dim(x, start, chunk_size),
                    ctxs_pad,
                )
            Me = jax.vmap(per_element)(ctx_chunk)
            chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(Me.dtype)
            Me = Me * chunk_valid[:, None, None]
            return jax.lax.dynamic_update_slice(
                data_flat,
                Me.reshape(chunk_size * space.n_ldofs * space.n_ldofs),
                (start * space.n_ldofs * space.n_ldofs,),
            )

        data = jax.lax.fori_loop(0, n_chunks, loop_body, data)
        data = data[: n_elems * space.n_ldofs * space.n_ldofs]

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
    elem_data: FormContext | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    chunk_build_context: bool | None = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> LinearReturn:
    """
    Expects form(ctx, params) -> (n_q, n_ldofs) and integrates Σ_q form * wJ for RHS.
    If kernel is provided: kernel(ctx) -> (n_ldofs,).
    """
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=chunk_build_context,
        pad_trace=pad_trace,
    )

    includes_measure = getattr(form, "_includes_measure", False)

    if kernel is None:
        def per_element(ctx: FormContext):
            integrand = form(ctx, params)  # (n_q, m)
            wJ = ctx.w * ctx.test.detJ     # (n_q,)
            return _integrate_q_linear(
                integrand,
                wJ,
                includes_measure=bool(includes_measure),
            )  # (m,)
    else:
        per_element = kernel

    if n_chunks is None or (jax.core.trace_ctx.is_top_level() and not chunk_build_context):
        if elem_data is None:
            elem_data = space.build_form_contexts(
                dep=dep,
                include_x_q=include_x_q,
                lightweight=lightweight_context,
            )
        F_e_all = jax.vmap(per_element)(elem_data)            # (n_elems, m)
        data = F_e_all.reshape(-1)
    else:
        n_elems = space.elem_dofs.shape[0]
        m = n_ldofs
        n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
            n_elems=int(n_elems),
            n_chunks=n_chunks,
            pad_trace=pad_trace,
        )

        use_chunk_context = bool(
            dep is None
            and (chunk_build_context or elem_data is None)
        )
        use_chunk_context, conn_pad, elem_ids, elem_data_pad = _prepare_chunk_context_source(
            space,
            n_pad=int(n_pad),
            pad=int(pad),
            dep=dep,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
            chunk_build_context=use_chunk_context,
            elem_data=elem_data,
        )
        sample_ctx_b = _chunk_context_from_source(
            space,
            start=0,
            chunk_size=1,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=elem_data_pad,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
        )
        sample_ctx = jax.tree_util.tree_map(lambda x: x[0], sample_ctx_b)

        sample_fe = per_element(sample_ctx)
        if sparse:
            data = jnp.zeros((n_pad * m,), dtype=sample_fe.dtype)

            def loop_body(i, data_flat):
                start = i * chunk_size
                ctx_chunk = _chunk_context_from_source(
                    space,
                    start=start,
                    chunk_size=chunk_size,
                    use_chunk_context=use_chunk_context,
                    conn_pad=conn_pad,
                    elem_ids=elem_ids,
                    ctxs_pad=elem_data_pad,
                    include_x_q=include_x_q,
                    lightweight_context=lightweight_context,
                )
                Fe = jax.vmap(per_element)(ctx_chunk)  # (chunk, m)
                chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(Fe.dtype)
                Fe = Fe * chunk_valid[:, None]
                fe_flat = Fe.reshape(chunk_size * m)
                return jax.lax.dynamic_update_slice(data_flat, fe_flat, (start * m,))

            data = jax.lax.fori_loop(0, n_chunks, loop_body, data)
            data = data[: n_elems * m]
        else:
            if pad:
                elem_dofs_pad = jnp.concatenate([elem_dofs, jnp.repeat(elem_dofs[-1:], pad, axis=0)], axis=0)
            else:
                elem_dofs_pad = elem_dofs
            sdn = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            F0 = jnp.zeros((n_dofs,), dtype=sample_fe.dtype)

            def loop_body_sum(i, F_acc):
                start = i * chunk_size
                ctx_chunk = _chunk_context_from_source(
                    space,
                    start=start,
                    chunk_size=chunk_size,
                    use_chunk_context=use_chunk_context,
                    conn_pad=conn_pad,
                    elem_ids=elem_ids,
                    ctxs_pad=elem_data_pad,
                    include_x_q=include_x_q,
                    lightweight_context=lightweight_context,
                )
                Fe = jax.vmap(per_element)(ctx_chunk)  # (chunk, m)
                chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(Fe.dtype)
                Fe = Fe * chunk_valid[:, None]
                fe_flat = Fe.reshape(chunk_size * m)
                rows_chunk = _slice_first_dim(elem_dofs_pad, start, chunk_size).reshape(-1)
                return jax.lax.scatter_add(F_acc, rows_chunk[:, None], fe_flat, sdn)

            return jax.lax.fori_loop(0, n_chunks, loop_body_sum, F0)

    rows = _get_elem_rows(space)

    if sparse:
        return rows, data, n_dofs

    F = jax.ops.segment_sum(data, rows, n_dofs)
    return F


def assemble_bilinear_linear_pair(
    space: SpaceLike,
    bilinear_form: FormKernel[P],
    bilinear_params: P,
    linear_form: FormKernel[P],
    linear_params: P,
    *,
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,
    dep: jnp.ndarray | None = None,
    elem_data: FormContext | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    chunk_build_context: bool | None = None,
    bilinear_kernel: ElementBilinearKernel | None = None,
    linear_kernel: ElementLinearKernel | None = None,
    jit: bool = True,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> PairReturn:
    from ..solver import FluxSparseMatrix
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=chunk_build_context,
        pad_trace=pad_trace,
    )

    if pattern is None:
        if hasattr(space, "get_sparsity_pattern"):
            pat = space.get_sparsity_pattern(with_idx=True)
        else:
            pat = make_sparsity_pattern(space, with_idx=True)
    else:
        pat = pattern

    if bilinear_kernel is None:
        bilinear_kernel = make_element_bilinear_kernel(bilinear_form, bilinear_params, jit=jit)
    if linear_kernel is None:
        linear_kernel = make_element_linear_kernel(linear_form, linear_params, jit=jit)

    n_elems = int(space.elem_dofs.shape[0])
    n_ldofs = int(space.n_ldofs)

    if n_chunks is None or (jax.core.trace_ctx.is_top_level() and not chunk_build_context):
        if elem_data is None:
            elem_data = space.build_form_contexts(
                dep=dep,
                include_x_q=include_x_q,
                lightweight=lightweight_context,
            )
        assert elem_data is not None
        Ke = jax.vmap(bilinear_kernel)(elem_data)
        Fe = jax.vmap(linear_kernel)(elem_data)
        K_data = Ke.reshape(-1)
        F_data = Fe.reshape(-1)
        rows = _get_elem_rows(space)
        F = jax.ops.segment_sum(F_data, rows, space.n_dofs)
        return FluxSparseMatrix(pat, K_data), F

    n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
        n_elems=int(n_elems),
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )

    use_chunk_context = bool(
        dep is None
        and (chunk_build_context or elem_data is None)
    )
    use_chunk_context, conn_pad, elem_ids, elem_data_pad = _prepare_chunk_context_source(
        space,
        n_pad=int(n_pad),
        pad=int(pad),
        dep=dep,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=use_chunk_context,
        elem_data=elem_data,
    )
    sample_ctx_b = _chunk_context_from_source(
        space,
        start=0,
        chunk_size=1,
        use_chunk_context=use_chunk_context,
        conn_pad=conn_pad,
        elem_ids=elem_ids,
        ctxs_pad=elem_data_pad,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
    )
    sample_ctx = jax.tree_util.tree_map(lambda x: x[0], sample_ctx_b)

    sample_ke = bilinear_kernel(sample_ctx)
    sample_fe = linear_kernel(sample_ctx)
    m = int(sample_ke.shape[0])
    K_data = jnp.zeros((n_pad * m * m,), dtype=sample_ke.dtype)
    F_data = jnp.zeros((n_pad * m,), dtype=sample_fe.dtype)

    def loop_body(i, carry):
        K_flat, F_flat = carry
        start = i * chunk_size
        ctx_chunk = _chunk_context_from_source(
            space,
            start=start,
            chunk_size=chunk_size,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=elem_data_pad,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
        )
        Ke = jax.vmap(bilinear_kernel)(ctx_chunk)
        Fe = jax.vmap(linear_kernel)(ctx_chunk)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(Ke.dtype)
        Ke = Ke * chunk_valid[:, None, None]
        Fe = Fe * chunk_valid[:, None]
        K_flat = jax.lax.dynamic_update_slice(K_flat, Ke.reshape(chunk_size * m * m), (start * m * m,))
        F_flat = jax.lax.dynamic_update_slice(F_flat, Fe.reshape(chunk_size * m), (start * m,))
        return (K_flat, F_flat)

    K_data, F_data = jax.lax.fori_loop(0, n_chunks, loop_body, (K_data, F_data))
    K_data = K_data[: n_elems * m * m]
    F_data = F_data[: n_elems * m]
    rows = _get_elem_rows(space)
    F = jax.ops.segment_sum(F_data, rows, space.n_dofs)
    return FluxSparseMatrix(pat, K_data), F


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
        fe = _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )  # (m,)
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
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

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
        fe = _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(form, "_includes_measure", False)),
        )
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
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

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
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_bilinear(
            integrand,
            wJ,
            includes_measure=bool(getattr(form, "_includes_measure", False)),
        )

    return jax.jit(per_element) if jit else per_element


def make_element_linear_kernel(
    form: FormKernel[P], params: P, *, jit: bool = True
) -> ElementLinearKernel:
    """Element kernel: (ctx) -> fe."""

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(form, "_includes_measure", False)),
        )

    return jax.jit(per_element) if jit else per_element


def make_element_residual_kernel(
    res_form: ResidualForm[P], params: P
) -> ElementResidualKernel:
    """Jitted element residual kernel: (ctx, u_elem) -> fe."""

    def per_element(ctx: FormContext, u_elem: jnp.ndarray):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

    return jax.jit(per_element)


def make_element_jacobian_kernel(
    res_form: ResidualForm[P], params: P
) -> ElementJacobianKernel:
    """Jitted element Jacobian kernel: (ctx, u_elem) -> Ke."""

    def fe_fun(u_elem, ctx: FormContext):
        integrand = res_form(ctx, u_elem, params)
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_linear(
            integrand,
            wJ,
            includes_measure=bool(getattr(res_form, "_includes_measure", False)),
        )

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
        wJ = ctx.w * ctx.test.detJ             # (n_q,)
        return _integrate_q_linear(integrand, wJ, includes_measure=bool(includes_measure))
    if hasattr(ctx, "fields") and ctx.fields is not None:
        return _integrate_q_named_fields(cast(Mapping[str, Array], integrand), ctx, includes_measure)
    return _integrate_q_tree(
        integrand,
        ctx.w * ctx.test.detJ,
        includes_measure=bool(includes_measure),
    )


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
            wJ = ctx.w * ctx.test.detJ
            return _integrate_q_linear(
                integrand,
                wJ,
                includes_measure=bool(getattr(form_linear, "_includes_measure", False)),
            )

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
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> Array:
    """
    Assemble only the numeric values for the Jacobian (pattern-free).
    """
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=None,
        lightweight_context=None,
        chunk_build_context=None,
        pad_trace=pad_trace,
    )
    ker = kernel if kernel is not None else make_element_jacobian_kernel(res_form, params)

    u_elems = u[space.elem_dofs]
    if n_chunks is None:
        ctxs = space.build_form_contexts(include_x_q=include_x_q, lightweight=lightweight_context)
        J_e_all = jax.vmap(ker)(u_elems, ctxs)  # (n_elem, m, m)
        return J_e_all.reshape(-1)

    n_elems = int(u_elems.shape[0])
    n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
        n_elems=int(n_elems),
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )
    m = int(space.n_ldofs)
    if pad:
        u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
    else:
        u_elems_pad = u_elems
    first_u = _slice_first_dim(u_elems_pad, 0, 1)

    use_chunk_context, conn_pad, elem_ids, ctxs_pad = _prepare_chunk_context_source(
        space,
        n_pad=int(n_pad),
        pad=int(pad),
        dep=None,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        chunk_build_context=chunk_build_context,
    )
    first_ctx = _chunk_context_from_source(
        space,
        start=0,
        chunk_size=1,
        use_chunk_context=use_chunk_context,
        conn_pad=conn_pad,
        elem_ids=elem_ids,
        ctxs_pad=ctxs_pad,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
    )
    sample_J = jax.vmap(ker)(first_u, first_ctx)[0]

    data = jnp.zeros((n_pad * m * m,), dtype=sample_J.dtype)

    def loop_body(i, data_flat):
        start = i * chunk_size
        ctx_chunk = _chunk_context_from_source(
            space,
            start=start,
            chunk_size=chunk_size,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=ctxs_pad,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
        )
        u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
        J_e = jax.vmap(ker)(u_chunk, ctx_chunk)
        chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(J_e.dtype)
        J_e = J_e * chunk_valid[:, None, None]
        return jax.lax.dynamic_update_slice(
            data_flat,
            J_e.reshape(chunk_size * m * m),
            (start * m * m,),
        )

    data = jax.lax.fori_loop(0, n_chunks, loop_body, data)
    return data[: n_elems * m * m]


def assemble_residual_scatter(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementResidualKernel | None = None,
    sparse: bool = False,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> LinearReturn:
    """
    Assemble residual using jitted element kernel + vmap + scatter_add.
    Avoids Python loops; good for JIT stability.

    Note: `res_form` should return the integrand only; quadrature weights and detJ
    are applied in the element kernel (make_element_residual_kernel). Do not multiply
    by w or detJ inside `res_form`.
    """
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=None,
        lightweight_context=None,
        chunk_build_context=None,
        pad_trace=pad_trace,
    )
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    if jax.core.trace_ctx.is_top_level():
        if np.max(elem_dofs) >= n_dofs:
            raise ValueError("elem_dofs contains index outside n_dofs")
        if np.min(elem_dofs) < 0:
            raise ValueError("elem_dofs contains negative index")
    ker = kernel if kernel is not None else make_element_residual_kernel(res_form, params)

    u_elems = u[elem_dofs]
    if n_chunks is None:
        ctxs = space.build_form_contexts(include_x_q=include_x_q, lightweight=lightweight_context)
        elem_res = jax.vmap(ker)(ctxs, u_elems)  # (n_elem, n_ldofs)
        data = elem_res.reshape(-1)
    else:
        n_elems = int(u_elems.shape[0])
        n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
            n_elems=int(n_elems),
            n_chunks=n_chunks,
            pad_trace=pad_trace,
        )
        if pad:
            u_elems_pad = jnp.concatenate([u_elems, jnp.repeat(u_elems[-1:], pad, axis=0)], axis=0)
        else:
            u_elems_pad = u_elems

        use_chunk_context, conn_pad, elem_ids, ctxs_pad = _prepare_chunk_context_source(
            space,
            n_pad=int(n_pad),
            pad=int(pad),
            dep=None,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
            chunk_build_context=chunk_build_context,
        )

        m = int(space.n_ldofs)
        if sparse:
            data = jnp.zeros((n_pad * m,), dtype=u_elems.dtype)

            def loop_body(i, data_flat):
                start = i * chunk_size
                ctx_chunk = _chunk_context_from_source(
                    space,
                    start=start,
                    chunk_size=chunk_size,
                    use_chunk_context=use_chunk_context,
                    conn_pad=conn_pad,
                    elem_ids=elem_ids,
                    ctxs_pad=ctxs_pad,
                    include_x_q=include_x_q,
                    lightweight_context=lightweight_context,
                )
                u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
                res_chunk = jax.vmap(ker)(ctx_chunk, u_chunk)
                chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(res_chunk.dtype)
                res_chunk = res_chunk * chunk_valid[:, None]
                return jax.lax.dynamic_update_slice(
                    data_flat,
                    res_chunk.reshape(chunk_size * m),
                    (start * m,),
                )

            data = jax.lax.fori_loop(0, n_chunks, loop_body, data)
            data = data[: n_elems * m]
        else:
            if pad:
                elem_dofs_pad = jnp.concatenate([elem_dofs, jnp.repeat(elem_dofs[-1:], pad, axis=0)], axis=0)
            else:
                elem_dofs_pad = elem_dofs
            sdn = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            F0 = jnp.zeros((n_dofs,), dtype=u_elems.dtype)

            def loop_body_sum(i, F_acc):
                start = i * chunk_size
                ctx_chunk = _chunk_context_from_source(
                    space,
                    start=start,
                    chunk_size=chunk_size,
                    use_chunk_context=use_chunk_context,
                    conn_pad=conn_pad,
                    elem_ids=elem_ids,
                    ctxs_pad=ctxs_pad,
                    include_x_q=include_x_q,
                    lightweight_context=lightweight_context,
                )
                u_chunk = _slice_first_dim(u_elems_pad, start, chunk_size)
                res_chunk = jax.vmap(ker)(ctx_chunk, u_chunk)
                chunk_valid = _slice_first_dim(valid_mask, start, chunk_size).astype(res_chunk.dtype)
                res_chunk = res_chunk * chunk_valid[:, None]
                data_chunk = res_chunk.reshape(chunk_size * m)
                rows_chunk = _slice_first_dim(elem_dofs_pad, start, chunk_size).reshape(-1)
                return jax.lax.scatter_add(F_acc, rows_chunk[:, None], data_chunk, sdn)

            F = jax.lax.fori_loop(0, n_chunks, loop_body_sum, F0)
            if jax.core.trace_ctx.is_top_level():
                if not bool(jax.block_until_ready(jnp.all(jnp.isfinite(F)))):
                    bad = int(jnp.count_nonzero(~jnp.isfinite(F)))
                    raise RuntimeError(f"[assemble_residual_scatter] residual vector nonfinite: {bad}")
            return F
    if jax.core.trace_ctx.is_top_level():
        if not bool(jax.block_until_ready(jnp.all(jnp.isfinite(data)))):
            bad = int(jnp.count_nonzero(~jnp.isfinite(data)))
            raise RuntimeError(f"[assemble_residual_scatter] residual data nonfinite: {bad}")

    rows = _get_elem_rows(space)

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
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> JacobianReturn:
    """
    Assemble Jacobian using jitted element kernel + vmap + scatter_add.
    If a SparsityPattern is provided, rows/cols are reused without regeneration.
    CONTRACT: The returned `data` ordering matches `pattern.rows/cols` exactly.
    Any change to pattern generation or data flattening must preserve this.
    """
    from ..solver import FluxSparseMatrix  # local import to avoid circular

    if pattern is not None:
        pat = pattern
    else:
        pat = _get_pattern(space, with_idx=not sparse)
        if pat is None:
            pat = make_sparsity_pattern(space, with_idx=not sparse)
    data = assemble_jacobian_values(
        space, res_form, u, params, kernel=kernel, n_chunks=n_chunks, pad_trace=pad_trace, policy=policy
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
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> LinearReturn:
    """
    Assemble the global residual vector (scatter-based).
    If kernel is provided: kernel(ctx, u_elem) -> (n_ldofs,).
    """
    return assemble_residual_scatter(
        space,
        form,
        u,
        params,
        kernel=kernel,
        sparse=sparse,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
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
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
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
        policy=policy,
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
    n_chunks, pad_trace = _resolve_chunk_policy(
        policy=policy,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )
    n_chunks, pad_trace = _resolve_chunk_policy(
        policy=policy,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )
    n_chunks, pad_trace = _resolve_chunk_policy(
        policy=policy,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
    )
