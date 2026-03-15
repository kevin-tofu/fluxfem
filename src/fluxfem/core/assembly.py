from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, TYPE_CHECKING, TypeAlias, TypeVar, Union, cast
import warnings
import numpy as np
import jax
import jax.numpy as jnp

try:
    from .._runtime_warn import warn_float32_assembly_once
except Exception:  # pragma: no cover
    import warnings

    _WARNED_FLOAT32_ASSEMBLY = False

    def warn_float32_assembly_once(*, context: str = "assembly") -> None:
        global _WARNED_FLOAT32_ASSEMBLY
        if _WARNED_FLOAT32_ASSEMBLY:
            return
        if bool(jax.config.read("jax_enable_x64")):
            return
        _WARNED_FLOAT32_ASSEMBLY = True
        warnings.warn(
            "Running in float32 mode (x64 disabled). "
            f"{context} can suffer from residual/conditioning degradation; "
            "use x64 for reliable diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
from ..mesh import HexMesh, StructuredHexBox
from .dtypes import INDEX_DTYPE
from .forms import FormContext
from .space import (
    FESpaceBase,
    NamedSpace,
    BilinearSpaces,
    LinearSpaces,
    ResidualSpaces,
    JacobianSpaces,
    build_form_contexts_pair,
)
from .assembly_numpy import (
    assemble_numpy_scalar_diffusion_body_force_data as _assemble_numpy_scalar_diffusion_body_force_data,
    assemble_numpy_scalar_diffusion_pair_fast as _assemble_numpy_scalar_diffusion_pair_fast,
    is_numpy_scalar_body_force_fast_path as _is_numpy_scalar_body_force_fast_path,
    is_numpy_scalar_diffusion_body_force_pair_fast_path as _is_numpy_scalar_diffusion_body_force_pair_fast_path,
    is_numpy_scalar_diffusion_fast_path as _is_numpy_scalar_diffusion_fast_path,
)

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
    from ..solver import FluxSparseMatrix, FluxSparseOperator, SparsityPattern
else:
    FluxSparseMatrix = Any
    FluxSparseOperator = Any
    SparsityPattern = Any

SparseCOO: TypeAlias = tuple[Array, Array, Array, int]
LinearCOO: TypeAlias = tuple[Array, Array, int]
JacobianReturn: TypeAlias = FluxSparseMatrix
BilinearReturn: TypeAlias = Union[Array, FluxSparseMatrix, SparseCOO]
LinearReturn: TypeAlias = Union[Array, LinearCOO]
MassReturn: TypeAlias = Union[FluxSparseMatrix, Array]
PairReturn: TypeAlias = tuple[FluxSparseMatrix, Array]


def _resolve_named_bilinear_spaces(
    space: Any,
) -> tuple[NamedSpace, NamedSpace] | None:
    if isinstance(space, BilinearSpaces):
        return space.test, space.trial
    if not isinstance(space, Mapping):
        return None
    warnings.warn(
        "Passing named bilinear roles as a dict is deprecated; "
        "use BilinearSpaces(test=..., trial=...) instead.",
        FutureWarning,
        stacklevel=3,
    )
    if "test" not in space or "trial" not in space:
        raise ValueError(
            "Named-space bilinear assembly expects a mapping with keys 'test' and 'trial'."
        )
    test_space = space["test"]
    trial_space = space["trial"]
    if not isinstance(test_space, NamedSpace) or not isinstance(trial_space, NamedSpace):
        raise TypeError(
            "Named-space bilinear assembly requires 'test' and 'trial' values to be NamedSpace instances."
        )
    return test_space, trial_space


def _resolve_named_linear_space(
    space: Any,
) -> NamedSpace | None:
    if isinstance(space, LinearSpaces):
        return space.test
    if not isinstance(space, Mapping):
        return None
    warnings.warn(
        "Passing named linear roles as a dict is deprecated; "
        "use LinearSpaces(test=...) instead.",
        FutureWarning,
        stacklevel=3,
    )
    if "test" not in space:
        raise ValueError("Named-space linear assembly expects a mapping with key 'test'.")
    test_space = space["test"]
    if not isinstance(test_space, NamedSpace):
        raise TypeError("Named-space linear assembly requires the 'test' value to be a NamedSpace instance.")
    return test_space


def _resolve_named_residual_spaces(
    space: Any,
) -> tuple[NamedSpace, NamedSpace] | None:
    if isinstance(space, ResidualSpaces):
        return space.test, space.unknown
    return None


def _resolve_named_jacobian_spaces(
    space: Any,
) -> tuple[NamedSpace, NamedSpace] | None:
    if isinstance(space, JacobianSpaces):
        return space.test, space.trial
    return None


@dataclass(frozen=True)
class AssemblyPolicy:
    """
    Shared execution policy for volume assembly.

    Use this instead of passing many low-level tuning knobs to each assemble call.
    Explicit function arguments still take precedence over policy values.
    """

    n_chunks: int | None = None
    include_x_q: bool = False
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


def _integrate_q_scalar(integrand: Array, wJ: Array, *, includes_measure: bool) -> Array:
    """Integrate scalar quadrature values with optional embedded measure."""
    if includes_measure:
        return jnp.einsum("q->", integrand)
    return jnp.einsum("q,q->", integrand, wJ)


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
            wJ = ctx.w * ctx.bindings[name].test.detJ
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


from .assembly_chunk_utils import (
    _chunk_context_from_source,
    _maybe_trace_pad,
    _prepare_chunk_context_source,
    _prepare_chunk_iteration,
    _slice_first_dim,
    chunk_pad_stats,
)
from .assembly_matrix import (
    accumulate_chunk_matrix_and_vector_scatter as _accumulate_chunk_matrix_and_vector_scatter,
    accumulate_chunk_matrix_and_vector_segment as _accumulate_chunk_matrix_and_vector_segment,
    accumulate_chunk_matrix_data as _accumulate_chunk_matrix_data,
)
from .assembly_vector import (
    accumulate_chunk_vector_data as _accumulate_chunk_vector_data,
    accumulate_chunk_vector_scatter as _accumulate_chunk_vector_scatter,
    accumulate_chunk_vector_segment as _accumulate_chunk_vector_segment,
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
    ) -> FluxSparseMatrix:
        """
        kernel(u_elem, ctx) -> (n_ldofs, n_ldofs)
        """
        from ..solver import FluxSparseMatrix  # local import to avoid circular

        u_elems = jnp.asarray(u)[self.elem_dofs]
        J_e = jax.vmap(kernel)(u_elems, self.elem_data)
        if mask is not None:
            J_e = J_e * jnp.asarray(mask)[:, None, None]
        data = J_e.reshape(-1)
        if self.pattern is not None:
            return FluxSparseMatrix(self.pattern, data)
        rows, cols = self._rows_cols()
        return FluxSparseMatrix(rows, cols, data, n_dofs=self.n_dofs)

    def assemble_jacobian(
        self,
        res_form: ResidualForm[P],
        u: Array,
        params: P,
        *,
        mask: Array | None = None,
        kernel: ElementJacobianKernel | None = None,
    ) -> FluxSparseMatrix:
        if kernel is None:
            kernel = make_element_jacobian_kernel(res_form, params)
        return self.assemble_jacobian_with_kernel(
            kernel,
            u,
            mask=mask,
        )

class SpaceLike(FESpaceBase, Protocol):
    pass


def assemble_bilinear_dense(
    space: SpaceLike,
    kernel: FormKernel[P],
    params: P,
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

    from ..solver import FluxSparseMatrix  # local import to avoid circular
    return FluxSparseMatrix(rows, cols, data, n_dofs).to_dense()


def assemble_bilinear_form(
    space: SpaceLike,
    form: FormKernel[P],
    params: P,
    *,
    backend: str = "jax",
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
) -> FluxSparseMatrix | FluxSparseOperator:
    warn_float32_assembly_once(context="volume assembly")
    """
    Assemble a sparse bilinear form into a FluxSparseMatrix.

    Expects form(ctx, params) -> (n_q, n_ldofs, n_ldofs).
    If kernel is provided: kernel(ctx) -> (n_ldofs, n_ldofs).
    """
    from ..solver import FluxSparseMatrix
    named_spaces = _resolve_named_bilinear_spaces(space)
    if named_spaces is not None:
        test_space, trial_space = named_spaces
        if test_space.space is trial_space.space:
            return assemble_bilinear_form(
                test_space.space,
                form,
                params,
                backend=backend,
                include_x_q=include_x_q,
                lightweight_context=lightweight_context,
                pattern=pattern,
                n_chunks=n_chunks,
                dep=dep,
                elem_data=elem_data,
                chunk_build_context=chunk_build_context,
                kernel=kernel,
                jit=jit,
                pad_trace=pad_trace,
                policy=policy,
            )
        if any(arg is not None for arg in (pattern, n_chunks, dep, elem_data, chunk_build_context, pad_trace, policy)):
            raise ValueError(
                "Distinct-space assemble_bilinear_form currently does not support pattern/n_chunks/dep/"
                "elem_data/chunk_build_context/pad_trace/policy overrides."
            )
        return _assemble_bilinear_form_pg_impl(
            test_space,
            trial_space,
            form,
            params,
            backend=backend,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
            kernel=kernel,
            jit=jit,
        )
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
    include_x_q_req = include_x_q
    if include_x_q_req is None and policy is None:
        include_x_q_req = False
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=include_x_q_req,
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
        kernel_jit = jit if backend == "jax" else False
        kernel = make_element_bilinear_kernel(form, params, jit=kernel_jit)
    if backend == "numpy":
        if _is_numpy_scalar_diffusion_fast_path(space, form):
            K_data, _F, _F_data = _assemble_numpy_scalar_diffusion_body_force_data(
                space,
                bilinear_params=float(params),
                linear_params=0.0,
                n_chunks=n_chunks,
                pad_trace=pad_trace,
            )
            return FluxSparseMatrix(pat, K_data)
        data_parts: list[np.ndarray] = []
        if elem_data is not None:
            raise NotImplementedError("backend='numpy' with explicit elem_data is not supported in generic mode.")
        chunk_size_np = int(space.elem_dofs.shape[0]) if n_chunks is None else max(1, int(np.ceil(int(space.elem_dofs.shape[0]) / int(n_chunks))))
        try:
            ctx_chunks = space.build_form_contexts_numpy_chunked(
                chunk_size=chunk_size_np,
                dep=dep,
                include_x_q=include_x_q,
                lightweight=bool(lightweight_context),
            )
        except Exception:
            if include_x_q:
                raise
            ctx_chunks = space.build_form_contexts_numpy_chunked(
                chunk_size=chunk_size_np,
                dep=dep,
                include_x_q=True,
                lightweight=bool(lightweight_context),
            )
        for ctxs in ctx_chunks:
            for ctx_e in ctxs:
                ke = np.asarray(kernel(ctx_e), dtype=float)
                data_parts.append(ke.reshape(-1))
        data_np = np.concatenate(data_parts, axis=0) if data_parts else np.zeros((0,), dtype=float)
        return FluxSparseMatrix(pat, data_np)

    vmapped_kernel = jax.vmap(kernel)

    if n_chunks is None or (jax.core.trace_ctx.is_top_level() and not chunk_build_context):
        if elem_data is not None:
            K_e_all = vmapped_kernel(elem_data)  # (n_elems, m, m)
        else:
            def _eval(include_x_q_eff: bool) -> Array:
                ctx = space.build_form_contexts(
                    dep=dep,
                    include_x_q=include_x_q_eff,
                    lightweight=lightweight_context,
                )
                return vmapped_kernel(ctx)

            try:
                K_e_all = _eval(include_x_q)
            except Exception:
                if include_x_q:
                    raise
                K_e_all = _eval(True)
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
    use_chunk_context = bool(dep is None and chunk_build_context)

    def _init_chunk(include_x_q_eff: bool) -> tuple[Array, bool, Array | None, Array | None, FormContext | None, Callable[[int], Array]]:
        use_ctx, conn_pad, elem_ids, elem_data_pad = _prepare_chunk_context_source(
            space,
            n_pad=int(n_pad),
            pad=int(pad),
            dep=dep,
            include_x_q=include_x_q_eff,
            lightweight_context=lightweight_context,
            chunk_build_context=use_chunk_context,
            elem_data=elem_data,
        )
        def chunk_values_fn(start: int) -> Array:
            ctx_chunk = _chunk_context_from_source(
                space,
                start=start,
                chunk_size=chunk_size,
                use_chunk_context=use_ctx,
                conn_pad=conn_pad,
                elem_ids=elem_ids,
                ctxs_pad=elem_data_pad,
                include_x_q=include_x_q_eff,
                lightweight_context=lightweight_context,
            )
            return vmapped_kernel(ctx_chunk)

        # Keep sample and loop batch shapes aligned to avoid extra recompiles.
        sample_ke = chunk_values_fn(0)[0]
        return sample_ke, use_ctx, conn_pad, elem_ids, elem_data_pad, chunk_values_fn

    try:
        sample_ke, use_chunk_context, conn_pad, elem_ids, elem_data_pad, chunk_values_fn = _init_chunk(include_x_q)
    except Exception:
        if include_x_q:
            raise
        sample_ke, use_chunk_context, conn_pad, elem_ids, elem_data_pad, chunk_values_fn = _init_chunk(True)

    if m is None:
        m = int(sample_ke.shape[0])

    data = _accumulate_chunk_matrix_data(
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        n_pad=n_pad,
        m=m,
        dtype=sample_ke.dtype,
        valid_mask=valid_mask,
        chunk_values_fn=chunk_values_fn,
    )
    data = data[: n_elems * m * m]
    return FluxSparseMatrix(pat, data)


def _assemble_bilinear_form_pg_impl(
    test_space: FESpaceBase | NamedSpace,
    trial_space: FESpaceBase | NamedSpace,
    form: FormKernel[P],
    params: P,
    *,
    backend: str = "jax",
    dep_test: jnp.ndarray | None = None,
    dep_trial: jnp.ndarray | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    kernel: ElementBilinearKernel | None = None,
    jit: bool = True,
) -> FluxSparseOperator:
    warn_float32_assembly_once(context="volume assembly")
    from ..solver import FluxSparseOperator

    if backend != "jax":
        raise NotImplementedError("assemble_bilinear_form_pg currently supports backend='jax' only.")

    test_name = test_space.name if isinstance(test_space, NamedSpace) else "V"
    trial_name = trial_space.name if isinstance(trial_space, NamedSpace) else "U"
    test_space_obj = test_space.space if isinstance(test_space, NamedSpace) else test_space
    trial_space_obj = trial_space.space if isinstance(trial_space, NamedSpace) else trial_space

    include_x_q_eff = False if include_x_q is None else bool(include_x_q)
    lightweight_eff = True if lightweight_context is None else bool(lightweight_context)
    ctxs = build_form_contexts_pair(
        test_space_obj,
        trial_space_obj,
        dep_test=dep_test,
        dep_trial=dep_trial,
        include_x_q=include_x_q_eff,
        lightweight=lightweight_eff,
        test_name=test_name,
        trial_name=trial_name,
    )
    if kernel is None:
        kernel = make_element_bilinear_kernel(form, params, jit=jit)

    K_e_all = jax.vmap(kernel)(ctxs)
    test_elem_dofs = jnp.asarray(test_space_obj.elem_dofs, dtype=INDEX_DTYPE)
    trial_elem_dofs = jnp.asarray(trial_space_obj.elem_dofs, dtype=INDEX_DTYPE)
    if test_elem_dofs.shape[0] != trial_elem_dofs.shape[0]:
        raise ValueError("assemble_bilinear_form_pg requires the same number of elements in test and trial spaces.")

    n_test_ldofs = int(test_elem_dofs.shape[1])
    n_trial_ldofs = int(trial_elem_dofs.shape[1])
    rows = jnp.repeat(test_elem_dofs, n_trial_ldofs, axis=1).reshape(-1)
    cols = jnp.tile(trial_elem_dofs, (1, n_test_ldofs)).reshape(-1)
    data = K_e_all.reshape(-1)
    return FluxSparseOperator(
        rows,
        cols,
        data,
        shape=(int(test_space_obj.n_dofs), int(trial_space_obj.n_dofs)),
        meta={"test_space": test_name, "trial_space": trial_name},
    )


def assemble_bilinear_form_pg(
    test_space: FESpaceBase | NamedSpace,
    trial_space: FESpaceBase | NamedSpace,
    form: FormKernel[P],
    params: P,
    *,
    backend: str = "jax",
    dep_test: jnp.ndarray | None = None,
    dep_trial: jnp.ndarray | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    kernel: ElementBilinearKernel | None = None,
    jit: bool = True,
) -> FluxSparseOperator:
    """
    Compatibility helper for bilinear assembly over distinct test/trial spaces.

    Prefer ``assemble_bilinear_form(BilinearSpaces(...), ...)`` in new code.
    Current constraints:
    - backend='jax' only
    - elementwise-aligned spaces only
    - quadrature points and weights must match between the spaces
    """
    warnings.warn(
        "assemble_bilinear_form_pg(...) is deprecated; "
        "use assemble_bilinear_form(BilinearSpaces(...), ...) instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _assemble_bilinear_form_pg_impl(
        test_space,
        trial_space,
        form,
        params,
        backend=backend,
        dep_test=dep_test,
        dep_trial=dep_trial,
        include_x_q=include_x_q,
        lightweight_context=lightweight_context,
        kernel=kernel,
        jit=jit,
    )


def assemble_mass_matrix(
    space: SpaceLike,
    *,
    backend: str = "jax",
    lumped: bool = False,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> MassReturn:
    warn_float32_assembly_once(context="mass-matrix assembly")
    """
    Assemble mass matrix M_ij = ∫ N_i N_j dΩ.
    Supports scalar and vector spaces. If lumped=True, rows are summed to diagonal.
    """
    from ..solver import FluxSparseMatrix  # local import to avoid circular
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
    n_ldofs = int(space.n_ldofs)
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

    if backend == "numpy":
        if n_chunks is not None:
            raise ValueError("backend='numpy' currently supports only non-chunked assembly.")
        ctxs = space.build_form_contexts(include_x_q=False, lightweight=lightweight_context)
        n_elems = int(space.elem_dofs.shape[0])
        data_parts: list[np.ndarray] = []
        for e in range(n_elems):
            ctx_e = jax.tree_util.tree_map(lambda x: x[e], ctxs)
            me = np.asarray(per_element(ctx_e), dtype=float)
            data_parts.append(me.reshape(-1))
        data = np.concatenate(data_parts, axis=0) if data_parts else np.zeros((0,), dtype=float)
        pat = _get_pattern(space, with_idx=False)
        if pat is None:
            elem_dofs = np.asarray(space.elem_dofs, dtype=int)
            rows = np.repeat(elem_dofs, n_ldofs, axis=1).reshape(-1)
            cols = np.tile(elem_dofs, (1, n_ldofs)).reshape(-1)
        else:
            rows = np.asarray(pat.rows, dtype=int)
            cols = np.asarray(pat.cols, dtype=int)
        if lumped:
            M = np.zeros((int(space.n_dofs),), dtype=float)
            if data.size:
                np.add.at(M, rows, data)
            return M
        return FluxSparseMatrix(rows, cols, data, n_dofs=space.n_dofs)

    if n_chunks is None:
        ctxs = space.build_form_contexts(include_x_q=False, lightweight=lightweight_context)
        M_e_all = jax.vmap(per_element)(ctxs)  # (n_elems, n_ldofs, n_ldofs)
        data = M_e_all.reshape(-1)
    else:
        n_elems = int(space.elem_dofs.shape[0])
        n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
            n_elems=n_elems,
            n_chunks=n_chunks,
            pad_trace=pad_trace,
        )
        use_chunk_context = bool(chunk_build_context)
        use_chunk_context, conn_pad, elem_ids, ctxs_pad = _prepare_chunk_context_source(
            space,
            n_pad=n_pad,
            pad=pad,
            dep=None,
            include_x_q=False,
            lightweight_context=lightweight_context,
            chunk_build_context=use_chunk_context,
        )
        first_ctx_b = _chunk_context_from_source(
            space,
            start=0,
            chunk_size=1,
            use_chunk_context=use_chunk_context,
            conn_pad=conn_pad,
            elem_ids=elem_ids,
            ctxs_pad=ctxs_pad,
            include_x_q=False,
            lightweight_context=lightweight_context,
        )
        sample_me = jax.vmap(per_element)(first_ctx_b)[0]
        m = int(sample_me.shape[0])

        def chunk_values_fn(start: int) -> Array:
            ctx_chunk = _chunk_context_from_source(
                space,
                start=start,
                chunk_size=chunk_size,
                use_chunk_context=use_chunk_context,
                conn_pad=conn_pad,
                elem_ids=elem_ids,
                ctxs_pad=ctxs_pad,
                include_x_q=False,
                lightweight_context=lightweight_context,
            )
            return jax.vmap(per_element)(ctx_chunk)

        data = _accumulate_chunk_matrix_data(
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            n_pad=n_pad,
            m=m,
            dtype=sample_me.dtype,
            valid_mask=valid_mask,
            chunk_values_fn=chunk_values_fn,
        )
        data = data[: n_elems * m * m]

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
    backend: str = "jax",
    kernel: ElementLinearKernel | None = None,
    sparse: bool = False,
    vector_accumulation: Literal["segment", "scatter"] = "scatter",
    n_chunks: Optional[int] = None,
    dep: jnp.ndarray | None = None,
    elem_data: FormContext | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    chunk_build_context: bool | None = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> LinearReturn:
    warn_float32_assembly_once(context="linear-form assembly")
    """
    Expects form(ctx, params) -> (n_q, n_ldofs) and integrates Σ_q form * wJ for RHS.
    If kernel is provided: kernel(ctx) -> (n_ldofs,).
    """
    named_test_space = _resolve_named_linear_space(space)
    if named_test_space is not None:
        return assemble_linear_form(
            named_test_space.space,
            form,
            params,
            backend=backend,
            sparse=sparse,
            vector_accumulation=vector_accumulation,
            kernel=kernel,
            n_chunks=n_chunks,
            dep=dep,
            elem_data=elem_data,
            include_x_q=include_x_q,
            lightweight_context=lightweight_context,
            chunk_build_context=chunk_build_context,
            pad_trace=pad_trace,
            policy=policy,
        )

    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
    elem_dofs = space.elem_dofs
    n_dofs = space.n_dofs
    n_ldofs = space.n_ldofs
    include_x_q_req = include_x_q
    if include_x_q_req is None and policy is None:
        include_x_q_req = bool(getattr(form, "_ff_requires_x_q", False))
    n_chunks, include_x_q, lightweight_context, chunk_build_context, pad_trace = _resolve_assembly_policy(
        policy=policy,
        n_chunks=n_chunks,
        include_x_q=include_x_q_req,
        lightweight_context=lightweight_context,
        chunk_build_context=chunk_build_context,
        pad_trace=pad_trace,
    )
    if vector_accumulation not in ("segment", "scatter"):
        raise ValueError(
            f"vector_accumulation must be 'segment' or 'scatter' (got {vector_accumulation!r})"
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

    if backend == "numpy":
        if _is_numpy_scalar_body_force_fast_path(space, form):
            _K_data, F, F_data = _assemble_numpy_scalar_diffusion_body_force_data(
                space,
                bilinear_params=0.0,
                linear_params=float(params),
                n_chunks=n_chunks,
                pad_trace=pad_trace,
            )
            if sparse:
                return np.asarray(_get_elem_rows(space), dtype=int), F_data, n_dofs
            return F
        data_parts: list[np.ndarray] = []
        if elem_data is not None:
            raise NotImplementedError("backend='numpy' with explicit elem_data is not supported in generic mode.")
        chunk_size_np = int(space.elem_dofs.shape[0]) if n_chunks is None else max(1, int(np.ceil(int(space.elem_dofs.shape[0]) / int(n_chunks))))
        for ctxs in space.build_form_contexts_numpy_chunked(
            chunk_size=chunk_size_np,
            dep=dep,
            include_x_q=include_x_q,
            lightweight=bool(lightweight_context),
        ):
            for ctx_e in ctxs:
                fe = np.asarray(per_element(ctx_e), dtype=float).reshape(-1)
                data_parts.append(fe)
        data = np.concatenate(data_parts, axis=0) if data_parts else np.zeros((0,), dtype=float)
        rows = np.asarray(_get_elem_rows(space), dtype=int)
        if sparse:
            return rows, data, n_dofs
        F = np.zeros((int(n_dofs),), dtype=float)
        if data.size:
            np.add.at(F, rows, data)
        return F

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

        use_chunk_context = bool(dep is None and chunk_build_context)
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
            def chunk_values_fn(start: int) -> Array:
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
                return jax.vmap(per_element)(ctx_chunk)

            data = _accumulate_chunk_vector_data(
                n_chunks=n_chunks,
                chunk_size=chunk_size,
                n_pad=n_pad,
                m=m,
                dtype=sample_fe.dtype,
                valid_mask=valid_mask,
                chunk_values_fn=chunk_values_fn,
            )
            data = data[: n_elems * m]
        else:
            if pad:
                elem_dofs_pad = jnp.concatenate([elem_dofs, jnp.repeat(elem_dofs[-1:], pad, axis=0)], axis=0)
            else:
                elem_dofs_pad = elem_dofs
            def chunk_values_fn(start: int) -> Array:
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
                return jax.vmap(per_element)(ctx_chunk)

            if vector_accumulation == "scatter":
                return _accumulate_chunk_vector_scatter(
                    n_chunks=n_chunks,
                    chunk_size=chunk_size,
                    m=m,
                    n_dofs=n_dofs,
                    dtype=sample_fe.dtype,
                    valid_mask=valid_mask,
                    elem_dofs_pad=elem_dofs_pad,
                    chunk_values_fn=chunk_values_fn,
                )
            return _accumulate_chunk_vector_segment(
                n_chunks=n_chunks,
                chunk_size=chunk_size,
                m=m,
                n_dofs=n_dofs,
                dtype=sample_fe.dtype,
                valid_mask=valid_mask,
                elem_dofs_pad=elem_dofs_pad,
                chunk_values_fn=chunk_values_fn,
            )

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
    backend: str = "jax",
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,
    dep: jnp.ndarray | None = None,
    elem_data: FormContext | None = None,
    include_x_q: bool | None = None,
    lightweight_context: bool | None = None,
    chunk_build_context: bool | None = None,
    bilinear_kernel: ElementBilinearKernel | None = None,
    linear_kernel: ElementLinearKernel | None = None,
    vector_accumulation: Literal["segment", "scatter"] = "segment",
    jit: bool = True,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> PairReturn:
    warn_float32_assembly_once(context="paired bilinear/linear assembly")
    from ..solver import FluxSparseMatrix
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
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
        bilinear_kernel = make_element_bilinear_kernel(
            bilinear_form,
            bilinear_params,
            jit=(jit if backend == "jax" else False),
        )
    if linear_kernel is None:
        linear_kernel = make_element_linear_kernel(
            linear_form,
            linear_params,
            jit=(jit if backend == "jax" else False),
        )
    if vector_accumulation not in ("segment", "scatter"):
        raise ValueError(
            f"vector_accumulation must be 'segment' or 'scatter' (got {vector_accumulation!r})"
        )

    n_elems = int(space.elem_dofs.shape[0])
    n_ldofs = int(space.n_ldofs)

    if backend == "numpy":
        if _is_numpy_scalar_diffusion_body_force_pair_fast_path(space, bilinear_form, linear_form):
            return _assemble_numpy_scalar_diffusion_pair_fast(
                space,
                bilinear_params=float(bilinear_params),
                linear_params=float(linear_params),
                pattern=pat,
                n_chunks=n_chunks,
                pad_trace=pad_trace,
            )
        bilinear_kernel_batched = jax.vmap(bilinear_kernel)
        linear_kernel_batched = jax.vmap(linear_kernel)
        if elem_data is None:
            elem_data = space.build_form_contexts(
                dep=dep,
                include_x_q=include_x_q,
                lightweight=lightweight_context,
            )
        if n_chunks is None:
            assert elem_data is not None
            Ke = np.asarray(bilinear_kernel_batched(elem_data), dtype=float)
            Fe = np.asarray(linear_kernel_batched(elem_data), dtype=float)
            K_data = Ke.reshape(-1)
            F_data = Fe.reshape(-1)
            rows = np.asarray(_get_elem_rows(space), dtype=int)
            F = np.zeros((int(space.n_dofs),), dtype=float)
            if F_data.size:
                np.add.at(F, rows, F_data)
            return FluxSparseMatrix(pat, K_data), F

        n_chunks, chunk_size, pad, n_pad, valid_mask = _prepare_chunk_iteration(
            n_elems=int(n_elems),
            n_chunks=n_chunks,
            pad_trace=pad_trace,
        )
        use_chunk_context = bool(dep is None and chunk_build_context)
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

        def chunk_values_np(start: int) -> tuple[np.ndarray, np.ndarray]:
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
            Ke_chunk = np.asarray(bilinear_kernel_batched(ctx_chunk), dtype=float)
            Fe_chunk = np.asarray(linear_kernel_batched(ctx_chunk), dtype=float)
            return Ke_chunk, Fe_chunk

        sample_ke, sample_fe = chunk_values_np(0)
        m = int(sample_ke.shape[-1])
        rows = np.asarray(_get_elem_rows(space), dtype=int)
        if pad:
            rows_pad = np.concatenate([rows, np.repeat(rows[-n_ldofs:], pad * n_ldofs)])
            valid_mask_np = np.asarray(valid_mask, dtype=bool)
        else:
            rows_pad = rows
            valid_mask_np = None

        K_parts: list[np.ndarray] = []
        F = np.zeros((int(space.n_dofs),), dtype=float)
        for start in range(0, int(n_pad), int(chunk_size)):
            Ke_chunk, Fe_chunk = chunk_values_np(start)
            if valid_mask_np is not None:
                mask = valid_mask_np[start : start + int(chunk_size)]
                Ke_chunk = Ke_chunk[mask]
                Fe_chunk = Fe_chunk[mask]
            K_parts.append(Ke_chunk.reshape(-1))
            if Fe_chunk.size:
                chunk_rows = rows_pad[start * n_ldofs : (start + Fe_chunk.shape[0]) * n_ldofs]
                np.add.at(F, chunk_rows, Fe_chunk.reshape(-1))
        K_data = np.concatenate(K_parts, axis=0) if K_parts else np.zeros((0,), dtype=float)
        return FluxSparseMatrix(pat, K_data), F

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

    use_chunk_context = bool(dep is None and chunk_build_context)
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
    def chunk_values_fn(start: int) -> tuple[Array, Array]:
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
        return jax.vmap(bilinear_kernel)(ctx_chunk), jax.vmap(linear_kernel)(ctx_chunk)

    sample_ke, sample_fe = chunk_values_fn(0)
    sample_ke = sample_ke[0]
    sample_fe = sample_fe[0]
    m = int(sample_ke.shape[0])
    if pad:
        elem_dofs_pad = jnp.concatenate([space.elem_dofs, jnp.repeat(space.elem_dofs[-1:], pad, axis=0)], axis=0)
    else:
        elem_dofs_pad = space.elem_dofs

    if vector_accumulation == "scatter":
        K_data, F = _accumulate_chunk_matrix_and_vector_scatter(
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            n_pad=n_pad,
            m=m,
            n_dofs=space.n_dofs,
            matrix_dtype=sample_ke.dtype,
            vector_dtype=sample_fe.dtype,
            valid_mask=valid_mask,
            elem_dofs_pad=elem_dofs_pad,
            chunk_values_fn=chunk_values_fn,
        )
    else:
        K_data, F = _accumulate_chunk_matrix_and_vector_segment(
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            n_pad=n_pad,
            m=m,
            n_dofs=space.n_dofs,
            matrix_dtype=sample_ke.dtype,
            vector_dtype=sample_fe.dtype,
            valid_mask=valid_mask,
            elem_dofs_pad=elem_dofs_pad,
            chunk_values_fn=chunk_values_fn,
        )
    K_data = K_data[: n_elems * m * m]
    return FluxSparseMatrix(pat, K_data), F


def assemble_functional(
    space: SpaceLike,
    form: FormKernel[P],
    params: P,
    *,
    backend: str = "jax",
) -> jnp.ndarray | np.ndarray:
    """
    Assemble scalar functional J = ∫ form(ctx, params) dΩ.
    Expects form(ctx, params) -> (n_q,) or (n_q, 1).
    """
    if backend not in {"jax", "numpy"}:
        raise ValueError("backend must be 'jax' or 'numpy'")
    elem_data = space.build_form_contexts()

    includes_measure = getattr(form, "_includes_measure", False)

    def per_element(ctx: FormContext):
        integrand = form(ctx, params)
        if integrand.ndim == 2 and integrand.shape[1] == 1:
            integrand = integrand[:, 0]
        wJ = ctx.w * ctx.test.detJ
        return _integrate_q_scalar(
            integrand,
            wJ,
            includes_measure=bool(includes_measure),
        )

    if backend == "numpy":
        n_elems = int(space.elem_dofs.shape[0])
        total = 0.0
        for e in range(n_elems):
            ctx_e = jax.tree_util.tree_map(lambda x: x[e], elem_data)
            total += float(np.asarray(per_element(ctx_e)))
        return np.asarray(total, dtype=float)

    vals = jax.vmap(per_element)(elem_data)
    return jnp.sum(vals)


def assemble_jacobian_global(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
) -> JacobianReturn:
    from .assembly_jacobian import assemble_jacobian_global as _impl
    return _impl(space, res_form, u, params)


def assemble_jacobian_elementwise(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
) -> JacobianReturn:
    from .assembly_jacobian import assemble_jacobian_elementwise as _impl
    return _impl(space, res_form, u, params)


def assemble_residual_global(
    space: SpaceLike,
    form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    sparse: bool = False
) -> LinearReturn:
    from .assembly_residual import assemble_residual_global as _impl
    return _impl(space, form, u, params, sparse=sparse)


def assemble_residual_elementwise(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    sparse: bool = False,
) -> LinearReturn:
    from .assembly_residual import assemble_residual_elementwise as _impl
    return _impl(space, res_form, u, params, sparse=sparse)


# Backward compatibility aliases (prefer assemble_*_elementwise).
assemble_jacobian_elementwise_xla = assemble_jacobian_elementwise
assemble_residual_elementwise_xla = assemble_residual_elementwise


def make_element_bilinear_kernel(
    form: FormKernel[P], params: P, *, jit: bool = True
) -> ElementBilinearKernel:
    from .assembly_kernels import make_element_bilinear_kernel as _impl
    return _impl(form, params, jit=jit)


def make_element_linear_kernel(
    form: FormKernel[P], params: P, *, jit: bool = True
) -> ElementLinearKernel:
    from .assembly_kernels import make_element_linear_kernel as _impl
    return _impl(form, params, jit=jit)


def make_element_residual_kernel(
    res_form: ResidualForm[P], params: P
) -> ElementResidualKernel:
    from .assembly_kernels import make_element_residual_kernel as _impl
    return _impl(res_form, params)


def make_element_jacobian_kernel(
    res_form: ResidualForm[P], params: P
) -> ElementJacobianKernel:
    from .assembly_kernels import make_element_jacobian_kernel as _impl
    return _impl(res_form, params)


def element_residual(
    res_form: ResidualFormLike[P], ctx: FormContext, u_elem: ResidualInput, params: P
) -> ResidualValue:
    from .assembly_kernels import element_residual as _impl
    return _impl(res_form, ctx, u_elem, params)


def element_jacobian(
    res_form: ResidualFormLike[P], ctx: FormContext, u_elem: ResidualInput, params: P
) -> ResidualValue:
    from .assembly_kernels import element_jacobian as _impl
    return _impl(res_form, ctx, u_elem, params)


def make_element_kernel(
    form: FormKernel[P] | ResidualForm[P],
    params: P,
    *,
    kind: Literal["bilinear", "linear", "residual", "jacobian"],
    jit: bool = True,
) -> ElementKernel:
    from .assembly_kernels import make_element_kernel as _impl
    return _impl(form, params, kind=kind, jit=jit)


def make_sparsity_pattern(space: SpaceLike, *, with_idx: bool = True) -> SparsityPattern:
    from .assembly_pattern import make_sparsity_pattern as _impl
    return _impl(space, with_idx=with_idx)


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
    from .assembly_jacobian import assemble_jacobian_values as _impl
    return _impl(
        space,
        res_form,
        u,
        params,
        kernel=kernel,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
    )


def assemble_residual_scatter(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementResidualKernel | None = None,
    sparse: bool = False,
    vector_accumulation: Literal["segment", "scatter"] = "scatter",
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> LinearReturn:
    from .assembly_residual import assemble_residual_scatter as _impl
    return _impl(
        space,
        res_form,
        u,
        params,
        kernel=kernel,
        sparse=sparse,
        vector_accumulation=vector_accumulation,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
    )


def assemble_jacobian_scatter(
    space: SpaceLike,
    res_form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    kernel: ElementJacobianKernel | None = None,
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> JacobianReturn:
    from .assembly_jacobian import assemble_jacobian_scatter as _impl
    return _impl(
        space,
        res_form,
        u,
        params,
        kernel=kernel,
        pattern=pattern,
        n_chunks=n_chunks,
        pad_trace=pad_trace,
        policy=policy,
    )


# Alias scatter-based assembly as the default public API
def assemble_residual(
    space: SpaceLike,
    form: ResidualForm[P],
    u: jnp.ndarray,
    params: P,
    *,
    backend: str = "jax",
    kernel: ElementResidualKernel | None = None,
    sparse: bool = False,
    vector_accumulation: Literal["segment", "scatter"] = "scatter",
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> LinearReturn:
    warn_float32_assembly_once(context="residual assembly")
    named_spaces = _resolve_named_residual_spaces(space)
    if named_spaces is not None:
        test_space, unknown_space = named_spaces
        if test_space.space is unknown_space.space:
            from .assembly_residual import assemble_residual as _impl

            return _impl(
                test_space.space,
                form,
                u,
                params,
                backend=backend,
                kernel=kernel,
                sparse=sparse,
                vector_accumulation=vector_accumulation,
                n_chunks=n_chunks,
                pad_trace=pad_trace,
                policy=policy,
            )
        if any(arg is not None for arg in (kernel, n_chunks, pad_trace, policy)):
            raise ValueError(
                "Distinct-space assemble_residual currently does not support kernel/n_chunks/pad_trace/policy overrides."
            )
        if backend != "jax":
            raise NotImplementedError("Named-space assemble_residual currently supports backend='jax' only.")
        ctxs = build_form_contexts_pair(
            test_space.space,
            unknown_space.space,
            include_x_q=False,
            lightweight=True,
            test_name=test_space.name,
            trial_name=unknown_space.name,
        )
        u_elems = u[unknown_space.space.elem_dofs]
        ker = make_element_residual_kernel(form, params)
        elem_res = jax.vmap(ker)(ctxs, u_elems)
        rows = jnp.asarray(test_space.space.elem_dofs, dtype=INDEX_DTYPE).reshape(-1)
        data = elem_res.reshape(-1)
        n_dofs = int(test_space.space.n_dofs)
        if sparse:
            return rows, data, n_dofs
        return jax.ops.segment_sum(data, rows, n_dofs)
    from .assembly_residual import assemble_residual as _impl
    return _impl(
        space,
        form,
        u,
        params,
        backend=backend,
        kernel=kernel,
        sparse=sparse,
        vector_accumulation=vector_accumulation,
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
    pattern: SparsityPattern | None = None,
    n_chunks: Optional[int] = None,
    pad_trace: bool | None = None,
    policy: AssemblyPolicy | None = None,
) -> JacobianReturn:
    warn_float32_assembly_once(context="jacobian assembly")
    named_spaces = _resolve_named_jacobian_spaces(space)
    if named_spaces is not None:
        test_space, trial_space = named_spaces
        if test_space.space is trial_space.space:
            from .assembly_jacobian import assemble_jacobian as _impl

            return _impl(
                test_space.space,
                res_form,
                u,
                params,
                kernel=kernel,
                pattern=pattern,
                n_chunks=n_chunks,
                pad_trace=pad_trace,
                policy=policy,
            )
        if any(arg is not None for arg in (kernel, pattern, n_chunks, pad_trace, policy)):
            raise ValueError(
                "Distinct-space assemble_jacobian currently does not support kernel/pattern/n_chunks/pad_trace/policy overrides."
            )
        ctxs = build_form_contexts_pair(
            test_space.space,
            trial_space.space,
            include_x_q=False,
            lightweight=True,
            test_name=test_space.name,
            trial_name=trial_space.name,
        )
        u_elems = u[trial_space.space.elem_dofs]
        ker = make_element_jacobian_kernel(res_form, params)
        J_e_all = jax.vmap(ker)(u_elems, ctxs)
        test_elem_dofs = jnp.asarray(test_space.space.elem_dofs, dtype=INDEX_DTYPE)
        trial_elem_dofs = jnp.asarray(trial_space.space.elem_dofs, dtype=INDEX_DTYPE)
        n_test_ldofs = int(test_elem_dofs.shape[1])
        n_trial_ldofs = int(trial_elem_dofs.shape[1])
        rows = jnp.repeat(test_elem_dofs, n_trial_ldofs, axis=1).reshape(-1)
        cols = jnp.tile(trial_elem_dofs, (1, n_test_ldofs)).reshape(-1)
        data = J_e_all.reshape(-1)
        from ..solver import FluxSparseOperator

        return FluxSparseOperator(
            rows,
            cols,
            data,
            shape=(int(test_space.space.n_dofs), int(trial_space.space.n_dofs)),
            meta={"test_space": test_space.name, "trial_space": trial_space.name},
        )
    from .assembly_jacobian import assemble_jacobian as _impl
    return _impl(
        space,
        res_form,
        u,
        params,
        kernel=kernel,
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
    _form._ff_requires_x_q = True  # type: ignore[attr-defined]
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
