from __future__ import annotations
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TYPE_CHECKING, TypeAlias, TypeVar, cast
import jax
import jax.numpy as jnp
import numpy as np
import warnings

P = TypeVar("P")

if TYPE_CHECKING:
    from .assembly import (
        ElementBilinearKernel,
        ElementJacobianKernel,
        ElementLinearKernel,
        ElementResidualKernel,
        Kernel,
        FormKernel,
        BilinearReturn,
        JacobianReturn,
        LinearReturn,
        MassReturn,
        ResidualForm,
    )
    from .weakform import BilinearForm, LinearForm
    from ..solver import FluxSparseMatrix, SparsityPattern
else:
    Kernel = Callable[..., Any]
    FormKernel = Callable[..., Any]
    ResidualForm = Callable[..., Any]
    ElementBilinearKernel = Callable[..., Any]
    ElementLinearKernel = Callable[..., Any]
    ElementResidualKernel = Callable[..., Any]
    ElementJacobianKernel = Callable[..., Any]
    BilinearReturn = Any
    JacobianReturn = Any
    LinearReturn = Any
    MassReturn = Any
    BilinearForm = Any
    LinearForm = Any
    FluxSparseMatrix = Any
    SparsityPattern = Any

KernelCacheKey: TypeAlias = tuple[str, int, int, bool]
KernelCache: TypeAlias = dict[
    KernelCacheKey,
    ElementBilinearKernel | ElementLinearKernel | ElementResidualKernel | ElementJacobianKernel,
]
PatternCache: TypeAlias = dict[bool, SparsityPattern]
PatternLike: TypeAlias = str | SparsityPattern | None

_WARNED_UNTAGGED_KERNELS: set[int] = set()


def _warn_untagged_kernel(form) -> None:
    key = id(form)
    if key in _WARNED_UNTAGGED_KERNELS:
        return
    _WARNED_UNTAGGED_KERNELS.add(key)
    warnings.warn(
        "Raw kernel has no _ff_kind metadata; prefer tagging with ff.kernel(...) or "
        "set _ff_kind/_ff_domain on the callable.",
        category=UserWarning,
        stacklevel=3,
    )

from .dtypes import INDEX_DTYPE
from ..mesh import (
    BaseMesh,
    BaseMeshPytree,
    HexMesh,
    HexMeshPytree,
    TetMesh,
    TetMeshPytree,
)
from .basis import (
    Basis3D,
    HexTriLinearBasis,
    HexTriLinearBasisPytree,
    HexSerendipityBasis20,
    HexSerendipityBasis20Pytree,
    HexTriQuadraticBasis27,
    HexTriQuadraticBasis27Pytree,
    TetLinearBasis,
    TetLinearBasisPytree,
    TetQuadraticBasis10,
    TetQuadraticBasis10Pytree,
    make_hex20_basis,
    make_hex20_basis_pytree,
    make_hex27_basis,
    make_hex27_basis_pytree,
    make_hex_basis,
    make_hex_basis_pytree,
    make_tet10_basis,
    make_tet10_basis_pytree,
    make_tet_basis,
    make_tet_basis_pytree,
)
from .forms import (
    ElementVector,
    FormContext,
    FormFieldLike,
    ScalarFormField,
    VectorFormField,
)
from .data import SpaceData


class FESpaceBase(Protocol):
    """
    Protocol for FE space objects used by assembly.

    This defines the minimal interface required by the core assembly routines:
    element-to-DOF connectivity, value dimension, and the ability to build
    per-element FormContext objects (test/trial fields plus quadrature data).
    """
    elem_dofs: jnp.ndarray
    value_dim: int

    @property
    def n_dofs(self) -> int: ...

    @property
    def n_ldofs(self) -> int: ...

    def build_form_contexts(self, dep: jnp.ndarray | None = None) -> FormContext: ...


@dataclass(eq=False)
class FESpaceClosure:
    """
    Finite element space built from a mesh, basis, and element dof map.

    This is the standard space used by fluxfem. It bundles:
    - a mesh (geometry and connectivity),
    - a basis (shape functions + quadrature),
    - an element-to-DOF map (elem_dofs),
    - and metadata such as value_dim and cached sparsity patterns.

    The class provides thin wrappers around assembly helpers and constructs
    FormContext objects for element-level integration.
    """
    mesh: BaseMesh
    basis: Basis3D
    elem_dofs: jnp.ndarray  # (n_elems, n_ldofs) int64
    value_dim: int = 1      # 1=scalar, 3=vector, etc.
    _n_dofs: int | None = None
    _n_ldofs: int | None = None
    data: SpaceData | None = None
    _pattern_cache: PatternCache = field(default_factory=dict, repr=False)
    _kernel_cache: KernelCache = field(default_factory=dict, repr=False)
    _elem_rows_cache: jnp.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # Ensure value_dim is a Python int (avoid tracers).
        self.value_dim = operator.index(self.value_dim)

        if self._n_ldofs is None:
            self._n_ldofs = int(self.elem_dofs.shape[1])
        if self._n_dofs is None:
            self._n_dofs = int(np.asarray(self.elem_dofs).max()) + 1

        n_nodes = int(self.mesh.element_coords().shape[1])
        expected = n_nodes * self.value_dim
        if self._n_ldofs != expected:
            raise ValueError(
                f"n_ldofs mismatch: elem_dofs has {self._n_ldofs}, "
                f"but n_nodes({n_nodes})*value_dim({self.value_dim})={expected}"
            )

        if self.data is None:
            self.data = SpaceData.from_space(self)

    @property
    def n_dofs(self) -> int:
        assert self._n_dofs is not None
        return self._n_dofs

    @property
    def n_ldofs(self) -> int:
        assert self._n_ldofs is not None
        return self._n_ldofs

    def get_elem_rows(self) -> jnp.ndarray:
        cached = self._elem_rows_cache
        if cached is not None:
            return cached
        rows = self.elem_dofs.reshape(-1)
        if jax.core.trace_ctx.is_top_level():
            self._elem_rows_cache = rows
        return rows

    def build_form_contexts(self, dep: jnp.ndarray | None = None) -> FormContext:
        def _tie_in(x, y):
            if x is None:
                return y
            try:
                return jax.lax.tie_in(x, y)
            except AttributeError:
                return y + jnp.sin(x) * 0

        vd = int(self.value_dim)
        mesh, basis = self.mesh, self.basis
        elem_coords = mesh.element_coords()      # (n_elems, n_nodes, 3)
        elem_coords = _tie_in(dep, elem_coords)

        N_ref = basis.shape_functions()          # (n_q, n_nodes)
        w_ref = basis.quad_weights               # (n_q,)
        x_q = jnp.einsum("qa,eai->eqi", N_ref, elem_coords)  # (n_elems, n_q, 3)
        w = jnp.broadcast_to(
            w_ref[None, :], (elem_coords.shape[0], w_ref.shape[0])
        )
        w = _tie_in(dep, w)

        if vd == 1:
            def make_field(Xe):
                return ScalarFormField(N=N_ref, elem_coords=Xe, basis=basis)
        else:
            def make_field(Xe):
                return VectorFormField(
                    N=N_ref, elem_coords=Xe, basis=basis, value_dim=vd
                )

        test = jax.vmap(make_field)(elem_coords)
        # Test/trial share the same field data for single-space bilinear forms.
        trial = test

        return FormContext(
            test=test, trial=trial, x_q=x_q,
            w=w, elem_id=jnp.arange(elem_coords.shape[0])
        )

    def make_batched_assembler(self, *, dep: jnp.ndarray | None = None, pattern=None):
        from .assembly import BatchedAssembler
        if pattern is None:
            pattern = self.get_sparsity_pattern(with_idx=True)
        return BatchedAssembler.from_space(self, dep=dep, pattern=pattern)

    def build_cg_operator(
        self,
        A,
        *,
        matvec: str = "flux",
        preconditioner=None,
        solver: str = "cg",
        dof_per_node: int | None = None,
        block_sizes=None,
    ):
        from ..solver.cg import build_cg_operator

        if dof_per_node is None:
            dof_per_node = int(self.value_dim)
        return build_cg_operator(
            A,
            matvec=matvec,
            preconditioner=preconditioner,
            solver=solver,
            dof_per_node=dof_per_node,
            block_sizes=block_sizes,
        )

    def _kernel_cache_key(self, kind: str, form, params, *, jit: bool) -> KernelCacheKey | None:
        if not jit:
            return None
        form_key = getattr(form, "__wrapped__", form)
        try:
            params_key = hash(params)
        except Exception:
            return None
        return (kind, id(form_key), params_key, True)

    def _get_cached_kernel(self, kind: str, form, params, *, jit: bool, maker):
        key = self._kernel_cache_key(kind, form, params, jit=jit)
        if key is None:
            return maker(form, params, jit=jit)
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        kernel = maker(form, params, jit=jit)
        if jax.core.trace_ctx.is_top_level():
            self._kernel_cache[key] = kernel
        return kernel

    def assemble(
        self,
        form: FormKernel[P] | BilinearForm | LinearForm,
        params: P | None = None,
        *,
        kind: str | None = None,
        n_chunks: int | None = None,
        dep: jnp.ndarray | None = None,
        jit: bool = True,
        pattern: PatternLike = "auto",
        kernel: ElementBilinearKernel | ElementLinearKernel | None = None,
        **kwargs,
    ):
        """
        High-level assembly entry point with optional kernel caching.

        kind: "bilinear" or "linear". If None, inferred from LinearForm/BilinearForm
            or compiled/kernels tagged with _ff_kind metadata.
        pattern: "auto" to reuse cached sparsity pattern for bilinear assembly.
        """
        from .weakform import BilinearForm, LinearForm
        from .assembly import make_element_bilinear_kernel, make_element_linear_kernel

        if kind is None:
            if isinstance(form, BilinearForm):
                kind = "bilinear"
                form = form.get_compiled()
            elif isinstance(form, LinearForm):
                kind = "linear"
                form = form.get_compiled()
            else:
                inferred_kind = getattr(form, "_ff_kind", None)
                inferred_domain = getattr(form, "_ff_domain", None)
                if inferred_kind is None:
                    raise ValueError(
                        f"kind is required for raw kernels without metadata (got {form!r}). "
                        "Use @ff.kernel(kind=..., domain=...) or pass kind=."
                    )
                if inferred_domain not in (None, "volume"):
                    raise ValueError(
                        f"Unsupported form domain '{inferred_domain}' for Space.assemble. "
                        "Use assemble_surface_linear_form or assemble_surface_bilinear_form."
                    )
                kind = inferred_kind
        else:
            inferred_kind = getattr(form, "_ff_kind", None)
            inferred_domain = getattr(form, "_ff_domain", None)
            if inferred_kind is None:
                _warn_untagged_kernel(form)
            if inferred_kind is not None and inferred_kind != kind:
                raise ValueError(
                    f"assemble kind '{kind}' does not match form kind '{inferred_kind}' "
                    f"for {form!r}. "
                    "Align kind= with the kernel metadata (or retag the kernel)."
                )
            if inferred_domain not in (None, "volume"):
                raise ValueError(
                    f"Unsupported form domain '{inferred_domain}' for Space.assemble. "
                    "Use assemble_surface_linear_form or assemble_surface_bilinear_form."
                )

        if kind == "bilinear":
            if kernel is None:
                kernel = self._get_cached_kernel(
                    "bilinear",
                    form,
                    params,
                    jit=jit,
                    maker=make_element_bilinear_kernel,
                )
            form_kernel = cast(FormKernel, form)
            pattern_use = None
            if pattern == "auto":
                pattern_use = self.get_sparsity_pattern(with_idx=True)
            else:
                pattern_use = pattern
            return self.assemble_bilinear_form(
                form_kernel,
                params,
                n_chunks=n_chunks,
                dep=dep,
                kernel=kernel,
                pattern=pattern_use,
                **kwargs,
            )
        if kind == "linear":
            if kernel is None:
                kernel = self._get_cached_kernel(
                    "linear",
                    form,
                    params,
                    jit=jit,
                    maker=make_element_linear_kernel,
                )
            form_kernel = cast(FormKernel, form)
            return self.assemble_linear_form(
                form_kernel,
                params,
                n_chunks=n_chunks,
                dep=dep,
                kernel=kernel,
                **kwargs,
            )
        raise ValueError(f"Unsupported assemble kind: {kind}")

    # --- Thin wrappers over functional assembly APIs (kept functional for JAX friendliness) ---
    def assemble_bilinear_form(
        self,
        form: FormKernel[P],
        params: P,
        *,
        n_chunks: int | None = None,
        dep: jnp.ndarray | None = None,
        kernel: ElementBilinearKernel | None = None,
        **kwargs,
    ) -> FluxSparseMatrix:
        """Assemble bilinear form; kernel(ctx) -> (n_ldofs, n_ldofs) if provided."""
        from .assembly import assemble_bilinear_form
        if "pattern" not in kwargs or kwargs.get("pattern") is None:
            kwargs["pattern"] = self.get_sparsity_pattern(with_idx=True)
        return assemble_bilinear_form(
            self, form, params, n_chunks=n_chunks, dep=dep, kernel=kernel, **kwargs
        )

    def assemble_linear_form(
        self,
        form: FormKernel[P],
        params: P,
        *,
        n_chunks: int | None = None,
        dep: jnp.ndarray | None = None,
        kernel: ElementLinearKernel | None = None,
        **kwargs,
    ) -> LinearReturn:
        """Assemble linear form; kernel(ctx) -> (n_ldofs,) if provided."""
        from .assembly import assemble_linear_form
        return assemble_linear_form(
            self, form, params, n_chunks=n_chunks, dep=dep, kernel=kernel, **kwargs
        )

    def assemble_functional(self, form: FormKernel[P], params: P) -> jnp.ndarray:
        from .assembly import assemble_functional
        return assemble_functional(self, form, params)

    def assemble_mass_matrix(
        self,
        *,
        n_chunks: int | None = None,
        **kwargs,
    ) -> MassReturn:
        from .assembly import assemble_mass_matrix
        return assemble_mass_matrix(self, n_chunks=n_chunks, **kwargs)

    def assemble_bilinear_dense(
        self,
        kernel: FormKernel[P],
        params: P,
        **kwargs,
    ) -> BilinearReturn:
        from .assembly import assemble_bilinear_dense
        return assemble_bilinear_dense(self, kernel, params, **kwargs)

    def assemble_residual(
        self,
        res_form: ResidualForm[P],
        u: jnp.ndarray,
        params: P,
        *,
        kernel: ElementResidualKernel | None = None,
        **kwargs,
    ) -> LinearReturn:
        """Assemble residual; kernel(ctx, u_elem) -> (n_ldofs,) if provided."""
        from .assembly import assemble_residual
        return assemble_residual(self, res_form, u, params, kernel=kernel, **kwargs)

    def assemble_jacobian(
        self,
        res_form: ResidualForm[P],
        u: jnp.ndarray,
        params: P,
        *,
        kernel: ElementJacobianKernel | None = None,
        **kwargs,
    ) -> JacobianReturn:
        """Assemble Jacobian; kernel(u_elem, ctx) -> (n_ldofs, n_ldofs) if provided."""
        from .assembly import assemble_jacobian
        return assemble_jacobian(self, res_form, u, params, kernel=kernel, **kwargs)

    def get_sparsity_pattern(self, *, with_idx: bool = True):
        cached = self._pattern_cache.get(with_idx)
        if cached is not None:
            return cached
        from .assembly import make_sparsity_pattern
        pat = make_sparsity_pattern(self, with_idx=with_idx)
        if jax.core.trace_ctx.is_top_level():
            self._pattern_cache[with_idx] = pat
        return pat


@jax.tree_util.register_pytree_node_class
class FESpacePytree(FESpaceClosure):
    """
    FESpaceClosure with JAX pytree support.

    Use this when a space must be carried through JAX transformations (jit/vmap),
    or stored inside other pytrees. Only mesh, basis, and elem_dofs are treated
    as children; metadata is preserved as auxiliary data.
    """
    def tree_flatten(self):
        children = (self.mesh, self.basis, self.elem_dofs)
        aux = {
            "value_dim": int(self.value_dim),
            "_n_dofs": self._n_dofs,
            "_n_ldofs": self._n_ldofs,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mesh, basis, elem_dofs = children
        return cls(
            mesh=mesh,
            basis=basis,
            elem_dofs=elem_dofs,
            value_dim=aux["value_dim"],
            _n_dofs=aux.get("_n_dofs", None),
            _n_ldofs=aux.get("_n_ldofs", None),
        )


FESpace = FESpaceClosure


def make_space(
    mesh: BaseMesh,
    basis: Basis3D,
    element: ElementVector | None = None,
) -> FESpace:
    """
    Build an FE space from a mesh and basis.

    element=None → scalar dof per node (elem_dofs = mesh.conn), value_dim=1
    element=ElementVector(dim) → vector dof per node, value_dim=dim
    """
    if element is None:
        elem_dofs = mesh.conn
        value_dim = 1
    else:
        elem_dofs = element.dof_map(mesh.conn)
        value_dim = int(element.dim)

    return FESpace(
        mesh=mesh,
        basis=basis,
        elem_dofs=jnp.asarray(elem_dofs, dtype=INDEX_DTYPE),
        value_dim=value_dim
    )


def _mesh_to_pytree(mesh: BaseMesh) -> BaseMeshPytree:
    if isinstance(mesh, HexMeshPytree) or isinstance(mesh, TetMeshPytree):
        return mesh
    if isinstance(mesh, HexMesh):
        return HexMeshPytree(
            coords=mesh.coords,
            conn=mesh.conn,
            cell_tags=mesh.cell_tags,
            node_tags=mesh.node_tags,
        )
    if isinstance(mesh, TetMesh):
        return TetMeshPytree(
            coords=mesh.coords,
            conn=mesh.conn,
            cell_tags=mesh.cell_tags,
            node_tags=mesh.node_tags,
        )
    raise TypeError(f"Unsupported mesh type for pytree: {type(mesh)}")


def _basis_to_pytree(basis):
    if isinstance(
        basis,
        (
            HexTriLinearBasisPytree,
            HexSerendipityBasis20Pytree,
            HexTriQuadraticBasis27Pytree,
            TetLinearBasisPytree,
            TetQuadraticBasis10Pytree,
        ),
    ):
        return basis
    if isinstance(basis, HexTriLinearBasis):
        return HexTriLinearBasisPytree(basis.quad_points, basis.quad_weights)
    if isinstance(basis, HexSerendipityBasis20):
        return HexSerendipityBasis20Pytree(basis.quad_points, basis.quad_weights)
    if isinstance(basis, HexTriQuadraticBasis27):
        return HexTriQuadraticBasis27Pytree(basis.quad_points, basis.quad_weights)
    if isinstance(basis, TetLinearBasis):
        return TetLinearBasisPytree(basis.quad_points, basis.quad_weights)
    if isinstance(basis, TetQuadraticBasis10):
        return TetQuadraticBasis10Pytree(basis.quad_points, basis.quad_weights)
    raise TypeError(f"Unsupported basis type for pytree: {type(basis)}")


def make_space_pytree(
    mesh: BaseMeshPytree,
    basis: Basis3D,
    element: ElementVector | None = None,
) -> FESpacePytree:
    """Build a pytree-compatible FE space."""
    if element is None:
        elem_dofs = mesh.conn
        value_dim = 1
    else:
        elem_dofs = element.dof_map(mesh.conn)
        value_dim = int(element.dim)

    mesh_py = _mesh_to_pytree(mesh)
    basis_py = _basis_to_pytree(basis)

    return FESpacePytree(
        mesh=mesh_py,
        basis=basis_py,
        elem_dofs=jnp.asarray(elem_dofs, dtype=INDEX_DTYPE),
        value_dim=value_dim,
    )


def make_tet10_space(
    mesh: TetMesh, dim: int = 1, intorder: int = 2
) -> FESpace:
    """Create a quadratic tet space (10-node elements)."""
    basis = make_tet10_basis(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space(mesh, basis, element)


def make_tet10_space_pytree(
    mesh: TetMesh, dim: int = 1, intorder: int = 2
) -> FESpacePytree:
    """Create a pytree quadratic tet space (10-node elements)."""
    basis = make_tet10_basis_pytree(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space_pytree(cast(BaseMeshPytree, mesh), basis, element)


def make_hex_space(mesh: HexMesh, dim: int = 1, intorder: int = 2) -> FESpace:
    """Create a trilinear hex space (8-node elements)."""
    basis = make_hex_basis(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space(mesh, basis, element)


def make_hex_space_pytree(
    mesh: HexMesh, dim: int = 1, intorder: int = 2
) -> FESpacePytree:
    """Create a pytree trilinear hex space (8-node elements)."""
    basis = make_hex_basis_pytree(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space_pytree(cast(BaseMeshPytree, mesh), basis, element)


def make_hex20_space(
    mesh: HexMesh, dim: int = 1, intorder: int = 2
) -> FESpace:
    """Create a serendipity hex space (20-node elements)."""
    basis = make_hex20_basis(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space(mesh, basis, element)


def make_hex20_space_pytree(
    mesh: HexMesh, dim: int = 1, intorder: int = 2
) -> FESpacePytree:
    """Create a pytree serendipity hex space (20-node elements)."""
    basis = make_hex20_basis_pytree(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space_pytree(cast(BaseMeshPytree, mesh), basis, element)


def make_hex27_space(
    mesh: HexMesh, dim: int = 1, intorder: int = 3
) -> FESpace:
    """Create a triquadratic hex space (27-node elements)."""
    basis = make_hex27_basis(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space(mesh, basis, element)


def make_hex27_space_pytree(
    mesh: HexMesh, dim: int = 1, intorder: int = 3
) -> FESpacePytree:
    """Create a pytree triquadratic hex space (27-node elements)."""
    basis = make_hex27_basis_pytree(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space_pytree(cast(BaseMeshPytree, mesh), basis, element)


def make_tet_space(mesh: TetMesh, dim: int = 1, intorder: int = 2) -> FESpace:
    """Create a linear or quadratic tet space based on mesh nodes."""
    n_nodes = mesh.conn.shape[1]
    if n_nodes == 10:
        basis: Basis3D = make_tet10_basis(intorder if intorder > 1 else 2)
    else:
        basis = make_tet_basis(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space(mesh, basis, element)


def make_tet_space_pytree(
    mesh: TetMesh, dim: int = 1, intorder: int = 2
) -> FESpacePytree:
    """Create a pytree linear or quadratic tet space based on mesh nodes."""
    n_nodes = mesh.conn.shape[1]
    if n_nodes == 10:
        basis: Basis3D = make_tet10_basis_pytree(intorder if intorder > 1 else 2)
    else:
        basis = make_tet_basis_pytree(intorder)
    element = None if dim == 1 else ElementVector(dim)
    return make_space_pytree(cast(BaseMeshPytree, mesh), basis, element)
