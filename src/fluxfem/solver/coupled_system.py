from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .cg import cg_solve_jax
from .legacy_coupled_system import ConstraintSpec, DirichletSpec
from .dirichlet import enforce_dirichlet_fluxsparse_jax
from .sparse import FluxSparseMatrix, block_diag_flux, concat_flux


@dataclass
class CoupledSystem:
    """JAX-native coupled system with sparse-first assembly."""

    K_u: FluxSparseMatrix
    F_u: jnp.ndarray

    @classmethod
    def from_structural(cls, K_u, F_u) -> "CoupledSystem":
        K = cls._as_flux_matrix(K_u)
        F = jnp.asarray(F_u)
        if F.shape != (K.n_dofs,):
            raise ValueError("F_u shape must match K_u size.")
        return cls(K_u=K, F_u=F)

    @classmethod
    def create(cls, K_u, F_u, *, backend: str = "jax"):
        """Create a coupled-system implementation for the requested backend."""
        if backend == "jax":
            return cls.from_structural(K_u, F_u)
        if backend == "numpy":
            from .legacy_coupled_system import LegacyCoupledSystem

            return LegacyCoupledSystem.from_structural(K_u, F_u)
        raise ValueError("backend must be 'numpy' or 'jax'.")

    @staticmethod
    def _as_flux_matrix(K_u) -> FluxSparseMatrix:
        if isinstance(K_u, FluxSparseMatrix):
            return K_u
        if sp.issparse(K_u):
            coo = K_u.tocoo()
            return FluxSparseMatrix(coo.row, coo.col, coo.data, int(K_u.shape[0]))
        K = jnp.asarray(K_u)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("K_u must be square.")
        n = int(K.shape[0])
        if n == 0:
            return FluxSparseMatrix(np.asarray([], dtype=int), np.asarray([], dtype=int), jnp.zeros((0,), dtype=K.dtype), 0)
        rows = jnp.repeat(jnp.arange(n, dtype=jnp.int32), n)
        cols = jnp.tile(jnp.arange(n, dtype=jnp.int32), n)
        data = K.reshape(-1)
        return FluxSparseMatrix(rows, cols, data, n)

    @property
    def dtype(self):
        return self.F_u.dtype if self.F_u.size else self.K_u.data.dtype

    @property
    def n_u(self) -> int:
        return int(self.K_u.n_dofs)

    def to_dense(self) -> jnp.ndarray:
        return self.K_u.to_dense()

    def _empty_flux(self, n_dofs: int) -> FluxSparseMatrix:
        return FluxSparseMatrix(
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            jnp.zeros((0,), dtype=self.dtype),
            n_dofs,
        )

    def _dense_block_to_flux(self, row_dofs, col_dofs, block, *, n_total: int | None = None) -> FluxSparseMatrix:
        row_idx = jnp.asarray(row_dofs, dtype=jnp.int32).reshape(-1)
        col_idx = jnp.asarray(col_dofs, dtype=jnp.int32).reshape(-1)
        block_arr = jnp.asarray(block, dtype=self.dtype)
        if block_arr.shape != (row_idx.shape[0], col_idx.shape[0]):
            raise ValueError("block shape mismatch for provided row/col DOFs.")
        rr = jnp.repeat(row_idx, col_idx.shape[0])
        cc = jnp.tile(col_idx, row_idx.shape[0])
        data = block_arr.reshape(-1)
        return FluxSparseMatrix(rr, cc, data, self.n_u if n_total is None else int(n_total))

    def add_local_stiffness(self, local_dofs, K_local, *, F_local=None) -> None:
        dofs = np.asarray(local_dofs, dtype=int).reshape(-1)
        if dofs.size == 0:
            return
        self.K_u = concat_flux(self.K_u, self._dense_block_to_flux(dofs, dofs, K_local))
        if F_local is not None:
            F_arr = jnp.asarray(F_local, dtype=self.F_u.dtype).reshape(-1)
            if F_arr.shape != (dofs.size,):
                raise ValueError("F_local shape mismatch.")
            self.F_u = self.F_u.at[jnp.asarray(dofs, dtype=jnp.int32)].add(F_arr)

    def add_local_kkt(
        self,
        local_dofs,
        C_local,
        *,
        Kuu_local=None,
        F_contact=None,
    ) -> None:
        dofs = np.asarray(local_dofs, dtype=int).reshape(-1)
        C_arr = jnp.asarray(C_local, dtype=self.dtype)
        if C_arr.ndim != 2 or C_arr.shape[1] != dofs.size:
            raise ValueError("Constraint matrix width must match selected DOF count.")
        n_prev = self.n_u
        n_l = int(C_arr.shape[0])
        n_total = n_prev + n_l
        mats = [FluxSparseMatrix(self.K_u.pattern.rows, self.K_u.pattern.cols, self.K_u.data, n_total)]
        if Kuu_local is not None:
            mats.append(self._dense_block_to_flux(dofs, dofs, Kuu_local, n_total=n_total))
        lambda_dofs = np.arange(n_prev, n_total, dtype=int)
        mats.append(self._dense_block_to_flux(dofs, lambda_dofs, C_arr.T, n_total=n_total))
        mats.append(self._dense_block_to_flux(lambda_dofs, dofs, C_arr, n_total=n_total))
        self.K_u = concat_flux(mats, n_dofs=n_total)
        F_full = jnp.zeros((n_total,), dtype=self.F_u.dtype)
        F_full = F_full.at[:n_prev].set(self.F_u)
        if F_contact is not None:
            F_arr = jnp.asarray(F_contact, dtype=self.F_u.dtype).reshape(-1)
            if F_arr.shape != (n_total,):
                raise ValueError("F_contact shape mismatch.")
            F_full = F_full + F_arr
        self.F_u = F_full

    def append_structural_block(
        self,
        K_block=None,
        F_block=None,
        *,
        n_dofs: int | None = None,
    ) -> slice:
        if K_block is None:
            if n_dofs is None:
                raise ValueError("n_dofs is required when K_block is omitted.")
            n_add = int(n_dofs)
            if n_add <= 0:
                raise ValueError("n_dofs must be positive.")
            K_add = self._empty_flux(n_add)
        else:
            K_add = self._as_flux_matrix(K_block)
            n_add = int(K_add.n_dofs)
            if n_dofs is not None and int(n_dofs) != n_add:
                raise ValueError("n_dofs does not match K_block shape.")

        if F_block is None:
            F_add = jnp.zeros((n_add,), dtype=self.dtype)
        else:
            F_add = jnp.asarray(F_block, dtype=self.F_u.dtype).reshape(-1)
            if F_add.shape != (n_add,):
                raise ValueError("F_block shape must match appended DOF count.")

        start = self.n_u
        stop = start + n_add

        self.K_u = block_diag_flux(self.K_u, K_add)
        self.F_u = jnp.concatenate([self.F_u, F_add], axis=0)
        return slice(start, stop)

    def add_constraint_kkt(
        self,
        C,
        *,
        F_contact=None,
        rho: float = 0.0,
    ) -> None:
        """
        Lift a DOF-level KKT constraint system directly into the sparse JAX system.
        """
        C_arr = jnp.asarray(C, dtype=self.dtype)
        if C_arr.ndim != 2 or C_arr.shape[1] != self.n_u:
            raise ValueError("Constraint matrix width must match the current structural DOF count.")
        Kuu = jnp.asarray(rho, dtype=self.dtype) * (C_arr.T @ C_arr)
        self.add_local_kkt(np.arange(self.n_u, dtype=int), C_arr, Kuu_local=Kuu, F_contact=F_contact)

    def solve(
        self,
        *,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        diagonal_shift: float = 0.0,
        solver: str = "cg",
        tol: float = 1e-8,
        maxiter: int | None = None,
    ):
        """
        Solve the coupled system with the sparse-first JAX CG path.
        """
        if solver != "cg":
            raise ValueError("CoupledSystem only supports solver='cg'. Use to_dense() for dense reference solves.")
        dir_dofs = np.asarray(dirichlet_dofs if dirichlet_dofs is not None else [], dtype=int)
        if dir_dofs.size > 0:
            K_bc, F_bc = enforce_dirichlet_fluxsparse_jax(self.K_u, self.F_u, dir_dofs, dirichlet_vals)
        else:
            K_bc, F_bc = self.K_u, self.F_u
        if float(diagonal_shift) != 0.0:
            diag = jnp.arange(K_bc.n_dofs, dtype=jnp.int32)
            K_bc = concat_flux(
                K_bc,
                FluxSparseMatrix(diag, diag, jnp.full((K_bc.n_dofs,), jnp.asarray(diagonal_shift, dtype=self.dtype)), K_bc.n_dofs),
                n_dofs=K_bc.n_dofs,
            )
        u, _info = cg_solve_jax(K_bc, F_bc, tol=tol, maxiter=maxiter)
        return u

    def compliance(self, load_vector, *, u=None):
        u_vec = self.solve() if u is None else jnp.asarray(u)
        return jnp.dot(jnp.asarray(load_vector, dtype=u_vec.dtype), u_vec)


@dataclass
class _JaxFieldBlock:
    name: str
    offset: int
    n_dofs: int
    value_dim: int
    n_nodes: int
    point: jnp.ndarray | None = None


class CoupledSystemBuilder:
    """Minimal JAX-native builder for structural blocks and remote springs."""

    def __init__(self, system: CoupledSystem):
        self.system = system
        self._blocks: dict[str, _JaxFieldBlock] = {}

    @classmethod
    def from_structural(cls, K_u, F_u) -> "CoupledSystemBuilder":
        return cls(CoupledSystem.from_structural(K_u, F_u))

    @classmethod
    def create(cls, K_u, F_u, *, backend: str = "jax"):
        """Create a coupled-system builder for the requested backend."""
        if backend == "jax":
            return cls.from_structural(K_u, F_u)
        if backend == "numpy":
            from .legacy_coupled_system import LegacyCoupledSystemBuilder

            return LegacyCoupledSystemBuilder.from_structural(K_u, F_u)
        raise ValueError("backend must be 'numpy' or 'jax'.")

    def _next_offset(self) -> int:
        if not self._blocks:
            return 0
        return max(b.offset + b.n_dofs for b in self._blocks.values())

    def _get_block(self, name: str) -> _JaxFieldBlock:
        try:
            return self._blocks[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown field '{name}'.") from exc

    def register_field(
        self,
        name: str,
        *,
        offset: int | None = None,
        n_dofs: int,
        value_dim: int = 1,
        n_nodes: int | None = None,
    ) -> None:
        key = str(name)
        if key in self._blocks:
            raise ValueError(f"Field '{key}' is already registered.")
        off = self._next_offset() if offset is None else int(offset)
        nd = int(n_dofs)
        vd = int(value_dim)
        if vd <= 0:
            raise ValueError("value_dim must be positive.")
        if nd <= 0:
            raise ValueError("n_dofs must be positive.")
        if n_nodes is None:
            if nd % vd != 0:
                raise ValueError("n_dofs must be divisible by value_dim when n_nodes is omitted.")
            nn = nd // vd
        else:
            nn = int(n_nodes)
        self._blocks[key] = _JaxFieldBlock(name=key, offset=off, n_dofs=nd, value_dim=vd, n_nodes=nn)

    def append_field(
        self,
        name: str,
        *,
        n_dofs: int,
        value_dim: int = 1,
        n_nodes: int | None = None,
        K_block=None,
        F_block=None,
    ) -> None:
        key = str(name)
        if key in self._blocks:
            raise ValueError(f"Field '{key}' is already registered.")
        new_slice = self.system.append_structural_block(K_block, F_block, n_dofs=n_dofs)
        self.register_field(
            key,
            offset=int(new_slice.start),
            n_dofs=n_dofs,
            value_dim=value_dim,
            n_nodes=n_nodes,
        )

    def append_remote_point(
        self,
        name: str,
        *,
        point,
        include_rotation: bool = True,
        F_block=None,
    ) -> None:
        dof_count = 6 if include_rotation else 3
        self.append_field(name, n_dofs=dof_count, value_dim=1, n_nodes=dof_count, F_block=F_block)
        block = self._get_block(name)
        block.point = jnp.asarray(point, dtype=self.system.dtype).reshape(-1)
        if block.point.shape != (3,):
            raise ValueError("remote point must be a 3D coordinate.")

    def add_constraint(self, spec: ConstraintSpec) -> None:
        """
        Add a supported constraint through the shared ConstraintSpec descriptor.
        """
        if not isinstance(spec, ConstraintSpec):
            raise TypeError("spec must be a ConstraintSpec instance.")

        kind = str(spec.kind).lower()
        if kind == "contact":
            self.add_contact(
                spec.contact_obj,
                master=spec.master,
                slave=spec.slave,
                family=spec.family,
                enforcement=spec.enforcement,
                law=spec.law,
                formulation=spec.formulation,
                value_dim=spec.value_dim,
                weak_form=spec.weak_form,
                state=spec.state,
                params=spec.params,
                residual=spec.residual,
                scale=spec.scale,
                residual_sign=spec.residual_sign,
                normal_source=spec.normal_source,
                sparse=spec.sparse,
                batch_jac=spec.batch_jac,
            )
            return
        if kind == "matrix":
            self.add_constraint_matrix(
                spec.C,
                master=spec.master,
                slave=spec.slave,
                value_dim=spec.value_dim,
                rho=spec.rho,
                F_contact=spec.F_contact,
            )
            return
        if kind == "matrix_dof":
            self.add_constraint_matrix_dof(
                spec.C,
                master=spec.master,
                slave=spec.slave,
                rho=spec.rho,
                F_contact=spec.F_contact,
            )
            return
        if kind == "rbe2":
            self.add_rbe2_constraint(
                master=spec.master,
                slave=spec.slave,
                ref_point=spec.ref_point,
                slave_coords=spec.slave_coords,
                rho=spec.rho,
                F_contact=spec.F_contact,
            )
            return
        if kind == "rbe3":
            self.add_rbe3_constraint(
                master=spec.master,
                slave=spec.slave,
                ref_point=spec.ref_point,
                slave_coords=spec.slave_coords,
                weights=spec.weights,
                normalize_weights=spec.normalize_weights,
                rho=spec.rho,
                F_contact=spec.F_contact,
            )
            return
        if kind == "embedding":
            self.add_embedding_constraint(
                spec.embedding,
                master=spec.master,
                slave=spec.slave,
                value_dim=spec.value_dim,
                rho=spec.rho,
                F_contact=spec.F_contact,
            )
            return
        raise NotImplementedError(
            f"CoupledSystemBuilder does not support ConstraintSpec(kind='{kind}') yet."
        )

    def resolve_dirichlet(
        self,
        specs: Sequence[DirichletSpec],
        *,
        default_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert field-based Dirichlet specs into `(dirichlet_dofs, dirichlet_vals)`.
        """
        dof_to_value: dict[int, float] = {}

        for spec in specs:
            if not isinstance(spec, DirichletSpec):
                raise TypeError("dirichlet specs must be DirichletSpec instances.")
            dofs = self.resolve_block_dofs(
                str(spec.field),
                nodes=spec.nodes,
                components=spec.components,
                local_dofs=spec.local_dofs,
            )
            val_obj = spec.value if spec.value is not None else default_value
            val_arr = np.asarray(val_obj, dtype=float).reshape(-1)
            if dofs.size == 0:
                continue
            if val_arr.size == 1:
                vals = np.full((dofs.size,), float(val_arr[0]), dtype=float)
            elif val_arr.size == dofs.size:
                vals = val_arr.astype(float, copy=False)
            else:
                raise ValueError("Dirichlet 'value' must be scalar or match the number of selected DOFs.")
            for d, v in zip(dofs, vals):
                dof_to_value[int(d)] = float(v)

        if not dof_to_value:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        items = sorted(dof_to_value.items(), key=lambda kv: kv[0])
        dofs = np.asarray([k for k, _ in items], dtype=int)
        vals = np.asarray([v for _, v in items], dtype=float)
        return dofs, vals

    def resolve_block_dofs(
        self,
        field: str,
        *,
        nodes: int | Sequence[int] | np.ndarray | None = None,
        components: int | Sequence[int] | np.ndarray | None = None,
        local_dofs: int | Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        b = self._get_block(field)
        if local_dofs is not None:
            if nodes is not None or components is not None:
                raise ValueError("Use either local_dofs or nodes/components, not both.")
            local = np.asarray(local_dofs, dtype=int).reshape(-1)
            if np.any(local < 0) or np.any(local >= b.n_dofs):
                raise ValueError("local_dofs out of range.")
            return b.offset + local

        if nodes is None:
            node_ids = np.arange(b.n_nodes, dtype=int)
        else:
            node_ids = np.asarray(nodes if np.ndim(nodes) > 0 else [nodes], dtype=int).reshape(-1)
        if np.any(node_ids < 0) or np.any(node_ids >= b.n_nodes):
            raise ValueError("nodes out of range.")

        if components is None:
            comp_ids = np.arange(b.value_dim, dtype=int)
        else:
            comp_ids = np.asarray(components if np.ndim(components) > 0 else [components], dtype=int).reshape(-1)
        if np.any(comp_ids < 0) or np.any(comp_ids >= b.value_dim):
            raise ValueError("components out of range.")

        local = np.asarray([b.value_dim * n + c for n in node_ids for c in comp_ids], dtype=int)
        return b.offset + local

    def add_field_matrix(self, field: str, K_local, *, F_local=None) -> None:
        b = self._get_block(field)
        K_arr = jnp.asarray(K_local, dtype=self.system.dtype)
        if K_arr.shape != (b.n_dofs, b.n_dofs):
            raise ValueError(f"K_local shape {K_arr.shape} does not match field '{b.name}' size {(b.n_dofs, b.n_dofs)}.")
        local_dofs = np.arange(b.offset, b.offset + b.n_dofs, dtype=int)
        if F_local is not None:
            F_arr = jnp.asarray(F_local, dtype=self.system.F_u.dtype).reshape(-1)
            if F_arr.shape != (b.n_dofs,):
                raise ValueError(f"F_local shape {F_arr.shape} does not match field '{b.name}' size {(b.n_dofs,)}.")
        else:
            F_arr = None
        self.system.add_local_stiffness(local_dofs, K_arr, F_local=F_arr)

    def add_contact_nitsche(
        self,
        ops_or_jacobian,
        *,
        master: str,
        slave: str,
        residual=None,
        scale: float = 1.0,
        residual_sign: float = -1.0,
        value_dim: int | None = None,
    ) -> None:
        m = self._get_block(master)
        s = self._get_block(slave)

        if value_dim is None:
            if m.value_dim != s.value_dim:
                raise ValueError("master/slave value_dim mismatch. Pass value_dim explicitly.")
            vd = m.value_dim
        else:
            vd = int(value_dim)
        if vd <= 0:
            raise ValueError("value_dim must be positive.")
        if m.n_dofs != vd * m.n_nodes or s.n_dofs != vd * s.n_nodes:
            raise ValueError("contact penalty currently requires full nodal field blocks.")

        jac = getattr(ops_or_jacobian, "jacobian", ops_or_jacobian)
        if residual is None and hasattr(ops_or_jacobian, "residual"):
            residual = ops_or_jacobian.residual

        n_master = vd * m.n_nodes
        n_slave = vd * s.n_nodes
        n_cu = n_master + n_slave
        J_if = jnp.asarray(jac, dtype=self.system.dtype)
        if J_if.shape != (n_cu, n_cu):
            raise ValueError("J_contact shape mismatch for provided node counts and value_dim.")
        local_dofs = np.concatenate(
            [
                np.arange(m.offset, m.offset + n_master, dtype=int),
                np.arange(s.offset, s.offset + n_slave, dtype=int),
            ]
        )
        s_scale = jnp.asarray(scale, dtype=self.system.dtype)
        self.system.add_local_stiffness(local_dofs, s_scale * J_if)

        if residual is not None:
            r_if = jnp.asarray(residual, dtype=self.system.F_u.dtype).reshape(-1)
            if r_if.shape != (n_cu,):
                raise ValueError("residual shape mismatch for provided node counts and value_dim.")
            r_scale = jnp.asarray(scale * residual_sign, dtype=self.system.F_u.dtype)
            self.system.F_u = self.system.F_u.at[jnp.asarray(local_dofs, dtype=jnp.int32)].add(r_scale * r_if)

    def add_contact(
        self,
        contact_obj,
        *,
        master: str,
        slave: str,
        family: str | None = None,
        enforcement: str | None = None,
        law: str | None = None,
        formulation: str | None = None,
        value_dim: int | None = None,
        weak_form=None,
        state=None,
        params=None,
        residual=None,
        scale: float = 1.0,
        residual_sign: float = -1.0,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
        F_contact=None,
        rho: float | None = None,
        multiplier=None,
        facet_conn_master=None,
    ) -> None:
        """
        JAX-native contact entry point for coupled assembly.

        Supported today
        - explicit penalty/nitsche contributions with ``jacobian``/``residual``
        - raw contact objects resolved through
          ``assemble_contact_penalty_operators(..., backend="jax")``

        Also supported
        - explicit mortar/multiplier contact operators assembled on the JAX path
        - raw contact objects resolved through
          ``assemble_contact_constraint_operators(..., backend="jax")``

        AD behavior
        - explicit penalty/nitsche contributions with ``jacobian``/``residual``
        - raw contact objects assembled through the JAX penalty path can
          participate in autodiff as long as the underlying contact assembly
          stays on the JAX path
        - explicit multiplier operators with JAX ``B``/``Kuu`` can also
          participate in autodiff on the JAX path

        Practical scope
        - use ``add_contact_nitsche(...)`` if you already have a JAX Jacobian
        - use ``add_contact(...)`` when you want the builder to assemble a raw
          penalty contact object for you
        """
        family_arg = None if family is None else str(family).lower()
        family_mode = None
        family_enforcement = None
        if family_arg is not None:
            if family_arg in {"constraint", "mortar"}:
                family_mode = "constraint"
                family_enforcement = "mortar"
            elif family_arg in {"penalty", "nitsche"}:
                family_mode = "penalty"
                family_enforcement = "nitsche"
            else:
                raise ValueError("family must be 'constraint' or 'penalty' (aliases: 'mortar', 'nitsche').")
        if (
            hasattr(contact_obj, "assemble_contact_constraint_operators")
            and not hasattr(contact_obj, "jacobian")
            and not hasattr(contact_obj, "coupling_aa")
        ):
            from ..mesh.contact import (
                assemble_contact_constraint_operators,
                assemble_contact_penalty_operators,
            )

            f_arg_guess = None if formulation is None else str(formulation).lower()
            has_penalty_inputs = (weak_form is not None) or (state is not None) or (params is not None)
            if family_mode is not None:
                resolved_family = family_mode
            elif f_arg_guess in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
                resolved_family = "constraint"
            elif f_arg_guess in {"penalty", "penalty_consistent", "nitsche"}:
                resolved_family = "penalty"
            elif has_penalty_inputs:
                resolved_family = "penalty"
            else:
                resolved_family = "constraint"

            if resolved_family == "constraint":
                if multiplier is None:
                    raise ValueError("constraint-family contact requires multiplier.")
                contact_obj = assemble_contact_constraint_operators(
                    contact_obj,
                    law=law,
                    formulation=formulation,
                    rho=0.0 if rho is None else float(rho),
                    multiplier=multiplier,
                    backend="jax",
                    weak_form=weak_form,
                    state=state,
                    params=params,
                    normal_source=normal_source,
                    sparse=sparse,
                    batch_jac=batch_jac,
                )
            else:
                contact_obj = assemble_contact_penalty_operators(
                    contact_obj,
                    law=law,
                    formulation=formulation,
                    backend="jax",
                    weak_form=weak_form,
                    state=state,
                    params=params,
                    normal_source=normal_source,
                    sparse=sparse,
                    batch_jac=batch_jac,
                )

        resolved = enforcement
        if resolved is None and family_enforcement is not None:
            resolved = family_enforcement
        if resolved is None and hasattr(contact_obj, "enforcement"):
            resolved = getattr(contact_obj, "enforcement")
        if resolved is None and formulation is not None:
            f_arg = str(formulation).lower()
            if f_arg in {"penalty", "penalty_consistent", "nitsche"}:
                resolved = "nitsche"
            elif f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
                resolved = "mortar"
            else:
                resolved = formulation
        if resolved is None:
            resolved = "nitsche"
        resolved = str(resolved).lower()
        if resolved in {"penalty", "nitsche"}:
            self.add_contact_nitsche(
                contact_obj,
                master=master,
                slave=slave,
                residual=residual,
                scale=scale,
                residual_sign=residual_sign,
                value_dim=value_dim,
            )
            return
        if resolved in {"constraint", "mortar"}:
            self.add_contact_mortar(
                contact_obj,
                master=master,
                slave=slave,
                value_dim=value_dim,
                F_contact=F_contact,
                rho=rho,
                multiplier=multiplier,
                facet_conn_master=facet_conn_master,
            )
            return
        raise ValueError("enforcement must resolve to 'nitsche' or 'mortar'.")

    def add_contact_mortar(
        self,
        ops_or_kkt,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        F_contact=None,
        rho: float | None = None,
        multiplier=None,
        facet_conn_master=None,
    ) -> None:
        """
        Add mortar/multiplier contact on the JAX path.

        Supported today
        - explicit contact operators with ``B`` and ``Kuu``
        - preassembled dense KKT matrices

        Not supported yet
        - sparse preassembled KKT inputs
        - implicit assembly from ``coupling_aa``/``coupling_ab`` only
        """
        m = self._get_block(master)
        s = self._get_block(slave)
        if value_dim is None:
            if m.value_dim != s.value_dim:
                raise ValueError("master/slave value_dim mismatch. Pass value_dim explicitly.")
            vd = m.value_dim
        else:
            vd = int(value_dim)

        if hasattr(ops_or_kkt, "B") and getattr(ops_or_kkt, "B", None) is not None:
            B_local = jnp.asarray(getattr(ops_or_kkt, "B"), dtype=self.system.dtype)
            Kuu_local_obj = getattr(ops_or_kkt, "Kuu", None)
            if Kuu_local_obj is None:
                rho_eff = getattr(ops_or_kkt, "rho", None) if rho is None else rho
                rho_eff = 0.0 if rho_eff is None else rho_eff
                Kuu_local = jnp.asarray(rho_eff, dtype=self.system.dtype) * (B_local.T @ B_local)
            else:
                Kuu_local = jnp.asarray(Kuu_local_obj, dtype=self.system.dtype)

            n_cu = int(m.n_dofs + s.n_dofs)
            if B_local.ndim != 2 or B_local.shape[1] != n_cu:
                raise ValueError("mortar operator B shape mismatch for provided master/slave blocks.")
            if Kuu_local.shape != (n_cu, n_cu):
                raise ValueError("mortar operator Kuu shape mismatch for provided master/slave blocks.")

            local_dofs = np.concatenate(
                [
                    np.arange(m.offset, m.offset + m.n_dofs, dtype=int),
                    np.arange(s.offset, s.offset + s.n_dofs, dtype=int),
                ]
            )
            self.system.add_local_kkt(local_dofs, B_local, Kuu_local=Kuu_local, F_contact=F_contact)
            return

        if not hasattr(ops_or_kkt, "shape") and not isinstance(ops_or_kkt, (np.ndarray, jnp.ndarray)):
            raise NotImplementedError(
                "CoupledSystemBuilder.add_contact_mortar currently accepts explicit mortar operators "
                "with B/Kuu or preassembled dense KKT matrices."
            )

        K_contact = jnp.asarray(ops_or_kkt, dtype=self.system.dtype)
        n_cu = int(m.n_dofs + s.n_dofs)
        if K_contact.ndim != 2 or K_contact.shape[0] != K_contact.shape[1]:
            raise ValueError("preassembled contact KKT must be square.")
        if K_contact.shape[0] < n_cu:
            raise ValueError("preassembled contact KKT is smaller than the contact DOF block.")
        n_l = int(K_contact.shape[0] - n_cu)
        if n_l < 0:
            raise ValueError("invalid preassembled contact KKT shape.")

        Kuu_local = K_contact[:n_cu, :n_cu]
        Kul_local = K_contact[:n_cu, n_cu:]
        Klu_local = K_contact[n_cu:, :n_cu]
        Kll = K_contact[n_cu:, n_cu:]
        local_dofs = np.concatenate(
            [
                np.arange(m.offset, m.offset + m.n_dofs, dtype=int),
                np.arange(s.offset, s.offset + s.n_dofs, dtype=int),
            ]
        )
        n_prev = self.system.n_u
        n_total = n_prev + n_l
        mats = [FluxSparseMatrix(self.system.K_u.pattern.rows, self.system.K_u.pattern.cols, self.system.K_u.data, n_total)]
        mats.append(self.system._dense_block_to_flux(local_dofs, local_dofs, Kuu_local, n_total=n_total))
        lambda_dofs = np.arange(n_prev, n_total, dtype=int)
        mats.append(self.system._dense_block_to_flux(local_dofs, lambda_dofs, Kul_local, n_total=n_total))
        mats.append(self.system._dense_block_to_flux(lambda_dofs, local_dofs, Klu_local, n_total=n_total))
        mats.append(self.system._dense_block_to_flux(lambda_dofs, lambda_dofs, Kll, n_total=n_total))
        self.system.K_u = concat_flux(mats, n_dofs=n_total)

        F_full = jnp.zeros((n_total,), dtype=self.system.F_u.dtype)
        F_full = F_full.at[: n_prev].set(self.system.F_u)
        if F_contact is not None:
            F_arr = jnp.asarray(F_contact, dtype=self.system.F_u.dtype).reshape(-1)
            if F_arr.shape != (n_total,):
                raise ValueError("F_contact shape mismatch.")
            F_full = F_full + F_arr
        self.system.F_u = F_full

    def add_constraint_matrix_dof(
        self,
        C,
        *,
        master: str,
        slave: str,
        rho: float = 0.0,
        F_contact=None,
    ) -> None:
        m = self._get_block(master)
        s = self._get_block(slave)
        C_arr = jnp.asarray(C, dtype=self.system.dtype)
        n_cu = int(m.n_dofs + s.n_dofs)
        if C_arr.ndim != 2 or C_arr.shape[1] != n_cu:
            raise ValueError("C shape mismatch for provided master/slave DOF counts.")
        local_dofs = np.concatenate(
            [
                np.arange(m.offset, m.offset + m.n_dofs, dtype=int),
                np.arange(s.offset, s.offset + s.n_dofs, dtype=int),
            ]
        )
        Kuu_local = jnp.asarray(rho, dtype=self.system.dtype) * (C_arr.T @ C_arr)
        self.system.add_local_kkt(local_dofs, C_arr, Kuu_local=Kuu_local, F_contact=F_contact)

    def add_constraint_matrix(
        self,
        C,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        rho: float = 0.0,
        F_contact=None,
    ) -> None:
        m = self._get_block(master)
        s = self._get_block(slave)
        if value_dim is None:
            if m.value_dim != s.value_dim:
                raise ValueError("master/slave value_dim mismatch. Pass value_dim explicitly.")
            vd = m.value_dim
        else:
            vd = int(value_dim)
        if vd <= 0:
            raise ValueError("value_dim must be positive.")

        C_arr = jnp.asarray(C, dtype=self.system.dtype)
        n_cu = vd * int(m.n_nodes + s.n_nodes)
        if C_arr.ndim != 2 or C_arr.shape[1] != n_cu:
            raise ValueError("C shape mismatch for provided master/slave node counts and value_dim.")
        self.add_constraint_matrix_dof(C_arr, master=master, slave=slave, rho=rho, F_contact=F_contact)

    def add_rbe2_constraint(
        self,
        *,
        master: str,
        slave: str,
        ref_point,
        slave_coords,
        rho: float = 0.0,
        F_contact=None,
    ) -> None:
        """
        Build and add a 3D RBE2-style rigid constraint matrix in JAX.

        Expected field layout:
        - ``master``: 6 DOFs ordered as ``[u_ref(3), omega_ref(3)]``
        - ``slave``: 3 DOFs per node ordered as nodal translations
        """
        m = self._get_block(master)
        s = self._get_block(slave)
        x_ref = jnp.asarray(ref_point, dtype=self.system.dtype).reshape(-1)
        x_s = jnp.asarray(slave_coords, dtype=self.system.dtype)
        if x_ref.shape != (3,):
            raise ValueError("ref_point must be 3D.")
        if x_s.ndim != 2 or x_s.shape[1] != 3:
            raise ValueError("slave_coords must have shape (n_slave, 3).")
        if m.n_dofs != 6:
            raise ValueError("RBE2 master field must have exactly 6 DOFs.")
        if s.n_dofs != 3 * int(x_s.shape[0]):
            raise ValueError("RBE2 slave field size must match 3 * n_slave_nodes.")

        n_s = int(x_s.shape[0])
        n_rows = 3 * n_s
        n_cols = 6 + 3 * n_s
        C = jnp.zeros((n_rows, n_cols), dtype=self.system.dtype)
        for i in range(n_s):
            rx, ry, rz = x_s[i] - x_ref
            r0 = 3 * i
            c_slave = 6 + 3 * i
            C = C.at[r0 + 0, 0].set(-1.0)
            C = C.at[r0 + 1, 1].set(-1.0)
            C = C.at[r0 + 2, 2].set(-1.0)
            C = C.at[r0 + 0, 4].set(-rz)
            C = C.at[r0 + 0, 5].set(+ry)
            C = C.at[r0 + 1, 3].set(+rz)
            C = C.at[r0 + 1, 5].set(-rx)
            C = C.at[r0 + 2, 3].set(-ry)
            C = C.at[r0 + 2, 4].set(+rx)
            C = C.at[r0 + 0, c_slave + 0].set(+1.0)
            C = C.at[r0 + 1, c_slave + 1].set(+1.0)
            C = C.at[r0 + 2, c_slave + 2].set(+1.0)
        self.add_constraint_matrix_dof(C, master=master, slave=slave, rho=rho, F_contact=F_contact)

    def add_embedding_constraint(
        self,
        embedding,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        rho: float = 0.0,
        F_contact=None,
    ) -> None:
        """
        Build and add embedding constraints from ``EmbeddingMap`` in JAX.
        """
        from ..mesh.contact import assemble_embedding_constraint_matrix

        m = self._get_block(master)
        s = self._get_block(slave)
        if value_dim is None:
            if m.value_dim != s.value_dim:
                raise ValueError("master/slave value_dim mismatch. Pass value_dim explicitly.")
            vd = m.value_dim
        else:
            vd = int(value_dim)
        C = assemble_embedding_constraint_matrix(
            embedding,
            n_master_nodes=m.n_nodes,
            n_slave_nodes=s.n_nodes,
            value_dim=vd,
            backend="jax",
        )
        self.add_constraint_matrix(
            C,
            master=master,
            slave=slave,
            value_dim=vd,
            rho=rho,
            F_contact=F_contact,
        )

    def add_rbe3_constraint(
        self,
        *,
        master: str,
        slave: str,
        ref_point,
        slave_coords,
        weights=None,
        normalize_weights: bool = True,
        rho: float = 0.0,
        F_contact=None,
    ) -> None:
        """
        Build and add a weighted 3D RBE3-style interpolation constraint in JAX.

        Expected field layout:
        - ``master``: 6 DOFs ordered as ``[u_ref(3), omega_ref(3)]``
        - ``slave``: 3 DOFs per node ordered as nodal translations
        """
        m = self._get_block(master)
        s = self._get_block(slave)
        x_ref = jnp.asarray(ref_point, dtype=self.system.dtype).reshape(-1)
        x_s = jnp.asarray(slave_coords, dtype=self.system.dtype)
        if x_ref.shape != (3,):
            raise ValueError("ref_point must be 3D.")
        if x_s.ndim != 2 or x_s.shape[1] != 3:
            raise ValueError("slave_coords must have shape (n_slave, 3).")
        n_s = int(x_s.shape[0])
        if n_s == 0:
            raise ValueError("slave_coords must contain at least one node.")
        if m.n_dofs != 6:
            raise ValueError("RBE3 master field must have exactly 6 DOFs.")
        if s.n_dofs != 3 * n_s:
            raise ValueError("RBE3 slave field size must match 3 * n_slave_nodes.")

        if weights is None:
            w = jnp.ones((n_s,), dtype=self.system.dtype)
        else:
            w = jnp.asarray(weights, dtype=self.system.dtype).reshape(-1)
            if w.shape != (n_s,):
                raise ValueError("weights must have shape (n_slave,).")
        if normalize_weights:
            w_sum = jnp.sum(w)
            if float(jnp.abs(w_sum)) <= 1e-15:
                raise ValueError("weights sum must be non-zero when normalize_weights=True.")
            w = w / w_sum

        def _bmat(point):
            rx, ry, rz = point - x_ref
            return jnp.array(
                [
                    [1.0, 0.0, 0.0, 0.0, rz, -ry],
                    [0.0, 1.0, 0.0, -rz, 0.0, rx],
                    [0.0, 0.0, 1.0, ry, -rx, 0.0],
                ],
                dtype=self.system.dtype,
            )

        M = jnp.zeros((6, 6), dtype=self.system.dtype)
        C = jnp.zeros((6, 6 + 3 * n_s), dtype=self.system.dtype)
        for i in range(n_s):
            Bi = _bmat(x_s[i])
            wi = w[i]
            M = M + wi * (Bi.T @ Bi)
            c0 = 6 + 3 * i
            C = C.at[:, c0 : c0 + 3].set(-wi * Bi.T)
        C = C.at[:, :6].set(M)
        self.add_constraint_matrix_dof(C, master=master, slave=slave, rho=rho, F_contact=F_contact)

    def _coerce_spring_matrix_and_reference(self, stiffness, reference_value, *, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        stiff = jnp.asarray(stiffness, dtype=self.system.dtype)
        if stiff.ndim == 0:
            K_sel = jnp.eye(n, dtype=self.system.dtype) * stiff
        elif stiff.ndim == 1:
            if stiff.shape != (n,):
                raise ValueError("stiffness vector must match selected DOF count.")
            K_sel = jnp.diag(stiff)
        elif stiff.ndim == 2:
            if stiff.shape != (n, n):
                raise ValueError("stiffness matrix must match selected DOF count.")
            K_sel = stiff
        else:
            raise ValueError("stiffness must be scalar, vector, or square matrix.")

        ref_arr = jnp.asarray(reference_value, dtype=self.system.F_u.dtype).reshape(-1)
        if ref_arr.size == 1:
            u_ref = jnp.full((n,), ref_arr[0], dtype=self.system.F_u.dtype)
        elif ref_arr.shape == (n,):
            u_ref = ref_arr
        else:
            raise ValueError("reference_value must be scalar or match selected DOF count.")
        return K_sel, u_ref

    def add_dof_spring(
        self,
        field: str,
        *,
        stiffness,
        reference_value=0.0,
        nodes: int | Sequence[int] | np.ndarray | None = None,
        components: int | Sequence[int] | np.ndarray | None = None,
        local_dofs: int | Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        dofs = self.resolve_block_dofs(
            field,
            nodes=nodes,
            components=components,
            local_dofs=local_dofs,
        )
        if dofs.size == 0:
            return dofs
        K_sel, u_ref = self._coerce_spring_matrix_and_reference(stiffness, reference_value, n=int(dofs.size))
        dofs_j = jnp.asarray(dofs, dtype=jnp.int32)
        self.system.add_local_stiffness(np.asarray(dofs, dtype=int), K_sel, F_local=K_sel @ u_ref)
        return np.asarray(dofs, dtype=int)

    def add_remote_spring(
        self,
        field: str,
        *,
        translational_stiffness=None,
        rotational_stiffness=None,
        translational_target=0.0,
        rotational_target=0.0,
    ) -> None:
        b = self._get_block(field)
        if b.n_dofs not in {3, 6}:
            raise ValueError("remote spring helper expects a 3-DOF or 6-DOF field.")
        if translational_stiffness is not None:
            self.add_dof_spring(
                field,
                local_dofs=np.arange(min(3, b.n_dofs)),
                stiffness=translational_stiffness,
                reference_value=translational_target,
            )
        if rotational_stiffness is not None:
            if b.n_dofs < 6:
                raise ValueError("rotational springs require a 6-DOF remote field.")
            self.add_dof_spring(
                field,
                local_dofs=np.arange(3, 6),
                stiffness=rotational_stiffness,
                reference_value=rotational_target,
            )

    def build(self) -> CoupledSystem:
        return self.system

    def solve(
        self,
        *,
        dirichlet_specs: Sequence[DirichletSpec] | None = None,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        diagonal_shift: float = 0.0,
        solver: str = "cg",
        tol: float = 1e-8,
        maxiter: int | None = None,
    ):
        """
        Build and solve with optional `DirichletSpec` constraints.

        The JAX coupled path is sparse-first and only exposes `solver="cg"`.
        """
        if dirichlet_specs is not None and dirichlet_dofs is not None:
            raise ValueError("Use either dirichlet_specs or dirichlet_dofs, not both.")
        if dirichlet_specs is not None:
            dirichlet_dofs, dirichlet_vals = self.resolve_dirichlet(dirichlet_specs)
        return self.system.solve(
            dirichlet_dofs=dirichlet_dofs,
            dirichlet_vals=dirichlet_vals,
            diagonal_shift=diagonal_shift,
            solver=solver,
            tol=tol,
            maxiter=maxiter,
        )


JAXCoupledSystem = CoupledSystem
JAXCoupledSystemBuilder = CoupledSystemBuilder

__all__ = [
    "CoupledSystem",
    "CoupledSystemBuilder",
    "JAXCoupledSystem",
    "JAXCoupledSystemBuilder",
]
