from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .dirichlet import enforce_dirichlet_fluxsparse, enforce_dirichlet_sparse
from .sparse import FluxSparseMatrix


@dataclass
class CoupledSystem:
    """
    Sparse coupled system for structural/contact assembly.

    Notes:
    - Contact lifting currently assumes scalar displacement per node
      (`value_dim=1`) for KKT blocks assembled by `assemble_contact_kkt`.
    - Lambda DOFs are appended after structural DOFs.
    """

    K_u: sp.csr_matrix
    F_u: np.ndarray
    K_contact_lifted: sp.csr_matrix | None = None
    F_contact_lifted: np.ndarray | None = None

    @classmethod
    def from_structural(cls, K_u, F_u):
        if isinstance(K_u, FluxSparseMatrix):
            K = K_u.to_csr()
        elif sp.issparse(K_u):
            K = K_u.tocsr()
        else:
            K = sp.csr_matrix(np.asarray(K_u, dtype=float))
        F = np.asarray(F_u, dtype=float)
        if K.shape[0] != K.shape[1]:
            raise ValueError("K_u must be square.")
        if F.shape != (K.shape[0],):
            raise ValueError("F_u shape must match K_u size.")
        return cls(K_u=K, F_u=F)

    @property
    def n_u(self) -> int:
        return int(self.K_u.shape[0])

    def _contact_projection(
        self,
        *,
        n_master_nodes: int,
        n_slave_nodes: int,
        master_offset: int,
        slave_offset: int,
        value_dim: int,
    ) -> sp.csr_matrix:
        n_master_nodes = int(n_master_nodes)
        n_slave_nodes = int(n_slave_nodes)
        master_offset = int(master_offset)
        slave_offset = int(slave_offset)
        value_dim = int(value_dim)
        if value_dim <= 0:
            raise ValueError("value_dim must be positive.")
        n_cu = value_dim * (n_master_nodes + n_slave_nodes)
        P = sp.lil_matrix((n_cu, self.n_u), dtype=float)
        row = 0
        for i in range(n_master_nodes):
            for d in range(value_dim):
                P[row, master_offset + value_dim * i + d] = 1.0
                row += 1
        for i in range(n_slave_nodes):
            for d in range(value_dim):
                P[row, slave_offset + value_dim * i + d] = 1.0
                row += 1
        return P.tocsr()

    def add_contact_kkt(
        self,
        K_contact,
        *,
        n_master_nodes: int,
        n_slave_nodes: int,
        master_offset: int,
        slave_offset: int,
        F_contact: np.ndarray | None = None,
        value_dim: int = 1,
    ) -> None:
        if isinstance(K_contact, FluxSparseMatrix):
            Kc = K_contact.to_csr()
        elif sp.issparse(K_contact):
            Kc = K_contact.tocsr()
        else:
            Kc = sp.csr_matrix(np.asarray(K_contact, dtype=float))

        value_dim = int(value_dim)
        n_cu = value_dim * int(n_master_nodes + n_slave_nodes)
        if Kc.shape[0] != Kc.shape[1] or Kc.shape[0] < n_cu:
            raise ValueError("K_contact shape is invalid.")
        n_l = int(Kc.shape[0] - n_cu)

        Kuu_c = Kc[:n_cu, :n_cu]
        Kul_c = Kc[:n_cu, n_cu:]
        Klu_c = Kc[n_cu:, :n_cu]
        Kll_c = Kc[n_cu:, n_cu:]

        # P maps contact displacement dofs -> structural displacement dofs
        P = self._contact_projection(
            n_master_nodes=n_master_nodes,
            n_slave_nodes=n_slave_nodes,
            master_offset=master_offset,
            slave_offset=slave_offset,
            value_dim=value_dim,
        )

        Kuu_lift = P.T @ Kuu_c @ P
        Kul_lift = P.T @ Kul_c
        Klu_lift = Klu_c @ P
        K_add = sp.bmat(
            [
                [Kuu_lift, Kul_lift],
                [Klu_lift, Kll_c],
            ],
            format="csr",
        )

        if self.K_contact_lifted is None:
            self.K_contact_lifted = K_add
            F_add = np.zeros((self.n_u + n_l,), dtype=float)
            if F_contact is not None:
                Fc = np.asarray(F_contact, dtype=float)
                if Fc.shape != F_add.shape:
                    raise ValueError("F_contact shape mismatch.")
                F_add += Fc
            self.F_contact_lifted = F_add
            return

        if self.K_contact_lifted.shape != K_add.shape:
            raise ValueError("Contact block size mismatch while accumulating contact contributions.")
        self.K_contact_lifted = self.K_contact_lifted + K_add
        if F_contact is not None:
            Fc = np.asarray(F_contact, dtype=float)
            if Fc.shape != self.F_contact_lifted.shape:
                raise ValueError("F_contact shape mismatch.")
            assert self.F_contact_lifted is not None
            self.F_contact_lifted += Fc

    def add_contact_nitsche(
        self,
        J_contact,
        *,
        n_master_nodes: int,
        n_slave_nodes: int,
        master_offset: int,
        slave_offset: int,
        value_dim: int = 1,
        residual: np.ndarray | None = None,
        scale: float = 1.0,
        residual_sign: float = -1.0,
    ) -> None:
        """
        Lift a two-body Nitsche interface Jacobian/residual into structural DOFs.

        Parameters
        ----------
        J_contact:
            Interface Jacobian ordered as [master dofs, slave dofs].
        residual:
            Optional interface residual with the same ordering.
            `residual_sign=-1` maps Newton form (K du = -R).
        """
        if isinstance(J_contact, FluxSparseMatrix):
            Jc = J_contact.to_csr()
        elif sp.issparse(J_contact):
            Jc = J_contact.tocsr()
        else:
            Jc = sp.csr_matrix(np.asarray(J_contact, dtype=float))

        value_dim = int(value_dim)
        n_cu = value_dim * int(n_master_nodes + n_slave_nodes)
        if Jc.shape != (n_cu, n_cu):
            raise ValueError("J_contact shape mismatch for provided node counts and value_dim.")

        P = self._contact_projection(
            n_master_nodes=n_master_nodes,
            n_slave_nodes=n_slave_nodes,
            master_offset=master_offset,
            slave_offset=slave_offset,
            value_dim=value_dim,
        )

        s = float(scale)
        self.K_u = self.K_u + s * (P.T @ Jc @ P)

        if residual is not None:
            r = np.asarray(residual, dtype=float).reshape(-1)
            if r.shape != (n_cu,):
                raise ValueError("residual shape mismatch for provided node counts and value_dim.")
            self.F_u = self.F_u + (s * float(residual_sign)) * np.asarray(P.T @ r, dtype=float)

    def assemble(self, *, format: str = "fluxsparse"):
        if format not in {"fluxsparse", "csr", "dense"}:
            raise ValueError("format must be 'fluxsparse', 'csr', or 'dense'.")

        if self.K_contact_lifted is None:
            K = self.K_u.copy()
            F = self.F_u.copy()
        else:
            K = self.K_contact_lifted.tolil()
            K[: self.n_u, : self.n_u] = K[: self.n_u, : self.n_u] + self.K_u
            K = K.tocsr()
            F = self.F_u.copy()
            if self.F_contact_lifted is not None:
                F_full = self.F_contact_lifted.copy()
                F_full[: self.n_u] += F
                F = F_full
            else:
                F_full = np.zeros((K.shape[0],), dtype=float)
                F_full[: self.n_u] = F
                F = F_full

        if format == "dense":
            return K.toarray(), F
        if format == "csr":
            return K, F

        coo = K.tocoo()
        return FluxSparseMatrix(coo.row, coo.col, coo.data, K.shape[0]), F

    def solve(
        self,
        *,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        format: str = "csr",
        diagonal_shift: float = 0.0,
    ):
        if format not in {"csr", "fluxsparse"}:
            raise ValueError("format must be 'csr' or 'fluxsparse'.")

        if format == "fluxsparse":
            K_flux, F = self.assemble(format="fluxsparse")
            dir_dofs = np.asarray(dirichlet_dofs if dirichlet_dofs is not None else [], dtype=int)
            K_bc, F_bc = enforce_dirichlet_fluxsparse(K_flux, F, dir_dofs, dirichlet_vals)
            if float(diagonal_shift) != 0.0:
                K_bc = K_bc + float(diagonal_shift) * sp.eye(K_bc.shape[0], format="csr")
            return spla.spsolve(K_bc, F_bc)

        K_csr, F = self.assemble(format="csr")
        if dirichlet_dofs is not None and np.asarray(dirichlet_dofs).size > 0:
            coo = K_csr.tocoo()
            K_flux = FluxSparseMatrix(coo.row, coo.col, coo.data, K_csr.shape[0])
            K_csr, F = enforce_dirichlet_sparse(
                K_flux,
                F,
                np.asarray(dirichlet_dofs, dtype=int),
                dirichlet_vals,
            )
        if float(diagonal_shift) != 0.0:
            K_csr = K_csr + float(diagonal_shift) * sp.eye(K_csr.shape[0], format="csr")
        return spla.spsolve(K_csr, F)


@dataclass
class _FieldBlock:
    name: str
    offset: int
    n_dofs: int
    value_dim: int
    n_nodes: int


@dataclass(frozen=True)
class DirichletSpec:
    """
    Field-aware Dirichlet selector for coupled systems.

    Exactly one selector style must be used:
    - `nodes` (+ optional `components`)
    - `local_dofs`
    """

    field: str
    nodes: int | Sequence[int] | np.ndarray | None = None
    components: int | Sequence[int] | np.ndarray | None = None
    local_dofs: int | Sequence[int] | np.ndarray | None = None
    value: float | Sequence[float] | np.ndarray = 0.0

    def __post_init__(self) -> None:
        has_nodes = self.nodes is not None
        has_local = self.local_dofs is not None
        if has_nodes == has_local:
            raise ValueError("DirichletSpec requires exactly one of nodes or local_dofs.")


@dataclass(frozen=True)
class ConstraintSpec:
    """
    Unified constraint descriptor for CoupledSystemBuilder.

    kind:
    - "contact": routes to add_contact(...)
    - "matrix": routes to add_constraint_matrix(...)
    - "matrix_dof": routes to add_constraint_matrix_dof(...)
    - "embedding": routes to add_embedding_constraint(...)
    """

    kind: str
    master: str
    slave: str
    value_dim: int | None = None
    rho: float = 0.0
    F_contact: np.ndarray | None = None
    backend: str = "numpy"

    contact_obj: Any | None = None
    C: Any | None = None
    embedding: Any | None = None

    # contact routing/options
    family: str | None = None
    enforcement: str | None = None
    law: str | None = None
    formulation: str | None = None
    weak_form: Any | None = None
    state: Any | None = None
    params: Any | None = None
    residual: np.ndarray | None = None
    scale: float = 1.0
    residual_sign: float = -1.0
    normal_source: str = "master"
    sparse: bool = False
    batch_jac: bool | None = None
    multiplier_space: str | None = None
    facet_conn_master: np.ndarray | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).lower()
        if kind not in {"contact", "matrix", "matrix_dof", "embedding"}:
            raise ValueError("ConstraintSpec.kind must be one of: contact, matrix, matrix_dof, embedding.")
        if kind == "contact" and self.contact_obj is None:
            raise ValueError("ConstraintSpec(kind='contact') requires contact_obj.")
        if kind in {"matrix", "matrix_dof"} and self.C is None:
            raise ValueError("ConstraintSpec(kind='matrix*') requires C.")
        if kind == "embedding" and self.embedding is None:
            raise ValueError("ConstraintSpec(kind='embedding') requires embedding.")


class CoupledSystemBuilder:
    """
    Helper to reduce manual offset/node bookkeeping for coupled contact assembly.
    """

    def __init__(self, system: CoupledSystem):
        self.system = system
        self._blocks: dict[str, _FieldBlock] = {}

    @classmethod
    def from_structural(cls, K_u, F_u) -> "CoupledSystemBuilder":
        return cls(CoupledSystem.from_structural(K_u, F_u))

    def _next_offset(self) -> int:
        if not self._blocks:
            return 0
        return max(b.offset + b.n_dofs for b in self._blocks.values())

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
        self._blocks[key] = _FieldBlock(name=key, offset=off, n_dofs=nd, value_dim=vd, n_nodes=nn)

    def register_space(
        self,
        name: str,
        space: Any,
        *,
        offset: int | None = None,
        value_dim: int = 1,
        n_nodes: int | None = None,
    ) -> None:
        nd = int(getattr(space, "n_dofs"))
        off = self._next_offset() if offset is None else int(offset)
        self.register_field(
            name,
            offset=off,
            n_dofs=nd,
            value_dim=value_dim,
            n_nodes=n_nodes,
        )

    def register_blocks(self, blocks: Sequence[Any]) -> None:
        """
        Register multiple blocks with auto-offset.

        Accepted entries:
        - (name, space)
        - (name, space, {"value_dim": ..., "n_nodes": ..., "offset": ...})
        - {"name": ..., "space": ...} or {"name": ..., "n_dofs": ...}
        """
        for item in blocks:
            if isinstance(item, dict):
                name = item["name"]
                if "space" in item:
                    self.register_space(
                        name,
                        item["space"],
                        offset=item.get("offset"),
                        value_dim=item.get("value_dim", 1),
                        n_nodes=item.get("n_nodes"),
                    )
                else:
                    self.register_field(
                        name,
                        offset=item.get("offset"),
                        n_dofs=item["n_dofs"],
                        value_dim=item.get("value_dim", 1),
                        n_nodes=item.get("n_nodes"),
                    )
                continue

            if isinstance(item, tuple):
                if len(item) == 2:
                    name, space = item
                    opts = {}
                elif len(item) == 3:
                    name, space, opts = item
                    if not isinstance(opts, dict):
                        raise TypeError("3-tuple register_blocks entry must use dict as third item.")
                else:
                    raise ValueError("tuple register_blocks entry must have length 2 or 3.")
                self.register_space(
                    name,
                    space,
                    offset=opts.get("offset"),
                    value_dim=opts.get("value_dim", 1),
                    n_nodes=opts.get("n_nodes"),
                )
                continue

            raise TypeError("register_blocks entries must be dict or tuple.")

    def _get_block(self, name: str) -> _FieldBlock:
        key = str(name)
        if key not in self._blocks:
            candidates = list(self._blocks.keys())
            hint = ""
            if candidates:
                close = difflib.get_close_matches(key, candidates, n=1)
                if close:
                    hint = f" Did you mean '{close[0]}'?"
            raise ValueError(f"Field '{key}' is not registered.{hint}")
        return self._blocks[key]

    def resolve_block_dofs(
        self,
        field: str,
        *,
        nodes: int | Sequence[int] | np.ndarray | None = None,
        components: int | Sequence[int] | np.ndarray | None = None,
        local_dofs: int | Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Resolve field-local node/component or local-dof indices to global DOFs.

        Parameters
        ----------
        field:
            Registered block name.
        nodes/components:
            Node/component selector in the field.
            `components=None` means all components in `[0, value_dim)`.
        local_dofs:
            Field-local DOF indices. Mutually exclusive with `nodes`.
        """
        b = self._get_block(field)

        has_nodes = nodes is not None
        has_local = local_dofs is not None
        if has_nodes and has_local:
            raise ValueError("Specify either nodes/components or local_dofs, not both.")
        if not has_nodes and not has_local:
            raise ValueError("One of nodes or local_dofs must be provided.")

        if has_local:
            ld = np.asarray(local_dofs, dtype=int).reshape(-1)
            if ld.size == 0:
                return ld
            if np.any(ld < 0) or np.any(ld >= b.n_dofs):
                raise ValueError(f"local_dofs out of range for field '{b.name}'.")
            return b.offset + ld

        node_arr = np.asarray(nodes, dtype=int).reshape(-1)
        if node_arr.size == 0:
            return np.asarray([], dtype=int)
        if np.any(node_arr < 0) or np.any(node_arr >= b.n_nodes):
            raise ValueError(f"nodes out of range for field '{b.name}'.")

        if components is None:
            comp_arr = np.arange(b.value_dim, dtype=int)
        else:
            comp_arr = np.asarray(components, dtype=int).reshape(-1)
        if comp_arr.size == 0:
            return np.asarray([], dtype=int)
        if np.any(comp_arr < 0) or np.any(comp_arr >= b.value_dim):
            raise ValueError(f"components out of range for field '{b.name}'.")

        local = (node_arr[:, None] * b.value_dim + comp_arr[None, :]).reshape(-1)
        return b.offset + local

    def resolve_dirichlet(
        self,
        specs: Sequence[DirichletSpec],
        *,
        default_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert field-based Dirichlet specs into `(dirichlet_dofs, dirichlet_vals)`.

        Supported fields:
        - `field` (required)
        - `nodes` and optional `components`
        - `local_dofs`
        - `value` (scalar or array-like)
        """
        dof_to_value: dict[int, float] = {}

        for spec in specs:
            if not isinstance(spec, DirichletSpec):
                raise TypeError("dirichlet specs must be DirichletSpec instances.")
            field = str(spec.field)
            dofs = self.resolve_block_dofs(
                field,
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
                raise ValueError(
                    "Dirichlet 'value' must be scalar or match the number of selected DOFs."
                )

            for d, v in zip(dofs, vals):
                dof_to_value[int(d)] = float(v)

        if not dof_to_value:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        items = sorted(dof_to_value.items(), key=lambda kv: kv[0])
        dofs = np.asarray([k for k, _ in items], dtype=int)
        vals = np.asarray([v for _, v in items], dtype=float)
        return dofs, vals

    def solve(
        self,
        *,
        dirichlet_specs: Sequence[DirichletSpec] | None = None,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        format: str = "csr",
        diagonal_shift: float = 0.0,
    ):
        """
        Build and solve with optional `DirichletSpec` constraints.
        """
        if dirichlet_specs is not None and dirichlet_dofs is not None:
            raise ValueError("Use either dirichlet_specs or dirichlet_dofs, not both.")
        if dirichlet_specs is not None:
            dirichlet_dofs, dirichlet_vals = self.resolve_dirichlet(dirichlet_specs)
        return self.system.solve(
            dirichlet_dofs=dirichlet_dofs,
            dirichlet_vals=dirichlet_vals,
            format=format,
            diagonal_shift=diagonal_shift,
        )

    def add_contact_nitsche(
        self,
        ops_or_jacobian,
        *,
        master: str,
        slave: str,
        residual: np.ndarray | None = None,
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

        jac = getattr(ops_or_jacobian, "jacobian", ops_or_jacobian)
        if residual is None and hasattr(ops_or_jacobian, "residual"):
            residual = ops_or_jacobian.residual

        self.system.add_contact_nitsche(
            jac,
            residual=residual,
            n_master_nodes=m.n_nodes,
            n_slave_nodes=s.n_nodes,
            master_offset=m.offset,
            slave_offset=s.offset,
            value_dim=vd,
            scale=scale,
            residual_sign=residual_sign,
        )

    def add_contact_mortar(
        self,
        ops_or_kkt,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        F_contact: np.ndarray | None = None,
        rho: float | None = None,
        multiplier_space: str | None = None,
        facet_conn_master: np.ndarray | None = None,
        backend: str = "numpy",
    ) -> None:
        """
        Add mortar contact through either:
        - operators from ``assemble_contact_constraint_operators(...)``, or
        - a preassembled KKT matrix.
        """
        m = self._get_block(master)
        s = self._get_block(slave)
        if value_dim is None:
            if m.value_dim != s.value_dim:
                raise ValueError("master/slave value_dim mismatch. Pass value_dim explicitly.")
            vd = m.value_dim
        else:
            vd = int(value_dim)

        if hasattr(ops_or_kkt, "coupling_aa") and hasattr(ops_or_kkt, "coupling_ab"):
            coupling_aa = getattr(ops_or_kkt, "coupling_aa")
            coupling_ab = getattr(ops_or_kkt, "coupling_ab")
            if coupling_aa is None or coupling_ab is None:
                raise ValueError("mortar operators must include coupling_aa and coupling_ab.")
            rho_eff = getattr(ops_or_kkt, "rho", None) if rho is None else float(rho)
            if rho_eff is None:
                rho_eff = 0.0
            mult_eff = getattr(ops_or_kkt, "multiplier_space", None) if multiplier_space is None else multiplier_space
            if mult_eff is None:
                mult_eff = "nodal"
            fc = facet_conn_master
            if fc is None and hasattr(ops_or_kkt, "facet_conn_master"):
                fc = getattr(ops_or_kkt, "facet_conn_master")
            from ..mesh.contact import assemble_contact_kkt as _assemble_contact_kkt

            K_contact = _assemble_contact_kkt(
                coupling_aa,
                coupling_ab,
                rho=float(rho_eff),
                multiplier_space=str(mult_eff),
                facet_conn_master=fc,
                backend=backend,
                format="fluxsparse",
            )
        else:
            K_contact = ops_or_kkt

        self.system.add_contact_kkt(
            K_contact,
            n_master_nodes=m.n_nodes,
            n_slave_nodes=s.n_nodes,
            master_offset=m.offset,
            slave_offset=s.offset,
            F_contact=F_contact,
            value_dim=vd,
        )

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
        # nitsche options
        residual: np.ndarray | None = None,
        scale: float = 1.0,
        residual_sign: float = -1.0,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
        # mortar options
        F_contact: np.ndarray | None = None,
        rho: float | None = None,
        multiplier_space: str | None = None,
        facet_conn_master: np.ndarray | None = None,
        backend: str = "numpy",
    ) -> None:
        """
        Unified contact entry point for penalty/constraint families.
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

        # Accept raw contact-space objects and assemble operators internally.
        if (
            hasattr(contact_obj, "assemble_contact_constraint_operators")
            and not hasattr(contact_obj, "jacobian")
            and not hasattr(contact_obj, "coupling_aa")
        ):
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
                from ..mesh.contact import assemble_contact_constraint_operators as _assemble_ops

                contact_obj = _assemble_ops(
                    contact_obj,
                    law=law,
                    formulation=formulation,
                    rho=0.0 if rho is None else float(rho),
                    multiplier_space="nodal" if multiplier_space is None else str(multiplier_space),
                    backend=backend,
                    weak_form=weak_form,
                    state=state,
                    params=params,
                    normal_source=normal_source,
                    sparse=sparse,
                    batch_jac=batch_jac,
                )
            else:
                from ..mesh.contact import assemble_contact_penalty_operators as _assemble_ops

                contact_obj = _assemble_ops(
                    contact_obj,
                    law=law,
                    formulation=formulation,
                    backend=backend,
                    weak_form=weak_form,
                    state=state,
                    params=params,
                    normal_source=normal_source,
                    sparse=sparse,
                    batch_jac=batch_jac,
                )

        # `law` is currently metadata only; routing is enforcement/formulation-based.
        _ = law
        e_arg = None if enforcement is None else str(enforcement).lower()
        if e_arg in {"constraint", "mortar"}:
            e_arg = "mortar"
        elif e_arg in {"penalty", "nitsche"}:
            e_arg = "nitsche"
        f_arg = None if formulation is None else str(formulation).lower()
        if family_enforcement is not None and e_arg is not None and e_arg != family_enforcement:
            raise ValueError("family conflicts with enforcement.")
        if family_mode == "constraint" and f_arg in {"penalty", "penalty_consistent", "nitsche"}:
            raise ValueError("family='constraint' conflicts with penalty formulation.")
        if family_mode == "penalty" and f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
            raise ValueError("family='penalty' conflicts with multiplier formulation.")

        m = e_arg
        if m is None and family_enforcement is not None:
            m = family_enforcement
        if m is None and f_arg is not None:
            if f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
                m = "mortar"
            elif f_arg in {"penalty", "penalty_consistent", "nitsche"}:
                m = "nitsche"
            else:
                raise ValueError("Unknown formulation. Supported: multiplier, augmented_lagrangian, penalty.")
        if m is None and hasattr(contact_obj, "enforcement"):
            m = getattr(contact_obj, "enforcement")
        if m is None and hasattr(contact_obj, "formulation"):
            f_obj = str(getattr(contact_obj, "formulation")).lower()
            if f_obj in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
                m = "mortar"
            elif f_obj in {"penalty", "penalty_consistent", "nitsche"}:
                m = "nitsche"
        if m is None:
            raise ValueError("enforcement or formulation is required when contact_obj has no routing metadata.")
        m = str(m).lower()
        if f_arg is not None:
            if m == "mortar" and f_arg in {"penalty", "penalty_consistent", "nitsche"}:
                raise ValueError("formulation suggests nitsche, but enforcement resolved to mortar.")
            if m == "nitsche" and f_arg in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
                raise ValueError("formulation suggests mortar, but enforcement resolved to nitsche.")
        if m == "nitsche":
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
        if m == "mortar":
            self.add_contact_mortar(
                contact_obj,
                master=master,
                slave=slave,
                value_dim=value_dim,
                F_contact=F_contact,
                rho=rho,
                multiplier_space=multiplier_space,
                facet_conn_master=facet_conn_master,
                backend=backend,
            )
            return
        raise ValueError("enforcement must be 'nitsche' or 'mortar'.")

    def add_constraint(self, spec: ConstraintSpec) -> None:
        """
        Add a constraint through a unified typed descriptor.
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
                F_contact=spec.F_contact,
                rho=spec.rho,
                multiplier_space=spec.multiplier_space,
                facet_conn_master=spec.facet_conn_master,
                backend=spec.backend,
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
        if kind == "embedding":
            self.add_embedding_constraint(
                spec.embedding,
                master=spec.master,
                slave=spec.slave,
                value_dim=spec.value_dim,
                rho=spec.rho,
                backend=spec.backend,
                F_contact=spec.F_contact,
            )
            return
        raise ValueError("Unsupported ConstraintSpec.kind.")

    def add_constraint_matrix(
        self,
        C,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        rho: float = 0.0,
        F_contact: np.ndarray | None = None,
    ) -> None:
        """
        Add generic two-block equality constraints:
            C * [u_master; u_slave] = 0
        using KKT assembly with optional AL-like regularization ``rho * C^T C``.

        Accepted ``C`` inputs:
        - SciPy sparse matrix (recommended for large systems)
        - dense array-like
        - ``FluxSparseMatrix`` (converted internally via ``to_csr()``)
        """
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

        if isinstance(C, FluxSparseMatrix):
            C_csr = C.to_csr()
        elif sp.issparse(C):
            C_csr = C.tocsr()
        else:
            C_csr = sp.csr_matrix(np.asarray(C, dtype=float))
        n_cu = vd * int(m.n_nodes + s.n_nodes)
        if C_csr.ndim != 2 or C_csr.shape[1] != n_cu:
            raise ValueError("C shape mismatch for provided master/slave node counts and value_dim.")
        n_l = int(C_csr.shape[0])
        Kuu = float(rho) * (C_csr.T @ C_csr)
        Zll = sp.csr_matrix((n_l, n_l), dtype=float)
        K_contact = sp.bmat([[Kuu, C_csr.T], [C_csr, Zll]], format="csr")

        self.system.add_contact_kkt(
            K_contact,
            n_master_nodes=m.n_nodes,
            n_slave_nodes=s.n_nodes,
            master_offset=m.offset,
            slave_offset=s.offset,
            F_contact=F_contact,
            value_dim=vd,
        )

    def add_constraint_matrix_dof(
        self,
        C,
        *,
        master: str,
        slave: str,
        rho: float = 0.0,
        F_contact: np.ndarray | None = None,
    ) -> None:
        """
        Add generic DOF-level equality constraints on concatenated block DOFs:
            C * [u_master_dof; u_slave_dof] = 0

        Unlike ``add_constraint_matrix``, this method does not use ``value_dim``/node
        interpretation and instead treats both blocks as pure DOF vectors.
        """
        m = self._get_block(master)
        s = self._get_block(slave)
        if isinstance(C, FluxSparseMatrix):
            C_csr = C.to_csr()
        elif sp.issparse(C):
            C_csr = C.tocsr()
        else:
            C_csr = sp.csr_matrix(np.asarray(C, dtype=float))

        n_cu = int(m.n_dofs + s.n_dofs)
        if C_csr.ndim != 2 or C_csr.shape[1] != n_cu:
            raise ValueError("C shape mismatch for provided master/slave DOF counts.")
        n_l = int(C_csr.shape[0])
        Kuu = float(rho) * (C_csr.T @ C_csr)
        Zll = sp.csr_matrix((n_l, n_l), dtype=float)
        K_contact = sp.bmat([[Kuu, C_csr.T], [C_csr, Zll]], format="csr")

        self.system.add_contact_kkt(
            K_contact,
            n_master_nodes=m.n_dofs,
            n_slave_nodes=s.n_dofs,
            master_offset=m.offset,
            slave_offset=s.offset,
            F_contact=F_contact,
            value_dim=1,
        )

    def add_embedding_constraint(
        self,
        embedding,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        rho: float = 0.0,
        backend: str = "numpy",
        F_contact: np.ndarray | None = None,
    ) -> None:
        """
        Build and add embedding constraints from ``EmbeddingMap``.
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
            backend=backend,
        )
        self.add_constraint_matrix(
            C,
            master=master,
            slave=slave,
            value_dim=vd,
            rho=rho,
            F_contact=F_contact,
        )

    def build(self) -> CoupledSystem:
        return self.system


__all__ = ["CoupledSystem", "CoupledSystemBuilder", "DirichletSpec", "ConstraintSpec"]
