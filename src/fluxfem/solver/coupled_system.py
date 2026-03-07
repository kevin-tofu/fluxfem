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
        # Accept raw contact-space objects and assemble operators internally.
        if (
            hasattr(contact_obj, "assemble_contact_constraint_operators")
            and not hasattr(contact_obj, "jacobian")
            and not hasattr(contact_obj, "coupling_aa")
        ):
            f_arg_guess = None if formulation is None else str(formulation).lower()
            has_penalty_inputs = (weak_form is not None) or (state is not None) or (params is not None)
            if f_arg_guess in {"multiplier", "lagrange_multiplier", "augmented_lagrangian"}:
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
        f_arg = None if formulation is None else str(formulation).lower()
        m = e_arg
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

    def build(self) -> CoupledSystem:
        return self.system


__all__ = ["CoupledSystem", "CoupledSystemBuilder"]
