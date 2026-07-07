from __future__ import annotations

import difflib
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .cg import cg_solve_jax
from .dirichlet import enforce_dirichlet_fluxsparse, enforce_dirichlet_sparse
from .sparse import FluxSparseMatrix, concat_flux


def _resolve_contact_multiplier_choice(
    contact_obj,
    multiplier,
    mortar,
    *,
    value_dim: int | None = None,
    mortar_rank: int | None = None,
    mortar_max_rank: int | None = None,
    mortar_energy_tol: float = 0.999,
    mortar_rtol: float = 1e-10,
):
    if multiplier is not None:
        if mortar is not None:
            raise ValueError("Provide either multiplier or mortar, not both.")
        if any(v is not None for v in (mortar_rank, mortar_max_rank)):
            raise ValueError("mortar_rank/mortar_max_rank require mortar='coarse_dual'.")
        return multiplier
    from ..mesh.contact import ContactMultiplierSpace

    vd = 1 if value_dim is None else int(value_dim)
    if mortar is None:
        return ContactMultiplierSpace.from_contact(contact_obj, value_dim=vd)
    key = str(mortar).lower()
    if key in {"dual", "dual_nodal", "default"}:
        if any(v is not None for v in (mortar_rank, mortar_max_rank)):
            raise ValueError("mortar_rank/mortar_max_rank require mortar='coarse_dual'.")
        return ContactMultiplierSpace.dual_mortar(value_dim=vd)
    if key in {"coarse_dual", "coarse-dual", "coarse", "coarse_dual_nodal"}:
        return ContactMultiplierSpace.coarse_dual_mortar(
            value_dim=vd,
            rank=mortar_rank,
            max_rank=mortar_max_rank,
            energy_tol=float(mortar_energy_tol),
            rtol=float(mortar_rtol),
        )
    if key in {"nodal", "legacy_nodal"}:
        if any(v is not None for v in (mortar_rank, mortar_max_rank)):
            raise ValueError("mortar_rank/mortar_max_rank require mortar='coarse_dual'.")
        return ContactMultiplierSpace.nodal_mortar(value_dim=vd)
    if key in {"p0", "p0_master"}:
        if any(v is not None for v in (mortar_rank, mortar_max_rank)):
            raise ValueError("mortar_rank/mortar_max_rank require mortar='coarse_dual'.")
        return ContactMultiplierSpace.p0_mortar(contact_obj, value_dim=vd)
    raise ValueError("mortar must be 'dual', 'coarse_dual', 'nodal', or 'p0'.")


@dataclass
class NumpyCoupledSystem:
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

    @classmethod
    def create(cls, K_u, F_u, *, backend: str | None = None):
        """
        Create a coupled-system implementation for the requested backend.
        """
        backend = "numpy" if backend is None else str(backend).lower()
        if backend == "numpy":
            return cls.from_structural(K_u, F_u)
        if backend == "jax":
            from .coupled_system import CoupledSystem as JAXCoupledSystem

            return JAXCoupledSystem.from_structural(K_u, F_u)
        raise ValueError("backend must be 'numpy' or 'jax'.")

    @property
    def n_u(self) -> int:
        return int(self.K_u.shape[0])

    def append_structural_block(
        self,
        K_block=None,
        F_block: np.ndarray | None = None,
        *,
        n_dofs: int | None = None,
    ) -> slice:
        """
        Append structural DOFs to the primary unknown block.

        This is intended for user-added fields such as remote nodes. Any existing
        lifted KKT/contact system is expanded consistently, with zeros inserted for
        the new structural rows/cols in the lifted contribution.
        """
        if K_block is None:
            if n_dofs is None:
                raise ValueError("n_dofs is required when K_block is omitted.")
            n_add = int(n_dofs)
            if n_add <= 0:
                raise ValueError("n_dofs must be positive.")
            K_add = sp.csr_matrix((n_add, n_add), dtype=self.K_u.dtype)
        else:
            if isinstance(K_block, FluxSparseMatrix):
                K_add = K_block.to_csr()
            elif sp.issparse(K_block):
                K_add = K_block.tocsr()
            else:
                K_add = sp.csr_matrix(np.asarray(K_block, dtype=float))
            if K_add.shape[0] != K_add.shape[1]:
                raise ValueError("K_block must be square.")
            n_add = int(K_add.shape[0])
            if n_dofs is not None and int(n_dofs) != n_add:
                raise ValueError("n_dofs does not match K_block shape.")

        if F_block is None:
            F_add = np.zeros((n_add,), dtype=float)
        else:
            F_add = np.asarray(F_block, dtype=float).reshape(-1)
            if F_add.shape != (n_add,):
                raise ValueError("F_block shape must match appended DOF count.")

        start = self.n_u
        stop = start + n_add

        self.K_u = sp.block_diag((self.K_u, K_add), format="csr")
        self.F_u = np.concatenate([self.F_u, F_add], axis=0)

        if self.K_contact_lifted is not None:
            K_prev = self.K_contact_lifted.tocsr()
            n_l = int(K_prev.shape[0] - start)
            if n_l < 0:
                raise ValueError("Invalid lifted contact matrix shape.")
            Kuu_prev = K_prev[:start, :start]
            Kul_prev = K_prev[:start, start:]
            Klu_prev = K_prev[start:, :start]
            Kll_prev = K_prev[start:, start:]
            z_u = sp.csr_matrix((start, n_add), dtype=K_prev.dtype)
            z_ut = sp.csr_matrix((n_add, start), dtype=K_prev.dtype)
            z_ul = sp.csr_matrix((n_add, n_l), dtype=K_prev.dtype)
            z_lu = sp.csr_matrix((n_l, n_add), dtype=K_prev.dtype)
            z_uu = sp.csr_matrix((n_add, n_add), dtype=K_prev.dtype)
            self.K_contact_lifted = sp.bmat(
                [
                    [Kuu_prev, z_u, Kul_prev],
                    [z_ut, z_uu, z_ul],
                    [Klu_prev, z_lu, Kll_prev],
                ],
                format="csr",
            )
            if self.F_contact_lifted is not None:
                F_prev = np.asarray(self.F_contact_lifted, dtype=float).reshape(-1)
                self.F_contact_lifted = np.concatenate(
                    [F_prev[:start], np.zeros((n_add,), dtype=float), F_prev[start:]],
                    axis=0,
                )

        return slice(start, stop)

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

        F_add = np.zeros((self.n_u + n_l,), dtype=float)
        if F_contact is not None:
            Fc = np.asarray(F_contact, dtype=float)
            if Fc.shape != F_add.shape:
                raise ValueError("F_contact shape mismatch.")
            F_add += Fc

        if self.K_contact_lifted is None:
            self.K_contact_lifted = K_add
            self.F_contact_lifted = F_add
            return

        # Accumulate multiple mortar/KKT contacts by appending new lambda blocks.
        # This allows each contact contribution to have its own lambda size.
        K_prev = self.K_contact_lifted.tocsr()
        n_prev_l = int(K_prev.shape[0] - self.n_u)
        if n_prev_l < 0:
            raise ValueError("Invalid lifted contact matrix shape.")

        Kuu_prev = K_prev[: self.n_u, : self.n_u]
        Kul_prev = K_prev[: self.n_u, self.n_u :]
        Klu_prev = K_prev[self.n_u :, : self.n_u]
        Kll_prev = K_prev[self.n_u :, self.n_u :]

        Kuu_new = K_add[: self.n_u, : self.n_u]
        Kul_new = K_add[: self.n_u, self.n_u :]
        Klu_new = K_add[self.n_u :, : self.n_u]
        Kll_new = K_add[self.n_u :, self.n_u :]

        z_prev_new = sp.csr_matrix((n_prev_l, n_l), dtype=float)
        z_new_prev = sp.csr_matrix((n_l, n_prev_l), dtype=float)
        self.K_contact_lifted = sp.bmat(
            [
                [Kuu_prev + Kuu_new, Kul_prev, Kul_new],
                [Klu_prev, Kll_prev, z_prev_new],
                [Klu_new, z_new_prev, Kll_new],
            ],
            format="csr",
        )

        F_prev = self.F_contact_lifted
        if F_prev is None:
            F_prev = np.zeros((self.n_u + n_prev_l,), dtype=float)
        self.F_contact_lifted = np.concatenate(
            [
                np.asarray(F_prev[: self.n_u], dtype=float) + F_add[: self.n_u],
                np.asarray(F_prev[self.n_u :], dtype=float),
                np.asarray(F_add[self.n_u :], dtype=float),
            ]
        )

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

    def assemble(self, *, format: str = "fluxsparse", backend: str | None = None):
        backend = "numpy" if backend is None else str(backend).lower()
        if backend not in {"numpy", "jax"}:
            raise ValueError("backend must be 'numpy' or 'jax'.")
        if format not in {"fluxsparse", "csr", "dense"}:
            raise ValueError("format must be 'fluxsparse', 'csr', or 'dense'.")
        if backend == "jax" and format != "fluxsparse":
            raise ValueError("backend='jax' only supports format='fluxsparse'. Use backend='numpy' for csr/dense assembly.")

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
        flux = FluxSparseMatrix(coo.row, coo.col, coo.data, K.shape[0])
        if backend == "jax":
            import jax.numpy as jnp

            return flux, jnp.asarray(F)
        return flux, F

    def _assemble_solve_system(
        self,
        *,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        format: str = "csr",
        diagonal_shift: float = 0.0,
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        if format not in {"csr", "fluxsparse"}:
            raise ValueError("format must be 'csr' or 'fluxsparse'.")

        dir_dofs = np.asarray(dirichlet_dofs if dirichlet_dofs is not None else [], dtype=int)
        if format == "fluxsparse":
            K_flux, F = self.assemble(format="fluxsparse")
            K_bc, F_bc = enforce_dirichlet_fluxsparse(K_flux, F, dir_dofs, dirichlet_vals)
            if float(diagonal_shift) != 0.0:
                K_bc = K_bc + float(diagonal_shift) * sp.eye(K_bc.shape[0], format="csr")
            return K_bc.tocsr(), np.asarray(F_bc, dtype=float), dir_dofs

        K_csr, F = self.assemble(format="csr")
        if dir_dofs.size > 0:
            coo = K_csr.tocoo()
            K_flux = FluxSparseMatrix(coo.row, coo.col, coo.data, K_csr.shape[0])
            K_csr, F = enforce_dirichlet_sparse(K_flux, F, dir_dofs, dirichlet_vals)
        if float(diagonal_shift) != 0.0:
            K_csr = K_csr + float(diagonal_shift) * sp.eye(K_csr.shape[0], format="csr")
        return K_csr.tocsr(), np.asarray(F, dtype=float), dir_dofs

    @staticmethod
    def _zero_dirichlet_matrix(mat, dirichlet_dofs: np.ndarray) -> sp.csr_matrix:
        if isinstance(mat, FluxSparseMatrix):
            out = mat.to_csr()
        elif sp.issparse(mat):
            out = mat.tocsr()
        else:
            out = sp.csr_matrix(np.asarray(mat, dtype=float))
        if dirichlet_dofs.size == 0:
            return out
        out = out.tolil()
        out[dirichlet_dofs, :] = 0.0
        out[:, dirichlet_dofs] = 0.0
        return out.tocsr()

    @staticmethod
    def _zero_dirichlet_vector(vec, dirichlet_dofs: np.ndarray, *, size: int) -> np.ndarray:
        out = np.asarray(vec, dtype=float).reshape(-1)
        if out.shape != (size,):
            raise ValueError(f"vector shape must be {(size,)}, got {out.shape}.")
        out = out.copy()
        if dirichlet_dofs.size > 0:
            out[dirichlet_dofs] = 0.0
        return out

    def solve(
        self,
        *,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        format: str = "csr",
        diagonal_shift: float = 0.0,
        backend: str | None = None,
        jax_solver: str = "cg",
        tol: float = 1e-8,
        maxiter: int | None = None,
    ):
        backend = "numpy" if backend is None else str(backend).lower()
        if backend not in {"numpy", "jax"}:
            raise ValueError("backend must be 'numpy' or 'jax'.")
        if backend == "jax" and format == "csr":
            format = "fluxsparse"
        if backend == "numpy":
            K_sys, F_sys, _dir_dofs = self._assemble_solve_system(
                dirichlet_dofs=dirichlet_dofs,
                dirichlet_vals=dirichlet_vals,
                format=format,
                diagonal_shift=diagonal_shift,
            )
            return spla.spsolve(K_sys, F_sys)

        if jax_solver != "cg":
            raise ValueError("CoupledSystem backend='jax' only supports jax_solver='cg'. Use assemble(..., format='dense', backend='jax') for dense reference matrices.")

        import jax.numpy as jnp
        from .dirichlet import enforce_dirichlet_fluxsparse_jax

        K_flux, F_flux = self.assemble(format="fluxsparse", backend="jax")
        dir_dofs = np.asarray(dirichlet_dofs if dirichlet_dofs is not None else [], dtype=int)
        if dir_dofs.size > 0:
            K_bc, F_bc = enforce_dirichlet_fluxsparse_jax(K_flux, F_flux, dir_dofs, dirichlet_vals)
        else:
            K_bc, F_bc = K_flux, F_flux
        if float(diagonal_shift) != 0.0:
            diag = jnp.arange(K_bc.n_dofs, dtype=jnp.int32)
            K_bc = concat_flux(
                K_bc,
                FluxSparseMatrix(diag, diag, jnp.full((K_bc.n_dofs,), jnp.asarray(diagonal_shift, dtype=F_bc.dtype)), K_bc.n_dofs),
                n_dofs=K_bc.n_dofs,
            )
        u, _info = cg_solve_jax(K_bc, F_bc, tol=tol, maxiter=maxiter)
        return u

    def linear_output_sensitivity(
        self,
        dK,
        *,
        output_vector,
        dF=None,
        u: np.ndarray | None = None,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        format: str = "csr",
        diagonal_shift: float = 0.0,
    ) -> float:
        """
        Differentiate a linear output ``J = output_vector^T u`` w.r.t. a parameter.

        The state ``u`` solves the coupled linear system assembled by this object.
        ``dK`` and ``dF`` are the derivatives of the assembled stiffness and RHS
        with respect to the parameter.
        """
        K_sys, F_sys, dir_dofs = self._assemble_solve_system(
            dirichlet_dofs=dirichlet_dofs,
            dirichlet_vals=dirichlet_vals,
            format=format,
            diagonal_shift=diagonal_shift,
        )
        n = int(K_sys.shape[0])
        c = self._zero_dirichlet_vector(output_vector, dir_dofs, size=n)
        dK_raw = self._zero_dirichlet_matrix(dK, np.asarray([], dtype=int))
        if dK_raw.shape == (self.n_u, self.n_u) and n != self.n_u:
            dK_lift = sp.lil_matrix((n, n), dtype=dK_raw.dtype)
            dK_lift[: self.n_u, : self.n_u] = dK_raw
            dK_raw = dK_lift.tocsr()
        dK_bc = self._zero_dirichlet_matrix(dK_raw, dir_dofs)
        if dK_bc.shape != K_sys.shape:
            raise ValueError(f"dK shape {dK_bc.shape} does not match assembled system shape {K_sys.shape}.")
        if dF is None:
            dF_bc = np.zeros((n,), dtype=float)
        else:
            dF_arr = np.asarray(dF, dtype=float).reshape(-1)
            if dF_arr.shape == (self.n_u,) and n != self.n_u:
                dF_full = np.zeros((n,), dtype=float)
                dF_full[: self.n_u] = dF_arr
                dF_arr = dF_full
            dF_bc = self._zero_dirichlet_vector(dF_arr, dir_dofs, size=n)

        u_vec = np.asarray(u, dtype=float).reshape(-1) if u is not None else spla.spsolve(K_sys, F_sys)
        if u_vec.shape != (n,):
            raise ValueError(f"u shape must be {(n,)}, got {u_vec.shape}.")

        lam = spla.spsolve(K_sys.T, c)
        dJ = lam @ (dF_bc - dK_bc @ u_vec)
        return float(dJ)

    def linear_compliance_sensitivity(
        self,
        dK,
        *,
        load_vector,
        dF=None,
        u: np.ndarray | None = None,
        dirichlet_dofs: np.ndarray | None = None,
        dirichlet_vals=0.0,
        format: str = "csr",
        diagonal_shift: float = 0.0,
    ) -> float:
        """
        Differentiate compliance ``C = load_vector^T u`` w.r.t. a parameter.

        Pass the external load of interest explicitly via ``load_vector``.
        When the parameter only modifies stiffness, leave ``dF=None``.
        """
        return self.linear_output_sensitivity(
            dK,
            output_vector=load_vector,
            dF=dF,
            u=u,
            dirichlet_dofs=dirichlet_dofs,
            dirichlet_vals=dirichlet_vals,
            format=format,
            diagonal_shift=diagonal_shift,
        )


@dataclass
class _FieldBlock:
    name: str
    offset: int
    n_dofs: int
    value_dim: int
    n_nodes: int
    point: np.ndarray | None = None


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
    - "rbe2": routes to add_rbe2_constraint(...)
    - "rbe3": routes to add_rbe3_constraint(...)

    Notes on ``kind="contact"``:
    - NumPy/SciPy builder: supports the existing contact routing surface
      (penalty/nitsche and mortar/constraint families).
    - JAX builder: currently supports only penalty/nitsche contact.
      Mortar/multiplier contact is not supported there yet.
    - For JAX, autodiff currently applies to contact contributions assembled
      through the JAX contact path. This includes dense and sparse Jacobian
      assembly for the penalty family, and the raw ``add_contact(...)`` path
      when it resolves through ``assemble_contact_penalty_operators(..., backend="jax")``.
    """

    kind: str
    master: str
    slave: str
    value_dim: int | None = None
    rho: float = 0.0
    F_contact: np.ndarray | None = None
    backend: str | None = None

    contact_obj: Any | None = None
    C: Any | None = None
    embedding: Any | None = None
    ref_point: np.ndarray | None = None
    slave_coords: np.ndarray | None = None
    weights: np.ndarray | None = None
    normalize_weights: bool = True

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
    multiplier: Any | None = None
    facet_conn_master: np.ndarray | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).lower()
        if kind not in {"contact", "matrix", "matrix_dof", "embedding", "rbe2", "rbe3"}:
            raise ValueError("ConstraintSpec.kind must be one of: contact, matrix, matrix_dof, embedding, rbe2, rbe3.")
        if kind == "contact" and self.contact_obj is None:
            raise ValueError("ConstraintSpec(kind='contact') requires contact_obj.")
        if kind in {"matrix", "matrix_dof"} and self.C is None:
            raise ValueError("ConstraintSpec(kind='matrix*') requires C.")
        if kind == "embedding" and self.embedding is None:
            raise ValueError("ConstraintSpec(kind='embedding') requires embedding.")
        if kind == "rbe2" and (self.ref_point is None or self.slave_coords is None):
            raise ValueError("ConstraintSpec(kind='rbe2') requires ref_point and slave_coords.")
        if kind == "rbe3" and (self.ref_point is None or self.slave_coords is None):
            raise ValueError("ConstraintSpec(kind='rbe3') requires ref_point and slave_coords.")


class NumpyCoupledSystemBuilder:
    """
    Helper to reduce manual offset/node bookkeeping for coupled contact assembly.
    """

    def __init__(self, system: CoupledSystem):
        self.system = system
        self._blocks: dict[str, _FieldBlock] = {}

    @classmethod
    def from_structural(cls, K_u, F_u) -> "CoupledSystemBuilder":
        return cls(CoupledSystem.from_structural(K_u, F_u))

    @classmethod
    def create(cls, K_u, F_u, *, backend: str | None = None):
        """
        Create a coupled-system builder for the requested backend.
        """
        backend = "numpy" if backend is None else str(backend).lower()
        if backend == "numpy":
            return cls.from_structural(K_u, F_u)
        if backend == "jax":
            from .coupled_system import CoupledSystemBuilder as JAXCoupledSystemBuilder

            return JAXCoupledSystemBuilder.from_structural(K_u, F_u)
        raise ValueError("backend must be 'numpy' or 'jax'.")

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

    def append_remote_point(
        self,
        name: str,
        *,
        point,
        include_rotation: bool = True,
        F_block: np.ndarray | None = None,
    ) -> None:
        """
        Append a remote-point field.

        By default this creates a 6-DOF field ordered as
        ``[u_ref(3), omega_ref(3)]``. With ``include_rotation=False`` it creates a
        translational 3-DOF remote point.
        """
        dof_count = 6 if include_rotation else 3
        self.append_field(name, n_dofs=dof_count, value_dim=1, n_nodes=dof_count, F_block=F_block)
        block = self._get_block(name)
        block.point = np.asarray(point, dtype=float).reshape(-1)
        if block.point.shape != (3,):
            raise ValueError("remote point must be a 3D coordinate.")

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

    def append_field(
        self,
        name: str,
        *,
        n_dofs: int,
        value_dim: int = 1,
        n_nodes: int | None = None,
        K_block=None,
        F_block: np.ndarray | None = None,
    ) -> None:
        """
        Append a new structural field at the end of the current unknown vector.

        This is the main extension hook for user-added remote nodes or auxiliary
        structural DOFs that should participate in subsequent constraints.
        """
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

    def append_dof_copy_field(
        self,
        name: str,
        *,
        source: str,
        source_dofs,
        rho: float = 0.0,
    ) -> None:
        """
        Append an auxiliary field tied to selected DOFs of an existing field.

        This replaces hand-built ``u_source[selected] - u_aux = 0`` matrices
        before applying RBE2/RBE3 constraints to the auxiliary field.
        """
        src = self._get_block(source)
        dofs = np.asarray(source_dofs, dtype=int).reshape(-1)
        if dofs.size == 0:
            raise ValueError("source_dofs must contain at least one DOF.")
        local = dofs - src.offset
        if np.any(local < 0) or np.any(local >= src.n_dofs):
            raise ValueError("source_dofs must lie inside the source field.")
        self.append_field(name, n_dofs=dofs.size, value_dim=1)
        rows = np.arange(dofs.size, dtype=int)
        all_rows = np.concatenate([rows, rows])
        all_cols = np.concatenate([local, src.n_dofs + rows])
        data = np.concatenate([np.ones(dofs.size), -np.ones(dofs.size)])
        C = sp.coo_matrix((data, (all_rows, all_cols)), shape=(dofs.size, src.n_dofs + dofs.size)).tocsr()
        self.add_constraint_matrix_dof(C, master=source, slave=name, rho=rho)

    def add_distributed_coupling(
        self,
        *,
        source: str,
        source_dofs,
        remote: str,
        point,
        slave_coords,
        weights: np.ndarray | None = None,
        copy_field: str | None = None,
        normalize_weights: bool = True,
        rho: float = 0.0,
        backend: str | None = None,
        validate_rank: bool = True,
        min_rank: int | None = None,
        dependent_components=None,
        slave_components=None,
    ) -> str:
        """
        Couple selected source DOFs to a 6-DOF remote point through RBE3-style averaging.

        This is a named convenience wrapper around ``append_dof_copy_field``,
        ``append_remote_point``, and ``add_rbe3_constraint``. It returns the
        generated auxiliary copy-field name. ``dependent_components`` selects
        remote components from ``[Tx, Ty, Tz, Rx, Ry, Rz]``.
        """
        dep = tuple(range(6)) if dependent_components is None else tuple(dependent_components)
        if validate_rank:
            from ..mesh.contact import assemble_rbe3_constraint_matrix

            local = assemble_rbe3_constraint_matrix(
                np.asarray(point, dtype=float),
                np.asarray(slave_coords, dtype=float),
                weights=None if weights is None else np.asarray(weights, dtype=float),
                normalize_weights=normalize_weights,
                dependent_components=dep,
                slave_components=slave_components,
                backend="numpy",
            )
            rank = int(np.linalg.matrix_rank(np.asarray(local)[:, :6]))
            required = len(dep) if min_rank is None else int(min_rank)
            if rank < required:
                raise ValueError(
                    f"distributed coupling remote reference block has rank {rank}/{required}; "
                    "use a larger/non-degenerate patch or fewer dependent components."
                )
        copy_name = f"{remote}_distributed_patch" if copy_field is None else str(copy_field)
        self.append_dof_copy_field(copy_name, source=source, source_dofs=source_dofs, rho=rho)
        self.append_remote_point(remote, point=point, include_rotation=True)
        self.add_rbe3_constraint(
            master=remote,
            slave=copy_name,
            ref_point=point,
            slave_coords=slave_coords,
            weights=weights,
            normalize_weights=normalize_weights,
            dependent_components=dep,
            slave_components=slave_components,
            rho=rho,
            backend=backend,
        )
        return copy_name

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
        backend: str | None = None,
        jax_solver: str = "cg",
        tol: float = 1e-8,
        maxiter: int | None = None,
    ):
        """
        Build and solve with optional `DirichletSpec` constraints.

        For `backend="jax"`, only sparse `jax_solver="cg"` is supported.
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
            backend=backend,
            jax_solver=jax_solver,
            tol=tol,
            maxiter=maxiter,
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
        multiplier=None,
        facet_conn_master: np.ndarray | None = None,
        backend: str | None = None,
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
            mult_obj = getattr(ops_or_kkt, "multiplier", None) if multiplier is None else multiplier
            mult_family = str(getattr(mult_obj, "family", "")).lower() if mult_obj is not None else ""
            if mult_family in {"p0_active", "p0_supermesh"}:
                B_obj = getattr(ops_or_kkt, "B", None)
                if B_obj is None:
                    raise ValueError(f"{mult_family} mortar operators must include B.")
                Kuu_obj = getattr(ops_or_kkt, "Kuu", None)
                if Kuu_obj is None:
                    raise ValueError(f"{mult_family} mortar operators must include Kuu.")
                B_csr = B_obj.tocsr() if hasattr(B_obj, "tocsr") else sp.csr_matrix(np.asarray(B_obj, dtype=float))
                Kuu_csr = (
                    Kuu_obj.tocsr()
                    if hasattr(Kuu_obj, "tocsr")
                    else sp.csr_matrix(np.asarray(Kuu_obj, dtype=float))
                )
                Zll = sp.csr_matrix((B_csr.shape[0], B_csr.shape[0]), dtype=Kuu_csr.dtype)
                K_contact = sp.bmat(
                    [[Kuu_csr, B_csr.T], [B_csr, Zll]],
                    format="csr",
                )
            else:
                coupling_aa = getattr(ops_or_kkt, "coupling_aa")
                coupling_ab = getattr(ops_or_kkt, "coupling_ab")
                if coupling_aa is None or coupling_ab is None:
                    raise ValueError("mortar operators must include coupling_aa and coupling_ab.")
                rho_eff = getattr(ops_or_kkt, "rho", None) if rho is None else float(rho)
                if rho_eff is None:
                    rho_eff = 0.0
                if mult_obj is None:
                    raise ValueError("Constraint-family contact requires multiplier (ContactMultiplierSpace).")
                fc = facet_conn_master
                if fc is None and hasattr(ops_or_kkt, "facet_conn_master"):
                    fc = getattr(ops_or_kkt, "facet_conn_master")
                from ..mesh.contact import assemble_contact_kkt as _assemble_contact_kkt

                K_contact = _assemble_contact_kkt(
                    coupling_aa,
                    coupling_ab,
                    rho=float(rho_eff),
                    multiplier=mult_obj,
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
        multiplier=None,
        mortar: str | None = None,
        mortar_rank: int | None = None,
        mortar_max_rank: int | None = None,
        mortar_energy_tol: float = 0.999,
        mortar_rtol: float = 1e-10,
        facet_conn_master: np.ndarray | None = None,
        backend: str | None = None,
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
            warnings.warn(
                "Passing a raw contact interface into CoupledSystemBuilder.add_contact(...) is a compatibility path. "
                "Prefer assembling an explicit contact contribution first and passing that contribution to add_contact(...).",
                DeprecationWarning,
                stacklevel=2,
            )
            f_arg_guess = None if formulation is None else str(formulation).lower()
            has_penalty_inputs = (weak_form is not None) or (state is not None) or (params is not None)
            eval_backend = "jax" if (has_penalty_inputs and backend == "numpy") else backend
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
                multiplier = _resolve_contact_multiplier_choice(
                    contact_obj,
                    multiplier,
                    mortar,
                    value_dim=value_dim,
                    mortar_rank=mortar_rank,
                    mortar_max_rank=mortar_max_rank,
                    mortar_energy_tol=mortar_energy_tol,
                    mortar_rtol=mortar_rtol,
                )

                contact_obj = _assemble_ops(
                    contact_obj,
                    law=law,
                    formulation=formulation,
                    rho=0.0 if rho is None else float(rho),
                    multiplier=multiplier,
                    backend=eval_backend,
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
                    backend=eval_backend,
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
                multiplier=multiplier,
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
                multiplier=spec.multiplier,
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
        if kind == "rbe2":
            self.add_rbe2_constraint(
                master=spec.master,
                slave=spec.slave,
                ref_point=spec.ref_point,
                slave_coords=spec.slave_coords,
                rho=spec.rho,
                backend=spec.backend,
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
                backend=spec.backend,
                F_contact=spec.F_contact,
            )
            return
        raise ValueError("Unsupported ConstraintSpec.kind.")

    def add_field_matrix(self, field: str, K_local, *, F_local: np.ndarray | None = None) -> None:
        """
        Add a local stiffness/load contribution directly onto one registered field.

        This is useful for user-added remote fields, springs, or reduced support
        models that should modify the structural block without introducing
        additional Lagrange multiplier DOFs.
        """
        b = self._get_block(field)
        if isinstance(K_local, FluxSparseMatrix):
            K_csr = K_local.to_csr()
        elif sp.issparse(K_local):
            K_csr = K_local.tocsr()
        else:
            K_csr = sp.csr_matrix(np.asarray(K_local, dtype=float))
        if K_csr.shape != (b.n_dofs, b.n_dofs):
            raise ValueError(f"K_local shape {K_csr.shape} does not match field '{b.name}' size {(b.n_dofs, b.n_dofs)}.")

        self.system.K_u = self.system.K_u.tolil()
        self.system.K_u[b.offset : b.offset + b.n_dofs, b.offset : b.offset + b.n_dofs] += K_csr
        self.system.K_u = self.system.K_u.tocsr()

        if F_local is not None:
            F_arr = np.asarray(F_local, dtype=float).reshape(-1)
            if F_arr.shape != (b.n_dofs,):
                raise ValueError(f"F_local shape {F_arr.shape} does not match field '{b.name}' size {(b.n_dofs,)}.")
            self.system.F_u[b.offset : b.offset + b.n_dofs] += F_arr

    @staticmethod
    def _coerce_spring_matrix_and_reference(stiffness, reference_value, *, n: int) -> tuple[np.ndarray, np.ndarray]:
        if np.isscalar(stiffness):
            K_sel = np.eye(n, dtype=float) * float(stiffness)
        else:
            stiff_arr = np.asarray(stiffness, dtype=float)
            if stiff_arr.ndim == 1:
                if stiff_arr.shape != (n,):
                    raise ValueError("stiffness vector must match selected DOF count.")
                K_sel = np.diag(stiff_arr)
            elif stiff_arr.ndim == 2:
                if stiff_arr.shape != (n, n):
                    raise ValueError("stiffness matrix must match selected DOF count.")
                K_sel = stiff_arr
            else:
                raise ValueError("stiffness must be scalar, vector, or square matrix.")

        ref_arr = np.asarray(reference_value, dtype=float).reshape(-1)
        if ref_arr.size == 1:
            u_ref = np.full((n,), float(ref_arr[0]), dtype=float)
        elif ref_arr.size == n:
            u_ref = ref_arr.astype(float, copy=False)
        else:
            raise ValueError("reference_value must be scalar or match selected DOF count.")
        return K_sel, u_ref

    def dof_spring_contribution(
        self,
        field: str,
        *,
        stiffness,
        reference_value=0.0,
        nodes: int | Sequence[int] | np.ndarray | None = None,
        components: int | Sequence[int] | np.ndarray | None = None,
        local_dofs: int | Sequence[int] | np.ndarray | None = None,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """
        Build the full-system stiffness/RHS contribution of a spring support.

        This is useful when a scalar parameter scales the spring and one wants
        ``dK/dp`` and ``dF/dp`` for sensitivity analysis.
        """
        dofs = self.resolve_block_dofs(
            field,
            nodes=nodes,
            components=components,
            local_dofs=local_dofs,
        )
        n_sys = self.system.n_u
        if dofs.size == 0:
            return sp.csr_matrix((n_sys, n_sys), dtype=float), np.zeros((n_sys,), dtype=float)

        K_sel, u_ref = self._coerce_spring_matrix_and_reference(stiffness, reference_value, n=int(dofs.size))
        dofs_i = np.asarray(dofs, dtype=int)
        rows = np.repeat(dofs_i, dofs_i.size)
        cols = np.tile(dofs_i, dofs_i.size)
        data = K_sel.reshape(-1)
        K_full = sp.csr_matrix((data, (rows, cols)), shape=(n_sys, n_sys), dtype=float)
        F_full = np.zeros((n_sys,), dtype=float)
        F_full[dofs_i] = np.asarray(K_sel @ u_ref, dtype=float)
        return K_full, F_full

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
        """
        Add linear springs to selected DOFs of a registered field.

        The spring contribution is:
            K += K_s
            F += K_s @ u_ref
        so a scalar or vector ``reference_value`` acts like the target displacement
        of a spring-to-ground support.
        """
        dofs = self.resolve_block_dofs(
            field,
            nodes=nodes,
            components=components,
            local_dofs=local_dofs,
        )
        if dofs.size == 0:
            return dofs
        K_add, F_add = self.dof_spring_contribution(
            field,
            stiffness=stiffness,
            reference_value=reference_value,
            nodes=nodes,
            components=components,
            local_dofs=local_dofs,
        )
        self.system.K_u = (self.system.K_u + K_add).tocsr()
        self.system.F_u += F_add
        return np.asarray(dofs, dtype=int)

    def remote_spring_contribution(
        self,
        field: str,
        *,
        translational_stiffness=None,
        rotational_stiffness=None,
        translational_force=None,
        rotational_force=None,
        translational_target=0.0,
        rotational_target=0.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """
        Build the full-system spring contribution for a remote point.

        This returns the same contribution that ``add_remote_spring`` would add,
        which is convenient for analytical sensitivities.
        """

        def _resolve_spring_stiffness(name: str, stiffness, force, target, n_expected: int):
            if stiffness is not None and force is not None:
                raise ValueError(f"{name}: specify either stiffness or force, not both.")
            if stiffness is not None:
                return stiffness
            if force is None:
                return None

            force_arr = np.asarray(force, dtype=float).reshape(-1)
            target_arr = np.asarray(target, dtype=float).reshape(-1)

            if force_arr.size == 1:
                force_arr = np.full((n_expected,), float(force_arr[0]), dtype=float)
            elif force_arr.size != n_expected:
                raise ValueError(f"{name}: force must be scalar or length {n_expected}.")

            if target_arr.size == 1:
                target_arr = np.full((n_expected,), float(target_arr[0]), dtype=float)
            elif target_arr.size != n_expected:
                raise ValueError(f"{name}: target must be scalar or length {n_expected}.")

            zero_mask = np.abs(target_arr) <= 1e-15
            if np.any(zero_mask & (np.abs(force_arr) > 1e-15)):
                raise ValueError(f"{name}: cannot infer stiffness when target is zero but force is non-zero.")

            stiff = np.zeros((n_expected,), dtype=float)
            nz = ~zero_mask
            stiff[nz] = force_arr[nz] / target_arr[nz]
            return stiff

        b = self._get_block(field)
        if b.n_dofs not in {3, 6}:
            raise ValueError("remote spring helper expects a 3-DOF or 6-DOF field.")

        n_sys = self.system.n_u
        K_full = sp.csr_matrix((n_sys, n_sys), dtype=float)
        F_full = np.zeros((n_sys,), dtype=float)

        translational_stiffness = _resolve_spring_stiffness(
            "translational spring",
            translational_stiffness,
            translational_force,
            translational_target,
            min(3, b.n_dofs),
        )
        if translational_stiffness is not None:
            K_add, F_add = self.dof_spring_contribution(
                field,
                local_dofs=np.arange(min(3, b.n_dofs)),
                stiffness=translational_stiffness,
                reference_value=translational_target,
            )
            K_full = K_full + K_add
            F_full += F_add

        rotational_stiffness = _resolve_spring_stiffness(
            "rotational spring",
            rotational_stiffness,
            rotational_force,
            rotational_target,
            3,
        )
        if rotational_stiffness is not None:
            if b.n_dofs < 6:
                raise ValueError("rotational springs require a 6-DOF remote field.")
            K_add, F_add = self.dof_spring_contribution(
                field,
                local_dofs=np.arange(3, 6),
                stiffness=rotational_stiffness,
                reference_value=rotational_target,
            )
            K_full = K_full + K_add
            F_full += F_add

        return K_full.tocsr(), F_full

    def add_remote_spring(
        self,
        field: str,
        *,
        translational_stiffness=None,
        rotational_stiffness=None,
        translational_force=None,
        rotational_force=None,
        translational_target=0.0,
        rotational_target=0.0,
    ) -> None:
        """
        Add translational and/or rotational springs to a remote-point field.

        Expected layouts:
        - 6 DOF: ``[u_ref(3), omega_ref(3)]``
        - 3 DOF: translational-only remote point

        For each block, either provide ``*_stiffness`` directly, or provide
        ``*_force`` together with the corresponding target and let the helper
        derive diagonal stiffness values from ``force / target``.
        """
        K_add, F_add = self.remote_spring_contribution(
            field,
            translational_stiffness=translational_stiffness,
            rotational_stiffness=rotational_stiffness,
            translational_force=translational_force,
            rotational_force=rotational_force,
            translational_target=translational_target,
            rotational_target=rotational_target,
        )
        self.system.K_u = (self.system.K_u + K_add).tocsr()
        self.system.F_u += F_add

    def add_bolt_preload(
        self,
        field: str,
        *,
        stiffness: float,
        direction,
        target_displacement: float = 0.0,
        local_dofs=None,
    ) -> np.ndarray:
        """
        Add a directional preload spring on a remote or structural field.

        The contribution is ``k d d^T`` with target displacement
        ``target_displacement * d`` on the selected DOFs. ``direction`` is
        normalized internally.
        """
        b = self._get_block(field)
        if local_dofs is None:
            direction_arr = np.asarray(direction, dtype=float).reshape(-1)
            if direction_arr.size == 3:
                dofs = np.arange(3, dtype=int)
            elif direction_arr.size == b.n_dofs:
                dofs = np.arange(b.n_dofs, dtype=int)
            else:
                raise ValueError("direction must have length 3 or match the field DOF count.")
        else:
            dofs = np.asarray(local_dofs, dtype=int).reshape(-1)
            direction_arr = np.asarray(direction, dtype=float).reshape(-1)
            if direction_arr.size != dofs.size:
                raise ValueError("direction length must match local_dofs.")
        if dofs.size == 0:
            raise ValueError("At least one preload DOF is required.")
        if np.any(dofs < 0) or np.any(dofs >= b.n_dofs):
            raise ValueError("local_dofs contains an index outside the field.")
        norm = float(np.linalg.norm(direction_arr))
        if norm <= 0.0:
            raise ValueError("direction must be nonzero.")
        unit = direction_arr / norm
        k_dir = float(stiffness) * np.outer(unit, unit)
        ref = float(target_displacement) * unit
        return self.add_dof_spring(field, local_dofs=dofs, stiffness=k_dir, reference_value=ref)

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

    def add_dof_tie_constraint(
        self,
        *,
        master: str,
        slave: str,
        master_dofs,
        slave_dofs,
        rhs=0.0,
        rho: float = 0.0,
    ) -> None:
        """Constrain selected field-local DOFs by ``u_master - u_slave = rhs``."""
        m = self._get_block(master)
        s = self._get_block(slave)
        master_arr = np.asarray(master_dofs, dtype=int).reshape(-1)
        slave_arr = np.asarray(slave_dofs, dtype=int).reshape(-1)
        if master_arr.shape != slave_arr.shape:
            raise ValueError("master_dofs and slave_dofs must have the same shape.")
        if master_arr.size == 0:
            raise ValueError("At least one tied DOF is required.")
        if np.any(master_arr < 0) or np.any(master_arr >= m.n_dofs):
            raise ValueError("master_dofs contains an index outside the master field.")
        if np.any(slave_arr < 0) or np.any(slave_arr >= s.n_dofs):
            raise ValueError("slave_dofs contains an index outside the slave field.")

        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        if rhs_arr.size == 1:
            rhs_arr = np.full((master_arr.size,), float(rhs_arr[0]), dtype=float)
        if rhs_arr.shape != master_arr.shape:
            raise ValueError("rhs must be scalar or match the tied DOF count.")

        rows = np.arange(master_arr.size, dtype=int)
        cols = np.concatenate([master_arr, m.n_dofs + slave_arr])
        data = np.concatenate([np.ones(master_arr.size, dtype=float), -np.ones(master_arr.size, dtype=float)])
        C = sp.coo_matrix(
            (data, (np.concatenate([rows, rows]), cols)),
            shape=(master_arr.size, m.n_dofs + s.n_dofs),
        ).tocsr()

        F_contact = None
        if np.any(rhs_arr != 0.0):
            F_contact = np.zeros((self.system.n_u + master_arr.size,), dtype=float)
            F_contact[self.system.n_u :] = rhs_arr
        self.add_constraint_matrix_dof(C, master=master, slave=slave, rho=rho, F_contact=F_contact)

    def add_embedding_constraint(
        self,
        embedding,
        *,
        master: str,
        slave: str,
        value_dim: int | None = None,
        rho: float = 0.0,
        backend: str | None = None,
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

    def add_rbe2_constraint(
        self,
        *,
        master: str,
        slave: str,
        ref_point: np.ndarray,
        slave_coords: np.ndarray,
        slave_components=None,
        rho: float = 0.0,
        backend: str | None = None,
        F_contact: np.ndarray | None = None,
    ) -> None:
        """
        Build and add a 3D RBE2-style rigid constraint matrix.

        Expected field layout:
        - ``master``: 6 DOFs ordered as ``[u_ref(3), omega_ref(3)]``
        - ``slave``: 3 DOFs per node ordered as nodal translations
        """
        from ..mesh.contact import assemble_rbe2_constraint_matrix

        m = self._get_block(master)
        s = self._get_block(slave)
        C = assemble_rbe2_constraint_matrix(ref_point, slave_coords, slave_components=slave_components, backend=backend)
        if m.n_dofs != 6:
            raise ValueError("RBE2 master field must have exactly 6 DOFs.")
        if s.n_dofs != 3 * int(np.asarray(slave_coords).shape[0]):
            raise ValueError("RBE2 slave field size must match 3 * n_slave_nodes.")
        self.add_constraint_matrix_dof(
            C,
            master=master,
            slave=slave,
            rho=rho,
            F_contact=F_contact,
        )

    def add_rbe3_constraint(
        self,
        *,
        master: str,
        slave: str,
        ref_point: np.ndarray,
        slave_coords: np.ndarray,
        weights: np.ndarray | None = None,
        normalize_weights: bool = True,
        dependent_components=None,
        slave_components=None,
        rho: float = 0.0,
        backend: str | None = None,
        F_contact: np.ndarray | None = None,
    ) -> None:
        """
        Build and add a weighted 3D RBE3-style interpolation constraint.

        Expected field layout:
        - ``master``: 6 DOFs ordered as ``[u_ref(3), omega_ref(3)]``
        - ``slave``: 3 DOFs per node ordered as nodal translations
        """
        from ..mesh.contact import assemble_rbe3_constraint_matrix

        m = self._get_block(master)
        s = self._get_block(slave)
        C = assemble_rbe3_constraint_matrix(
            ref_point,
            slave_coords,
            weights=weights,
            normalize_weights=normalize_weights,
            dependent_components=dependent_components,
            slave_components=slave_components,
            backend=backend,
        )
        if m.n_dofs != 6:
            raise ValueError("RBE3 master field must have exactly 6 DOFs.")
        if s.n_dofs != 3 * int(np.asarray(slave_coords).shape[0]):
            raise ValueError("RBE3 slave field size must match 3 * n_slave_nodes.")
        self.add_constraint_matrix_dof(
            C,
            master=master,
            slave=slave,
            rho=rho,
            F_contact=F_contact,
        )

    def build(self) -> NumpyCoupledSystem:
        return self.system


CoupledSystem = NumpyCoupledSystem
CoupledSystemBuilder = NumpyCoupledSystemBuilder

__all__ = [
    "NumpyCoupledSystem",
    "NumpyCoupledSystemBuilder",
    "CoupledSystem",
    "CoupledSystemBuilder",
    "DirichletSpec",
    "ConstraintSpec",
]
