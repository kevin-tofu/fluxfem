from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def infer_contact_side_facets(contact, *, side: str) -> np.ndarray | None:
    side_norm = str(side).lower()
    if side_norm not in {"master", "slave"}:
        raise ValueError("side must be 'master' or 'slave'.")

    if hasattr(contact, "surface_master") and hasattr(contact, "surface_slave"):
        surf = contact.surface_master if side_norm == "master" else contact.surface_slave
        return np.asarray(surf.conn, dtype=int)

    if hasattr(contact, "contacts") and len(getattr(contact, "contacts")) > 0:
        if side_norm == "slave":
            return None
        first = contact.contacts[0]
        if hasattr(first, "surface_master"):
            return np.asarray(first.surface_master.conn, dtype=int)
    return None


def infer_contact_multiplier_patch_ids(contact, *, family: str, side: str = "master") -> np.ndarray | None:
    fam = str(family).lower()
    if str(side).lower() != "master":
        return None
    if fam == "p0_supermesh" and hasattr(contact, "source_facets_master"):
        return np.asarray(contact.source_facets_master, dtype=int)
    if fam == "p0_active" and hasattr(contact, "source_facets_master"):
        source_facets = np.asarray(contact.source_facets_master, dtype=int)
        if source_facets.size == 0:
            return np.zeros((0,), dtype=int)
        return np.unique(source_facets)
    if fam == "p0":
        facets = infer_contact_side_facets(contact, side=side)
        if facets is not None:
            return np.arange(int(np.asarray(facets).shape[0]), dtype=int)
    return None


@dataclass(frozen=True)
class ContactMultiplierSpace:
    """Discrete LM-space description used by constraint-family contact assembly."""

    family: str = "dual_nodal"  # "dual_nodal" | "nodal" | "coarse_p1" | "p0" | "p0_active" | "p0_supermesh"
    side: str = "master"  # For p0-like families, current implementation supports only "master".
    value_dim: int = 1
    facet_conn: np.ndarray | None = None
    coarse_rank: int | None = None
    coarse_projection: np.ndarray | None = None
    coarse_mode: str | None = None
    coarse_energy_tol: float | None = None
    coarse_rtol: float | None = None
    coarse_max_rank: int | None = None
    coarse_patch_ids: np.ndarray | None = None
    coarse_basis: np.ndarray | None = None
    constraint_scaling: str = "none"

    def __post_init__(self) -> None:
        fam = str(self.family).lower()
        if fam not in {"nodal", "dual_nodal", "coarse_p1", "p0", "p0_active", "p0_supermesh"}:
            raise ValueError(
                "ContactMultiplierSpace.family must be 'nodal', 'dual_nodal', 'coarse_p1', "
                "'p0', 'p0_active', or 'p0_supermesh'."
            )
        side = str(self.side).lower()
        if side not in {"master", "slave"}:
            raise ValueError("ContactMultiplierSpace.side must be 'master' or 'slave'.")
        if fam == "coarse_p1" and side != "master":
            raise NotImplementedError(
                "coarse_p1 multipliers currently support only side='master' "
                "(coarse basis is defined in the master-side nodal space)."
            )
        if int(self.value_dim) <= 0:
            raise ValueError("ContactMultiplierSpace.value_dim must be positive.")
        if self.coarse_rank is not None and int(self.coarse_rank) <= 0:
            raise ValueError("ContactMultiplierSpace.coarse_rank must be positive when provided.")
        if self.coarse_projection is not None and np.asarray(self.coarse_projection).ndim != 2:
            raise ValueError("ContactMultiplierSpace.coarse_projection must be a 2D matrix.")
        if self.coarse_mode is not None and str(self.coarse_mode).lower() not in {
            "qr",
            "svd",
            "auto",
            "algebraic_qr",
            "patch_qr",
            "row_qr",
            "pivoted_qr",
        }:
            raise ValueError(
                "ContactMultiplierSpace.coarse_mode must be 'qr', 'svd', 'auto', "
                "'algebraic_qr', 'patch_qr', 'row_qr', or 'pivoted_qr'."
            )
        if self.coarse_energy_tol is not None and not (0.0 < float(self.coarse_energy_tol) <= 1.0):
            raise ValueError("ContactMultiplierSpace.coarse_energy_tol must be in (0, 1].")
        if self.coarse_rtol is not None and float(self.coarse_rtol) < 0.0:
            raise ValueError("ContactMultiplierSpace.coarse_rtol must be non-negative.")
        if self.coarse_max_rank is not None and int(self.coarse_max_rank) <= 0:
            raise ValueError("ContactMultiplierSpace.coarse_max_rank must be positive when provided.")
        if self.coarse_patch_ids is not None:
            patch_ids = np.asarray(self.coarse_patch_ids, dtype=int).reshape(-1)
            if patch_ids.size == 0:
                raise ValueError("ContactMultiplierSpace.coarse_patch_ids must be non-empty when provided.")
            if np.any(patch_ids < 0):
                raise ValueError("ContactMultiplierSpace.coarse_patch_ids must not contain negative ids.")
            if fam not in {"p0", "p0_active", "p0_supermesh"}:
                raise ValueError("ContactMultiplierSpace.coarse_patch_ids are supported only for p0-like families.")
        if self.coarse_basis is not None:
            basis = np.asarray(self.coarse_basis, dtype=float)
            if basis.ndim != 2:
                raise ValueError("ContactMultiplierSpace.coarse_basis must be a 2D matrix.")
            if basis.shape[0] <= 0 or basis.shape[1] <= 0:
                raise ValueError("ContactMultiplierSpace.coarse_basis must be non-empty.")
            if fam != "coarse_p1":
                raise ValueError("ContactMultiplierSpace.coarse_basis is supported only for family='coarse_p1'.")
        if fam == "coarse_p1" and self.coarse_basis is None:
            raise ValueError("ContactMultiplierSpace.coarse_basis is required when family='coarse_p1'.")
        if str(self.constraint_scaling).lower() not in {"none", "l2"}:
            raise ValueError("ContactMultiplierSpace.constraint_scaling must be 'none' or 'l2'.")

    @classmethod
    def from_contact(
        cls,
        contact,
        *,
        family: str = "dual_nodal",
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        coarse_rank: int | None = None,
        coarse_projection: np.ndarray | None = None,
        coarse_mode: str | None = None,
        coarse_energy_tol: float | None = None,
        coarse_rtol: float | None = None,
        coarse_max_rank: int | None = None,
        coarse_patch_ids: np.ndarray | None = None,
        coarse_basis: np.ndarray | None = None,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        fc = None if facet_conn is None else np.asarray(facet_conn, dtype=int)
        if str(family).lower() in {"p0", "p0_active", "p0_supermesh"} and fc is None:
            fc = infer_contact_side_facets(contact, side=str(side))
        return cls(
            family=str(family).lower(),
            side=str(side).lower(),
            value_dim=int(value_dim),
            facet_conn=fc,
            coarse_rank=None if coarse_rank is None else int(coarse_rank),
            coarse_projection=None if coarse_projection is None else np.asarray(coarse_projection, dtype=float),
            coarse_mode=None if coarse_mode is None else str(coarse_mode).lower(),
            coarse_energy_tol=None if coarse_energy_tol is None else float(coarse_energy_tol),
            coarse_rtol=None if coarse_rtol is None else float(coarse_rtol),
            coarse_max_rank=None if coarse_max_rank is None else int(coarse_max_rank),
            coarse_patch_ids=None if coarse_patch_ids is None else np.asarray(coarse_patch_ids, dtype=int),
            coarse_basis=None if coarse_basis is None else np.asarray(coarse_basis, dtype=float),
            constraint_scaling=str(constraint_scaling).lower(),
        )

    @classmethod
    def dual_mortar(
        cls,
        *,
        side: str = "master",
        value_dim: int = 1,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        return cls(
            family="dual_nodal",
            side=side,
            value_dim=int(value_dim),
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def nodal_mortar(
        cls,
        *,
        side: str = "master",
        value_dim: int = 1,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        return cls(
            family="nodal",
            side=side,
            value_dim=int(value_dim),
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def coarse_dual_mortar(
        cls,
        *,
        mode: str = "auto",
        rank: int | None = None,
        energy_tol: float = 0.999,
        rtol: float = 1e-10,
        max_rank: int | None = None,
        projection: np.ndarray | None = None,
        side: str = "master",
        value_dim: int = 1,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        coarse_mode = "qr" if rank is not None and str(mode).lower() == "auto" else str(mode).lower()
        return cls(
            family="dual_nodal",
            side=side,
            value_dim=int(value_dim),
            coarse_rank=None if rank is None else int(rank),
            coarse_projection=None if projection is None else np.asarray(projection, dtype=float),
            coarse_mode=coarse_mode,
            coarse_energy_tol=float(energy_tol),
            coarse_rtol=float(rtol),
            coarse_max_rank=None if max_rank is None else int(max_rank),
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def p0_mortar(
        cls,
        contact=None,
        *,
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        family: str = "p0",
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        if contact is None and facet_conn is None:
            raise ValueError("p0_mortar requires contact or facet_conn.")
        return cls.from_contact(
            contact,
            family=family,
            side=side,
            value_dim=value_dim,
            facet_conn=facet_conn,
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def coarse_p0_mortar(
        cls,
        contact=None,
        *,
        patch_ids: np.ndarray,
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        family: str = "p0",
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        """Facet-integrated coarse P0 mortar grouped by patch ids."""

        if contact is None and facet_conn is None:
            raise ValueError("coarse_p0_mortar requires contact or facet_conn.")
        return cls.from_contact(
            contact,
            family=family,
            side=side,
            value_dim=value_dim,
            facet_conn=facet_conn,
            coarse_patch_ids=np.asarray(patch_ids, dtype=int),
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def algebraic_qr_mortar(
        cls,
        contact=None,
        *,
        family: str = "p0_supermesh",
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        rtol: float = 1e-10,
        max_rank: int | None = None,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        """Rank-revealing row-selected mortar using pivoted QR on ``B.T``."""

        return cls.from_contact(
            contact,
            family=family,
            side=side,
            value_dim=value_dim,
            facet_conn=facet_conn,
            coarse_mode="algebraic_qr",
            coarse_rtol=float(rtol),
            coarse_max_rank=None if max_rank is None else int(max_rank),
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def patch_qr_mortar(
        cls,
        contact=None,
        *,
        patch_ids: np.ndarray | None = None,
        family: str = "p0_supermesh",
        side: str = "master",
        value_dim: int = 1,
        facet_conn: np.ndarray | None = None,
        rtol: float = 1e-10,
        max_rank: int | None = None,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        """Patch-local row-selected mortar using pivoted QR on each patch block."""

        patch_arr = patch_ids
        if patch_arr is None and contact is not None:
            patch_arr = infer_contact_multiplier_patch_ids(contact, family=str(family), side=side)
        if patch_arr is None:
            raise ValueError("patch_qr_mortar requires patch_ids or a contact with inferable row patches.")
        return cls.from_contact(
            contact,
            family=family,
            side=side,
            value_dim=value_dim,
            facet_conn=facet_conn,
            coarse_mode="patch_qr",
            coarse_rtol=float(rtol),
            coarse_max_rank=None if max_rank is None else int(max_rank),
            coarse_patch_ids=np.asarray(patch_arr, dtype=int),
            constraint_scaling=constraint_scaling,
        )

    @classmethod
    def coarse_p1_mortar(
        cls,
        *,
        basis: np.ndarray,
        side: str = "master",
        value_dim: int = 1,
        constraint_scaling: str = "none",
    ) -> "ContactMultiplierSpace":
        """Integrated coarse P1 mortar from coarse master-side nodal basis rows.

        ``basis`` has shape ``(n_coarse_nodes, n_master_nodes)``.  Each row is a
        coarse multiplier shape function represented in the fine master nodal
        basis, and the assembled rows are ``basis @ M_aa`` and ``basis @ M_ab``.
        """

        return cls(
            family="coarse_p1",
            side=side,
            value_dim=int(value_dim),
            coarse_basis=np.asarray(basis, dtype=float),
            constraint_scaling=constraint_scaling,
        )


MultiplierSpec = ContactMultiplierSpace

