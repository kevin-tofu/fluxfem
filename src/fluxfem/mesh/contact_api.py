from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np

from .base import BaseMesh
from .surface import SurfaceMesh


def _warn_contact_legacy_name(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


@dataclass(frozen=True)
class ContactSide:
    surface: SurfaceMesh
    elem_conn: np.ndarray | None
    value_dim: int
    space: object | None = None

    @classmethod
    def from_facets(
        cls,
        mesh: BaseMesh,
        facets: np.ndarray,
        space=None,
        *,
        value_dim: int | None = None,
        mode: str = "touching",
    ):
        side = mesh.surface_with_elem_conn_from_facets(facets, mode=mode)
        if value_dim is None:
            if space is None:
                raise ValueError("space or value_dim is required for ContactSide.from_facets")
            value_dim = int(getattr(space, "value_dim", 1))
        return cls(surface=side.surface, elem_conn=side.elem_conn, value_dim=int(value_dim), space=space)

    @classmethod
    def from_surfaces(
        cls,
        surface: SurfaceMesh,
        *,
        elem_conn: np.ndarray | None = None,
        value_dim: int = 1,
        space: object | None = None,
    ):
        return cls(surface=surface, elem_conn=elem_conn, value_dim=int(value_dim), space=space)


@dataclass(frozen=True)
class ContactSpaces:
    """Public spec that binds contact roles to contact sides."""

    master: ContactSide
    slave: ContactSide
    field_master: str = "a"
    field_slave: str = "b"

    def __post_init__(self) -> None:
        if not str(self.field_master):
            raise ValueError("ContactSpaces.field_master must be non-empty.")
        if not str(self.field_slave):
            raise ValueError("ContactSpaces.field_slave must be non-empty.")

    def to_contact_surface_space(
        self,
        *,
        quad_order: int = 0,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ):
        _warn_contact_legacy_name("ContactSpaces.to_contact_surface_space()", "ContactPairSpec.prepare()")
        return self.prepare(
            quad_order=quad_order,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
        )

    def prepare(
        self,
        *,
        quad_order: int = 0,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ):
        from .contact import ContactSurfaceSpace

        return ContactSurfaceSpace.from_sides(
            self.master,
            self.slave,
            field_master=str(self.field_master),
            field_slave=str(self.field_slave),
            quad_order=int(quad_order),
            tol=float(tol),
            backend=backend,
            batch_jac=batch_jac,
        )


@dataclass(frozen=True)
class ContactGroupSpaces:
    """Public spec that binds one-master/many-slave contact roles."""

    master: ContactSide
    slaves: Sequence[ContactSide]
    field_master: str = "master"
    field_slave: str = "slave"

    def __post_init__(self) -> None:
        if len(self.slaves) == 0:
            raise ValueError("ContactGroupSpaces.slaves must contain at least one ContactSide.")
        if not str(self.field_master):
            raise ValueError("ContactGroupSpaces.field_master must be non-empty.")
        if not str(self.field_slave):
            raise ValueError("ContactGroupSpaces.field_slave must be non-empty.")

    def to_contact_surface_space(
        self,
        *,
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ):
        _warn_contact_legacy_name("ContactGroupSpaces.to_contact_surface_space()", "ContactGroupSpec.prepare()")
        return self.prepare(
            quad_order=quad_order,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    def prepare(
        self,
        *,
        quad_order: int = 0,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ):
        from .contact import OneToManyContactSurfaceSpace

        return OneToManyContactSurfaceSpace.from_sides(
            self.master,
            list(self.slaves),
            field_master=str(self.field_master),
            field_slave=str(self.field_slave),
            quad_order=int(quad_order),
            space_mode_master=str(space_mode_master),
            space_mode_slave=str(space_mode_slave),
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            normal_sign=normal_sign,
            tol=float(tol),
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )


@dataclass(frozen=True)
class OneSidedContactSpaces:
    """Public spec that binds one-sided contact roles to a contact side."""

    side: ContactSide
    surface_master: SurfaceMesh | None = None
    elem_conn_master: np.ndarray | None = None
    facet_to_elem_master: np.ndarray | None = None

    def to_contact_surface_space(
        self,
        *,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ):
        _warn_contact_legacy_name("OneSidedContactSpaces.to_contact_surface_space()", "OneSidedContactSpec.prepare()")
        return self.prepare(
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
        )

    def prepare(
        self,
        *,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ):
        from .contact import OneSidedContactSurfaceSpace

        return OneSidedContactSurfaceSpace.from_side(
            self.side,
            surface_master=self.surface_master,
            elem_conn_master=self.elem_conn_master,
            facet_to_elem_master=self.facet_to_elem_master,
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
        )


ContactSideSpec = ContactSide
ContactPairSpec = ContactSpaces
ContactGroupSpec = ContactGroupSpaces
OneSidedContactSpec = OneSidedContactSpaces
