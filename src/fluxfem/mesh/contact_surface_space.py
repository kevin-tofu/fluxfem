from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast
import warnings

import numpy as np
import numpy.typing as npt

from .base import BaseMesh
from .contact_api import ContactSide
from .contact_interface import (
    assemble_contact_coupling_matrices as _assemble_contact_coupling_matrices,
    assemble_contact_interface_jacobian as _assemble_contact_interface_jacobian,
    assemble_contact_interface_residual as _assemble_contact_interface_residual,
    assemble_onesided_bilinear,
    build_supermesh_triangle_quadrature_cache,
    map_surface_facets_to_hex_elements,
    map_surface_facets_to_tet_elements,
)
from .contact_surface_helpers import (
    contact_space_side_n_dofs,
    facet_map_for_elem_conn,
    summarize_contact_field_state,
)
from .supermesh import build_surface_supermesh
from .surface import SurfaceMesh

if TYPE_CHECKING:
    from .contact import (
        ContactBilinearLike,
        ContactJacobianReturn,
        ContactOperators,
        ContactState,
        MixedSurfaceResidualForm,
        PenaltyContactContribution,
    )
    from .mortar_multiplier import ContactMultiplierSpace
    from ..core.weakform import Params as WeakParams


_CONTACT_SETUP_CACHE: dict[tuple, "ContactSurfaceSpace"] = {}


def _warn_contact_legacy_name(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


@dataclass(eq=False)
class OneSidedContactSurfaceSpace:
    """Surface wrapper for one-sided (Dirichlet) contact assembly."""

    surface_slave: SurfaceMesh
    elem_conn_slave: np.ndarray
    facet_to_elem_slave: np.ndarray
    value_dim: int = 1
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    surface_master: SurfaceMesh | None = None
    elem_conn_master: np.ndarray | None = None
    facet_to_elem_master: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        surface_master: SurfaceMesh | None = None,
        elem_conn_master: np.ndarray | None = None,
        facet_to_elem_master: np.ndarray | None = None,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
    ) -> "OneSidedContactSurfaceSpace":
        if side.elem_conn is None:
            raise ValueError("side.elem_conn is required for one-sided assembly")
        facet_map_slave = facet_map_for_elem_conn(side.surface, side.elem_conn)
        facet_map_master = facet_to_elem_master
        if surface_master is not None and elem_conn_master is not None and facet_map_master is None:
            facet_map_master = facet_map_for_elem_conn(surface_master, elem_conn_master)
        return cls(
            surface_slave=side.surface,
            elem_conn_slave=np.asarray(side.elem_conn, dtype=int),
            facet_to_elem_slave=np.asarray(facet_map_slave, dtype=int),
            value_dim=int(side.value_dim),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            surface_master=surface_master,
            elem_conn_master=None if elem_conn_master is None else np.asarray(elem_conn_master, dtype=int),
            facet_to_elem_master=None if facet_map_master is None else np.asarray(facet_map_master, dtype=int),
        )

    @classmethod
    def from_facets(
        cls,
        mesh: BaseMesh,
        facets: np.ndarray,
        space=None,
        *,
        surface_master: SurfaceMesh | None = None,
        elem_conn_master: np.ndarray | None = None,
        facet_to_elem_master: np.ndarray | None = None,
        value_dim: int | None = None,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        mode: str = "touching",
    ) -> "OneSidedContactSurfaceSpace":
        side = ContactSide.from_facets(mesh, facets, space, value_dim=value_dim, mode=mode)
        return cls.from_side(
            side,
            surface_master=surface_master,
            elem_conn_master=elem_conn_master,
            facet_to_elem_master=facet_to_elem_master,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
        )

    def initialize_state(self, *, metadata: Mapping[str, Any] | None = None):
        from .contact import ContactState

        return ContactState(
            interface_kind="one_sided",
            geometry="reference",
            iteration=0,
            active_set=None,
            field_summary={"slave": (int(self.elem_conn_slave.max()) + 1, int(self.value_dim))},
            metadata=dict(metadata or {}),
        )

    def update_state(
        self,
        *,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        contact_state: Any | None = None,
        geometry: str = "current",
        active_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        base = self.initialize_state() if contact_state is None else contact_state
        merged_metadata = dict(base.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return replace(
            base,
            geometry=str(geometry),
            iteration=int(base.iteration) + 1,
            active_set=active_set if active_set is not None else base.active_set,
            field_summary=summarize_contact_field_state(state),
            metadata=merged_metadata,
        )

    def assemble_bilinear(
        self,
        u_hat_fn: Any | None,
        params: Any,
        *,
        u_master: np.ndarray | None = None,
        grad_source: str = "volume",
        dof_source: str = "volume",
        quad_order: int | None = None,
        normal_sign: float | None = None,
        tol: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return assemble_onesided_bilinear(
            self.surface_slave,
            u_hat_fn,
            params,
            surface_master=self.surface_master,
            u_master=u_master,
            value_dim=self.value_dim,
            elem_conn=self.elem_conn_slave,
            facet_to_elem=self.facet_to_elem_slave,
            elem_conn_master=self.elem_conn_master,
            facet_to_elem_master=self.facet_to_elem_master,
            grad_source=grad_source,
            dof_source=dof_source,
            quad_order=self.quad_order if quad_order is None else int(quad_order),
            normal_sign=self.normal_sign if normal_sign is None else float(normal_sign),
            tol=self.tol if tol is None else float(tol),
        )


@dataclass(eq=False)
class ContactSurfaceSpace:
    """Surface interface wrapper for contact assembly on a supermesh."""

    surface_master: SurfaceMesh
    surface_slave: SurfaceMesh
    supermesh_coords: np.ndarray
    supermesh_conn: np.ndarray
    source_facets_master: np.ndarray
    source_facets_slave: np.ndarray
    elem_conn_master: np.ndarray | None
    elem_conn_slave: np.ndarray | None
    facet_to_elem_master: np.ndarray | None
    facet_to_elem_slave: np.ndarray | None
    field_master: str = "a"
    field_slave: str = "b"
    value_dim_master: int = 1
    value_dim_slave: int = 1
    space_mode_master: str = "nodal"
    space_mode_slave: str = "nodal"
    facet_dofs_master: np.ndarray | None = None
    facet_dofs_slave: np.ndarray | None = None
    trial_value_dim_master: int | None = None
    trial_value_dim_slave: int | None = None
    trial_space_mode_master: str | None = None
    trial_space_mode_slave: str | None = None
    trial_facet_dofs_master: np.ndarray | None = None
    trial_facet_dofs_slave: np.ndarray | None = None
    quad_order: int = 0
    normal_sign: float | None = None
    tol: float = 1e-8
    backend: str = "jax"
    batch_jac: bool | None = None
    supermesh_quad_cache: Any | None = None
    _compiled_bilinear_cache: dict[tuple[int, str], MixedSurfaceResidualForm] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_surfaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slave: SurfaceMesh,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        trial_value_dim_master: int | None = None,
        trial_value_dim_slave: int | None = None,
        trial_space_mode_master: str | None = None,
        trial_space_mode_slave: str | None = None,
        trial_facet_dofs_master: np.ndarray | None = None,
        trial_facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        import hashlib
        import os
        backend = "jax" if backend is None else str(backend).lower()

        if setup_cache_enabled is None:
            setup_cache_enabled = os.getenv("FLUXFEM_CONTACT_SETUP_CACHE", "0") not in ("0", "", "false", "False")
        if setup_cache_trace is None:
            setup_cache_trace = os.getenv("FLUXFEM_CONTACT_SETUP_CACHE_TRACE", "0") not in ("0", "", "false", "False")

        def _array_sig(arr: np.ndarray) -> tuple:
            arr_c = np.ascontiguousarray(arr)
            h = hashlib.blake2b(arr_c.view(np.uint8), digest_size=8).hexdigest()
            return (arr_c.shape, str(arr_c.dtype), h)

        if setup_cache_enabled:
            global _CONTACT_SETUP_CACHE
            try:
                _CONTACT_SETUP_CACHE
            except NameError:
                _CONTACT_SETUP_CACHE = {}
            key = (
                _array_sig(np.asarray(surface_master.coords)),
                _array_sig(np.asarray(surface_master.conn)),
                _array_sig(np.asarray(surface_slave.coords)),
                _array_sig(np.asarray(surface_slave.conn)),
                None if elem_conn_master is None else _array_sig(np.asarray(elem_conn_master)),
                None if elem_conn_slave is None else _array_sig(np.asarray(elem_conn_slave)),
                field_master,
                field_slave,
                int(value_dim_master),
                int(value_dim_slave),
                str(space_mode_master),
                str(space_mode_slave),
                None if facet_dofs_master is None else _array_sig(np.asarray(facet_dofs_master)),
                None if facet_dofs_slave is None else _array_sig(np.asarray(facet_dofs_slave)),
                None if trial_value_dim_master is None else int(trial_value_dim_master),
                None if trial_value_dim_slave is None else int(trial_value_dim_slave),
                None if trial_space_mode_master is None else str(trial_space_mode_master),
                None if trial_space_mode_slave is None else str(trial_space_mode_slave),
                None if trial_facet_dofs_master is None else _array_sig(np.asarray(trial_facet_dofs_master)),
                None if trial_facet_dofs_slave is None else _array_sig(np.asarray(trial_facet_dofs_slave)),
                int(quad_order),
                float(normal_sign) if normal_sign is not None else None,
                float(tol),
                backend,
                bool(batch_jac) if batch_jac is not None else None,
            )
            cached = _CONTACT_SETUP_CACHE.get(key)
            if cached is not None:
                if setup_cache_trace:
                    print(
                        f"[contact] setup cache hit n_tris={int(cached.supermesh_conn.shape[0])}",
                        flush=True,
                    )
                return cached

        sm = build_surface_supermesh(surface_master, surface_slave, tol=tol)
        facet_map_master = None
        facet_map_slave = None
        if elem_conn_master is not None:
            if elem_conn_master.shape[1] in {4, 10}:
                facet_map_master = map_surface_facets_to_tet_elements(surface_master, elem_conn_master)
            elif elem_conn_master.shape[1] in {8, 20, 27}:
                facet_map_master = map_surface_facets_to_hex_elements(surface_master, elem_conn_master)
            else:
                raise NotImplementedError("elem_conn_master must be tet4/tet10/hex8/hex20/hex27")
        if elem_conn_slave is not None:
            if elem_conn_slave.shape[1] in {4, 10}:
                facet_map_slave = map_surface_facets_to_tet_elements(surface_slave, elem_conn_slave)
            elif elem_conn_slave.shape[1] in {8, 20, 27}:
                facet_map_slave = map_surface_facets_to_hex_elements(surface_slave, elem_conn_slave)
            else:
                raise NotImplementedError("elem_conn_slave must be tet4/tet10/hex8/hex20/hex27")
        obj = cls(
            surface_master=surface_master,
            surface_slave=surface_slave,
            supermesh_coords=sm.coords,
            supermesh_conn=sm.conn,
            source_facets_master=sm.source_facets_a,
            source_facets_slave=sm.source_facets_b,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            facet_to_elem_master=facet_map_master,
            facet_to_elem_slave=facet_map_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=None if facet_dofs_master is None else np.asarray(facet_dofs_master, dtype=int),
            facet_dofs_slave=None if facet_dofs_slave is None else np.asarray(facet_dofs_slave, dtype=int),
            trial_value_dim_master=None if trial_value_dim_master is None else int(trial_value_dim_master),
            trial_value_dim_slave=None if trial_value_dim_slave is None else int(trial_value_dim_slave),
            trial_space_mode_master=None if trial_space_mode_master is None else str(trial_space_mode_master),
            trial_space_mode_slave=None if trial_space_mode_slave is None else str(trial_space_mode_slave),
            trial_facet_dofs_master=None if trial_facet_dofs_master is None else np.asarray(trial_facet_dofs_master, dtype=int),
            trial_facet_dofs_slave=None if trial_facet_dofs_slave is None else np.asarray(trial_facet_dofs_slave, dtype=int),
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            supermesh_quad_cache=build_supermesh_triangle_quadrature_cache(
                sm.coords,
                sm.conn,
                quad_order=int(quad_order),
                tol=float(tol),
            ),
        )
        if setup_cache_enabled:
            _CONTACT_SETUP_CACHE[key] = obj
            if setup_cache_trace:
                print(
                    f"[contact] setup cache store n_tris={int(obj.supermesh_conn.shape[0])}",
                    flush=True,
                )
        return obj

    @classmethod
    def from_facets(
        cls,
        coords: np.ndarray,
        facets: np.ndarray,
        *,
        elem_conn: np.ndarray | None = None,
        value_dim: int = 1,
        space_mode: str = "nodal",
        facet_dofs: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        surface = SurfaceMesh.from_facets(coords, facets)
        return cls.from_surfaces(
            surface,
            surface,
            elem_conn_master=elem_conn,
            elem_conn_slave=elem_conn,
            value_dim_master=value_dim,
            value_dim_slave=value_dim,
            space_mode_master=space_mode,
            space_mode_slave=space_mode,
            facet_dofs_master=facet_dofs,
            facet_dofs_slave=facet_dofs,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod
    def from_surfaces_and_spaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slave: SurfaceMesh,
        space_master,
        space_slave,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int | None = None,
        value_dim_slave: int | None = None,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> "ContactSurfaceSpace":
        if value_dim_master is None:
            value_dim_master = int(getattr(space_master, "value_dim", 1))
        if value_dim_slave is None:
            value_dim_slave = int(getattr(space_slave, "value_dim", 1))
        return cls.from_surfaces(
            surface_master,
            surface_slave,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
        )

    @classmethod
    def from_sides(
        cls,
        master: ContactSide,
        slave: ContactSide,
        *,
        field_master: str = "a",
        field_slave: str = "b",
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
    ) -> "ContactSurfaceSpace":
        return cls.from_surfaces(
            master.surface,
            slave.surface,
            elem_conn_master=master.elem_conn,
            elem_conn_slave=slave.elem_conn,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=master.value_dim,
            value_dim_slave=slave.value_dim,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    @classmethod  # type: ignore[no-redef]
    def from_facets(
        cls,
        coords_master: np.ndarray,
        facets_master: np.ndarray,
        coords_slave: np.ndarray,
        facets_slave: np.ndarray,
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slave: np.ndarray | None = None,
        field_master: str = "a",
        field_slave: str = "b",
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        space_mode_master: str = "nodal",
        space_mode_slave: str = "nodal",
        facet_dofs_master: np.ndarray | None = None,
        facet_dofs_slave: np.ndarray | None = None,
        quad_order: int = 0,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str | None = None,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        surface_master = SurfaceMesh.from_facets(coords_master, facets_master)
        surface_slave = SurfaceMesh.from_facets(coords_slave, facets_slave)
        return cls.from_surfaces(
            surface_master,
            surface_slave,
            elem_conn_master=elem_conn_master,
            elem_conn_slave=elem_conn_slave,
            field_master=field_master,
            field_slave=field_slave,
            value_dim_master=value_dim_master,
            value_dim_slave=value_dim_slave,
            space_mode_master=space_mode_master,
            space_mode_slave=space_mode_slave,
            facet_dofs_master=facet_dofs_master,
            facet_dofs_slave=facet_dofs_slave,
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            batch_jac=batch_jac,
            setup_cache_enabled=setup_cache_enabled,
            setup_cache_trace=setup_cache_trace,
        )

    def _split_fields(self, u: Mapping[str, np.ndarray] | Sequence[np.ndarray]):
        if isinstance(u, Mapping):
            return u[self.field_master], u[self.field_slave]
        if len(u) != 2:
            raise ValueError("u must be a mapping or a length-2 sequence")
        return u[0], u[1]

    def _auto_normal_sign(self) -> float:
        if not hasattr(self.surface_master, "facet_normals"):
            return 1.0
        normals = self.surface_master.facet_normals()
        coords = np.asarray(self.surface_master.coords)
        coords_slave = np.asarray(self.surface_slave.coords)
        facets_m = np.asarray(self.surface_master.conn, dtype=int)
        facets_s = np.asarray(self.surface_slave.conn, dtype=int)
        dots = []
        for fa, fb in zip(self.source_facets_master, self.source_facets_slave):
            n = normals[int(fa)]
            cm = np.mean(coords[facets_m[int(fa)]], axis=0)
            cs = np.mean(coords_slave[facets_s[int(fb)]], axis=0)
            dots.append(float(np.dot(n, cs - cm)))
        if not dots:
            return 1.0
        return 1.0 if np.sum(dots) >= 0.0 else -1.0

    def _resolve_backend(self, backend: str | None) -> str:
        use_backend = self.backend if backend is None else backend
        if use_backend not in {"jax", "numpy"}:
            raise ValueError("backend must be 'jax' or 'numpy'")
        return use_backend

    def _trial_layout(self, *, side: str) -> tuple[int, str, np.ndarray | None]:
        if side == "master":
            value_dim = int(self.trial_value_dim_master or self.value_dim_master)
            space_mode = str(self.trial_space_mode_master or self.space_mode_master)
            facet_dofs = self.trial_facet_dofs_master if self.trial_facet_dofs_master is not None else self.facet_dofs_master
            return value_dim, space_mode, facet_dofs
        if side == "slave":
            value_dim = int(self.trial_value_dim_slave or self.value_dim_slave)
            space_mode = str(self.trial_space_mode_slave or self.space_mode_slave)
            facet_dofs = self.trial_facet_dofs_slave if self.trial_facet_dofs_slave is not None else self.facet_dofs_slave
            return value_dim, space_mode, facet_dofs
        raise ValueError("side must be 'master' or 'slave'")

    def _validate_square_trial_layout(self) -> None:
        test_master = contact_space_side_n_dofs(self, side="master", role="test")
        test_slave = contact_space_side_n_dofs(self, side="slave", role="test")
        trial_master = contact_space_side_n_dofs(self, side="master", role="trial")
        trial_slave = contact_space_side_n_dofs(self, side="slave", role="trial")
        if test_master != trial_master or test_slave != trial_slave:
            raise NotImplementedError(
                "Distinct contact trial layouts currently require the same total DOF counts as the test layouts. "
                "Rectangular contact operators are not enabled yet."
            )

    def initialize_state(self, *, metadata: Mapping[str, Any] | None = None) -> ContactState:
        from .contact import ContactState

        return ContactState(
            interface_kind="pair",
            geometry="reference",
            iteration=0,
            active_set=None,
            field_summary={
                self.field_master: (contact_space_side_n_dofs(self, side="master", role="trial"),),
                self.field_slave: (contact_space_side_n_dofs(self, side="slave", role="trial"),),
            },
            metadata=dict(metadata or {}),
        )

    def update_state(
        self,
        *,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        contact_state: ContactState | None = None,
        geometry: str = "current",
        active_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContactState:
        base = self.initialize_state() if contact_state is None else contact_state
        merged_metadata = dict(base.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return replace(
            base,
            geometry=str(geometry),
            iteration=int(base.iteration) + 1,
            active_set=active_set if active_set is not None else base.active_set,
            field_summary=summarize_contact_field_state(state),
            metadata=merged_metadata,
        )

    def assemble_contact_coupling_matrices(self) -> tuple["ContactCouplingMatrix", "ContactCouplingMatrix"]:
        """Return (M_aa, M_ab) coupling matrices on this contact interface."""
        return _assemble_contact_coupling_matrices(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            tol=self.tol,
            quad_order=self.quad_order,
        )

    def assemble_contact_kkt(
        self,
        *,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        format: str = "fluxsparse",
        return_blocks: bool = False,
    ):
        m_aa, m_ab = self.assemble_contact_coupling_matrices()
        from .contact import assemble_contact_kkt

        return assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=rho,
            multiplier=multiplier,
            facet_conn_master=np.asarray(self.surface_master.conn, dtype=int),
            backend=backend,
            format=format,
            return_blocks=return_blocks,
        )

    def assemble_contact_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        from .contact import assemble_contact_constraint_operators

        return assemble_contact_constraint_operators(
            self,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Legacy alias for assemble_multiplier()."""
        _warn_contact_legacy_name("PreparedContactInterface.assemble_constraint_operators()", "PreparedContactInterface.assemble_multiplier()")
        return self.assemble_contact_constraint_operators(
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_multiplier(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Preferred public alias for assemble_contact_constraint_operators()."""
        return self.assemble_contact_constraint_operators(
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        from .contact import assemble_contact_penalty_operators

        return assemble_contact_penalty_operators(
            self,
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Legacy alias for assemble_penalty()."""
        _warn_contact_legacy_name("PreparedContactInterface.assemble_penalty_operators()", "PreparedContactInterface.assemble_penalty()")
        return self.assemble_contact_penalty_operators(
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_penalty(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Preferred public alias for assemble_contact_penalty_operators()."""
        return self.assemble_contact_penalty_operators(
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_operators(
        self,
        *,
        enforcement: str | None = None,
        method: str | None = None,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: ContactMultiplierSpace | None = None,
        backend: str | None = None,
        weak_form: MixedSurfaceResidualForm | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        res_form: MixedSurfaceResidualForm | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> ContactOperators:
        """Unified public alias that routes to penalty or constraint assembly."""
        from .contact import assemble_contact_operators

        return assemble_contact_operators(
            self,
            enforcement=enforcement,
            method=method,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_residual(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
        params: "WeakParams",
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
    ) -> np.ndarray:
        self._validate_square_trial_layout()
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        trial_value_dim_master, trial_space_mode_master, trial_facet_dofs_master = self._trial_layout(side="master")
        trial_value_dim_slave, trial_space_mode_slave, trial_facet_dofs_slave = self._trial_layout(side="slave")
        return _assemble_contact_interface_residual(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            res_form,
            u_master,
            u_slave,
            params,
            value_dim_a=self.value_dim_master,
            value_dim_b=self.value_dim_slave,
            trial_value_dim_a=trial_value_dim_master,
            trial_value_dim_b=trial_value_dim_slave,
            space_mode_a=self.space_mode_master,
            space_mode_b=self.space_mode_slave,
            trial_space_mode_a=trial_space_mode_master,
            trial_space_mode_b=trial_space_mode_slave,
            facet_dofs_a=self.facet_dofs_master,
            facet_dofs_b=self.facet_dofs_slave,
            trial_facet_dofs_a=trial_facet_dofs_master,
            trial_facet_dofs_b=trial_facet_dofs_slave,
            field_a=self.field_master,
            field_b=self.field_slave,
            elem_conn_a=self.elem_conn_master,
            elem_conn_b=self.elem_conn_slave,
            facet_to_elem_a=self.facet_to_elem_master,
            facet_to_elem_b=self.facet_to_elem_slave,
            normal_source=normal_source,
            normal_from="master",
            master_field=self.field_master,
            normal_sign=normal_sign,
            grad_source="volume",
            dof_source="volume",
            quad_order=self.quad_order,
            tol=self.tol,
        )

    def assemble_jacobian(
        self,
        res_form: MixedSurfaceResidualForm,
        u: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
        params: "WeakParams",
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
        sparse: bool = True,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> ContactJacobianReturn:
        self._validate_square_trial_layout()
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        use_backend = self._resolve_backend(backend)
        use_batch_jac = self.batch_jac if batch_jac is None else batch_jac
        trial_value_dim_master, trial_space_mode_master, trial_facet_dofs_master = self._trial_layout(side="master")
        trial_value_dim_slave, trial_space_mode_slave, trial_facet_dofs_slave = self._trial_layout(side="slave")
        return _assemble_contact_interface_jacobian(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
            res_form,
            u_master,
            u_slave,
            params,
            value_dim_a=self.value_dim_master,
            value_dim_b=self.value_dim_slave,
            trial_value_dim_a=trial_value_dim_master,
            trial_value_dim_b=trial_value_dim_slave,
            space_mode_a=self.space_mode_master,
            space_mode_b=self.space_mode_slave,
            trial_space_mode_a=trial_space_mode_master,
            trial_space_mode_b=trial_space_mode_slave,
            facet_dofs_a=self.facet_dofs_master,
            facet_dofs_b=self.facet_dofs_slave,
            trial_facet_dofs_a=trial_facet_dofs_master,
            trial_facet_dofs_b=trial_facet_dofs_slave,
            field_a=self.field_master,
            field_b=self.field_slave,
            elem_conn_a=self.elem_conn_master,
            elem_conn_b=self.elem_conn_slave,
            facet_to_elem_a=self.facet_to_elem_master,
            facet_to_elem_b=self.facet_to_elem_slave,
            normal_source=normal_source,
            normal_from="master",
            master_field=self.field_master,
            normal_sign=normal_sign,
            grad_source="volume",
            dof_source="volume",
            quad_order=self.quad_order,
            tol=self.tol,
            sparse=sparse,
            backend=use_backend,
            batch_jac=use_batch_jac,
            supermesh_quad_cache=self.supermesh_quad_cache,
        )

    def compile_bilinear(
        self,
        bilin: ContactBilinearLike,
        *,
        backend: str | None = None,
        use_cache: bool = True,
    ) -> MixedSurfaceResidualForm:
        """Compile a contact bilinear callable to a reusable mixed-surface residual form."""
        from .contact import _compile_contact_bilinear, _is_compiled_contact_bilinear

        if _is_compiled_contact_bilinear(bilin):
            return cast("MixedSurfaceResidualForm", bilin)
        use_backend = self._resolve_backend(backend)
        cache_key = (id(bilin), use_backend)
        if use_cache:
            cached = self._compiled_bilinear_cache.get(cache_key)
            if cached is not None:
                return cached
        res_form = _compile_contact_bilinear(
            bilin,
            field_master=self.field_master,
            field_slave=self.field_slave,
            backend=use_backend,
        )
        if use_cache:
            self._compiled_bilinear_cache[cache_key] = res_form
        return res_form

    def assemble_bilinear(
        self,
        bilin: ContactBilinearLike,
        u_master: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | npt.ArrayLike,
        u_slave: npt.ArrayLike | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        """
        Assemble a mixed surface bilinear form with signature (v1, v2, u1, u2, params).

        Notes:
        - v1/v2/u1/u2 are symbolic field refs; use .val/.grad/.sym_grad in the expression.
        - The bilinear must be linear in v1 and v2 and include ds() in its expression.
        - When building dot products, prefer dot(v1, ...) and dot(v2, ...) to keep shapes consistent.
        - Normal orientation, grad_source, and dof_source are fixed internally for simplicity.
        - u_master/u_slave can be passed as a single mapping/length-2 sequence; in that case,
          pass params as the next positional arg or a keyword.
        """
        def _is_field_pair(obj) -> bool:
            if isinstance(obj, Mapping):
                return True
            return isinstance(obj, Sequence) and not hasattr(obj, "shape")

        if params is None:
            if u_slave is None:
                raise TypeError("params is required")
            if _is_field_pair(u_master):
                params = u_slave
                u_master, u_slave = self._split_fields(u_master)
            else:
                raise TypeError("params is required")
        elif u_slave is None:
            u_master, u_slave = self._split_fields(u_master)

        use_backend = self._resolve_backend(None)
        res_form = self.compile_bilinear(bilin, backend=use_backend)
        return self.assemble_jacobian(
            res_form,
            {self.field_master: u_master, self.field_slave: u_slave},
            params,
            normal_sign=None,
            normal_source=normal_source,
            sparse=sparse,
            backend=use_backend,
        )

    def assemble_bilinear_form(
        self,
        bilin: ContactBilinearLike,
        params: "WeakParams",
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> ContactJacobianReturn:
        """Assemble an interface bilinear form without requiring a state vector."""
        n_master = contact_space_side_n_dofs(self, side="master", role="trial")
        n_slave = contact_space_side_n_dofs(self, side="slave", role="trial")
        u_master = np.zeros((n_master,), dtype=float)
        u_slave = np.zeros((n_slave,), dtype=float)
        return self.assemble_bilinear(
            bilin,
            u_master,
            u_slave,
            params,
            sparse=sparse,
            normal_source=normal_source,
        )

    def assemble_pair_nitsche(
        self,
        params: "WeakParams",
        *,
        sparse: bool = False,
        normal_source: str = "master",
        use_penalty: float | None = None,
        use_traction: float | None = None,
        backend_fastpath: str = "numpy_local_kernel",
    ) -> PenaltyContactContribution:
        """Assemble pair-Nitsche terms over this prepared contact supermesh."""
        from .contact import assemble_pair_nitsche_supermesh

        return assemble_pair_nitsche_supermesh(
            self,
            params,
            sparse=sparse,
            normal_source=normal_source,
            use_penalty=use_penalty,
            use_traction=use_traction,
            backend_fastpath=backend_fastpath,
        )


@dataclass(eq=False)
class OneToManyContactSurfaceSpace:
    """One-master/multi-slave wrapper built from pairwise ContactSurfaceSpace objects."""

    contacts: tuple["ContactSurfaceSpace", ...]
    field_master: str = "master"
    field_slave: str = "slave"
    _compiled_bilinear_cache: dict[tuple[int, str], "MixedSurfaceResidualForm"] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_meshes(
        cls,
        master_mesh: BaseMesh,
        slave_meshes: Sequence[BaseMesh],
        *,
        master_facets: np.ndarray | None = None,
        slave_facets_list: Sequence[np.ndarray] | None = None,
        master_facet_selector: Callable[[BaseMesh], np.ndarray] | None = None,
        slave_facet_selectors: Sequence[Callable[[BaseMesh], np.ndarray] | None] | Callable[[BaseMesh], np.ndarray] | None = None,
        master_space: object | None = None,
        slave_spaces: Sequence[object | None] | object | None = None,
        value_dim_master: int | None = None,
        value_dim_slaves: Sequence[int | None] | int | None = None,
        mode_master: str = "touching",
        mode_slave: str = "touching",
        field_master: str = "master",
        field_slave: str = "slave",
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
    ) -> "OneToManyContactSurfaceSpace":
        if len(slave_meshes) == 0:
            raise ValueError("slave_meshes must contain at least one mesh.")
        n_slaves = len(slave_meshes)

        if master_facets is None:
            if master_facet_selector is None:
                raise ValueError("Provide either master_facets or master_facet_selector.")
            master_facets = np.asarray(master_facet_selector(master_mesh), dtype=int)
        else:
            master_facets = np.asarray(master_facets, dtype=int)

        if slave_facets_list is None:
            if slave_facet_selectors is None:
                raise ValueError("Provide either slave_facets_list or slave_facet_selectors.")
            if callable(slave_facet_selectors):
                slave_facets_list = [np.asarray(slave_facet_selectors(mesh), dtype=int) for mesh in slave_meshes]
            else:
                if len(slave_facet_selectors) != n_slaves:
                    raise ValueError("slave_facet_selectors length must match slave_meshes length.")
                out_facets: list[np.ndarray] = []
                for mesh, sel in zip(slave_meshes, slave_facet_selectors):
                    if sel is None:
                        raise ValueError("slave_facet_selectors contains None; provide a selector for each slave.")
                    out_facets.append(np.asarray(sel(mesh), dtype=int))
                slave_facets_list = out_facets
        else:
            if len(slave_facets_list) != n_slaves:
                raise ValueError("slave_facets_list length must match slave_meshes length.")
            slave_facets_list = [np.asarray(facets, dtype=int) for facets in slave_facets_list]

        if slave_spaces is None:
            slave_spaces = [None] * n_slaves
        elif isinstance(slave_spaces, Sequence) and not isinstance(slave_spaces, (str, bytes)):
            if len(slave_spaces) != n_slaves:
                raise ValueError("slave_spaces length must match slave_meshes length.")
            slave_spaces = list(slave_spaces)
        else:
            slave_spaces = [slave_spaces] * n_slaves

        if value_dim_slaves is None:
            value_dim_slaves = [None] * n_slaves
        elif isinstance(value_dim_slaves, Sequence) and not isinstance(value_dim_slaves, (str, bytes)):
            if len(value_dim_slaves) != n_slaves:
                raise ValueError("value_dim_slaves length must match slave_meshes length.")
            value_dim_slaves = list(value_dim_slaves)
        else:
            value_dim_slaves = [int(value_dim_slaves)] * n_slaves

        master_side = ContactSide.from_facets(
            master_mesh,
            master_facets,
            master_space,
            value_dim=value_dim_master,
            mode=mode_master,
        )
        slave_sides = [
            ContactSide.from_facets(
                mesh,
                np.asarray(facets, dtype=int),
                space,
                value_dim=value_dim,
                mode=mode_slave,
            )
            for mesh, facets, space, value_dim in zip(slave_meshes, slave_facets_list, slave_spaces, value_dim_slaves)
        ]
        return cls.from_sides(
            master_side,
            slave_sides,
            field_master=field_master,
            field_slave=field_slave,
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

    @classmethod
    def from_sides(
        cls,
        master: ContactSide,
        slaves: Sequence[ContactSide],
        *,
        field_master: str = "master",
        field_slave: str = "slave",
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
    ) -> "OneToManyContactSurfaceSpace":
        if len(slaves) == 0:
            raise ValueError("slaves must contain at least one ContactSide.")
        contacts = tuple(
            ContactSurfaceSpace.from_sides(
                master,
                slave,
                field_master=field_master,
                field_slave=field_slave,
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
            for slave in slaves
        )
        return cls(contacts=contacts, field_master=field_master, field_slave=field_slave)

    @classmethod
    def from_surfaces(
        cls,
        surface_master: SurfaceMesh,
        surface_slaves: Sequence[SurfaceMesh],
        *,
        elem_conn_master: np.ndarray | None = None,
        elem_conn_slaves: Sequence[np.ndarray | None] | None = None,
        value_dim_master: int = 1,
        value_dim_slave: int = 1,
        field_master: str = "master",
        field_slave: str = "slave",
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
    ) -> "OneToManyContactSurfaceSpace":
        if len(surface_slaves) == 0:
            raise ValueError("surface_slaves must contain at least one surface.")
        if elem_conn_slaves is None:
            elem_conn_slaves = [None] * len(surface_slaves)
        if len(elem_conn_slaves) != len(surface_slaves):
            raise ValueError("elem_conn_slaves length must match surface_slaves length.")
        contacts = tuple(
            ContactSurfaceSpace.from_surfaces(
                surface_master,
                surface_slave,
                elem_conn_master=elem_conn_master,
                elem_conn_slave=elem_conn_slave,
                field_master=field_master,
                field_slave=field_slave,
                value_dim_master=value_dim_master,
                value_dim_slave=value_dim_slave,
                space_mode_master=space_mode_master,
                space_mode_slave=space_mode_slave,
                facet_dofs_master=facet_dofs_master,
                facet_dofs_slave=facet_dofs_slave,
                quad_order=quad_order,
                normal_sign=normal_sign,
                tol=tol,
                backend=backend,
                batch_jac=batch_jac,
                setup_cache_enabled=setup_cache_enabled,
                setup_cache_trace=setup_cache_trace,
            )
            for surface_slave, elem_conn_slave in zip(surface_slaves, elem_conn_slaves)
        )
        return cls(contacts=contacts, field_master=field_master, field_slave=field_slave)

    def _split_fields(
        self, u: Mapping[str, npt.ArrayLike] | Sequence[Any]
    ) -> tuple[npt.ArrayLike, list[npt.ArrayLike]]:
        if isinstance(u, Mapping):
            if self.field_master not in u:
                raise KeyError(f"u mapping must contain master field '{self.field_master}'.")
            if "slaves" not in u:
                raise KeyError("u mapping must contain key 'slaves' with per-slave states.")
            u_master = u[self.field_master]
            u_slaves = list(u["slaves"])
        else:
            if len(u) != 2:
                raise ValueError("u must be a mapping or a sequence like (u_master, u_slaves).")
            u_master = u[0]
            u_slaves = list(u[1])
        if len(u_slaves) != len(self.contacts):
            raise ValueError(
                f"u_slaves length mismatch: got {len(u_slaves)}, expected {len(self.contacts)}."
            )
        return u_master, u_slaves

    def _dof_layout(self) -> tuple[int, list[int], int]:
        if len(self.contacts) == 0:
            return 0, [], 0
        n_master = contact_space_side_n_dofs(self.contacts[0], side="master")
        slave_sizes = [contact_space_side_n_dofs(contact, side="slave") for contact in self.contacts]
        total = int(n_master + sum(slave_sizes))
        return n_master, slave_sizes, total

    def initialize_state(self, *, metadata: Mapping[str, Any] | None = None) -> "ContactState":
        from .contact import ContactState

        n_master, slave_sizes, _ = self._dof_layout()
        return ContactState(
            interface_kind="one_to_many",
            geometry="reference",
            iteration=0,
            active_set=None,
            field_summary={
                self.field_master: (n_master,),
                self.field_slave: tuple(int(n) for n in slave_sizes),
            },
            metadata=dict(metadata or {}),
        )

    def update_state(
        self,
        *,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        contact_state: "ContactState" | None = None,
        geometry: str = "current",
        active_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ContactState":
        base = self.initialize_state() if contact_state is None else contact_state
        merged_metadata = dict(base.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        return replace(
            base,
            geometry=str(geometry),
            iteration=int(base.iteration) + 1,
            active_set=active_set if active_set is not None else base.active_set,
            field_summary=summarize_contact_field_state(state),
            metadata=merged_metadata,
        )

    def _resolve_backend(self, backend: str | None) -> str:
        if backend is not None:
            return str(backend)
        if len(self.contacts) == 0:
            return "jax"
        return str(self.contacts[0].backend)

    def compile_bilinear(
        self,
        bilin: "ContactBilinearLike",
        *,
        backend: str | None = None,
        use_cache: bool = True,
    ) -> "MixedSurfaceResidualForm":
        """Compile a one-to-many contact bilinear once and reuse it across all pair contacts."""
        from .contact import _compile_contact_bilinear, _is_compiled_contact_bilinear

        use_backend = self._resolve_backend(backend)
        if _is_compiled_contact_bilinear(bilin):
            return cast("MixedSurfaceResidualForm", bilin)
        cache_key = (id(bilin), use_backend)
        if use_cache:
            cached = self._compiled_bilinear_cache.get(cache_key)
            if cached is not None:
                return cached
        res_form = _compile_contact_bilinear(
            bilin,
            field_master=self.field_master,
            field_slave=self.field_slave,
            backend=use_backend,
        )
        if use_cache:
            self._compiled_bilinear_cache[cache_key] = res_form
        return res_form

    @staticmethod
    def _scatter_pair_indices(local_idx: np.ndarray, *, n_master: int, slave_offset: int) -> np.ndarray:
        idx = np.asarray(local_idx, dtype=int)
        out = np.empty_like(idx)
        master_mask = idx < int(n_master)
        out[master_mask] = idx[master_mask]
        out[~master_mask] = int(n_master) + int(slave_offset) + (idx[~master_mask] - int(n_master))
        return out

    def assemble_residual(
        self,
        res_form: "MixedSurfaceResidualForm",
        u: Mapping[str, npt.ArrayLike] | Sequence[Any],
        params: "WeakParams",
        *,
        normal_source: str = "master",
    ) -> np.ndarray:
        u_master, u_slaves = self._split_fields(u)
        n_master, slave_sizes, n_total = self._dof_layout()
        R = np.zeros((n_total,), dtype=float)
        slave_offset = 0
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            r_local = np.asarray(
                contact.assemble_residual(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                ),
                dtype=float,
            )
            if r_local.shape[0] != n_master + n_slave:
                raise ValueError("Pair residual size mismatch while assembling one-to-many residual.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            R[idx] += r_local
            slave_offset += n_slave
        return R

    def assemble_jacobian(
        self,
        res_form: "MixedSurfaceResidualForm",
        u: Mapping[str, npt.ArrayLike] | Sequence[Any],
        params: "WeakParams",
        *,
        normal_source: str = "master",
        sparse: bool = True,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> "ContactJacobianReturn":
        from .contact import _contact_sparse_to_coo

        u_master, u_slaves = self._split_fields(u)
        n_master, slave_sizes, n_total = self._dof_layout()
        slave_offset = 0
        if sparse:
            from ..solver import FluxSparseMatrix

            rows_all: list[np.ndarray] = []
            cols_all: list[np.ndarray] = []
            data_all: list[np.ndarray] = []
            for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
                j_local = contact.assemble_jacobian(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                    sparse=True,
                    backend=backend,
                    batch_jac=batch_jac,
                )
                rows, cols, data, n_pair = _contact_sparse_to_coo(j_local)
                if n_pair != n_master + n_slave:
                    raise ValueError("Pair Jacobian size mismatch while assembling sparse one-to-many Jacobian.")
                rows_all.append(
                    self._scatter_pair_indices(rows, n_master=n_master, slave_offset=slave_offset)
                )
                cols_all.append(
                    self._scatter_pair_indices(cols, n_master=n_master, slave_offset=slave_offset)
                )
                data_all.append(data)
                slave_offset += n_slave
            if rows_all:
                rows_out = np.concatenate(rows_all)
                cols_out = np.concatenate(cols_all)
                data_out = np.concatenate(data_all)
            else:
                rows_out = np.zeros((0,), dtype=int)
                cols_out = np.zeros((0,), dtype=int)
                data_out = np.zeros((0,), dtype=float)
            return FluxSparseMatrix(rows_out, cols_out, data_out, n_dofs=n_total)

        K = np.zeros((n_total, n_total), dtype=float)
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            j_local = np.asarray(
                contact.assemble_jacobian(
                    res_form,
                    {self.field_master: u_master, self.field_slave: u_slave},
                    params,
                    normal_source=normal_source,
                    sparse=False,
                    backend=backend,
                    batch_jac=batch_jac,
                ),
                dtype=float,
            )
            if j_local.shape != (n_master + n_slave, n_master + n_slave):
                raise ValueError("Pair Jacobian shape mismatch while assembling dense one-to-many Jacobian.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            K[np.ix_(idx, idx)] += j_local
            slave_offset += n_slave
        return K

    def assemble_bilinear(
        self,
        bilin: "ContactBilinearLike",
        u_master: Mapping[str, npt.ArrayLike] | Sequence[Any] | npt.ArrayLike,
        u_slaves: Sequence[npt.ArrayLike] | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> "ContactJacobianReturn":
        from .contact import _contact_sparse_to_coo

        if params is None:
            if u_slaves is None:
                raise TypeError("params is required")
            if isinstance(u_master, Mapping) or (isinstance(u_master, Sequence) and not hasattr(u_master, "shape")):
                params = u_slaves  # type: ignore[assignment]
                u_master, u_slaves = self._split_fields(u_master)  # type: ignore[arg-type]
            else:
                raise TypeError("params is required")
        elif u_slaves is None:
            u_master, u_slaves = self._split_fields(u_master)  # type: ignore[arg-type]
        assert params is not None
        assert u_slaves is not None
        res_form = self.compile_bilinear(bilin)

        n_master, slave_sizes, n_total = self._dof_layout()
        slave_offset = 0
        if sparse:
            from ..solver import FluxSparseMatrix

            rows_all: list[np.ndarray] = []
            cols_all: list[np.ndarray] = []
            data_all: list[np.ndarray] = []
            for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
                j_local = contact.assemble_bilinear(
                    res_form,
                    u_master,
                    u_slave,
                    params,
                    sparse=True,
                    normal_source=normal_source,
                )
                rows, cols, data, n_pair = _contact_sparse_to_coo(j_local)
                if n_pair != n_master + n_slave:
                    raise ValueError("Pair Jacobian size mismatch while assembling sparse one-to-many bilinear.")
                rows_all.append(
                    self._scatter_pair_indices(rows, n_master=n_master, slave_offset=slave_offset)
                )
                cols_all.append(
                    self._scatter_pair_indices(cols, n_master=n_master, slave_offset=slave_offset)
                )
                data_all.append(data)
                slave_offset += n_slave
            if rows_all:
                rows_out = np.concatenate(rows_all)
                cols_out = np.concatenate(cols_all)
                data_out = np.concatenate(data_all)
            else:
                rows_out = np.zeros((0,), dtype=int)
                cols_out = np.zeros((0,), dtype=int)
                data_out = np.zeros((0,), dtype=float)
            return FluxSparseMatrix(rows_out, cols_out, data_out, n_dofs=n_total)

        K = np.zeros((n_total, n_total), dtype=float)
        for contact, u_slave, n_slave in zip(self.contacts, u_slaves, slave_sizes):
            j_local = np.asarray(
                contact.assemble_bilinear(
                    res_form,
                    u_master,
                    u_slave,
                    params,
                    sparse=False,
                    normal_source=normal_source,
                ),
                dtype=float,
            )
            if j_local.shape != (n_master + n_slave, n_master + n_slave):
                raise ValueError("Pair Jacobian shape mismatch while assembling dense one-to-many bilinear.")
            idx = np.concatenate(
                (
                    np.arange(n_master, dtype=int),
                    n_master + slave_offset + np.arange(n_slave, dtype=int),
                )
            )
            K[np.ix_(idx, idx)] += j_local
            slave_offset += n_slave
        return K

    def assemble_bilinear_form(
        self,
        bilin: "ContactBilinearLike",
        params: "WeakParams",
        *,
        sparse: bool = True,
        normal_source: str = "master",
    ) -> "ContactJacobianReturn":
        """Assemble a one-to-many interface bilinear form without requiring states."""
        n_master, slave_sizes, _ = self._dof_layout()
        u_master = np.zeros((n_master,), dtype=float)
        u_slaves = [np.zeros((n_slave,), dtype=float) for n_slave in slave_sizes]
        return self.assemble_bilinear(
            bilin,
            u_master,
            u_slaves,
            params,
            sparse=sparse,
            normal_source=normal_source,
        )

    def assemble_pair_nitsche(
        self,
        params: "WeakParams",
        *,
        sparse: bool = False,
        normal_source: str = "master",
        use_penalty: float | None = None,
        use_traction: float | None = None,
        backend_fastpath: str = "numpy_local_kernel",
    ) -> "PenaltyContactContribution":
        """Assemble pair-Nitsche terms over this one-to-many contact supermesh."""
        from .contact import assemble_pair_nitsche_supermesh

        return assemble_pair_nitsche_supermesh(
            self,
            params,
            sparse=sparse,
            normal_source=normal_source,
            use_penalty=use_penalty,
            use_traction=use_traction,
            backend_fastpath=backend_fastpath,
        )

    def assemble_contact_coupling_matrices(self):
        from .contact_interface import ContactCouplingMatrix

        n_master, slave_sizes, _ = self._dof_layout()
        n_slaves_total = int(sum(slave_sizes))
        rows_mm: list[np.ndarray] = []
        cols_mm: list[np.ndarray] = []
        data_mm: list[np.ndarray] = []
        rows_ms: list[np.ndarray] = []
        cols_ms: list[np.ndarray] = []
        data_ms: list[np.ndarray] = []

        slave_offset = 0
        for contact, n_slave in zip(self.contacts, slave_sizes):
            m_mm, m_ms_local = contact.assemble_contact_coupling_matrices()
            rows_mm.append(np.asarray(m_mm.rows, dtype=int))
            cols_mm.append(np.asarray(m_mm.cols, dtype=int))
            data_mm.append(np.asarray(m_mm.data, dtype=float))
            rows_ms.append(np.asarray(m_ms_local.rows, dtype=int))
            cols_ms.append(np.asarray(m_ms_local.cols, dtype=int) + slave_offset)
            data_ms.append(np.asarray(m_ms_local.data, dtype=float))
            if m_mm.shape != (n_master, n_master):
                raise ValueError("Pair M_aa shape mismatch while assembling one-to-many coupling matrices.")
            if m_ms_local.shape != (n_master, n_slave):
                raise ValueError("Pair M_ab shape mismatch while assembling one-to-many coupling matrices.")
            slave_offset += n_slave

        mm = ContactCouplingMatrix(
            rows=np.concatenate(rows_mm) if rows_mm else np.zeros((0,), dtype=int),
            cols=np.concatenate(cols_mm) if cols_mm else np.zeros((0,), dtype=int),
            data=np.concatenate(data_mm) if data_mm else np.zeros((0,), dtype=float),
            shape=(n_master, n_master),
        )
        ms = ContactCouplingMatrix(
            rows=np.concatenate(rows_ms) if rows_ms else np.zeros((0,), dtype=int),
            cols=np.concatenate(cols_ms) if cols_ms else np.zeros((0,), dtype=int),
            data=np.concatenate(data_ms) if data_ms else np.zeros((0,), dtype=float),
            shape=(n_master, n_slaves_total),
        )
        return mm, ms

    def assemble_contact_kkt(
        self,
        *,
        rho: float = 0.0,
        multiplier: "ContactMultiplierSpace" | None = None,
        backend: str | None = None,
        format: str = "fluxsparse",
        return_blocks: bool = False,
    ):
        from .contact import assemble_contact_kkt

        m_aa, m_ab = self.assemble_contact_coupling_matrices()
        master_facets = np.asarray(self.contacts[0].surface_master.conn, dtype=int)
        return assemble_contact_kkt(
            m_aa,
            m_ab,
            rho=rho,
            multiplier=multiplier,
            facet_conn_master=master_facets,
            backend=backend,
            format=format,
            return_blocks=return_blocks,
        )

    def assemble_contact_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: "ContactMultiplierSpace" | None = None,
        backend: str | None = None,
        weak_form: "MixedSurfaceResidualForm" | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: "MixedSurfaceResidualForm" | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> "ContactOperators":
        from .contact import assemble_contact_constraint_operators

        return assemble_contact_constraint_operators(
            self,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_constraint_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: "ContactMultiplierSpace" | None = None,
        backend: str | None = None,
        weak_form: "MixedSurfaceResidualForm" | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: "MixedSurfaceResidualForm" | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> "ContactOperators":
        """Alias for assemble_contact_constraint_operators()."""
        return self.assemble_contact_constraint_operators(
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: "MixedSurfaceResidualForm" | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: "MixedSurfaceResidualForm" | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> "ContactOperators":
        from .contact import assemble_contact_penalty_operators

        return assemble_contact_penalty_operators(
            self,
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_penalty_operators(
        self,
        *,
        law: str | None = None,
        formulation: str | None = None,
        backend: str | None = None,
        weak_form: "MixedSurfaceResidualForm" | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: "MixedSurfaceResidualForm" | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> "ContactOperators":
        """Alias for assemble_contact_penalty_operators()."""
        return self.assemble_contact_penalty_operators(
            law=law,
            formulation=formulation,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )

    def assemble_contact_operators(
        self,
        *,
        enforcement: str | None = None,
        method: str | None = None,
        law: str | None = None,
        formulation: str | None = None,
        rho: float = 0.0,
        multiplier: "ContactMultiplierSpace" | None = None,
        backend: str | None = None,
        weak_form: "MixedSurfaceResidualForm" | None = None,
        state: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        res_form: "MixedSurfaceResidualForm" | None = None,
        u: Mapping[str, npt.ArrayLike] | Sequence[Any] | None = None,
        params: "WeakParams" | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        batch_jac: bool | None = None,
    ) -> "ContactOperators":
        """Unified public alias that routes to penalty or constraint assembly."""
        from .contact import assemble_contact_operators

        return assemble_contact_operators(
            self,
            enforcement=enforcement,
            method=method,
            law=law,
            formulation=formulation,
            rho=rho,
            multiplier=multiplier,
            backend=backend,
            weak_form=weak_form,
            state=state,
            res_form=res_form,
            u=u,
            params=params,
            normal_source=normal_source,
            sparse=sparse,
            batch_jac=batch_jac,
        )
