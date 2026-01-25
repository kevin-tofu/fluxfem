from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

from .mortar import (
    assemble_mixed_surface_jacobian,
    assemble_mixed_surface_residual,
    assemble_onesided_bilinear,
    assemble_contact_onesided_floor,
    assemble_mortar_matrices,
    map_surface_facets_to_tet_elements,
    map_surface_facets_to_hex_elements,
)
from .supermesh import build_surface_supermesh
from .surface import SurfaceMesh
from .base import BaseMesh

if TYPE_CHECKING:
    from .mortar import MortarMatrix
    from ..core.weakform import Params as WeakParams
    from .mortar import SurfaceMixedFormContext

ContactJacobianReturn: TypeAlias = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, int]
MixedSurfaceResidualForm: TypeAlias = Callable[
    ["SurfaceMixedFormContext", Mapping[str, npt.ArrayLike], Any],
    Mapping[str, npt.ArrayLike],
]
SurfaceHatFn: TypeAlias = Callable[[np.ndarray], npt.ArrayLike]

_CONTACT_SETUP_CACHE: dict[tuple, "ContactSurfaceSpace"] = {}


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


def _facet_map_for_elem_conn(surface: SurfaceMesh, elem_conn: np.ndarray | None) -> np.ndarray:
    if elem_conn is None:
        raise ValueError("elem_conn is required to build facet_to_elem mapping.")
    if elem_conn.shape[1] in {4, 10}:
        return map_surface_facets_to_tet_elements(surface, elem_conn)
    if elem_conn.shape[1] in {8, 20, 27}:
        return map_surface_facets_to_hex_elements(surface, elem_conn)
    raise NotImplementedError("elem_conn must be tet4/tet10/hex8/hex20/hex27")


def facet_gap_values(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
) -> tuple[np.ndarray, float]:
    """
    Compute per-facet gap values for a one-sided contact plane.

    Returns (g_f, min_g_all) where g_f is reduced per facet and min_g_all is
    the global minimum node gap.
    """
    coords_np = np.asarray(coords, dtype=float)
    if value_dim is None:
        value_dim = int(coords_np.shape[1])
    u_nodes = np.asarray(u, dtype=float).reshape(-1, value_dim)
    x_cur = coords_np + u_nodes
    g_all = np.dot(x_cur, np.asarray(n, dtype=float)) - float(c)
    min_g_all = float(np.min(g_all)) if g_all.size else 0.0
    if facets is None or len(facets) == 0:
        return np.zeros((0,), dtype=float), min_g_all
    if reduce == "min":
        g_f = np.array([np.min(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    elif reduce == "mean":
        g_f = np.array([np.mean(g_all[np.asarray(facet, dtype=int)]) for facet in facets], dtype=float)
    else:
        raise ValueError("reduce must be 'min' or 'mean'")
    return g_f, min_g_all


def active_contact_facets(
    coords: np.ndarray,
    facets: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    c: float,
    *,
    value_dim: int | None = None,
    reduce: str = "min",
    threshold: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return active facet indices and global minimum gap for one-sided contact."""
    g_f, min_g_all = facet_gap_values(
        coords,
        facets,
        u,
        n,
        c,
        value_dim=value_dim,
        reduce=reduce,
    )
    active_ids = np.nonzero(g_f < threshold)[0]
    return active_ids, min_g_all


@dataclass(frozen=True)
class OneSidedContact:
    side: ContactSide
    n: np.ndarray | None
    c: float
    k: float
    beta: float
    quad_order: int = 2
    normal_sign: float = 1.0
    tol: float = 1e-8
    facet_map: np.ndarray | None = None

    @classmethod
    def from_side(
        cls,
        side: ContactSide,
        *,
        n: np.ndarray | None,
        c: float,
        k: float,
        beta: float,
        quad_order: int = 2,
        normal_sign: float = 1.0,
        tol: float = 1e-8,
        facet_map: np.ndarray | None = None,
    ) -> "OneSidedContact":
        if facet_map is None:
            facet_map = _facet_map_for_elem_conn(side.surface, side.elem_conn)
        return cls(
            side=side,
            n=n,
            c=float(c),
            k=float(k),
            beta=float(beta),
            quad_order=int(quad_order),
            normal_sign=float(normal_sign),
            tol=float(tol),
            facet_map=facet_map,
        )

    def assemble(self, u, *, return_metrics: bool = False):
        return assemble_contact_onesided_floor(
            self.side.surface,
            np.asarray(u, dtype=float),
            n=None if self.n is None else np.asarray(self.n, dtype=float),
            c=self.c,
            k=self.k,
            beta=self.beta,
            value_dim=self.side.value_dim,
            elem_conn=np.asarray(self.side.elem_conn) if self.side.elem_conn is not None else None,
            facet_to_elem=self.facet_map,
            quad_order=self.quad_order,
            normal_sign=self.normal_sign,
            tol=self.tol,
            return_metrics=return_metrics,
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
        facet_map_slave = _facet_map_for_elem_conn(side.surface, side.elem_conn)
        facet_map_master = facet_to_elem_master
        if surface_master is not None and elem_conn_master is not None and facet_map_master is None:
            facet_map_master = _facet_map_for_elem_conn(surface_master, elem_conn_master)
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

    def assemble_bilinear(
        self,
        u_hat_fn: SurfaceHatFn | None,
        params: "WeakParams",
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
    quad_order: int = 1
    normal_sign: float | None = None
    tol: float = 1e-8
    backend: str = "jax"
    fd_eps: float = 1e-6
    fd_mode: str = "central"
    fd_block_size: int = 1
    batch_jac: bool | None = None

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
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
        batch_jac: bool | None = None,
        setup_cache_enabled: bool | None = None,
        setup_cache_trace: bool | None = None,
    ) -> "ContactSurfaceSpace":
        import hashlib
        import os

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
                int(quad_order),
                float(normal_sign) if normal_sign is not None else None,
                float(tol),
                backend,
                float(fd_eps),
                fd_mode,
                int(fd_block_size),
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
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
            batch_jac=batch_jac,
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
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
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
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
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
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
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
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
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
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
        fd_block_size: int = 1,
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
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
            fd_block_size=fd_block_size,
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
        quad_order: int = 1,
        normal_sign: float | None = None,
        tol: float = 1e-8,
        backend: str = "jax",
        fd_eps: float = 1e-6,
        fd_mode: str = "central",
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
            quad_order=quad_order,
            normal_sign=normal_sign,
            tol=tol,
            backend=backend,
            fd_eps=fd_eps,
            fd_mode=fd_mode,
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

    def assemble_mortar_matrices(self) -> tuple["MortarMatrix", "MortarMatrix"]:
        """Return (M_aa, M_ab) mortar coupling matrices."""
        return assemble_mortar_matrices(
            self.supermesh_coords,
            self.supermesh_conn,
            self.source_facets_master,
            self.source_facets_slave,
            self.surface_master,
            self.surface_slave,
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
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        return assemble_mixed_surface_residual(
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
        sparse: bool = False,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ) -> ContactJacobianReturn:
        u_master, u_slave = self._split_fields(u)
        if normal_sign is None:
            normal_sign = self.normal_sign
        if normal_sign is None:
            normal_sign = self._auto_normal_sign()
        use_backend = self._resolve_backend(backend)
        use_batch_jac = self.batch_jac if batch_jac is None else batch_jac
        return assemble_mixed_surface_jacobian(
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
            fd_eps=self.fd_eps,
            fd_mode=self.fd_mode,
            fd_block_size=self.fd_block_size,
        )

    def assemble_bilinear(
        self,
        bilin: Callable[..., Any],
        u_master: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | npt.ArrayLike,
        u_slave: npt.ArrayLike | None = None,
        params: "WeakParams" | None = None,
        *,
        sparse: bool = False,
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
        from ..core.weakform import (
            compile_mixed_surface_residual,
            compile_mixed_surface_residual_numpy,
            unknown_ref,
            test_ref,
            param_ref,
            zero_ref,
        )

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

        v1 = test_ref(self.field_master)
        v2 = test_ref(self.field_slave)
        u1 = unknown_ref(self.field_master)
        u2 = unknown_ref(self.field_slave)
        z1 = zero_ref(self.field_master)
        z2 = zero_ref(self.field_slave)
        p = param_ref()

        expr_a = bilin(v1, z2, u1, u2, p)
        expr_b = bilin(z1, v2, u1, u2, p)
        use_backend = self._resolve_backend(None)
        if use_backend == "numpy":
            res_form = compile_mixed_surface_residual_numpy({self.field_master: expr_a, self.field_slave: expr_b})
        else:
            res_form = compile_mixed_surface_residual({self.field_master: expr_a, self.field_slave: expr_b})
        return self.assemble_jacobian(
            res_form,
            {self.field_master: u_master, self.field_slave: u_slave},
            params,
            normal_sign=None,
            normal_source=normal_source,
            sparse=sparse,
            backend=use_backend,
        )


__all__ = [
    "ContactSide",
    "OneSidedContact",
    "OneSidedContactSurfaceSpace",
    "ContactSurfaceSpace",
    "facet_gap_values",
    "active_contact_facets",
]
