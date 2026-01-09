from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .mortar import (
    assemble_mixed_surface_jacobian,
    assemble_mixed_surface_residual,
    assemble_mortar_matrices,
    map_surface_facets_to_tet_elements,
    map_surface_facets_to_hex_elements,
)
from .supermesh import build_surface_supermesh
from .surface import SurfaceMesh
from .base import BaseMesh


@dataclass(frozen=True)
class ContactSide:
    surface: SurfaceMesh
    elem_conn: np.ndarray | None
    value_dim: int
    space: object | None = None

    @classmethod
    def from_facets(cls, mesh: BaseMesh, facets: np.ndarray, space, *, mode: str = "touching"):
        side = mesh.surface_with_elem_conn_from_facets(facets, mode=mode)
        value_dim = int(getattr(space, "value_dim", 1))
        return cls(surface=side.surface, elem_conn=side.elem_conn, value_dim=value_dim, space=space)

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
        batch_jac: bool | None = None,
    ) -> "ContactSurfaceSpace":
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
        return cls(
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
            batch_jac=batch_jac,
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
        batch_jac: bool | None = None,
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
            batch_jac=batch_jac,
        )

    @classmethod
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

    def assemble_mortar_matrices(self):
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
        res_form,
        u,
        params,
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
    ):
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
        res_form,
        u,
        params,
        *,
        normal_sign: float | None = None,
        normal_source: str = "master",
        sparse: bool = False,
        backend: str | None = None,
        batch_jac: bool | None = None,
    ):
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
        )

    def assemble_bilinear(
        self,
        bilin,
        u_master,
        u_slave=None,
        params=None,
        *,
        sparse: bool = False,
        normal_source: str = "master",
    ):
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
        from ..core.weakform import compile_mixed_surface_residual, unknown_ref, test_ref, param_ref, zero_ref

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
        res_form = compile_mixed_surface_residual({self.field_master: expr_a, self.field_slave: expr_b})
        return self.assemble_jacobian(
            res_form,
            {self.field_master: u_master, self.field_slave: u_slave},
            params,
            normal_sign=None,
            normal_source=normal_source,
            sparse=sparse,
        )


__all__ = ["ContactSurfaceSpace"]
