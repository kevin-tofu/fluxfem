import os
from typing import Mapping

import numpy as np

from ..mesh import BaseMesh, HexMesh, TetMesh


VTK_HEXAHEDRON = 12
VTK_TETRA = 10


def _cell_type_for_mesh(mesh: BaseMesh) -> int:
    if isinstance(mesh, HexMesh):
        return VTK_HEXAHEDRON
    if isinstance(mesh, TetMesh):
        return VTK_TETRA
    raise TypeError(f"Unsupported mesh type for VTU export: {type(mesh)}")


def _write_dataarray(name: str, data: np.ndarray, ncomp: int = 1) -> str:
    flat = data.reshape(-1)
    comp_attr = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    values = " ".join(f"{v:.8e}" for v in flat)
    return f'<DataArray type="Float32" Name="{name}" format="ascii"{comp_attr}>{values}</DataArray>'


def write_vtu(
    mesh: BaseMesh,
    filepath: str,
    *,
    point_data: Mapping[str, np.ndarray] | None = None,
    cell_data: Mapping[str, np.ndarray] | None = None,
) -> None:
    """
    Write an UnstructuredGrid VTU for HexMesh or TetMesh.
    point_data/cell_data: dict name -> ndarray. Point data length must match n_points;
    cell data length must match n_cells.
    """
    coords = np.asarray(mesh.coords, dtype=float)
    conn = np.asarray(mesh.conn, dtype=np.int32)
    n_cells, n_nodes_per_cell = conn.shape
    n_points = coords.shape[0]
    cell_type = _cell_type_for_mesh(mesh)

    offsets = np.cumsum(np.full(n_cells, n_nodes_per_cell, dtype=np.int32))
    types = np.full(n_cells, cell_type, dtype=np.int32)

    point_data = point_data or {}
    cell_data = cell_data or {}

    # Build XML
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">')
    lines.append('  <UnstructuredGrid>')
    lines.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">')

    # PointData
    lines.append("      <PointData>")
    for name, arr in point_data.items():
        arr_np = np.asarray(arr)
        ncomp = 1 if arr_np.ndim == 1 else arr_np.shape[1]
        lines.append("        " + _write_dataarray(name, arr_np, ncomp))
    lines.append("      </PointData>")

    # CellData
    lines.append("      <CellData>")
    for name, arr in cell_data.items():
        arr_np = np.asarray(arr)
        ncomp = 1 if arr_np.ndim == 1 else arr_np.shape[1]
        lines.append("        " + _write_dataarray(name, arr_np, ncomp))
    lines.append("      </CellData>")

    # Points
    lines.append("      <Points>")
    lines.append("        " + _write_dataarray("Points", coords, 3))
    lines.append("      </Points>")

    # Cells
    lines.append("      <Cells>")
    lines.append("        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" + " ".join(str(int(v)) for v in conn.reshape(-1)) + "</DataArray>")
    lines.append("        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" + " ".join(str(int(v)) for v in offsets) + "</DataArray>")
    lines.append("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" + " ".join(str(int(v)) for v in types) + "</DataArray>")
    lines.append("      </Cells>")

    lines.append("    </Piece>")
    lines.append("  </UnstructuredGrid>")
    lines.append("</VTKFile>")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def write_displacement_vtu(
    mesh: BaseMesh,
    u: np.ndarray,
    filepath: str,
    *,
    name: str = "displacement",
) -> None:
    """
    Convenience wrapper: reshape displacement vector to point data and write VTU.
    Assumes 3 dof/node ordering [u0,v0,w0, u1,v1,w1, ...].
    """
    u_arr = np.asarray(u)
    n_nodes = mesh.coords.shape[0]
    disp = u_arr.reshape(n_nodes, -1)
    write_vtu(mesh, filepath, point_data={name: disp})


__all__ = ["write_vtu", "write_displacement_vtu"]
