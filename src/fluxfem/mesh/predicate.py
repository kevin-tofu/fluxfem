from __future__ import annotations

import numpy as np


def plane_predicate(axis: int, value: float, tol: float = 1e-8):
    """Return predicate True when all nodes lie on plane x[axis]=value (within tol)."""
    def pred(pts: np.ndarray) -> bool:
        return bool(np.allclose(pts[:, axis], value, atol=tol))
    return pred


def axis_plane_predicate(axis: int, value: float, tol: float = 1e-8):
    """Alias of plane_predicate for readability."""
    return plane_predicate(axis, value, tol=tol)


def slab_predicate(axis: int, min_val: float, max_val: float, tol: float = 1e-8):
    """Return predicate True when nodes lie in a slab min<=x[axis]<=max (within tol)."""
    def pred(pts: np.ndarray) -> np.ndarray:
        pts_np = np.asarray(pts, dtype=float)
        return (pts_np[:, axis] >= min_val - tol) & (pts_np[:, axis] <= max_val + tol)
    return pred


def bbox_predicate(mins: np.ndarray, maxs: np.ndarray, tol: float = 1e-8):
    """Return predicate True for nodes on the axis-aligned bounding box."""
    mins_np = np.asarray(mins, dtype=float)
    maxs_np = np.asarray(maxs, dtype=float)

    def pred(pts: np.ndarray) -> np.ndarray:
        pts_np = np.asarray(pts, dtype=float)
        on_min = np.isclose(pts_np, mins_np[None, :], atol=tol)
        on_max = np.isclose(pts_np, maxs_np[None, :], atol=tol)
        return np.any(on_min | on_max, axis=1)

    return pred


__all__ = [
    "plane_predicate",
    "axis_plane_predicate",
    "slab_predicate",
    "bbox_predicate",
]
