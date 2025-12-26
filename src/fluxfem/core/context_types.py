from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class VolumeContext(Protocol):
    """Minimum interface for volume weak-form evaluation."""

    test: Any
    trial: Any
    w: Any


@runtime_checkable
class SurfaceContext(Protocol):
    """Minimum interface for surface weak-form evaluation."""

    v: Any
    w: Any
    detJ: Any
    normal: Any


@runtime_checkable
class FormFieldLike(Protocol):
    """Minimum interface for form fields used in weak-form evaluation."""

    N: Any
    gradN: Any
    detJ: Any
    value_dim: int


UElement: TypeAlias = Any
ParamsLike: TypeAlias = Any
