from __future__ import annotations

from typing import Any, Mapping, Protocol, TYPE_CHECKING, TypeAlias, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from jax import Array as JaxArray

    ArrayLike: TypeAlias = np.ndarray | JaxArray
else:
    ArrayLike: TypeAlias = np.ndarray


@runtime_checkable
class VolumeContext(Protocol):
    """Minimum interface for volume weak-form evaluation."""

    test: "FormFieldLike"
    trial: "FormFieldLike"
    w: ArrayLike


@runtime_checkable
class SurfaceContext(Protocol):
    """Minimum interface for surface weak-form evaluation."""

    v: "FormFieldLike"
    w: ArrayLike
    detJ: ArrayLike
    normal: ArrayLike


@runtime_checkable
class FormFieldLike(Protocol):
    """Minimum interface for form fields used in weak-form evaluation."""

    N: ArrayLike
    gradN: ArrayLike
    detJ: ArrayLike
    value_dim: int
    basis: Any


@runtime_checkable
class WeakFormContext(Protocol):
    """Context interface used when resolving field references."""

    test: FormFieldLike
    trial: FormFieldLike
    v: FormFieldLike
    unknown: FormFieldLike | None
    fields: Mapping[str, Any] | None
    test_fields: Mapping[str, FormFieldLike] | None
    trial_fields: Mapping[str, FormFieldLike] | None
    unknown_fields: Mapping[str, FormFieldLike] | None


UElement: TypeAlias = ArrayLike | Mapping[str, ArrayLike]
ParamsLike: TypeAlias = Any
