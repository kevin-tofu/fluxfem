from __future__ import annotations

import warnings

import jax

_WARNED_FLOAT32_ASSEMBLY = False


def warn_float32_assembly_once(*, context: str = "assembly") -> None:
    """
    Emit a one-time RuntimeWarning when x64 is disabled.
    """
    global _WARNED_FLOAT32_ASSEMBLY
    if _WARNED_FLOAT32_ASSEMBLY:
        return
    if bool(jax.config.read("jax_enable_x64")):
        return
    _WARNED_FLOAT32_ASSEMBLY = True
    warnings.warn(
        "Running in float32 mode (x64 disabled). "
        f"{context} can suffer from residual/conditioning degradation; "
        "use x64 for reliable diagnostics.",
        RuntimeWarning,
        stacklevel=2,
    )

