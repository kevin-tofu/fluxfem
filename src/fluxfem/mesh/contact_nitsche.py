from __future__ import annotations

from typing import Any, Callable

import numpy as np

_FF_CONTACT_FORMULATION_ATTR = "_ff_contact_formulation"
_FF_CONTACT_FASTPATH_ATTR = "_ff_contact_backend_fastpath"


def _tag_pair_nitsche_penalty_bilinear(
    fn: Callable[..., Any],
    *,
    backend_fastpath: str = "numpy_local_kernel",
) -> Callable[..., Any]:
    setattr(fn, _FF_CONTACT_FORMULATION_ATTR, "pair_nitsche_penalty")
    if backend_fastpath is not None:
        setattr(fn, _FF_CONTACT_FASTPATH_ATTR, str(backend_fastpath))
    return fn


def make_pair_nitsche_supermesh_bilinear(
    *,
    backend_fastpath: str = "numpy_local_kernel",
) -> Callable[..., Any]:
    """Build the symmetric pair-Nitsche bilinear used on contact supermeshes."""
    import fluxfem.helpers_wf as h_wf
    from ..core.weakform import einsum as wf_einsum

    def _bilin(v1, v2, u1, u2, p):
        n = h_wf.normal()
        ju = u1.val - u2.val
        t_u = 0.5 * (h_wf.traction(u1, n, p) + h_wf.traction(u2, n, p))
        t_v1 = h_wf.traction(v1, n, p)
        t_v2 = h_wf.traction(v2, n, p)
        penalty = p.use_penalty * (p.alpha * p.inv_h) * (
            h_wf.dot(v1, ju) - h_wf.dot(v2, ju)
        )
        traction = p.use_traction * (-h_wf.dot(v1, t_u) + h_wf.dot(v2, t_u))
        traction -= p.use_traction * 0.5 * wf_einsum("qia,qi->qa", t_v1, ju)
        traction -= p.use_traction * 0.5 * wf_einsum("qia,qi->qa", t_v2, ju)
        return (penalty + traction) * h_wf.ds()

    return _tag_pair_nitsche_penalty_bilinear(
        _bilin,
        backend_fastpath=backend_fastpath,
    )


def params_with_pair_nitsche_defaults(
    params: Any,
    *,
    use_penalty: float | None,
    use_traction: float | None,
) -> Any:
    defaults = {
        "use_penalty": 1.0 if use_penalty is None else float(use_penalty),
        "use_traction": 1.0 if use_traction is None else float(use_traction),
    }
    data = dict(getattr(params, "_data", {}))
    if not data:
        data = dict(vars(params))
    changed = False
    for name, value in defaults.items():
        if name not in data or (name == "use_penalty" and use_penalty is not None) or (
            name == "use_traction" and use_traction is not None
        ):
            data[name] = value
            changed = True
    if not changed:
        return params
    from ..core.weakform import Params

    return Params(**data)


def assemble_pair_nitsche_supermesh_impl(
    contact,
    params: Any,
    *,
    contribution_cls: type,
    sparse: bool = False,
    normal_source: str = "master",
    use_penalty: float | None = None,
    use_traction: float | None = None,
    backend_fastpath: str = "numpy_local_kernel",
):
    """
    Assemble pair-Nitsche contact terms over a prepared contact supermesh.

    The contact object must provide ``assemble_bilinear_form``; prepared
    ``ContactSurfaceSpace`` and ``OneToManyContactSurfaceSpace`` objects do.
    """
    if not hasattr(contact, "assemble_bilinear_form"):
        raise TypeError("contact must provide assemble_bilinear_form() for pair-Nitsche supermesh assembly.")
    params_eff = params_with_pair_nitsche_defaults(
        params,
        use_penalty=use_penalty,
        use_traction=use_traction,
    )
    bilin = make_pair_nitsche_supermesh_bilinear(backend_fastpath=backend_fastpath)
    jacobian = contact.assemble_bilinear_form(
        bilin,
        params_eff,
        sparse=sparse,
        normal_source=normal_source,
    )
    diagnostics: dict[str, Any] = {}
    if hasattr(contact, "supermesh_conn"):
        diagnostics["supermesh_triangles"] = int(np.asarray(contact.supermesh_conn).shape[0])
    diagnostics["use_penalty"] = float(getattr(params_eff, "use_penalty", 1.0))
    diagnostics["use_traction"] = float(getattr(params_eff, "use_traction", 1.0))
    return contribution_cls(
        enforcement="nitsche",
        law="frictionless_tied",
        formulation="pair_nitsche_penalty",
        jacobian=jacobian,
        diagnostics=diagnostics,
    )
