from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import jax

from ..physics import operators as _ops


class Expr:
    """Expression tree node evaluated against a FormContext."""

    def __init__(self, op: str, *args):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", args)

    def eval(self, ctx, params=None, u_elem=None):
        return _eval_expr(self, ctx, params, u_elem=u_elem)

    def _binop(self, other, op):
        return Expr(op, self, _as_expr(other))

    def __add__(self, other):
        return self._binop(other, "add")

    def __radd__(self, other):
        return _as_expr(other)._binop(self, "add")

    def __sub__(self, other):
        return self._binop(other, "sub")

    def __rsub__(self, other):
        return _as_expr(other)._binop(self, "sub")

    def __mul__(self, other):
        return self._binop(other, "mul")

    def __rmul__(self, other):
        return _as_expr(other)._binop(self, "mul")

    def __matmul__(self, other):
        return self._binop(other, "matmul")

    def __rmatmul__(self, other):
        return _as_expr(other)._binop(self, "matmul")

    def __or__(self, other):
        return self._binop(other, "inner")

    def __ror__(self, other):
        return _as_expr(other)._binop(self, "inner")

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise ValueError("modulo is not supported for Expr exponentiation.")
        return Expr("pow", self, _as_expr(power))

    def __neg__(self):
        return Expr("neg", self)

    @property
    def T(self):
        return Expr("transpose", self)


@dataclass(frozen=True)
class FieldRef(Expr):
    """Symbolic reference to trial/test/unknown field, optionally by name."""

    role: str
    name: str | None = None

    def __init__(self, role: str, name: str | None = None):
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "name", name)
        super().__init__("field", role, name)

    @property
    def val(self):
        return Expr("value", self)

    @property
    def grad(self):
        return Expr("grad", self)

    @property
    def sym_grad(self):
        return Expr("sym_grad", self)

    def __mul__(self, other):
        if isinstance(other, FieldRef):
            return Expr("outer", self, other)
        return Expr("mul", Expr("value", self), _as_expr(other))

    def __rmul__(self, other):
        if isinstance(other, FieldRef):
            return Expr("outer", other, self)
        return Expr("mul", _as_expr(other), Expr("value", self))

    def __add__(self, other):
        return Expr("add", Expr("value", self), _as_expr(other))

    def __radd__(self, other):
        return Expr("add", _as_expr(other), Expr("value", self))

    def __sub__(self, other):
        return Expr("sub", Expr("value", self), _as_expr(other))

    def __rsub__(self, other):
        return Expr("sub", _as_expr(other), Expr("value", self))

    def __or__(self, other):
        if isinstance(other, FieldRef):
            return Expr("inner", self, other)
        return Expr("sdot", self, _as_expr(other))

    def __ror__(self, other):
        if isinstance(other, FieldRef):
            return Expr("inner", other, self)
        return Expr("sdot", _as_expr(other), self)


@dataclass(frozen=True)
class ParamRef(Expr):
    """Symbolic reference to params passed into the kernel."""

    def __init__(self):
        super().__init__("param")

    def __getattr__(self, name: str):
        return Expr("getattr", self, name)


@jax.tree_util.register_pytree_node_class
class Params:
    """Simple params container with attribute access (JAX pytree)."""

    def __init__(self, **kwargs):
        self._data = dict(kwargs)

    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str):
        return self._data[key]

    def tree_flatten(self):
        keys = tuple(sorted(self._data.keys()))
        values = tuple(self._data[k] for k in keys)
        return values, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(**dict(zip(keys, values)))


def trial_ref(name: str | None = "u") -> FieldRef:
    """Create a symbolic trial field reference."""
    return FieldRef(role="trial", name=name)


def test_ref(name: str | None = "v") -> FieldRef:
    """Create a symbolic test field reference."""
    return FieldRef(role="test", name=name)


def unknown_ref(name: str | None = "u") -> FieldRef:
    """Create a symbolic unknown (current solution) field reference."""
    return FieldRef(role="unknown", name=name)


def param_ref() -> ParamRef:
    """Create a symbolic params reference."""
    return ParamRef()


def _as_expr(obj) -> Expr:
    if isinstance(obj, Expr):
        return obj
    return Expr("lit", obj)


def _eval_field(obj: Any, ctx, params):
    if isinstance(obj, FieldRef):
        if obj.name is not None:
            mixed_fields = getattr(ctx, "fields", None)
            if mixed_fields is not None and obj.name in mixed_fields:
                group = mixed_fields[obj.name]
                if hasattr(group, "trial") and obj.role == "trial":
                    return group.trial
                if hasattr(group, "test") and obj.role == "test":
                    return group.test
                if hasattr(group, "unknown") and obj.role == "unknown":
                    return group.unknown if group.unknown is not None else group.trial
            if obj.role == "trial" and getattr(ctx, "trial_fields", None) is not None:
                if obj.name in ctx.trial_fields:
                    return ctx.trial_fields[obj.name]
            if obj.role == "test" and getattr(ctx, "test_fields", None) is not None:
                if obj.name in ctx.test_fields:
                    return ctx.test_fields[obj.name]
            if obj.role == "unknown" and getattr(ctx, "unknown_fields", None) is not None:
                if obj.name in ctx.unknown_fields:
                    return ctx.unknown_fields[obj.name]
            fields = getattr(ctx, "fields", None)
            if fields is not None and obj.name in fields:
                group = fields[obj.name]
                if isinstance(group, dict):
                    if obj.role in group:
                        return group[obj.role]
                    if "field" in group:
                        return group["field"]
                return group
        if obj.role == "trial":
            return ctx.trial
        if obj.role == "test":
            if hasattr(ctx, "test"):
                return ctx.test
            if hasattr(ctx, "v"):
                return ctx.v
            raise ValueError("Surface context is missing test field.")
        if obj.role == "unknown":
            return getattr(ctx, "unknown", ctx.trial)
        raise ValueError(f"Unknown field role: {obj.role}")
    if isinstance(obj, Expr):
        val = obj.eval(ctx, params)
        if hasattr(val, "N"):
            return val
    raise TypeError("Expected a field reference for this operator.")


def _eval_value(obj: Any, ctx, params, u_elem=None):
    if isinstance(obj, FieldRef):
        field = _eval_field(obj, ctx, params)
        if obj.role == "unknown":
            return _eval_unknown_value(obj, field, u_elem)
        return field.N
    if isinstance(obj, Expr):
        return obj.eval(ctx, params, u_elem=u_elem)
    return obj


def _extract_unknown_elem(field_ref: FieldRef, u_elem):
    if u_elem is None:
        raise ValueError("u_elem is required to evaluate unknown field value.")
    if isinstance(u_elem, dict):
        name = field_ref.name or "u"
        if name not in u_elem:
            raise ValueError(f"u_elem is missing key '{name}'.")
        return u_elem[name]
    return u_elem


def _eval_unknown_value(field_ref: FieldRef, field, u_elem):
    u_local = _extract_unknown_elem(field_ref, u_elem)
    value_dim = int(getattr(field, "value_dim", 1))
    if value_dim == 1:
        return jnp.einsum("qa,a->q", field.N, u_local)
    u_nodes = u_local.reshape((-1, value_dim))
    return jnp.einsum("qa,ai->qi", field.N, u_nodes)


def _eval_unknown_grad(field_ref: FieldRef, field, u_elem):
    u_local = _extract_unknown_elem(field_ref, u_elem)
    if u_local is None:
        raise ValueError("u_elem is required to evaluate unknown field gradient.")
    value_dim = int(getattr(field, "value_dim", 1))
    if value_dim == 1:
        return jnp.einsum("qaj,a->qj", field.gradN, u_local)
    u_nodes = u_local.reshape((-1, value_dim))
    return jnp.einsum("qaj,ai->qij", field.gradN, u_nodes)


def grad(field) -> Expr:
    """Return basis gradients for a scalar or vector FormField."""
    return Expr("grad", _as_expr(field))


def sym_grad(field) -> Expr:
    """Return symmetric-gradient B-matrix for a vector FormField."""
    return Expr("sym_grad", _as_expr(field))


def dot(a, b) -> Expr:
    """Dot product or vector load helper."""
    return Expr("dot", _as_expr(a), _as_expr(b))


def sdot(a, b) -> Expr:
    """Surface dot product or vector load helper."""
    return Expr("sdot", _as_expr(a), _as_expr(b))


def ddot(a, b, c=None) -> Expr:
    """Double contraction or a^T b c."""
    if c is None:
        return Expr("ddot", _as_expr(a), _as_expr(b))
    return Expr("ddot", _as_expr(a), _as_expr(b), _as_expr(c))


def inner(a, b) -> Expr:
    """Inner product over the last axis."""
    return Expr("inner", _as_expr(a), _as_expr(b))


def action(v, s) -> Expr:
    """Test-function action: v.val * s -> (q, n_ldofs)."""
    return Expr("action", _as_expr(v), _as_expr(s))


def gaction(v, q) -> Expr:
    """Gradient action: v.grad · q -> (q, n_ldofs)."""
    return Expr("gaction", _as_expr(v), _as_expr(q))


def normal() -> Expr:
    """Surface normal vector (from SurfaceFormContext)."""
    return Expr("surface_normal")


def ds() -> Expr:
    """Surface quadrature measure (w * detJ)."""
    return Expr("surface_measure")


def dOmega() -> Expr:
    """Volume quadrature measure (w * detJ)."""
    return Expr("volume_measure")


def I(dim: int) -> Expr:
    """Identity matrix of size dim."""
    return Expr("eye", dim)


def det(a) -> Expr:
    """Determinant of a square matrix."""
    return Expr("det", _as_expr(a))


def inv(a) -> Expr:
    """Matrix inverse."""
    return Expr("inv", _as_expr(a))


def transpose(a) -> Expr:
    """Swap the last two axes."""
    return Expr("transpose", _as_expr(a))


def log(a) -> Expr:
    """Natural logarithm."""
    return Expr("log", _as_expr(a))


def transpose_last2(a) -> Expr:
    """Swap the last two axes."""
    return Expr("transpose_last2", _as_expr(a))


def einsum(subscripts: str, *args) -> Expr:
    """Einsum wrapper that supports Expr inputs."""
    return Expr("einsum", subscripts, *[_as_expr(arg) for arg in args])


def _call_user(fn, *args, params):
    try:
        return fn(*args, params)
    except TypeError:
        return fn(*args)


def compile_bilinear(fn):
    """Compile a bilinear weak form (u, v, params) -> Expr into a kernel."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        u = trial_ref()
        v = test_ref()
        p = param_ref()
        try:
            expr = fn(u, v, p)
        except TypeError:
            expr = fn(u, v)

    includes_measure = _expr_contains(expr, "volume_measure")
    if not includes_measure:
        raise ValueError("Volume bilinear form must include dOmega().")

    def _form(ctx, params):
        return _as_expr(expr).eval(ctx, params)

    _form._includes_measure = includes_measure
    return _form


def compile_linear(fn):
    """Compile a linear weak form (v, params) -> Expr into a kernel."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        v = test_ref()
        p = param_ref()
        try:
            expr = fn(v, p)
        except TypeError:
            expr = fn(v)

    includes_measure = _expr_contains(expr, "volume_measure")
    if not includes_measure:
        raise ValueError("Volume linear form must include dOmega().")

    def _form(ctx, params):
        return _as_expr(expr).eval(ctx, params)

    _form._includes_measure = includes_measure
    return _form


def _expr_contains(expr: Expr, op: str) -> bool:
    if not isinstance(expr, Expr):
        return False
    if expr.op == op:
        return True
    return any(_expr_contains(arg, op) for arg in expr.args if isinstance(arg, Expr))


def compile_surface_linear(fn):
    """Compile a surface linear form into a kernel (ctx, params) -> ndarray."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        v = test_ref()
        p = param_ref()
        expr = None
        try:
            expr = fn(v, p)
        except TypeError:
            try:
                expr = fn(v)
            except TypeError:
                expr = None

    if not isinstance(expr, Expr):
        raise ValueError("Surface linear form must return an Expr; use ds() in the expression.")

    includes_measure = _expr_contains(expr, "surface_measure")
    if not includes_measure:
        raise ValueError("Surface linear form must include ds().")

    def _form(ctx, params):
        return _as_expr(expr).eval(ctx, params)

    _form._includes_measure = includes_measure  # type: ignore[attr-defined]
    return _form


class LinearForm:
    """Linear form wrapper with volume/surface backends."""

    def __init__(self, fn, *, kind: str):
        self.fn = fn
        self.kind = kind

    @classmethod
    def volume(cls, fn):
        return cls(fn, kind="volume")

    @classmethod
    def surface(cls, fn):
        return cls(fn, kind="surface")

    def compile(self, *, ctx_kind: str | None = None):
        kind = self.kind if ctx_kind is None else ctx_kind
        if kind == "volume":
            return compile_linear(self.fn)
        if kind == "surface":
            return compile_surface_linear(self.fn)
        raise ValueError(f"Unknown linear form kind: {kind}")


class BilinearForm:
    """Bilinear form wrapper (volume only for now)."""

    def __init__(self, fn):
        self.fn = fn

    @classmethod
    def volume(cls, fn):
        return cls(fn)

    def compile(self):
        return compile_bilinear(self.fn)


class ResidualForm:
    """Residual form wrapper (volume only for now)."""

    def __init__(self, fn):
        self.fn = fn

    @classmethod
    def volume(cls, fn):
        return cls(fn)

    def compile(self):
        return compile_residual(self.fn)


def compile_residual(fn):
    """Compile a residual weak form (v, u, params) -> Expr into a kernel."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        v = test_ref()
        u = unknown_ref()
        p = param_ref()
        try:
            expr = fn(v, u, p)
        except TypeError:
            expr = fn(v, u)

    includes_measure = _expr_contains(expr, "volume_measure")
    if not includes_measure:
        raise ValueError("Volume residual form must include dOmega().")

    def _form(ctx, u_elem, params):
        return _as_expr(expr).eval(ctx, params, u_elem=u_elem)

    _form._includes_measure = includes_measure
    return _form


def compile_mixed_residual(residuals: dict[str, Callable]):
    """Compile mixed residuals keyed by field name."""
    compiled = {}
    includes_measure = {}
    for name, fn in residuals.items():
        if isinstance(fn, Expr):
            expr = fn
        else:
            v = test_ref(name)
            u = unknown_ref(name)
            p = param_ref()
            try:
                expr = fn(v, u, p)
            except TypeError:
                expr = fn(v, u)
        compiled[name] = _as_expr(expr)
        includes_measure[name] = _expr_contains(compiled[name], "volume_measure")
        if not includes_measure[name]:
            raise ValueError(f"Mixed residual '{name}' must include dOmega().")

    def _form(ctx, u_elem, params):
        return {name: expr.eval(ctx, params, u_elem=u_elem) for name, expr in compiled.items()}

    _form._includes_measure = includes_measure
    return _form


class MixedWeakForm:
    """Container for mixed weak-form residuals keyed by field name."""

    def __init__(self, *, residuals: dict[str, Callable]):
        self.residuals = residuals

    def compile(self):
        if not self.residuals:
            raise ValueError("residuals are not defined")
        return compile_mixed_residual(self.residuals)


def _eval_expr(expr: Expr, ctx, params, u_elem=None):
    op = expr.op
    args = expr.args

    if op == "lit":
        return args[0]
    if op == "param":
        return params
    if op == "getattr":
        base = _eval_value(args[0], ctx, params, u_elem=u_elem)
        name = args[1]
        if isinstance(base, dict):
            return base[name]
        return getattr(base, name)
    if op == "field":
        role, name = args
        if name is not None:
            if role == "trial" and getattr(ctx, "trial_fields", None) is not None:
                if name in ctx.trial_fields:
                    return ctx.trial_fields[name]
            if role == "test" and getattr(ctx, "test_fields", None) is not None:
                if name in ctx.test_fields:
                    return ctx.test_fields[name]
            if role == "unknown" and getattr(ctx, "unknown_fields", None) is not None:
                if name in ctx.unknown_fields:
                    return ctx.unknown_fields[name]
            fields = getattr(ctx, "fields", None)
            if fields is not None and name in fields:
                group = fields[name]
                if isinstance(group, dict):
                    if role in group:
                        return group[role]
                    if "field" in group:
                        return group["field"]
                return group
        if role == "trial":
            return ctx.trial
        if role == "test":
            return ctx.test
        if role == "unknown":
            return getattr(ctx, "unknown", ctx.trial)
        raise ValueError(f"Unknown field role: {role}")
    if op == "value":
        field = _eval_field(args[0], ctx, params)
        if isinstance(args[0], FieldRef) and args[0].role == "unknown":
            return _eval_unknown_value(args[0], field, u_elem)
        return field.N
    if op == "grad":
        field = _eval_field(args[0], ctx, params)
        if isinstance(args[0], FieldRef) and args[0].role == "unknown":
            return _eval_unknown_grad(args[0], field, u_elem)
        return field.gradN
    if op == "pow":
        base = _eval_value(args[0], ctx, params, u_elem=u_elem)
        exp = _eval_value(args[1], ctx, params, u_elem=u_elem)
        return base**exp
    if op == "eye":
        return jnp.eye(int(args[0]))
    if op == "det":
        return jnp.linalg.det(_eval_value(args[0], ctx, params, u_elem=u_elem))
    if op == "inv":
        return jnp.linalg.inv(_eval_value(args[0], ctx, params, u_elem=u_elem))
    if op == "transpose":
        return jnp.swapaxes(_eval_value(args[0], ctx, params, u_elem=u_elem), -1, -2)
    if op == "log":
        return jnp.log(_eval_value(args[0], ctx, params, u_elem=u_elem))
    if op == "surface_normal":
        normal = getattr(ctx, "normal", None)
        if normal is None:
            raise ValueError("surface normal is not available in context")
        return normal
    if op == "surface_measure":
        if not hasattr(ctx, "w") or not hasattr(ctx, "detJ"):
            raise ValueError("surface measure requires surface context with w and detJ.")
        return ctx.w * ctx.detJ
    if op == "volume_measure":
        if not hasattr(ctx, "w") or not hasattr(ctx, "test"):
            raise ValueError("volume measure requires FormContext with w and test.detJ.")
        return ctx.w * ctx.test.detJ
    if op == "sym_grad":
        field = _eval_field(args[0], ctx, params)
        if isinstance(args[0], FieldRef) and args[0].role == "unknown":
            if u_elem is None:
                raise ValueError("u_elem is required to evaluate unknown sym_grad.")
            u_local = _extract_unknown_elem(args[0], u_elem)
            return _ops.sym_grad_u(field, u_local)
        return _ops.sym_grad(field)
    if op == "outer":
        a, b = args
        if not isinstance(a, FieldRef) or not isinstance(b, FieldRef):
            raise TypeError("outer expects FieldRef operands.")
        if a.role == b.role:
            raise ValueError("outer requires one trial and one test field.")
        test = a if a.role == "test" else b
        trial = b if a.role == "test" else a
        v_field = _eval_field(test, ctx, params)
        u_field = _eval_field(trial, ctx, params)
        if getattr(v_field, "value_dim", 1) != 1 or getattr(u_field, "value_dim", 1) != 1:
            raise ValueError("u*v is only defined for scalar fields; use dot/inner for vectors.")
        vN = v_field.N
        uN = u_field.N
        return jnp.einsum("qi,qj->qij", vN, uN)
    if op == "add":
        return _eval_value(args[0], ctx, params, u_elem=u_elem) + _eval_value(args[1], ctx, params, u_elem=u_elem)
    if op == "sub":
        return _eval_value(args[0], ctx, params, u_elem=u_elem) - _eval_value(args[1], ctx, params, u_elem=u_elem)
    if op == "mul":
        a = _eval_value(args[0], ctx, params, u_elem=u_elem)
        b = _eval_value(args[1], ctx, params, u_elem=u_elem)
        if hasattr(a, "ndim") and hasattr(b, "ndim"):
            if a.ndim == 1 and b.ndim == 2 and a.shape[0] == b.shape[0]:
                a = a[:, None]
            elif b.ndim == 1 and a.ndim == 2 and b.shape[0] == a.shape[0]:
                b = b[:, None]
            elif a.ndim >= 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
                b = b.reshape((b.shape[0],) + (1,) * (a.ndim - 1))
            elif b.ndim >= 2 and a.ndim == 1 and b.shape[0] == a.shape[0]:
                a = a.reshape((a.shape[0],) + (1,) * (b.ndim - 1))
        return a * b
    if op == "matmul":
        a = _eval_value(args[0], ctx, params, u_elem=u_elem)
        b = _eval_value(args[1], ctx, params, u_elem=u_elem)
        if (
            hasattr(a, "ndim")
            and hasattr(b, "ndim")
            and a.ndim == 3
            and b.ndim == 3
            and a.shape[0] == b.shape[0]
            and a.shape[-1] == b.shape[-1]
        ):
            return jnp.einsum("qia,qja->qij", a, b)
        return a @ b
    if op == "neg":
        return -_eval_value(args[0], ctx, params, u_elem=u_elem)
    if op == "dot":
        if isinstance(args[0], FieldRef):
            return _ops.dot(_eval_field(args[0], ctx, params), _eval_value(args[1], ctx, params, u_elem=u_elem))
        a = _eval_value(args[0], ctx, params, u_elem=u_elem)
        b = _eval_value(args[1], ctx, params, u_elem=u_elem)
        if hasattr(a, "ndim") and hasattr(b, "ndim") and a.ndim == 3 and b.ndim == 3 and a.shape[-1] == b.shape[-1]:
            return jnp.einsum("qia,qja->qij", a, b)
        return jnp.matmul(a, b)
    if op == "sdot":
        if isinstance(args[0], FieldRef):
            return _ops.dot(_eval_field(args[0], ctx, params), _eval_value(args[1], ctx, params, u_elem=u_elem))
        a = _eval_value(args[0], ctx, params, u_elem=u_elem)
        b = _eval_value(args[1], ctx, params, u_elem=u_elem)
        if hasattr(a, "ndim") and hasattr(b, "ndim") and a.ndim == 3 and b.ndim == 3 and a.shape[-1] == b.shape[-1]:
            return jnp.einsum("qia,qja->qij", a, b)
        return jnp.matmul(a, b)
    if op == "ddot":
        if len(args) == 2:
            a = _eval_value(args[0], ctx, params, u_elem=u_elem)
            b = _eval_value(args[1], ctx, params, u_elem=u_elem)
            if (
                hasattr(a, "ndim")
                and hasattr(b, "ndim")
                and a.ndim == 3
                and b.ndim == 3
                and a.shape[0] == b.shape[0]
                and a.shape[1] == b.shape[1]
            ):
                return jnp.einsum("qik,qim->qkm", a, b)
            return _ops.ddot(a, b)
        return _ops.ddot(
            _eval_value(args[0], ctx, params, u_elem=u_elem),
            _eval_value(args[1], ctx, params, u_elem=u_elem),
            _eval_value(args[2], ctx, params, u_elem=u_elem),
        )
    if op == "inner":
        a = _eval_value(args[0], ctx, params, u_elem=u_elem)
        b = _eval_value(args[1], ctx, params, u_elem=u_elem)
        return jnp.einsum("...i,...i->...", a, b)
    if op == "action":
        if isinstance(args[1], FieldRef):
            raise ValueError("action expects a scalar expression; use u.val for unknowns.")
        v_field = _eval_field(args[0], ctx, params)
        s = _eval_value(args[1], ctx, params, u_elem=u_elem)
        value_dim = int(getattr(v_field, "value_dim", 1))
        if value_dim == 1:
            if v_field.N.ndim != 2:
                raise ValueError("action expects scalar test field with N shape (q, ndofs).")
            if hasattr(s, "ndim") and s.ndim not in (0, 1):
                raise ValueError("action expects scalar s with shape (q,) or scalar.")
            return v_field.N * s
        if hasattr(s, "ndim") and s.ndim not in (1, 2):
            raise ValueError("action expects vector s with shape (q, dim) or (dim,).")
        return _ops.dot(v_field, s)
    if op == "gaction":
        v_field = _eval_field(args[0], ctx, params)
        q = _eval_value(args[1], ctx, params, u_elem=u_elem)
        if v_field.gradN.ndim != 3:
            raise ValueError("gaction expects test gradient with shape (q, ndofs, dim).")
        if not hasattr(q, "ndim"):
            raise ValueError("gaction expects q with shape (q, dim) or (q, dim, dim).")
        if q.ndim == 2:
            return jnp.einsum("qaj,qj->qa", v_field.gradN, q)
        if q.ndim == 3:
            if int(getattr(v_field, "value_dim", 1)) == 1:
                raise ValueError("gaction tensor flux requires vector test field.")
            return jnp.einsum("qij,qaj->qai", q, v_field.gradN).reshape(q.shape[0], -1)
        raise ValueError("gaction expects q with shape (q, dim) or (q, dim, dim).")
    if op == "transpose_last2":
        return _ops.transpose_last2(_eval_value(args[0], ctx, params, u_elem=u_elem))
    if op == "einsum":
        subscripts = args[0]
        operands = [_eval_value(arg, ctx, params, u_elem=u_elem) for arg in args[1:]]
        return jnp.einsum(subscripts, *operands)

    raise ValueError(f"Unknown Expr op: {op}")


__all__ = [
    "Expr",
    "FieldRef",
    "ParamRef",
    "trial_ref",
    "test_ref",
    "unknown_ref",
    "param_ref",
    "Params",
    "MixedWeakForm",
    "ResidualForm",
    "compile_bilinear",
    "compile_linear",
    "compile_residual",
    "compile_mixed_residual",
    "grad",
    "sym_grad",
    "dot",
    "ddot",
    "inner",
    "action",
    "gaction",
    "I",
    "det",
    "inv",
    "transpose",
    "log",
    "transpose_last2",
    "einsum",
]
