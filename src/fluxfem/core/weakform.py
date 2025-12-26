from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, get_args
import inspect

import numpy as np

import jax.numpy as jnp
import jax

from ..physics import operators as _ops
from .context_types import FormFieldLike, ParamsLike, SurfaceContext, UElement, VolumeContext


OpName = Literal[
    "lit",
    "getattr",
    "value",
    "grad",
    "pow",
    "eye",
    "det",
    "inv",
    "transpose",
    "log",
    "surface_normal",
    "surface_measure",
    "volume_measure",
    "sym_grad",
    "outer",
    "add",
    "sub",
    "mul",
    "matmul",
    "matmul_std",
    "neg",
    "dot",
    "sdot",
    "ddot",
    "inner",
    "action",
    "gaction",
    "transpose_last2",
    "einsum",
]

# Use OpName as the single source of truth for valid ops.
_OP_NAMES: frozenset[str] = frozenset(get_args(OpName))


_PRECEDENCE: dict[str, int] = {
    "add": 10,
    "sub": 10,
    "mul": 20,
    "matmul": 20,
    "matmul_std": 20,
    "inner": 20,
    "dot": 20,
    "sdot": 20,
    "ddot": 20,
    "pow": 30,
    "neg": 40,
    "transpose": 50,
}


def _pretty_render_arg(arg, prec: int | None = None) -> str:
    if isinstance(arg, Expr):
        return _pretty_expr(arg, prec or 0)
    if isinstance(arg, FieldRef):
        if arg.name is None:
            return f"{arg.role}"
        return f"{arg.role}:{arg.name}"
    if isinstance(arg, ParamRef):
        return "param"
    return repr(arg)


def _pretty_wrap(text: str, prec: int, parent_prec: int) -> str:
    if prec < parent_prec:
        return f"({text})"
    return text


def _pretty_expr(expr: Expr, parent_prec: int = 0) -> str:
    op = expr.op
    args = expr.args

    if op == "lit":
        return repr(args[0])
    if op == "getattr":
        base = _pretty_render_arg(args[0], _PRECEDENCE.get("transpose", 50))
        return f"{base}.{args[1]}"
    if op == "value":
        return f"val({_pretty_render_arg(args[0])})"
    if op == "grad":
        return f"grad({_pretty_render_arg(args[0])})"
    if op == "sym_grad":
        return f"sym_grad({_pretty_render_arg(args[0])})"
    if op == "neg":
        inner = _pretty_render_arg(args[0], _PRECEDENCE["neg"])
        return _pretty_wrap(f"-{inner}", _PRECEDENCE["neg"], parent_prec)
    if op == "transpose":
        inner = _pretty_render_arg(args[0], _PRECEDENCE["transpose"])
        return _pretty_wrap(f"{inner}.T", _PRECEDENCE["transpose"], parent_prec)
    if op == "pow":
        base = _pretty_render_arg(args[0], _PRECEDENCE["pow"])
        exp = _pretty_render_arg(args[1], _PRECEDENCE["pow"] + 1)
        return _pretty_wrap(f"{base}**{exp}", _PRECEDENCE["pow"], parent_prec)
    if op in {"add", "sub", "mul", "matmul", "dot", "sdot", "ddot"}:
        left = _pretty_render_arg(args[0], _PRECEDENCE[op])
        right = _pretty_render_arg(args[1], _PRECEDENCE[op] + 1)
        symbol = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "matmul": "@",
            "inner": "|",
            "dot": "dot",
            "sdot": "sdot",
            "ddot": "ddot",
        }[op]
        if symbol in {"dot", "sdot", "ddot"}:
            text = f"{symbol}({left}, {right})"
        else:
            text = f"{left} {symbol} {right}"
        return _pretty_wrap(text, _PRECEDENCE[op], parent_prec)
    if op == "inner":
        return f"inner({_pretty_render_arg(args[0])}, {_pretty_render_arg(args[1])})"
    if op in {"action", "gaction"}:
        return f"{op}({_pretty_render_arg(args[0])}, {_pretty_render_arg(args[1])})"
    if op == "matmul_std":
        return f"matmul_std({_pretty_render_arg(args[0])}, {_pretty_render_arg(args[1])})"
    if op == "outer":
        return f"outer({_pretty_render_arg(args[0])}, {_pretty_render_arg(args[1])})"
    if op in {
        "eye",
        "det",
        "inv",
        "log",
        "surface_normal",
        "surface_measure",
        "volume_measure",
        "transpose_last2",
        "einsum",
    }:
        rendered = ", ".join(_pretty_render_arg(arg) for arg in args)
        return f"{op}({rendered})"
    rendered = ", ".join(_pretty_render_arg(arg) for arg in args)
    return f"{op}({rendered})"


def _as_expr(obj) -> Expr | FieldRef | ParamRef:
    """Normalize inputs into Expr/FieldRef/ParamRef nodes."""
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, FieldRef):
        return obj
    if isinstance(obj, ParamRef):
        return obj
    if isinstance(obj, (int, float, bool, str)):
        return Expr("lit", obj)
    if isinstance(obj, np.generic):
        return Expr("lit", obj.item())
    if isinstance(obj, tuple):
        try:
            hash(obj)
        except TypeError as exc:
            raise TypeError(
                "Expr tuple literal must be hashable; use only immutable items."
            ) from exc
        return Expr("lit", obj)
    raise TypeError(
        "Expr literal must be a scalar or hashable tuple. "
        "Arrays are not allowed; pass them via params (ParamRef/params.xxx)."
    )


@dataclass(frozen=True, slots=True, init=False)
class Expr:
    """Expression tree node evaluated against a FormContext.

    Compile flow (recommended):
    - build an Expr via operators/refs
    - compile_* builds an EvalPlan (postorder nodes + index)
    - eval_with_plan(plan, ctx, params, u_elem) evaluates per element

    Expr.eval is a debug/single-shot path that creates a plan on demand.
    """

    op: OpName
    args: tuple[Any, ...]

    def __init__(self, op: OpName, *args):
        if op not in _OP_NAMES:
            raise ValueError(f"Unknown Expr op: {op!r}")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", args)

    def eval(self, ctx, params=None, u_elem=None):
        """Evaluate the expression against a context (debug/single-shot path)."""
        return _eval_expr(self, ctx, params, u_elem=u_elem)

    def children(self) -> tuple[Any, ...]:
        """Return direct child nodes (Expr/FieldRef/ParamRef) for traversal."""
        return tuple(arg for arg in self.args if isinstance(arg, (Expr, FieldRef, ParamRef)))

    def walk(self) -> Iterator[Any]:
        """Depth-first walk over nodes, including leaf FieldRef/ParamRef."""
        yield self
        for child in self.children():
            if isinstance(child, Expr):
                yield from child.walk()
            else:
                yield child

    def postorder(self) -> Iterator[Any]:
        """Postorder walk over nodes, including leaf FieldRef/ParamRef."""
        for child in self.children():
            if isinstance(child, Expr):
                yield from child.postorder()
            else:
                yield child
        yield self

    def postorder_expr(self) -> Iterator["Expr"]:
        """Postorder walk over Expr nodes only (for eval planning)."""
        for arg in self.args:
            if isinstance(arg, Expr):
                yield from arg.postorder_expr()
        yield self

    def _binop(self, other, op):
        return Expr(op, self, _as_expr(other))

    def __add__(self, other):
        """Add expressions: `a + b`."""
        return self._binop(other, "add")

    def __radd__(self, other):
        """Right-add expressions: `1 + expr`."""
        return Expr("add", _as_expr(other), self)

    def __sub__(self, other):
        """Subtract expressions: `a - b`."""
        return self._binop(other, "sub")

    def __rsub__(self, other):
        """Right-subtract expressions: `1 - expr`."""
        return Expr("sub", _as_expr(other), self)

    def __mul__(self, other):
        """Multiply expressions: `a * b`."""
        return self._binop(other, "mul")

    def __rmul__(self, other):
        """Right-multiply expressions: `2 * expr`."""
        return Expr("mul", _as_expr(other), self)

    def __matmul__(self, other):
        """Matrix product: `a @ b` (FEM-specific contraction semantics)."""
        return self._binop(other, "matmul")

    def __rmatmul__(self, other):
        """Right-matmul: `A @ expr`."""
        return Expr("matmul", _as_expr(other), self)

    def __or__(self, other):
        """Tensor inner product: `a | b` (use .val/.grad for FieldRef)."""
        if isinstance(other, FieldRef):
            raise TypeError("FieldRef | FieldRef is not supported; use outer(test, trial).")
        return Expr("inner", self, _as_expr(other))

    def __ror__(self, other):
        """Tensor inner product: `a | b` (use .val/.grad for FieldRef)."""
        if isinstance(other, FieldRef):
            raise TypeError("FieldRef | FieldRef is not supported; use outer(test, trial).")
        return Expr("inner", _as_expr(other), self)

    def __pow__(self, power, modulo=None):
        """Power: `a ** p` (no modulo support)."""
        if modulo is not None:
            raise ValueError("modulo is not supported for Expr exponentiation.")
        return Expr("pow", self, _as_expr(power))

    def __neg__(self):
        """Unary negation: `-expr`."""
        return Expr("neg", self)

    @property
    def T(self):
        """Transpose view: `expr.T`."""
        return Expr("transpose", self)

    def __repr__(self) -> str:
        return self.pretty()

    def pretty(self) -> str:
        return _pretty_expr(self)


@dataclass(frozen=True, slots=True)
class FieldRef:
    """Symbolic reference to trial/test/unknown field, optionally by name."""

    role: str
    name: str | None = None

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
            raise TypeError(
                "FieldRef * FieldRef is ambiguous; use outer(v, u) (test, trial), "
                "action(v, s), or dot(v, q)."
            )
        return Expr("mul", Expr("value", self), _as_expr(other))

    def __rmul__(self, other):
        if isinstance(other, FieldRef):
            raise TypeError(
                "FieldRef * FieldRef is ambiguous; use outer(v, u) (test, trial), "
                "action(v, s), or dot(v, q)."
            )
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
            raise TypeError(
                "FieldRef | FieldRef is not supported; use outer(test, trial) for basis kernels."
            )
        return Expr("dot", self, _as_expr(other))

    def __ror__(self, other):
        if isinstance(other, FieldRef):
            raise TypeError(
                "FieldRef | FieldRef is not supported; use outer(test, trial) for basis kernels."
            )
        return Expr("dot", _as_expr(other), self)


@dataclass(frozen=True, slots=True)
class ParamRef:
    """Symbolic reference to params passed into the kernel."""

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


def _eval_field(
    obj: Any,
    ctx: VolumeContext | SurfaceContext,
    params: ParamsLike,
) -> FormFieldLike:
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
    raise TypeError("Expected a field reference for this operator.")


# def _eval_value(obj: Any, ctx, params, u_elem=None):
#     if isinstance(obj, FieldRef):
#         field = _eval_field(obj, ctx, params)
#         if obj.role == "unknown":
#             return _eval_unknown_value(obj, field, u_elem)
#         return field.N
#     if isinstance(obj, ParamRef):
#         return params
#     if isinstance(obj, Expr):
#         return obj.eval(ctx, params, u_elem=u_elem)
#     return obj


def _extract_unknown_elem(field_ref: FieldRef, u_elem: UElement):
    if u_elem is None:
        raise ValueError("u_elem is required to evaluate unknown field value.")
    if isinstance(u_elem, dict):
        name = field_ref.name or "u"
        if name not in u_elem:
            raise ValueError(f"u_elem is missing key '{name}'.")
        return u_elem[name]
    return u_elem


def _basis_outer(test: FieldRef, trial: FieldRef, ctx, params):
    v_field = _eval_field(test, ctx, params)
    u_field = _eval_field(trial, ctx, params)
    if getattr(v_field, "value_dim", 1) != 1 or getattr(u_field, "value_dim", 1) != 1:
        raise ValueError(
            "inner/outer is only defined for scalar fields; use dot/action/einsum for vector/tensor cases."
        )
    return jnp.einsum("qi,qj->qij", v_field.N, u_field.N)


def _eval_unknown_value(field_ref: FieldRef, field: FormFieldLike, u_elem: UElement):
    u_local = _extract_unknown_elem(field_ref, u_elem)
    value_dim = int(getattr(field, "value_dim", 1))
    if value_dim == 1:
        return jnp.einsum("qa,a->q", field.N, u_local)
    u_nodes = u_local.reshape((-1, value_dim))
    return jnp.einsum("qa,ai->qi", field.N, u_nodes)


def _eval_unknown_grad(field_ref: FieldRef, field: FormFieldLike, u_elem: UElement):
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


def outer(a, b) -> Expr:
    """Outer product of scalar fields: `outer(v, u)` (test, trial)."""
    if not isinstance(a, FieldRef) or not isinstance(b, FieldRef):
        raise TypeError("outer expects FieldRef operands.")
    if a.role != "test" or b.role != "trial":
        raise TypeError("outer expects outer(test, trial).")
    return Expr("outer", a, b)


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
    """Inner product over the last axis (tensor-level)."""
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


def matmul(a, b) -> Expr:
    """FEM-specific batched contraction (same semantics as `@`)."""
    return Expr("matmul", _as_expr(a), _as_expr(b))


def matmul_std(a, b) -> Expr:
    """Standard matrix product (`jnp.matmul` semantics)."""
    return Expr("matmul_std", _as_expr(a), _as_expr(b))


def einsum(subscripts: str, *args) -> Expr:
    """Einsum wrapper that supports Expr inputs."""
    return Expr("einsum", subscripts, *[_as_expr(arg) for arg in args])


def _call_user(fn, *args, params):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, params)

    params_list = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params_list):
        return fn(*args, params)
    positional = [
        p
        for p in params_list
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    max_positional = len(positional)
    if len(args) + 1 <= max_positional:
        return fn(*args, params)
    return fn(*args)


def compile_bilinear(fn):
    """get_compiled a bilinear weak form (u, v, params) -> Expr into a kernel."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        u = trial_ref()
        v = test_ref()
        p = param_ref()
        expr = _call_user(fn, u, v, params=p)
    expr = _as_expr(expr)
    if not isinstance(expr, Expr):
        raise TypeError("Bilinear form must return an Expr.")

    volume_count = _count_op(expr, "volume_measure")
    surface_count = _count_op(expr, "surface_measure")
    if volume_count == 0:
        raise ValueError("Volume bilinear form must include dOmega().")
    if volume_count > 1:
        raise ValueError("Volume bilinear form must include dOmega() exactly once.")
    if surface_count > 0:
        raise ValueError("Volume bilinear form must not include ds().")

    plan = make_eval_plan(expr)

    def _form(ctx, params):
        return eval_with_plan(plan, ctx, params)

    _form._includes_measure = True
    return _form


def compile_linear(fn):
    """get_compiled a linear weak form (v, params) -> Expr into a kernel."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        v = test_ref()
        p = param_ref()
        expr = _call_user(fn, v, params=p)
    expr = _as_expr(expr)
    if not isinstance(expr, Expr):
        raise TypeError("Linear form must return an Expr.")

    volume_count = _count_op(expr, "volume_measure")
    surface_count = _count_op(expr, "surface_measure")
    if volume_count == 0:
        raise ValueError("Volume linear form must include dOmega().")
    if volume_count > 1:
        raise ValueError("Volume linear form must include dOmega() exactly once.")
    if surface_count > 0:
        raise ValueError("Volume linear form must not include ds().")

    plan = make_eval_plan(expr)

    def _form(ctx, params):
        return eval_with_plan(plan, ctx, params)

    _form._includes_measure = True
    return _form


def _expr_contains(expr: Expr, op: str) -> bool:
    if not isinstance(expr, Expr):
        return False
    if expr.op == op:
        return True
    return any(_expr_contains(arg, op) for arg in expr.args if isinstance(arg, Expr))


def _count_op(expr: Expr, op: str) -> int:
    if not isinstance(expr, Expr):
        return 0
    count = 1 if expr.op == op else 0
    for arg in expr.args:
        if isinstance(arg, Expr):
            count += _count_op(arg, op)
    return count


@dataclass(frozen=True, slots=True)
class EvalPlan:
    expr: Expr
    nodes: tuple[Expr, ...]
    index: dict[int, int]


def _validate_eval_plan(nodes: tuple[Expr, ...]) -> None:
    fieldref_arg_ops = {
        "value",
        "grad",
        "sym_grad",
        "dot",
        "sdot",
        "action",
        "gaction",
        "outer",
    }
    for node in nodes:
        op = node.op
        args = node.args
        if op not in fieldref_arg_ops:
            if any(isinstance(arg, FieldRef) for arg in args):
                raise TypeError(f"{op} cannot take FieldRef directly; wrap with .val/.grad/.sym_grad.")
        if op in {"value", "grad", "sym_grad"}:
            if len(args) != 1 or not isinstance(args[0], FieldRef):
                raise TypeError(f"{op} expects FieldRef.")
        elif op in {"dot", "sdot"}:
            if len(args) != 2:
                raise TypeError(f"{op} expects two arguments.")
            if any(isinstance(arg, FieldRef) for arg in args):
                if not isinstance(args[0], FieldRef):
                    raise TypeError(f"{op} expects FieldRef as the first argument.")
                if isinstance(args[1], FieldRef):
                    raise TypeError(f"{op} expects an expression for the second argument; use .val/.grad.")
        elif op in {"action", "gaction"}:
            if len(args) != 2 or not isinstance(args[0], FieldRef):
                raise TypeError(f"{op} expects FieldRef as the first argument.")
            if op == "action" and isinstance(args[1], FieldRef):
                raise ValueError("action expects a scalar expression; use u.val for unknowns.")
            if op == "gaction" and isinstance(args[1], FieldRef):
                raise TypeError("gaction expects an expression for the second argument; use .grad.")
        elif op == "outer":
            if len(args) != 2 or not all(isinstance(arg, FieldRef) for arg in args):
                raise TypeError("outer expects two FieldRef operands.")
            if args[0].role != "test" or args[1].role != "trial":
                raise TypeError("outer expects outer(test, trial).")


def make_eval_plan(expr: Expr) -> EvalPlan:
    nodes = tuple(expr.postorder_expr())
    _validate_eval_plan(nodes)
    index: dict[int, int] = {}
    for i, node in enumerate(nodes):
        index.setdefault(id(node), i)
    return EvalPlan(expr=expr, nodes=nodes, index=index)


def eval_with_plan(
    plan: EvalPlan,
    ctx: VolumeContext | SurfaceContext,
    params: ParamsLike,
    u_elem: UElement | None = None,
):
    nodes = plan.nodes
    index = plan.index
    vals: list[Any] = [None] * len(nodes)

    def get(obj):
        if isinstance(obj, Expr):
            return vals[index[id(obj)]]
        if isinstance(obj, FieldRef):
            raise TypeError(
                "FieldRef must be wrapped with .val/.grad/.sym_grad or used as the first arg of dot/action."
            )
        if isinstance(obj, ParamRef):
            return params
        return obj

    for i, node in enumerate(nodes):
        op = node.op
        args = node.args

        if op == "lit":
            vals[i] = args[0]
            continue
        if op == "getattr":
            base = get(args[0])
            name = args[1]
            if isinstance(base, dict):
                vals[i] = base[name]
            else:
                vals[i] = getattr(base, name)
            continue
        if op == "value":
            ref = args[0]
            assert isinstance(ref, FieldRef)
            field = _eval_field(ref, ctx, params)
            if ref.role == "unknown":
                vals[i] = _eval_unknown_value(ref, field, u_elem)
            else:
                vals[i] = field.N
            continue
        if op == "grad":
            ref = args[0]
            assert isinstance(ref, FieldRef)
            field = _eval_field(ref, ctx, params)
            if ref.role == "unknown":
                vals[i] = _eval_unknown_grad(ref, field, u_elem)
            else:
                vals[i] = field.gradN
            continue
        if op == "pow":
            base = get(args[0])
            exp = get(args[1])
            vals[i] = base**exp
            continue
        if op == "eye":
            vals[i] = jnp.eye(int(args[0]))
            continue
        if op == "det":
            vals[i] = jnp.linalg.det(get(args[0]))
            continue
        if op == "inv":
            vals[i] = jnp.linalg.inv(get(args[0]))
            continue
        if op == "transpose":
            vals[i] = jnp.swapaxes(get(args[0]), -1, -2)
            continue
        if op == "log":
            vals[i] = jnp.log(get(args[0]))
            continue
        if op == "surface_normal":
            normal = getattr(ctx, "normal", None)
            if normal is None:
                raise ValueError("surface normal is not available in context")
            vals[i] = normal
            continue
        if op == "surface_measure":
            if not hasattr(ctx, "w") or not hasattr(ctx, "detJ"):
                raise TypeError("surface measure requires SurfaceContext.")
            vals[i] = ctx.w * ctx.detJ
            continue
        if op == "volume_measure":
            if not hasattr(ctx, "w") or not hasattr(ctx, "test"):
                raise TypeError("volume measure requires VolumeContext.")
            vals[i] = ctx.w * ctx.test.detJ
            continue
        if op == "sym_grad":
            ref = args[0]
            assert isinstance(ref, FieldRef)
            field = _eval_field(ref, ctx, params)
            if ref.role == "unknown":
                if u_elem is None:
                    raise ValueError("u_elem is required to evaluate unknown sym_grad.")
                u_local = _extract_unknown_elem(ref, u_elem)
                vals[i] = _ops.sym_grad_u(field, u_local)
            else:
                vals[i] = _ops.sym_grad(field)
            continue
        if op == "outer":
            a, b = args
            if not isinstance(a, FieldRef) or not isinstance(b, FieldRef):
                raise TypeError("outer expects FieldRef operands.")
            test, trial = a, b
            vals[i] = _basis_outer(test, trial, ctx, params)
            continue
        if op == "add":
            vals[i] = get(args[0]) + get(args[1])
            continue
        if op == "sub":
            vals[i] = get(args[0]) - get(args[1])
            continue
        if op == "mul":
            a = get(args[0])
            b = get(args[1])
            if hasattr(a, "ndim") and hasattr(b, "ndim"):
                if a.ndim == 1 and b.ndim == 2 and a.shape[0] == b.shape[0]:
                    a = a[:, None]
                elif b.ndim == 1 and a.ndim == 2 and b.shape[0] == a.shape[0]:
                    b = b[:, None]
                elif a.ndim >= 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
                    b = b.reshape((b.shape[0],) + (1,) * (a.ndim - 1))
                elif b.ndim >= 2 and a.ndim == 1 and b.shape[0] == a.shape[0]:
                    a = a.reshape((a.shape[0],) + (1,) * (b.ndim - 1))
            vals[i] = a * b
            continue
        if op == "matmul":
            a = get(args[0])
            b = get(args[1])
            if (
                hasattr(a, "ndim")
                and hasattr(b, "ndim")
                and a.ndim == 3
                and b.ndim == 3
                and a.shape[0] == b.shape[0]
                and a.shape[-1] == b.shape[-1]
            ):
                vals[i] = jnp.einsum("qia,qja->qij", a, b)
            else:
                raise TypeError(
                    "Expr '@' (matmul) is FEM-specific; use matmul_std(a, b) for standard matmul."
                )
            continue
        if op == "matmul_std":
            a = get(args[0])
            b = get(args[1])
            vals[i] = jnp.matmul(a, b)
            continue
        if op == "neg":
            vals[i] = -get(args[0])
            continue
        if op == "dot":
            ref = args[0]
            if isinstance(ref, FieldRef):
                vals[i] = _ops.dot(_eval_field(ref, ctx, params), get(args[1]))
            else:
                a = get(args[0])
                b = get(args[1])
                if (
                    hasattr(a, "ndim")
                    and hasattr(b, "ndim")
                    and a.ndim == 3
                    and b.ndim == 3
                    and a.shape[-1] == b.shape[-1]
                ):
                    vals[i] = jnp.einsum("qia,qja->qij", a, b)
                else:
                    vals[i] = jnp.matmul(a, b)
            continue
        if op == "sdot":
            ref = args[0]
            if isinstance(ref, FieldRef):
                vals[i] = _ops.dot(_eval_field(ref, ctx, params), get(args[1]))
            else:
                a = get(args[0])
                b = get(args[1])
                if (
                    hasattr(a, "ndim")
                    and hasattr(b, "ndim")
                    and a.ndim == 3
                    and b.ndim == 3
                    and a.shape[-1] == b.shape[-1]
                ):
                    vals[i] = jnp.einsum("qia,qja->qij", a, b)
                else:
                    vals[i] = jnp.matmul(a, b)
            continue
        if op == "ddot":
            if len(args) == 2:
                a = get(args[0])
                b = get(args[1])
                if (
                    hasattr(a, "ndim")
                    and hasattr(b, "ndim")
                    and a.ndim == 3
                    and b.ndim == 3
                    and a.shape[0] == b.shape[0]
                    and a.shape[1] == b.shape[1]
                ):
                    vals[i] = jnp.einsum("qik,qim->qkm", a, b)
                else:
                    vals[i] = _ops.ddot(a, b)
            else:
                vals[i] = _ops.ddot(get(args[0]), get(args[1]), get(args[2]))
            continue
        if op == "inner":
            a = get(args[0])
            b = get(args[1])
            vals[i] = jnp.einsum("...i,...i->...", a, b)
            continue
        if op == "action":
            ref = args[0]
            assert isinstance(ref, FieldRef)
            if isinstance(args[1], FieldRef):
                raise ValueError("action expects a scalar expression; use u.val for unknowns.")
            v_field = _eval_field(ref, ctx, params)
            s = get(args[1])
            value_dim = int(getattr(v_field, "value_dim", 1))
            # action maps a test field with a scalar/vector expression into nodal space.
            if value_dim == 1:
                if v_field.N.ndim != 2:
                    raise ValueError("action expects scalar test field with N shape (q, ndofs).")
                if hasattr(s, "ndim") and s.ndim not in (0, 1):
                    raise ValueError("action expects scalar s with shape (q,) or scalar.")
                vals[i] = v_field.N * s
            else:
                if hasattr(s, "ndim") and s.ndim not in (1, 2):
                    raise ValueError("action expects vector s with shape (q, dim) or (dim,).")
                vals[i] = _ops.dot(v_field, s)
            continue
        if op == "gaction":
            ref = args[0]
            assert isinstance(ref, FieldRef)
            v_field = _eval_field(ref, ctx, params)
            q = get(args[1])
            # gaction maps a flux-like expression to nodal space via test gradients.
            if v_field.gradN.ndim != 3:
                raise ValueError("gaction expects test gradient with shape (q, ndofs, dim).")
            if not hasattr(q, "ndim"):
                raise ValueError("gaction expects q with shape (q, dim) or (q, dim, dim).")
            if q.ndim == 2:
                vals[i] = jnp.einsum("qaj,qj->qa", v_field.gradN, q)
            elif q.ndim == 3:
                if int(getattr(v_field, "value_dim", 1)) == 1:
                    raise ValueError("gaction tensor flux requires vector test field.")
                vals[i] = jnp.einsum("qij,qaj->qai", q, v_field.gradN).reshape(q.shape[0], -1)
            else:
                raise ValueError("gaction expects q with shape (q, dim) or (q, dim, dim).")
            continue
        if op == "transpose_last2":
            vals[i] = _ops.transpose_last2(get(args[0]))
            continue
        if op == "einsum":
            subscripts = args[0]
            operands = [get(arg) for arg in args[1:]]
            vals[i] = jnp.einsum(subscripts, *operands)
            continue

        raise ValueError(f"Unknown Expr op: {op}")

    return vals[index[id(plan.expr)]]


def compile_surface_linear(fn):
    """get_compiled a surface linear form into a kernel (ctx, params) -> ndarray."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        v = test_ref()
        p = param_ref()
        expr = _call_user(fn, v, params=p)

    expr = _as_expr(expr)
    if not isinstance(expr, Expr):
        raise ValueError("Surface linear form must return an Expr; use ds() in the expression.")

    surface_count = _count_op(expr, "surface_measure")
    volume_count = _count_op(expr, "volume_measure")
    if surface_count == 0:
        raise ValueError("Surface linear form must include ds().")
    if surface_count > 1:
        raise ValueError("Surface linear form must include ds() exactly once.")
    if volume_count > 0:
        raise ValueError("Surface linear form must not include dOmega().")

    plan = make_eval_plan(expr)

    def _form(ctx, params):
        return eval_with_plan(plan, ctx, params)

    _form._includes_measure = True  # type: ignore[attr-defined]
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

    def get_compiled(self, *, ctx_kind: str | None = None):
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

    def get_compiled(self):
        return compile_bilinear(self.fn)


class ResidualForm:
    """Residual form wrapper (volume only for now)."""

    def __init__(self, fn):
        self.fn = fn

    @classmethod
    def volume(cls, fn):
        return cls(fn)

    def get_compiled(self):
        return compile_residual(self.fn)


def compile_residual(fn):
    """get_compiled a residual weak form (v, u, params) -> Expr into a kernel."""
    if isinstance(fn, Expr):
        expr = fn
    else:
        v = test_ref()
        u = unknown_ref()
        p = param_ref()
        expr = _call_user(fn, v, u, params=p)
    expr = _as_expr(expr)
    if not isinstance(expr, Expr):
        raise TypeError("Residual form must return an Expr.")

    volume_count = _count_op(expr, "volume_measure")
    surface_count = _count_op(expr, "surface_measure")
    if volume_count == 0:
        raise ValueError("Volume residual form must include dOmega().")
    if volume_count > 1:
        raise ValueError("Volume residual form must include dOmega() exactly once.")
    if surface_count > 0:
        raise ValueError("Volume residual form must not include ds().")

    plan = make_eval_plan(expr)

    def _form(ctx, u_elem, params):
        return eval_with_plan(plan, ctx, params, u_elem=u_elem)

    _form._includes_measure = True
    return _form


def compile_mixed_residual(residuals: dict[str, Callable]):
    """get_compiled mixed residuals keyed by field name."""
    compiled = {}
    plans = {}
    includes_measure = {}
    for name, fn in residuals.items():
        if isinstance(fn, Expr):
            expr = fn
        else:
            v = test_ref(name)
            u = unknown_ref(name)
            p = param_ref()
            expr = _call_user(fn, v, u, params=p)
        expr = _as_expr(expr)
        if not isinstance(expr, Expr):
            raise TypeError(f"Mixed residual '{name}' must return an Expr.")
        compiled[name] = expr
        plans[name] = make_eval_plan(expr)
        volume_count = _count_op(compiled[name], "volume_measure")
        surface_count = _count_op(compiled[name], "surface_measure")
        includes_measure[name] = volume_count == 1
        if volume_count == 0:
            raise ValueError(f"Mixed residual '{name}' must include dOmega().")
        if volume_count > 1:
            raise ValueError(f"Mixed residual '{name}' must include dOmega() exactly once.")
        if surface_count > 0:
            raise ValueError(f"Mixed residual '{name}' must not include ds().")

    def _form(ctx, u_elem, params):
        return {name: eval_with_plan(plan, ctx, params, u_elem=u_elem) for name, plan in plans.items()}

    _form._includes_measure = includes_measure
    return _form


class MixedWeakForm:
    """Container for mixed weak-form residuals keyed by field name."""

    def __init__(self, *, residuals: dict[str, Callable]):
        self.residuals = residuals

    def get_compiled(self):
        if not self.residuals:
            raise ValueError("residuals are not defined")
        return compile_mixed_residual(self.residuals)


def _eval_expr(
    expr: Expr,
    ctx: VolumeContext | SurfaceContext,
    params: ParamsLike,
    u_elem: UElement | None = None,
):
    plan = make_eval_plan(expr)
    return eval_with_plan(plan, ctx, params, u_elem=u_elem)


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
    "outer",
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
    "matmul",
    "matmul_std",
    "einsum",
]
