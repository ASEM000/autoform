# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IR data structures, primitives, interpreters, and IR building"""

from __future__ import annotations

import functools as ft
import itertools as it
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator, Hashable
from contextlib import contextmanager
from contextvars import ContextVar
from operator import setitem
from threading import RLock
from typing import Any, ClassVar, NoReturn, Protocol, Self, TypeGuard, cast

import autoform.utils as utils

type Tree[T] = utils.Tree[T]

__all__ = [
    # base types
    "AVal",
    "StrAVal",
    "IntAVal",
    "FloatAVal",
    "BoolAVal",
    "Val",
    "trace_types",
    "aval_rules",
    "is_traceable",
    "avalof",
    # ir vals
    "Var",
    "is_var",
    "aval_if_var",
    # primitive
    "Prim",
    # rule registries
    "impl_rules",
    "abstract_rules",
    "batch_rules",
    "push_rules",
    "pull_fwd_rules",
    "pull_bwd_rules",
    "InterpreterRule",
    "AsyncInterpreterRule",
    "ImplRule",
    "AImplRule",
    "AbstractRule",
    "AAbstractRule",
    "PushforwardRule",
    "APushforwardRule",
    "PullbackFwdRule",
    "APullbackFwdRule",
    "PullbackBwdRule",
    "APullbackBwdRule",
    "BatchRule",
    "ABatchRule",
    # ir structures
    "Eqn",
    "IR",
    # interpreters
    "BaseInterpreter",
    "BoxedInterpreter",
    "Interpreter",
    "EvalInterpreter",
    "TraceBox",
    "trace_add_rules",
    "trace_sub_rules",
    "trace_mul_rules",
    "trace_truediv_rules",
    "trace_matmul_rules",
    "trace_eq_rules",
    "TraceInterpreter",
    "active_interpreter",
    "using_interpreter",
    "active_tags",
    "tag",
    # ir building and execution
    "fold",
    "trace",
    "check_static_inputs",
    "walk",
]

# ==================================================================================================
# BASE TYPES
# ==================================================================================================


class AVal:
    """Base class for abstract values used by traced programs.

    Abstract values carry trace-time information about runtime values. Extension
    domains subclass ``AVal`` to describe the information primitive abstract
    rules need, such as shape, dtype, schema, or other static metadata.

    Example:
        >>> import autoform.extend as afe
        >>> class ArrayAVal(afe.AVal):
        ...     def __init__(self, shape, dtype):
        ...         self.shape = shape
        ...         self.dtype = dtype
    """

    __slots__ = []


class ScalarAVal(AVal):
    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


class StrAVal(ScalarAVal):
    """Abstract value for ``str`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)("x")
        >>> (x,) = ir.in_tree
        >>> x.aval
        StrAVal()
    """

    __slots__ = []


class IntAVal(ScalarAVal):
    """Abstract value for ``int`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1)
        >>> (x,) = ir.in_tree
        >>> x.aval
        IntAVal()
    """

    __slots__ = []


class FloatAVal(ScalarAVal):
    """Abstract value for ``float`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1.0)
        >>> (x,) = ir.in_tree
        >>> x.aval
        FloatAVal()
    """

    __slots__ = []


class BoolAVal(ScalarAVal):
    """Abstract value for ``bool`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(True)
        >>> (x,) = ir.in_tree
        >>> x.aval
        BoolAVal()
    """

    __slots__ = []


type Val = str | int | float | bool

trace_types: set[type] = {str, int, float, bool}

aval_rules: dict[type, Callable[[Any], AVal]] = {}
aval_rules[str] = lambda _: StrAVal()
aval_rules[int] = lambda _: IntAVal()
aval_rules[float] = lambda _: FloatAVal()
aval_rules[bool] = lambda _: BoolAVal()


def is_traceable(x) -> TypeGuard[Val]:
    return type(x) in trace_types


def is_aval(x) -> TypeGuard[AVal]:
    return isinstance(x, AVal)


type EvalType = AVal | Val


def avalof(x, /) -> AVal:
    """Return the abstract value for a traceable leaf.

    ``avalof`` applies the registered aval rule for ``type(x)``. It is the
    concrete-to-abstract direction used by :func:`autoform.trace`, zeros, and
    extension code that needs to inspect a value domain.

    Args:
        x: Concrete value, symbolic zero, or IR value with a registered aval
            rule.

    Returns:
        The abstract value for ``x``.

    Raises:
        TypeError: If no aval rule is registered for ``type(x)``.
    """
    rule = aval_rules.get(type(x))
    if rule is None:
        raise TypeError(f"Unsupported input leaf type for `trace`: {type(x).__name__}.")
    aval = rule(x)
    assert is_aval(aval), f"aval rule for {type(x).__name__} returned {aval!r}"
    return aval


# ==================================================================================================
# IR VARS
# ==================================================================================================


# NOTE(asem): wrapped IR leaves are variables (placeholders) for user inputs.
# Concrete literals are kept as plain Python values in IR trees.
class Var:
    """Symbolic variable stored in IR trees.

    ``Var`` leaves stand for runtime values inside traced programs. Each
    variable carries an :class:`AVal` describing its abstract value, and an
    optional source variable used by transforms that create rewritten IR.

    Args:
        aval: Abstract value for the runtime value represented by this variable.
        source: Optional original variable this one was derived from.
    """

    __slots__ = ["id", "source", "aval"]
    counter: ClassVar[it.count[int]] = it.count(0)
    lock: ClassVar[RLock] = RLock()

    def __init__(self, /, *, aval: AVal, source: Var | None = None):
        self.id = next(self.counter)
        assert is_var(source) or source is None
        assert is_aval(aval)
        self.source = source
        self.aval = aval

    @classmethod
    def fresh(cls, *, aval: AVal, source: Var | None = None) -> Self:
        with cls.lock:
            return cls(source=source, aval=aval)

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source else ""
        return f"{type(self).__name__}[{self.aval!r}](id={self.id}{source})"


def is_var(x) -> TypeGuard[Var]:
    """Return ``True`` if input is an :class:`Var`."""

    return isinstance(x, Var)


def aval_if_var(x, /):
    """Return the aval for an IR variable, otherwise return input unchanged.

    This is useful when constructing new IR trees from existing ones: concrete
    literals stay concrete, while symbolic variables are replaced by the
    abstract values needed to create fresh variables or abstract outputs.
    """

    return x.aval if is_var(x) else x


aval_rules[Var] = lambda var: var.aval

# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


class Prim:
    """Primitive operation key used by interpreter rule registries.

    A primitive has no behavior by itself. Runtime, abstract, batching, and AD
    behavior are attached by registering rules keyed by the ``Prim`` instance.

    Args:
        name: The name of the primitive.

    Example:
        >>> import autoform.extend as afe
        >>> add = afe.Prim("add")
    """

    __slots__ = ["name"]

    def __init__(self, name: str):
        assert isinstance(name, str), f"Invalid name type: {type(name)=}"
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def bind(self, value: Tree, /, **params):
        return active_interpreter.get().interpret(self, value, **params)

    async def abind(self, value: Tree, /, **params):
        return await active_interpreter.get().ainterpret(self, value, **params)


# ==================================================================================================
# TAGS
# ==================================================================================================


active_tags: ContextVar[frozenset[Hashable]] = ContextVar("active_tags", default=frozenset())


@contextmanager
def tag(*tags: Hashable) -> Generator[tuple[Hashable, ...], None, None]:
    """Attach tags to equations at trace time.

    Equations built inside nested ``tag`` blocks receive the tags from all active
    blocks. Equations built after a block exits do not receive that block's tags.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     with af.tag("outer"):
        ...         head = af.concat(x, "!")
        ...         with af.tag("inner"):
        ...             return af.concat(head, "?")
        >>> ir = af.trace(program)("seed")
        >>> ir.eqns[0].tags == frozenset({"outer"})
        True
        >>> ir.eqns[1].tags == frozenset({"outer", "inner"})
        True
    """

    for value in tags:
        try:
            hash(value)
        except TypeError as e:
            raise TypeError(f"Tags must be hashable, got {value!r}") from e
    token = active_tags.set(active_tags.get() | frozenset(tags))
    try:
        yield tags
    finally:
        active_tags.reset(token)


# ==================================================================================================
# IR
# ==================================================================================================


class Eqn:
    """One primitive application inside an :class:`IR`.

    An equation records the primitive to execute, the IR-shaped input and output
    trees, static primitive parameters, and the tags active when the equation
    was traced. Calling :meth:`bind` executes the primitive under those tags.

    Args:
        prim: Primitive represented by this equation.
        in_tree: Input tree containing IR variables and concrete literals.
        out_tree: Output tree containing IR variables and concrete literals.
        params: Static parameters passed to the primitive rule.
        tags: Tags associated with this equation.
    """

    __slots__ = ["prim", "in_tree", "out_tree", "params", "tags"]

    def __init__(
        self,
        prim: Prim,
        in_tree: Tree,
        out_tree: Tree,
        params: dict[str, Any] | None = None,
        tags: frozenset[Hashable] = frozenset(),
    ):
        assert isinstance(prim, Prim)
        assert isinstance(params, dict) or params is None
        assert isinstance(tags, frozenset)
        self.prim = prim
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.params = params if params is not None else {}
        self.tags = tags

    def bind(self, in_tree: Tree, /, **params):
        with tag(*self.tags):
            return self.prim.bind(in_tree, **params)

    async def abind(self, in_tree: Tree, /, **params):
        with tag(*self.tags):
            return await self.prim.abind(in_tree, **params)

    def using(self, **kwargs) -> Eqn:
        return Eqn(self.prim, self.in_tree, self.out_tree, self.params | kwargs, self.tags)


class IR[*A, R]:
    """A traced AutoForm program.

    An ``IR`` contains the ordered equations produced by tracing, plus the input
    and output IR trees that describe how runtime arguments and results are
    structured. Extension transforms may construct new ``IR`` values when they
    rewrite or wrap a program.

    Args:
        eqns: Ordered primitive equations.
        in_tree: Tree describing the runtime input structure.
        out_tree: Tree describing the runtime output structure.
    """

    __slots__ = ["eqns", "in_tree", "out_tree"]

    def __init__(self, eqns: list[Eqn], in_tree: Tree, out_tree: Tree):
        assert isinstance(eqns, list)
        eqns = tuple(eqns)
        assert all(isinstance(eqn, Eqn) for eqn in eqns)
        self.eqns = eqns
        self.in_tree = in_tree
        self.out_tree = out_tree

    def __repr__(self) -> str:
        return generate_text_code(ir=self, expand_ir=True)

    def call(self, *args: *A) -> R:
        """Run IR with concrete runtime inputs.

        Use this after `trace(...)` has produced an `IR`. Pass values with the same
        pytree structure as `in_tree`; the method executes the stored equations
        in order and returns the final output tree.

        Example:
            >>> import autoform as af
            >>> def wrap(x):
            ...     return af.format("[{}]", x)
            >>> ir = af.trace(wrap)("x")
            >>> ir.call("y")
            '[y]'
        """
        return call(self)(*args)

    async def acall(self, *args: *A) -> R:
        """Run IR asynchronously with concrete runtime inputs.

        Use this when execution may cross async primitive rules. The inputs follow
        the same conventions as `IR.call(...)`, but the method returns an awaitable
        and each equation is driven through `abind(...)`.

        Example:
            >>> import autoform as af
            >>> import asyncio
            >>> def wrap(x):
            ...     return af.format("[{}]", x)
            >>> ir = af.trace(wrap)("x")
            >>> asyncio.run(ir.acall("y"))
            '[y]'
        """
        return await acall(self)(*args)

    def walk(self, *args: *A) -> Generator[tuple[Eqn | None, Tree], Tree, None]:
        """Step through this IR one equation at a time.

        Manual control over IR execution. Start with `next(gen)` to receive `(eqn, in_values)`,
        compute or override the equation output, using `eqn.bind(in_values, **eqn.params)`
        for synchronous execution or `await eqn.abind(in_values, **eqn.params)` for async
        execution, and send that output back with `gen.send(...)`. After the last equation,
        the generator yields `(None, out_tree)`.

        Example:
            >>> import autoform as af
            >>> def wrap(x):
            ...     punctuated = af.concat(x, "!")
            ...     return af.format("[{}]", punctuated)
            >>> ir = af.trace(wrap)("x")
            >>> gen = ir.walk("y")
            >>> eqn, in_values = next(gen)
            >>> eqn.prim.name
            'concat'
            >>> step = gen.send(eqn.bind(in_values, **eqn.params))
            >>> eqn, in_values = step
            >>> eqn.prim.name
            'format'
            >>> done, out = gen.send(eqn.bind(in_values, **eqn.params))
            >>> done is None, out
            (True, '[y!]')
        """
        return walk(self)(*args)


def generate_text_code(ir: IR, indent: int = 2, *, expand_ir: bool = False) -> str:
    assert isinstance(indent, int) and indent >= 0
    sp = " " * indent

    def format_ir_val(ir_val) -> str:
        if is_var(ir_val):
            var_type = type(ir_val).__name__
            aval_info = repr(ir_val.aval)
            type_info = f"[{aval_info}]"
            return f"%{ir_val.id}:{var_type}{type_info}"
        val = ir_val
        if isinstance(val, IR):
            if expand_ir:
                sub_code = generate_text_code(val, indent, expand_ir=True)
                return f"<IR:{{\n{sub_code}\n}}>"
            else:
                prim_names = ",".join(e.prim.name for e in val.eqns)
                if len(prim_names) > 20:
                    prim_names = prim_names[:17] + "..."
                return f"<IR:[{prim_names}]>"
        else:
            val_repr = repr(val)
            if len(val_repr) > 30:
                val_repr = val_repr[:27] + "..."
            return f"{val_repr}:Lit"

    def format_tree(tree: Tree) -> str:
        leaves = utils.tree.leaves(tree)
        return ", ".join(format_ir_val(leaf) for leaf in leaves) if leaves else "()"

    in_sig = format_tree(ir.in_tree)
    out_sig = format_tree(ir.out_tree)

    header = f"func({in_sig}) -> ({out_sig}) {{"
    lines = [header]

    for eqn in ir.eqns:
        lhs = format_tree(eqn.out_tree)
        rhs = format_tree(eqn.in_tree)
        eqn_args = [rhs]
        eqn_args.extend(f"{k}={eqn.params[k]!r}" for k in (eqn.params or {}))
        if eqn.tags:
            tags = ", ".join(sorted(repr(tag) for tag in eqn.tags))
            eqn_args.append(f"tags={{{tags}}}")
        lines.append(f"{sp}({lhs}) = {eqn.prim.name}({', '.join(eqn_args)})")

    lines.append("}")
    return "\n".join(lines)


# ==================================================================================================
# INTERPRETER
# ==================================================================================================


class BaseInterpreter(ABC):
    __slots__ = []

    @abstractmethod
    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Any: ...

    @abstractmethod
    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Any: ...


class Interpreter(BaseInterpreter):
    """Base class for runtime primitive interpreters.

    Subclass ``Interpreter`` to build an execution-time extension context. A
    custom interpreter usually stores the current :data:`active_interpreter` as
    its parent, overrides ``interpret`` and ``ainterpret`` to handle new primitives.
    """

    __slots__ = []


class BoxedInterpreter[T](BaseInterpreter):
    __slots__ = []
    # NOTE(asem): boxed interpreters own a transform-specific value wrapper.
    # plain interpreters only override primitive dispatch but boxed interpreters
    # also define how values are boxed before primitive evaluation and unboxed
    # when rules need the underlying payload.

    @abstractmethod
    def box(self, *value) -> T: ...

    @abstractmethod
    def unbox(self, value: Tree, /): ...


@contextmanager
def using_interpreter[T: BaseInterpreter](interpreter: T) -> Generator[T, None, None]:
    """Run primitive dispatch through an interpreter inside the context."""

    token = active_interpreter.set(interpreter)
    try:
        yield interpreter
    finally:
        active_interpreter.reset(token)


# ==================================================================================================
# EVAL
# ==================================================================================================


class EvalInterpreter(Interpreter):
    __slots__ = []

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return impl_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return await impl_rules.aget(prim)(in_tree, **params)


active_interpreter = ContextVar[BaseInterpreter]("active_interpreter", default=EvalInterpreter())


# ==================================================================================================
# TRACING
# ==================================================================================================


fold_flag: ContextVar[bool] = ContextVar("fold_mode", default=False)


@contextmanager
def fold() -> Generator[None, None, None]:
    """Evaluate immediately within the context.

    Inside ``af.trace(...)``, primitive calls normally build IR equations. A
    ``fold`` block instead runs primitive implementations while tracing and
    returns concrete values that can be embedded as literals in the surrounding
    IR. If a primitive inside the block depends on a dynamic traced value, an
    ``AssertionError`` is raised. Outside tracing, ``fold`` is a no-op.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     with af.fold():
        ...         prefix = af.concat("hello", " ")
        ...     return af.concat(prefix, x)
        >>> ir = af.trace(program)("seed")
        >>> len(ir.eqns)
        1
        >>> ir.call("world")
        'hello world'

    Fold is useful when a trace-time computation should decide ordinary Python
    control flow. Autoform cannot stage Python branches whose conditions depend
    on dynamic IR values; those conditions must be known while tracing. A folded
    computation runs immediately, so its concrete result can safely choose the
    branch that is traced into the IR.

    Example:
        >>> def program(x):
        ...     with af.fold():
        ...         route = af.concat("priority", ": high")
        ...     if route == "priority: high":
        ...         return af.concat("yes: ", x)
        ...     return af.concat("no: ", x)
        >>> ir = af.trace(program)("seed")
        >>> ir.call("answer")
        'yes: answer'
    """
    token = fold_flag.set(True)
    try:
        yield
    finally:
        fold_flag.reset(token)


TRACE_UNSUPPORTED_OP_ERROR = (
    "Cannot use {desc} on a traced value."
    "During af.trace(), values only carry abstract type information; "
    "Python {desc} needs a concrete runtime value and cannot be staged "
    "implicitly. If this value should be known while tracing, mark it static with "
    "af.trace(..., static=...) or compute this operation outside the traced function. "
    "If you need this operation at runtime in the IR, define an explicit autoform "
    "primitive for it."
)

TRACE_MISSING_RULE_ERROR = "No trace rule for {desc} on values of type {aval!r}. "

type TraceRule = Callable[[Any, Any], Any]


trace_eq_rules: dict[type[AVal], TraceRule] = {}
trace_add_rules: dict[type[AVal], TraceRule] = {}
trace_sub_rules: dict[type[AVal], TraceRule] = {}
trace_mul_rules: dict[type[AVal], TraceRule] = {}
trace_truediv_rules: dict[type[AVal], TraceRule] = {}
trace_matmul_rules: dict[type[AVal], TraceRule] = {}


class TraceBox:
    __slots__ = ["owner", "var"]

    def __init__(self, /, *, owner: TraceInterpreter, var: Var):
        assert isinstance(owner, TraceInterpreter)
        assert is_var(var)
        self.owner = owner
        self.var = var

    @property
    def aval(self) -> AVal:
        return self.var.aval

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.var!r})"

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other) -> Any:
        if rule := trace_eq_rules.get(type(self.aval)):
            return rule(self, other)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="==", aval=self.aval))

    def __add__(self, other) -> Any:
        if rule := trace_add_rules.get(type(self.aval)):
            return rule(self, other)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="+", aval=self.aval))

    def __radd__(self, other) -> Any:
        if rule := trace_add_rules.get(type(self.aval)):
            return rule(other, self)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="+", aval=self.aval))

    def __sub__(self, other) -> Any:
        if rule := trace_sub_rules.get(type(self.aval)):
            return rule(self, other)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="-", aval=self.aval))

    def __rsub__(self, other) -> Any:
        if rule := trace_sub_rules.get(type(self.aval)):
            return rule(other, self)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="-", aval=self.aval))

    def __mul__(self, other) -> Any:
        if rule := trace_mul_rules.get(type(self.aval)):
            return rule(self, other)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="*", aval=self.aval))

    def __rmul__(self, other) -> Any:
        if rule := trace_mul_rules.get(type(self.aval)):
            return rule(other, self)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="*", aval=self.aval))

    def __truediv__(self, other) -> Any:
        if rule := trace_truediv_rules.get(type(self.aval)):
            return rule(self, other)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="/", aval=self.aval))

    def __rtruediv__(self, other) -> Any:
        if rule := trace_truediv_rules.get(type(self.aval)):
            return rule(other, self)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="/", aval=self.aval))

    def __matmul__(self, other) -> Any:
        if rule := trace_matmul_rules.get(type(self.aval)):
            return rule(self, other)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="@", aval=self.aval))

    def __rmatmul__(self, other) -> Any:
        if rule := trace_matmul_rules.get(type(self.aval)):
            return rule(other, self)
        raise TypeError(TRACE_MISSING_RULE_ERROR.format(desc="@", aval=self.aval))

    def __bool__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="truthiness"))

    def __bytes__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="bytes coercion"))

    def __complex__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="complex number coercion"))

    def __contains__(self, _) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="membership testing"))

    def __float__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="float coercion"))

    def __format__(self, _) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="string formatting"))

    def __getitem__(self, _) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="indexing"))

    def __index__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="integer-index coercion"))

    def __int__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="integer coercion"))

    def __iter__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="iteration"))

    def __len__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="length"))

    def __str__(self) -> NoReturn:
        raise TypeError(TRACE_UNSUPPORTED_OP_ERROR.format(desc="string coercion"))


def assert_foldable(prim: Prim, value: Tree) -> None:
    traced_values = [x for x in utils.tree.leaves(value) if isinstance(x, TraceBox)]
    assert not traced_values, (
        f"Cannot evaluate {prim.name} in af.fold() because it depends on traced values "
        f"{traced_values!r}. Mark the dependencies static or move this computation outside af.fold()."
    )


class TraceInterpreter(BoxedInterpreter[TraceBox]):
    __slots__ = ["eqns"]

    def __init__(self):
        self.eqns: list[Eqn] = []

    def box(self, value, /) -> Tree:
        return TraceBox(owner=self, var=value) if is_var(value) else value

    def unbox(self, value: Tree, /) -> Tree:
        def func(value, /):
            if not isinstance(value, TraceBox):
                # NOTE(asem): basically literals case.
                return value
            assert value.owner is self, "Encountered TraceBox from a different tracer."
            # NOTE(asem): this catches leaked live trace values.
            # >>> leaked = {}
            # >>> def first_func(x):
            # ...     leaked["first"] = x
            # ...     return x
            # >>> def second_func(y):
            # ...     return concat(leaked["first"], y)
            # >>> ir1 = af.trace(first_func)("input")
            # >>> ir2 = af.trace(second_func)("input")
            return value.var

        return utils.tree.map(func, value)

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        if fold_flag.get():
            return self.eval(prim, in_tree, **params)
        return self.stage(prim, in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        if fold_flag.get():
            return await self.aeval(prim, in_tree, **params)
        return self.stage(prim, in_tree, **params)

    def eval(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        assert_foldable(prim, (in_tree, params))
        with using_interpreter(EvalInterpreter()):
            out_tree = prim.bind(in_tree, **params)
        assert_foldable(prim, out_tree)
        return out_tree

    async def aeval(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        assert_foldable(prim, (in_tree, params))
        with using_interpreter(EvalInterpreter()):
            out_tree = await prim.abind(in_tree, **params)
        assert_foldable(prim, out_tree)
        return out_tree

    def stage(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        def to_in_ir_atom(value):
            if not is_var(value):
                hash(value)
            return value

        def to_concrete(leaf, value):
            assert not is_var(value), f"Unexpected variable at {'/'.join(map(str, leaf))}"
            return value

        in_tree = self.unbox(in_tree)
        params = self.unbox(params)
        params = utils.tree.map_with_path(to_concrete, params)

        in_tree = utils.tree.map(to_in_ir_atom, in_tree)
        in_aval_tree = utils.tree.map(aval_if_var, in_tree)
        out_aval_tree = abstract_rules.get(prim)(in_aval_tree, **params)

        def to_out_ir_atom(x):
            # NOTE(asem): abstract rules return `AVal`/ python leaves.
            # `AVal` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return Var.fresh(aval=x) if is_aval(x) else x

        out_tree = utils.tree.map(to_out_ir_atom, out_aval_tree)
        self.eqns.append(Eqn(prim, in_tree, out_tree, params, active_tags.get()))
        return utils.tree.map(self.box, out_tree)


def trace[*A, R](
    func: Callable[[*A], R],
    /,
    *,
    static: Tree[bool] = False,
) -> Callable[[*A], IR[*A, R]]:
    """Build an IR by tracing a function's execution.

    Args:
        func: A callable that uses autoform primitives (format, concat, lm_call, etc.).
        static: Bool pytree matching the positional input structure.
            Mark a leaf ``True`` to keep that value fixed at trace time.
            Mark a leaf ``False`` to keep it as a normal runtime input.
            This is useful for ordinary Python control flow such as ``if``
            statements. Later calls must pass the same values for leaves
            marked static.

    Returns:
        A tracer callable that takes positional arguments and returns an IR.

    When a flag is marked static, tracing follows only the branch selected by
    that flag at trace time.

    Example:
        >>> import autoform as af
        >>> def label(is_error):
        ...     if is_error:
        ...         return "error"
        ...     return "ok"
        >>> ir = af.trace(label, static=True)(True)
        >>> ir.call(True)
        'error'
    """

    def is_static_spec(x) -> bool:
        return isinstance(x, bool)

    def to_in_ir_atom(x, is_static: bool):
        if is_static:
            hash(x)
            return x
        return to_var(x)

    def to_var(x, /) -> Var:
        assert not is_var(x), "Inputs to `trace` must be normal python types"
        assert is_traceable(x), f"Unsupported input leaf type for `trace`: {type(x).__name__}. "
        return Var.fresh(aval=avalof(x))

    @ft.wraps(func)
    def wrapper(*args: *A) -> IR[*A, R]:
        arg_tree = args
        in_static_tree = utils.tree.broadcast_prefix(static, arg_tree, is_leaf=is_static_spec)
        in_tree = utils.tree.map(to_in_ir_atom, arg_tree, in_static_tree, is_leaf=is_traceable)
        with using_interpreter(TraceInterpreter()) as tracer:
            out_trace_tree = func(*cast(tuple, utils.tree.map(tracer.box, in_tree)))
        out_tree = tracer.unbox(out_trace_tree)
        return IR(eqns=tracer.eqns, in_tree=in_tree, out_tree=out_tree)

    return wrapper


# ==================================================================================================
# WALK
# ==================================================================================================

type GenStep = tuple[Eqn | None, Tree]


def check_static_inputs(atoms: Tree, args: Tree, /) -> None:
    """Validate runtime inputs against static literals in an IR input tree."""

    def check_input(atom, value: Any):
        if not is_var(atom):
            expected = atom
            msg = f"Static input mismatch: expected {expected!r}, got {value!r}"
            assert expected == value, msg

    utils.tree.map(check_input, atoms, args)


@ft.partial(utils.lru_cache, maxsize=256)
def walk[*A, R](ir: IR[*A, R], /) -> Callable[[*A], Generator[GenStep, Tree, None]]:
    """Walk an IR one equation at a time."""
    # NOTE(asem): the key idea here is to hide the environment management
    # from the user.
    # TODO(asem): if user is using bind/abind, walk itself can be traced into another IR. maybe
    # add it to walk docs to clarify this point.

    def func(*args: *A) -> Generator[GenStep, Tree, None]:
        assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
        env: dict[Var, Any] = {}

        def read(ir_val) -> Any:
            return env[ir_val] if is_var(ir_val) else ir_val

        def write(ir_val, value: Any):
            is_var(ir_val) and setitem(env, ir_val, value)

        utils.tree.map(write, ir.in_tree, args)

        for eqn in ir.eqns:
            in_values = utils.tree.map(read, eqn.in_tree)
            out_values = yield eqn, in_values
            utils.tree.map(write, eqn.out_tree, out_values)

        yield None, utils.tree.map(read, ir.out_tree)

    return func


# ==================================================================================================
# CALL
# ==================================================================================================


@ft.partial(utils.lru_cache, maxsize=256)
def call[*A, R](ir: IR[*A, R], /) -> Callable[[*A], R]:
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def func(*args: *A) -> R:
        check_static_inputs(ir.in_tree, args)
        eqn, in_values = next(gen := walk(ir)(*args))
        while eqn:
            eqn, in_values = gen.send(eqn.bind(in_values, **eqn.params))
        return in_values

    return func


@ft.partial(utils.lru_cache, maxsize=256)
def acall[*A, R](ir: IR[*A, R], /) -> Callable[[*A], Awaitable[R]]:
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    async def func(*args: *A) -> R:
        check_static_inputs(ir.in_tree, args)
        eqn, in_values = next(gen := walk(ir)(*args))
        while eqn:
            eqn, in_values = gen.send(await eqn.abind(in_values, **eqn.params))
        return in_values

    return func


# ==================================================================================================
# RULES
# ==================================================================================================


class InterpreterRule[R](Protocol):
    def __call__(self, in_tree: Tree, /, **params: Any) -> R: ...


type TreePair = tuple[Tree, Tree]
type BatchRuleResult = tuple[Tree, Tree[bool] | bool]
type AsyncInterpreterRule[R] = InterpreterRule[Awaitable[R]]
type ImplRule = InterpreterRule[Tree]
type AImplRule = AsyncInterpreterRule[Tree]
type AbstractRule = InterpreterRule[Tree[EvalType]]
type AAbstractRule = AsyncInterpreterRule[Tree[EvalType]]
type PushforwardRule = InterpreterRule[TreePair]
type APushforwardRule = AsyncInterpreterRule[TreePair]
type PullbackFwdRule = InterpreterRule[TreePair]
type APullbackFwdRule = AsyncInterpreterRule[TreePair]
type PullbackBwdRule = InterpreterRule[Tree]
type APullbackBwdRule = AsyncInterpreterRule[Tree]
type BatchRule = InterpreterRule[BatchRuleResult]
type ABatchRule = AsyncInterpreterRule[BatchRuleResult]


class InterpreterRuleMapping[Rule: InterpreterRule[Any], ARule: AsyncInterpreterRule[Any]]:
    __slots__ = ["map", "amap", "lock"]

    def __init__(self):
        self.map: dict[Prim, Rule] = {}
        self.amap: dict[Prim, ARule] = {}
        self.lock = RLock()

    def set[R: Rule](self, prim: Prim, rule: R, /, *, replace: bool = False) -> R:
        assert isinstance(prim, Prim), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert isinstance(replace, bool), f"Expected bool for replace, got {type(replace)}"
        assert replace or prim not in self.map, f"Rule for primitive {prim} already defined"

        with self.lock:
            self.map[prim] = rule
        return rule

    def aset[AR: ARule](self, prim: Prim, rule: AR, /, *, replace: bool = False) -> AR:
        assert isinstance(prim, Prim), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert isinstance(replace, bool), f"Expected bool for replace, got {type(replace)}"
        assert replace or prim not in self.amap, f"Async rule for primitive {prim} already defined"

        with self.lock:
            self.amap[prim] = rule
        return rule

    def get(self, prim: Prim) -> Rule:
        with self.lock:
            if prim not in self.map:
                raise KeyError(f"No {type(self).__name__} rule defined for primitive {prim}")
            return self.map[prim]

    def aget(self, prim: Prim) -> ARule:
        with self.lock:
            if prim not in self.amap:
                raise KeyError(f"No async {type(self).__name__} rule defined for primitive {prim}")
            return self.amap[prim]


impl_rules = InterpreterRuleMapping[ImplRule, AImplRule]()
batch_rules = InterpreterRuleMapping[BatchRule, ABatchRule]()
push_rules = InterpreterRuleMapping[PushforwardRule, APushforwardRule]()
pull_fwd_rules = InterpreterRuleMapping[PullbackFwdRule, APullbackFwdRule]()
pull_bwd_rules = InterpreterRuleMapping[PullbackBwdRule, APullbackBwdRule]()
abstract_rules = InterpreterRuleMapping[AbstractRule, AAbstractRule]()
