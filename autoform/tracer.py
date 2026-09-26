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

"""Tracing."""

from __future__ import annotations

import functools as ft
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, TypeGuard, cast

import autoform.abstract as abstract
import autoform.core as core
import autoform.utils as utils

type Tree[T] = utils.Tree[T]

trace_types: set[type] = set()


def is_traceable(x) -> TypeGuard[str | int | float | bool]:
    return type(x) in trace_types


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
        >>> increment = af.trace(lambda value: value + 1)(1.0)
        >>> def program(x):
        ...     with af.fold():
        ...         prefix = f"v{increment.call(1.0)}: "
        ...     return prefix + x
        >>> ir = af.trace(program)("seed")
        >>> len(ir.eqns)
        1
        >>> ir.call("world")
        'v2.0: world'

    Fold is useful when a trace-time computation should decide ordinary Python
    control flow. Autoform cannot stage Python branches whose conditions depend
    on dynamic IR values; those conditions must be known while tracing. A folded
    computation runs immediately, so its concrete result can safely choose the
    branch that is traced into the IR.

    Example:
        >>> def program(x):
        ...     with af.fold():
        ...         route = increment.call(1.0)
        ...     if route == 2:
        ...         return "yes: " + x
        ...     return "no: " + x
        >>> ir = af.trace(program)("seed")
        >>> ir.call("answer")
        'yes: answer'
    """
    token = fold_flag.set(True)
    try:
        yield
    finally:
        fold_flag.reset(token)


class Dunder(Enum):
    NEG = "neg"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    MATMUL = "matmul"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    BOOL = "bool"
    BYTES = "bytes"
    COMPLEX = "complex"
    CONTAINS = "contains"
    FLOAT = "float"
    FORMAT = "format"
    GETITEM = "getitem"
    INDEX = "index"
    INT = "int"
    ITER = "iter"
    LEN = "len"
    STR = "str"


type DunderRule = Callable[..., Any]

dunder_rules: dict[tuple[Dunder, type[abstract.AVal]], DunderRule] = {}


class TraceBox(core.Box):
    __slots__ = ["var"]

    def __init__(self, /, *, owner: TraceInterpreter, var: core.Var):
        assert isinstance(owner, TraceInterpreter)
        assert core.is_var(var)
        super().__init__(owner)
        self.var = var

    @property
    def aval(self) -> abstract.AVal:
        return self.var.aval

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.var!r})"

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other) -> Any:
        return apply_dunder(Dunder.EQ, self, self, other)

    def __ne__(self, other) -> Any:
        return apply_dunder(Dunder.NE, self, self, other)

    def __lt__(self, other) -> Any:
        return apply_dunder(Dunder.LT, self, self, other)

    def __le__(self, other) -> Any:
        return apply_dunder(Dunder.LE, self, self, other)

    def __gt__(self, other) -> Any:
        return apply_dunder(Dunder.GT, self, self, other)

    def __ge__(self, other) -> Any:
        return apply_dunder(Dunder.GE, self, self, other)

    def __neg__(self) -> Any:
        return apply_dunder(Dunder.NEG, self, self)

    def __add__(self, other) -> Any:
        return apply_dunder(Dunder.ADD, self, self, other)

    def __radd__(self, other) -> Any:
        return apply_dunder(Dunder.ADD, self, other, self)

    def __sub__(self, other) -> Any:
        return apply_dunder(Dunder.SUB, self, self, other)

    def __rsub__(self, other) -> Any:
        return apply_dunder(Dunder.SUB, self, other, self)

    def __mul__(self, other) -> Any:
        return apply_dunder(Dunder.MUL, self, self, other)

    def __rmul__(self, other) -> Any:
        return apply_dunder(Dunder.MUL, self, other, self)

    def __truediv__(self, other) -> Any:
        return apply_dunder(Dunder.DIV, self, self, other)

    def __rtruediv__(self, other) -> Any:
        return apply_dunder(Dunder.DIV, self, other, self)

    def __pow__(self, other) -> Any:
        return apply_dunder(Dunder.POW, self, self, other)

    def __rpow__(self, other) -> Any:
        return apply_dunder(Dunder.POW, self, other, self)

    def __matmul__(self, other) -> Any:
        return apply_dunder(Dunder.MATMUL, self, self, other)

    def __rmatmul__(self, other) -> Any:
        return apply_dunder(Dunder.MATMUL, self, other, self)

    def __bool__(self) -> bool:
        return apply_dunder(Dunder.BOOL, self, self)

    def __bytes__(self) -> bytes:
        return apply_dunder(Dunder.BYTES, self, self)

    def __complex__(self) -> complex:
        return apply_dunder(Dunder.COMPLEX, self, self)

    def __contains__(self, item) -> bool:
        return apply_dunder(Dunder.CONTAINS, self, self, item)

    def __float__(self) -> float:
        return apply_dunder(Dunder.FLOAT, self, self)

    def __format__(self, format_spec: str) -> str:
        return apply_dunder(Dunder.FORMAT, self, self, format_spec)

    def __getitem__(self, key) -> Any:
        return apply_dunder(Dunder.GETITEM, self, self, key)

    def __index__(self) -> int:
        return apply_dunder(Dunder.INDEX, self, self)

    def __int__(self) -> int:
        return apply_dunder(Dunder.INT, self, self)

    def __iter__(self) -> Iterator[Any]:
        return apply_dunder(Dunder.ITER, self, self)

    def __len__(self) -> int:
        return apply_dunder(Dunder.LEN, self, self)

    def __str__(self) -> str:
        return apply_dunder(Dunder.STR, self, self)


def apply_dunder(dunder: Dunder, box: TraceBox, *operands):
    if (rule := dunder_rules.get((dunder, type(box.aval)))) is None:
        raise TypeError(f"No trace rule for {dunder.value} on values of type {box.aval!r}.")
    return rule(*operands)


def assert_foldable(prim: core.Prim, value: Tree) -> None:
    traced_values = [x for x in utils.tree.leaves(value) if isinstance(x, TraceBox)]
    assert not traced_values, (
        f"Cannot evaluate {prim.name} in af.fold() because it depends on traced values "
        f"{traced_values!r}. Mark the dependencies static or move this computation outside af.fold()."
    )


class TraceInterpreter(core.Interpreter[TraceBox]):
    __slots__ = ["eqns"]

    def __init__(self):
        self.eqns: list[core.Eqn] = []

    def box(self, value, /) -> Tree:
        return utils.tree.map(lambda v: TraceBox(owner=self, var=v) if core.is_var(v) else v, value)

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

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        if fold_flag.get():
            return self.eval(prim, in_tree, **params)
        return self.stage(prim, in_tree, **params)

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        if fold_flag.get():
            return await self.aeval(prim, in_tree, **params)
        return self.stage(prim, in_tree, **params)

    def eval(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        assert_foldable(prim, (in_tree, params))
        with core.using_interpreter(core.EvalInterpreter()):
            out_tree = prim.bind(in_tree, **params)
        assert_foldable(prim, out_tree)
        return out_tree

    async def aeval(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        assert_foldable(prim, (in_tree, params))
        with core.using_interpreter(core.EvalInterpreter()):
            out_tree = await prim.abind(in_tree, **params)
        assert_foldable(prim, out_tree)
        return out_tree

    def stage(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        def to_in_ir_atom(value):
            if not core.is_var(value):
                hash(value)
            return value

        def to_concrete(leaf, value):
            assert not core.is_var(value), f"Unexpected variable at {'/'.join(map(str, leaf))}"
            return value

        in_tree = self.unbox(in_tree)
        params = self.unbox(params)
        params = utils.tree.map_with_path(to_concrete, params)

        in_tree = utils.tree.map(to_in_ir_atom, in_tree)
        in_aval_tree = utils.tree.map(core.aval_if_var, in_tree)
        out_aval_tree = core.abstract_rules[prim](in_aval_tree, **params)

        def to_out_ir_atom(x):
            # NOTE(asem): abstract rules return `AVal`/ python leaves.
            # `AVal` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return core.Var.fresh(aval=x) if isinstance(x, abstract.AVal) else x

        out_tree = utils.tree.map(to_out_ir_atom, out_aval_tree)
        self.eqns.append(core.Eqn(prim, in_tree, out_tree, params, core.active_tags.get()))
        return self.box(out_tree)


def trace[*A, R](
    func: Callable[[*A], R],
    /,
    *,
    static: Tree[bool] = False,
) -> Callable[[*A], core.IR[*A, R]]:
    """Build an IR by tracing a function's execution.

    Args:
        func: A callable that uses autoform primitives (string.concat, lm.complete, etc.).
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

    def to_var(x, /) -> core.Var:
        assert not core.is_var(x), "Inputs to `trace` must be normal python types"
        assert is_traceable(x), f"Unsupported input leaf type for `trace`: {type(x).__name__}. "
        return core.Var.fresh(aval=abstract.avalof(x))

    @ft.wraps(func)
    def wrapper(*args: *A) -> core.IR[*A, R]:
        arg_tree = args
        in_static_tree = utils.tree.broadcast_prefix(static, arg_tree, is_leaf=is_static_spec)
        in_tree = utils.tree.map(to_in_ir_atom, arg_tree, in_static_tree, is_leaf=is_traceable)
        with core.using_interpreter(TraceInterpreter()) as tracer:
            out_trace_tree = func(*cast(tuple, tracer.box(in_tree)))
        out_tree = tracer.unbox(out_trace_tree)
        return core.IR(eqns=tracer.eqns, in_tree=in_tree, out_tree=out_tree)

    return wrapper
