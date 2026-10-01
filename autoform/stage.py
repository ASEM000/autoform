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

"""IR construction, analysis, tracing, and execution."""

from __future__ import annotations

import functools as ft
import itertools as it
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Generator, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from operator import setitem
from typing import Any, ClassVar, Self, TypeGuard, cast

import autoform.core as core
import autoform.utils as utils

type Tree[T] = utils.Tree[T]


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
        ...         head = x + "!"
        ...         with af.tag("inner"):
        ...             return head + "?"
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

    def __init__(self, /, *, aval: core.AVal, source: Var | None = None):
        self.id = next(self.counter)
        assert is_var(source) or source is None
        assert isinstance(aval, core.AVal)
        self.source = source
        self.aval = aval

    @classmethod
    def fresh(cls, *, aval: core.AVal, source: Var | None = None) -> Self:
        return cls(source=source, aval=aval)

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source else ""
        return f"{type(self).__name__}[{self.aval!r}](id={self.id}{source})"


core.aval_types[Var] = lambda value: value.aval


def is_var(x) -> TypeGuard[Var]:
    """Return ``True`` if input is an :class:`Var`."""

    return isinstance(x, Var)


def aval_if_var(x, /):
    """Return the aval for an IR variable, otherwise return input unchanged."""
    return x.aval if is_var(x) else x


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
        prim: core.Prim,
        in_tree: Tree,
        out_tree: Tree,
        params: dict[str, Any] | None = None,
        tags: frozenset[Hashable] = frozenset(),
    ):
        assert isinstance(prim, core.Prim)
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

        Example:
            >>> import autoform as af
            >>> def wrap(x):
            ...     return "[" + x + "]"
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
            ...     return "[" + x + "]"
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
            ...     punctuated = x + "!"
            ...     return "[" + punctuated + "]"
            >>> ir = af.trace(wrap)("x")
            >>> gen = ir.walk("y")
            >>> eqn, in_values = next(gen)
            >>> eqn.prim.name
            'concat'
            >>> step = gen.send(eqn.bind(in_values, **eqn.params))
            >>> eqn, in_values = step
            >>> eqn.prim.name
            'concat'
            >>> eqn, in_values = gen.send(eqn.bind(in_values, **eqn.params))
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
# WALK
# ==================================================================================================

type GenStep = tuple[Eqn | None, Tree]


def check_inputs(atoms: Tree, args: Tree, /) -> None:

    def check_input(atom, value: Any):
        if is_var(atom):
            atom.aval.check(value)
        else:
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
        check_inputs(ir.in_tree, args)
        eqn, in_values = next(gen := walk(ir)(*args))
        while eqn:
            eqn, in_values = gen.send(eqn.bind(in_values, **eqn.params))
        return in_values

    return func


@ft.partial(utils.lru_cache, maxsize=256)
def acall[*A, R](ir: IR[*A, R], /) -> Callable[[*A], Awaitable[R]]:
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    async def func(*args: *A) -> R:
        check_inputs(ir.in_tree, args)
        eqn, in_values = next(gen := walk(ir)(*args))
        while eqn:
            eqn, in_values = gen.send(await eqn.abind(in_values, **eqn.params))
        return in_values

    return func


# ==================================================================================================
# ANALYSIS
# ==================================================================================================

type UsedTree = Tree[bool]
type LiveSet = set[Var]
type Liveness = list[LiveSet]


def is_same_structure(lhs: IR, rhs: IR, /) -> bool:
    """Compare IR input/output structures"""

    assert isinstance(lhs, IR)
    assert isinstance(rhs, IR)

    def same_atom(x, y):
        if is_var(x) and is_var(y):
            return x.aval == y.aval
        # NOTE(asem): check for literals.
        return type(x) is type(y) and x == y

    left, right = (lhs.in_tree, lhs.out_tree), (rhs.in_tree, rhs.out_tree)
    if utils.tree.structure(left) != utils.tree.structure(right):
        return False
    return utils.tree.all(utils.tree.map(same_atom, left, right))


def var_leaves(tree: Tree, /) -> list[Var]:
    """Return Vars from an IR tree in leaf order."""

    return [cast(Var, x) for x in utils.tree.leaves(tree) if is_var(x)]


def var_producers(ir: IR, /) -> dict[Var, Eqn]:
    """Return the top-level producer equation for each Var defined by ``ir``."""

    producers: dict[Var, Eqn] = {}
    for eqn in ir.eqns:
        for var in var_leaves(eqn.out_tree):
            assert producers.get(var) is None
            producers[var] = eqn
    return producers


def eqn_graph(ir: IR, /) -> dict[Eqn, list[Eqn]]:
    """Return top-level equation dependencies as parent -> children adjacency."""

    var_to_parent = var_producers(ir)
    adjacency_list: dict[Eqn, list[Eqn]] = {eqn: [] for eqn in ir.eqns}
    for eqn in ir.eqns:
        seen_parents: set[Eqn] = set()
        for in_var in var_leaves(eqn.in_tree):
            if (p := var_to_parent.get(in_var)) is not None and p not in seen_parents:
                adjacency_list[p].append(eqn)
                seen_parents.add(p)

    return adjacency_list


@ft.partial(utils.lru_cache, maxsize=256)
def toposort_levels(ir: IR, /) -> list[list[Eqn]]:
    """Group IR equations into dependency levels."""

    # NOTE(asem): equations form a dag where edges are defined by shared irvars.
    # if equation a produces $x and equation b uses $x, then a -> b.
    # this function groups equations into levels where:
    # 1. equations in the same level are independent (can run in parallel)
    # 2. level n must complete before level n+1 starts

    # NOTE(asem): three-step process:
    # 1. map each var to its creator equation
    # 2. build adjacency list (parent -> children) from var flow
    # 3. topological sort into levels using kahn's algorithm

    # NOTE(asem): step 1/2: build adjacency list (parent -> children) from var flow
    adjacency_list = eqn_graph(ir)
    in_degree = defaultdict(lambda: 0)
    for children in adjacency_list.values():
        for child in children:
            in_degree[child] += 1

    # NOTE(asem): step 3: kahn's algorithm
    # basically prune nodes with 0 indegree then update the children indegree
    queue = deque(eqn for eqn in ir.eqns if in_degree[eqn] == 0)
    levels = []

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node)
            for child in adjacency_list[node]:
                in_degree[child] -= 1
                in_degree[child] == 0 and queue.append(child)
        levels.append(level)
    return levels


def liveness(ir: IR, /, *, out_used: UsedTree | None = None) -> Liveness:
    """Return live Vars at each IR boundary."""

    # NOTE(asem): liveness is a backward dataflow analysis that computes Vars live
    # at each boundary. The result length is len(ir.eqns) + 1: the first item is
    # the live input boundary, and the last item is the selected output boundary.

    if out_used is None:
        live_after = set(var_leaves(ir.out_tree))
    else:
        assert utils.tree.all(isinstance(leaf, bool) for leaf in utils.tree.leaves(out_used))
        assert utils.tree.structure(out_used) == utils.tree.structure(ir.out_tree)
        # NOTE(asem): with a partial output mask, only the selected output Vars are live.
        # >>> def program(x):
        # ...     a = x + "!"
        # ...     b = x + "?"
        # ...     return a, b
        # >>> liveness(ir, out_used=(True, False))[-1]
        # {a}
        live_after = set(var_leaves(utils.mask(ir.out_tree, out_used)))

    liveness: Liveness = [set() for _ in range(len(ir.eqns) + 1)]
    liveness[-1] = live_after

    for i, eqn in reversed(tuple(enumerate(ir.eqns))):
        # NOTE(asem): move in reversed order of equation list starting from the output Vars
        # with each step up the live before is basically all the live Vars used + live after
        # without the Vars defined by the current equation.
        uses: LiveSet = set(var_leaves(eqn.in_tree))
        defs: LiveSet = set(var_leaves(eqn.out_tree))
        live_before = uses | (live_after - defs)
        liveness[i] = live_before
        live_after = live_before

    return liveness


# ==================================================================================================
# TRACING
# ==================================================================================================

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
    POS, NEG, ABS, INVERT = "pos", "neg", "abs", "invert"
    ADD, SUB, MUL, DIV = "add", "sub", "mul", "div"
    FLOORDIV, MOD, DIVMOD = "floordiv", "mod", "divmod"
    POW, MATMUL = "pow", "matmul"
    AND, OR, XOR = "and", "or", "xor"
    LSHIFT, RSHIFT = "lshift", "rshift"
    EQ, NE, LT, LE, GT, GE = "eq", "ne", "lt", "le", "gt", "ge"
    BOOL, INT, FLOAT, COMPLEX = "bool", "int", "float", "complex"
    BYTES, STR, FORMAT = "bytes", "str", "format"
    CALL, GETITEM = "call", "getitem"
    CONTAINS, INDEX, LEN = "contains", "index", "len"
    ITER, NEXT, REVERSED = "iter", "next", "reversed"
    ROUND, FLOOR, CEIL, TRUNC = "round", "floor", "ceil", "trunc"


type DunderRule = Callable[..., Any]

dunder_rules: dict[tuple[Dunder, type[core.AVal]], DunderRule] = {}


class TraceBox:
    __slots__ = ["owner", "var"]

    def __init__(self, /, *, owner: TraceInterpreter, var: Var):
        assert isinstance(owner, TraceInterpreter)
        assert is_var(var)
        self.owner = owner
        self.var = var

    @property
    def aval(self) -> core.AVal:
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

    def __pos__(self) -> Any:
        return apply_dunder(Dunder.POS, self, self)

    def __neg__(self) -> Any:
        return apply_dunder(Dunder.NEG, self, self)

    def __abs__(self) -> Any:
        return apply_dunder(Dunder.ABS, self, self)

    def __invert__(self) -> Any:
        return apply_dunder(Dunder.INVERT, self, self)

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

    def __floordiv__(self, other) -> Any:
        return apply_dunder(Dunder.FLOORDIV, self, self, other)

    def __rfloordiv__(self, other) -> Any:
        return apply_dunder(Dunder.FLOORDIV, self, other, self)

    def __mod__(self, other) -> Any:
        return apply_dunder(Dunder.MOD, self, self, other)

    def __rmod__(self, other) -> Any:
        return apply_dunder(Dunder.MOD, self, other, self)

    def __divmod__(self, other) -> Any:
        return apply_dunder(Dunder.DIVMOD, self, self, other)

    def __rdivmod__(self, other) -> Any:
        return apply_dunder(Dunder.DIVMOD, self, other, self)

    def __pow__(self, other) -> Any:
        return apply_dunder(Dunder.POW, self, self, other)

    def __rpow__(self, other) -> Any:
        return apply_dunder(Dunder.POW, self, other, self)

    def __matmul__(self, other) -> Any:
        return apply_dunder(Dunder.MATMUL, self, self, other)

    def __rmatmul__(self, other) -> Any:
        return apply_dunder(Dunder.MATMUL, self, other, self)

    def __and__(self, other) -> Any:
        return apply_dunder(Dunder.AND, self, self, other)

    def __rand__(self, other) -> Any:
        return apply_dunder(Dunder.AND, self, other, self)

    def __or__(self, other) -> Any:
        return apply_dunder(Dunder.OR, self, self, other)

    def __ror__(self, other) -> Any:
        return apply_dunder(Dunder.OR, self, other, self)

    def __xor__(self, other) -> Any:
        return apply_dunder(Dunder.XOR, self, self, other)

    def __rxor__(self, other) -> Any:
        return apply_dunder(Dunder.XOR, self, other, self)

    def __lshift__(self, other) -> Any:
        return apply_dunder(Dunder.LSHIFT, self, self, other)

    def __rlshift__(self, other) -> Any:
        return apply_dunder(Dunder.LSHIFT, self, other, self)

    def __rshift__(self, other) -> Any:
        return apply_dunder(Dunder.RSHIFT, self, self, other)

    def __rrshift__(self, other) -> Any:
        return apply_dunder(Dunder.RSHIFT, self, other, self)

    def __bool__(self) -> bool:
        return apply_dunder(Dunder.BOOL, self, self)

    def __bytes__(self) -> bytes:
        return apply_dunder(Dunder.BYTES, self, self)

    def __call__(self, /, *args, **kwargs) -> Any:
        return apply_dunder(Dunder.CALL, self, self, *args, **kwargs)

    def __ceil__(self) -> Any:
        return apply_dunder(Dunder.CEIL, self, self)

    def __complex__(self) -> complex:
        return apply_dunder(Dunder.COMPLEX, self, self)

    def __contains__(self, item) -> bool:
        return apply_dunder(Dunder.CONTAINS, self, self, item)

    def __float__(self) -> float:
        return apply_dunder(Dunder.FLOAT, self, self)

    def __floor__(self) -> Any:
        return apply_dunder(Dunder.FLOOR, self, self)

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

    def __next__(self) -> Any:
        return apply_dunder(Dunder.NEXT, self, self)

    def __reversed__(self) -> Iterator[Any]:
        return apply_dunder(Dunder.REVERSED, self, self)

    def __round__(self, ndigits: int | None = None) -> Any:
        if ndigits is None:
            return apply_dunder(Dunder.ROUND, self, self)
        return apply_dunder(Dunder.ROUND, self, self, ndigits)

    def __str__(self) -> str:
        return apply_dunder(Dunder.STR, self, self)

    def __trunc__(self) -> Any:
        return apply_dunder(Dunder.TRUNC, self, self)


core.aval_types[TraceBox] = lambda value: value.aval


def apply_dunder(dunder: Dunder, box: TraceBox, /, *operands, **kwargs):
    if (rule := dunder_rules.get((dunder, type(box.aval)))) is None:
        raise TypeError(f"No trace rule for {dunder.value} on values of type {box.aval!r}.")
    return rule(*operands, **kwargs)


def assert_foldable(prim: core.Prim, value: Tree) -> None:
    traced_values = [x for x in utils.tree.leaves(value) if isinstance(x, TraceBox)]
    assert not traced_values, (
        f"Cannot evaluate {prim.name} in af.fold() because it depends on traced values "
        f"{traced_values!r}. Mark the dependencies static or move this computation outside af.fold()."
    )


class TraceInterpreter(core.Interpreter):
    __slots__ = ["eqns"]

    def __init__(self):
        self.eqns: list[Eqn] = []

    def box(self, value, /) -> Tree:
        return utils.tree.map(lambda v: TraceBox(owner=self, var=v) if is_var(v) else v, value)

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
        out_aval_tree = core.abstract_rules.get(prim)(in_aval_tree, **params)

        def to_out_ir_atom(x):
            # NOTE(asem): abstract rules return `AVal`/ python leaves.
            # `AVal` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return Var.fresh(aval=x) if isinstance(x, core.AVal) else x

        out_tree = utils.tree.map(to_out_ir_atom, out_aval_tree)
        self.eqns.append(Eqn(prim, in_tree, out_tree, params, active_tags.get()))
        return self.box(out_tree)


def trace[*A, R](
    func: Callable[[*A], R],
    /,
    *,
    static: Tree[bool] = False,
) -> Callable[[*A], IR[*A, R]]:
    """Build an IR by tracing a function's execution.

    Args:
        func: A callable that uses autoform primitives such as
            :func:`autoform.string.concat` and :func:`autoform.lm.fill`.
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
        return Var.fresh(aval=core.avalof(x))

    @ft.wraps(func)
    def wrapper(*args: *A) -> IR[*A, R]:
        arg_tree = args
        in_static_tree = utils.tree.broadcast_prefix(static, arg_tree, is_leaf=is_static_spec)
        in_tree = utils.tree.map(to_in_ir_atom, arg_tree, in_static_tree, is_leaf=is_traceable)
        with core.using_interpreter(TraceInterpreter()) as tracer:
            out_trace_tree = func(*cast(tuple, tracer.box(in_tree)))
        out_tree = tracer.unbox(out_trace_tree)
        return IR(eqns=tracer.eqns, in_tree=in_tree, out_tree=out_tree)

    return wrapper
