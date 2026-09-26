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

"""Program construction and execution, independent of tracing."""

from __future__ import annotations

import functools as ft
import itertools as it
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator, Hashable
from contextlib import contextmanager
from contextvars import ContextVar
from operator import setitem
from typing import Any, ClassVar, Self, TypeGuard

import autoform.abstract as abstract
import autoform.utils as utils

type Tree[T] = utils.Tree[T]

__all__ = [
    # ir vals
    "Var",
    "is_var",
    "aval_if_var",
    # primitive
    "Prim",
    # rule registries
    "impl_rules",
    "aimpl_rules",
    "abstract_rules",
    "batch_rules",
    "abatch_rules",
    "push_rules",
    "apush_rules",
    "pull_fwd_rules",
    "apull_fwd_rules",
    "pull_bwd_rules",
    "apull_bwd_rules",
    # ir structures
    "Eqn",
    "IR",
    # interpreters
    "Interpreter",
    "Box",
    "EvalInterpreter",
    "active_interpreter",
    "using_interpreter",
    "active_tags",
    "tag",
    # ir building and execution
    "check_static_inputs",
]

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

    def __init__(self, /, *, aval: abstract.AVal, source: Var | None = None):
        self.id = next(self.counter)
        assert is_var(source) or source is None
        assert isinstance(aval, abstract.AVal)
        self.source = source
        self.aval = aval

    @classmethod
    def fresh(cls, *, aval: abstract.AVal, source: Var | None = None) -> Self:
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
# INTERPRETER
# ==================================================================================================


class Box:
    __slots__ = ["owner"]

    def __init__(self, owner):
        self.owner = owner


class Interpreter[T](ABC):
    """Primitive dispatch and value boxing."""

    __slots__ = []

    @abstractmethod
    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Any: ...

    @abstractmethod
    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Any: ...

    @abstractmethod
    def box(self, value, /) -> Tree[T]: ...

    @abstractmethod
    def unbox(self, value: Tree, /): ...


@contextmanager
def using_interpreter[T: Interpreter](interpreter: T) -> Generator[T, None, None]:
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

    def box(self, value, /):
        return value

    def unbox(self, value, /):
        return value

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return impl_rules[prim](in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return await aimpl_rules[prim](in_tree, **params)


active_interpreter = ContextVar[Interpreter]("active_interpreter", default=EvalInterpreter())


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


type RuleMapping[T] = dict[Prim, Callable[..., T]]

impl_rules: RuleMapping[Tree] = {}
aimpl_rules: RuleMapping[Awaitable[Tree]] = {}
batch_rules: RuleMapping[tuple[Tree, Tree[bool]]] = {}
abatch_rules: RuleMapping[Awaitable[tuple[Tree, Tree[bool]]]] = {}
push_rules: RuleMapping[tuple[Tree, Tree]] = {}
apush_rules: RuleMapping[Awaitable[tuple[Tree, Tree]]] = {}
pull_fwd_rules: RuleMapping[tuple[Tree, Tree]] = {}
apull_fwd_rules: RuleMapping[Awaitable[tuple[Tree, Tree]]] = {}
pull_bwd_rules: RuleMapping[Tree] = {}
apull_bwd_rules: RuleMapping[Awaitable[Tree]] = {}
abstract_rules: RuleMapping[Tree[Any]] = {}

abstract.aval_types[Var] = lambda value: value.aval
