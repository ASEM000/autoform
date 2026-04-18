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
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from operator import setitem
from threading import RLock
from typing import Any, ClassVar, Self, TypeGuard, cast

from autoform.utils import Tree, lru_cache, treelib

__all__ = [
    # base types
    "AVal",
    "TypedAVal",
    "Val",
    "val_types",
    "is_val",
    # ir vals
    "IRVar",
    "is_irvar",
    "ir_aval",
    # tags
    "PrimTag",
    "TransformationTag",
    # primitive
    "Prim",
    # rule registries
    "impl_rules",
    "abstract_rules",
    "batch_rules",
    "push_rules",
    "pull_fwd_rules",
    "pull_bwd_rules",
    # ir structures
    "IREqn",
    "IR",
    # interpreters
    "Interpreter",
    "EvalInterpreter",
    "TracingInterpreter",
    "active_interpreter",
    "using_interpreter",
    "Tag",
    "active_tags",
    "tag",
    # ir building and execution
    "trace",
    "walk",
]

# ==================================================================================================
# BASE TYPES
# ==================================================================================================

val_types: set[type] = {str, int, float, bool}

type Val = str | int | float | bool


def is_val(x) -> bool:
    return isinstance(x, tuple(val_types))


class AVal:
    __slots__ = ()


class TypedAVal(AVal):
    __slots__ = "type"

    def __init__(self, type: type):
        self.type = type

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.type.__name__})"

    def __eq__(self, other) -> bool:
        return isinstance(other, TypedAVal) and self.type is other.type

    def __hash__(self) -> int:
        return hash((type(self), self.type))


def is_aval(x) -> TypeGuard[AVal]:
    return isinstance(x, AVal)


type EvalType = AVal | Val


def typeof(x, /) -> type:
    return x.type if is_aval(x) else type(x)


# ==================================================================================================
# IR VARS
# ==================================================================================================


# NOTE(asem): wrapped IR leaves are variables (placeholders) for user inputs.
# Concrete literals are kept as plain Python values in IR trees.
class IRVar:
    __slots__ = ("id", "source", "aval")
    counter: ClassVar[it.count[int]] = it.count(0)
    lock: ClassVar[RLock] = RLock()

    def __init__(self, /, *, aval: AVal, source: IRVar | None = None):
        self.id = next(self.counter)
        assert is_irvar(source) or source is None
        assert is_aval(aval)
        self.source = source
        self.aval = aval

    @classmethod
    def fresh(cls, *, aval: AVal, source: IRVar | None = None) -> Self:
        with cls.lock:
            return cls(source=source, aval=aval)

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source else ""
        aval = self.aval.type.__name__ if isinstance(self.aval, TypedAVal) else repr(self.aval)
        return f"{type(self).__name__}[{aval}](id={self.id}{source})"


def is_irvar(x) -> TypeGuard[IRVar]:
    return isinstance(x, IRVar)


def ir_aval(x, /):
    return x.aval if is_irvar(x) else x


# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


# NOTE(asem): tags are used to group primitives into categories
# to be later targeted by interceptors, ...
class PrimTag: ...


class TransformationTag(PrimTag): ...


class Prim:
    # NOTE(asem): primitive is a key used for matching against rules
    # defined in ``InterpreterRuleMapping``
    __slots__ = ("name", "tag")
    __match_args__ = ("name", "tag")

    def __init__(self, name: str, tag: set[type[PrimTag]] | None = None):
        assert isinstance(name, str), f"Invalid name type: {type(name)=}"
        assert tag is None or all(issubclass(t, PrimTag) for t in tag), f"Invalid tag: {tag=}"
        self.name = name
        self.tag: frozenset[type[PrimTag]] = frozenset(tag) if tag else frozenset()

    def __repr__(self) -> str:
        return self.name

    def bind(self, value: Tree, /, **params):
        return active_interpreter.get().interpret(self, value, **params)

    async def abind(self, value: Tree, /, **params):
        return await active_interpreter.get().ainterpret(self, value, **params)


# ==================================================================================================
# TAG
# ==================================================================================================


class Tag:
    """Base class for structured equation tags.

    Subclasses must be hashable.

    Example:
        >>> from dataclasses import dataclass
        >>> import autoform as af
        >>> @dataclass(frozen=True)
        ... class Label(af.Tag):
        ...     name: str
        >>> with af.tag(Label("draft")):
        ...     ir = af.trace(lambda x: af.concat(x, "!"))("seed")
        >>> ir.ir_eqns[0].tags == frozenset({Label("draft")})
        True
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        assert cls is not Tag, "Tag cannot be instantiated directly"
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert cls.__hash__ is not None, "Tag subclasses must be hashable"


active_tags: ContextVar[frozenset[Tag]] = ContextVar("active_tags", default=frozenset())


@contextmanager
def tag(*tags: Tag) -> Generator[tuple[Tag, ...], None, None]:
    """Attach tags to equations at trace time.

    Equations built inside nested ``tag`` blocks receive the tags from all active
    blocks. Equations built after a block exits do not receive that block's tags.

    Example:
        >>> from dataclasses import dataclass
        >>> import autoform as af
        >>> @dataclass(frozen=True)
        ... class Label(af.Tag):
        ...     name: str
        >>> def program(x):
        ...     with af.tag(Label("outer")):
        ...         head = af.concat(x, "!")
        ...         with af.tag(Label("inner")):
        ...             return af.concat(head, "?")
        >>> ir = af.trace(program)("seed")
        >>> ir.ir_eqns[0].tags == frozenset({Label("outer")})
        True
        >>> ir.ir_eqns[1].tags == frozenset({Label("outer"), Label("inner")})
        True
    """

    assert all(isinstance(tag, Tag) for tag in tags), f"Expected Tag instances, got {tags!r}"
    token = active_tags.set(active_tags.get() | frozenset(tags))
    try:
        yield tags
    finally:
        active_tags.reset(token)


# ==================================================================================================
# IR
# ==================================================================================================


class IREqn:
    __slots__ = ("prim", "in_ir_tree", "out_ir_tree", "params", "tags")
    __match_args__ = ("prim", "in_ir_tree", "out_ir_tree", "params", "tags")

    def __init__(
        self,
        prim: Prim,
        in_ir_tree: Tree,
        out_ir_tree: Tree,
        params: dict[str, Any] | None = None,
        tags: frozenset[Tag] = frozenset(),
    ):
        assert isinstance(prim, Prim)
        assert isinstance(params, dict) or params is None
        assert isinstance(tags, frozenset)
        self.prim = prim
        self.in_ir_tree = in_ir_tree
        self.out_ir_tree = out_ir_tree
        self.params = params if params is not None else {}
        assert all(isinstance(tag, Tag) for tag in tags), f"Expected Tag instances, got {tags!r}"
        self.tags = tags

    def bind(self, in_tree: Tree, /, **params):
        with tag(*self.tags):
            return self.prim.bind(in_tree, **params)

    async def abind(self, in_tree: Tree, /, **params):
        with tag(*self.tags):
            return await self.prim.abind(in_tree, **params)

    def using(self, **kwargs) -> IREqn:
        return IREqn(self.prim, self.in_ir_tree, self.out_ir_tree, self.params | kwargs, self.tags)


class IR[*A, R]:
    __slots__ = ("ir_eqns", "in_ir_tree", "out_ir_tree")
    __match_args__ = ("ir_eqns", "in_ir_tree", "out_ir_tree")

    def __init__(self, ir_eqns: list[IREqn], in_ir_tree: Tree, out_ir_tree: Tree):
        assert isinstance(ir_eqns, list)
        ir_eqns = tuple(ir_eqns)
        assert all(isinstance(ir_eqn, IREqn) for ir_eqn in ir_eqns)
        self.ir_eqns = ir_eqns
        self.in_ir_tree = in_ir_tree
        self.out_ir_tree = out_ir_tree

    def __repr__(self) -> str:
        return generate_text_code(ir=self, expand_ir=True)

    def call(self, *args: *A) -> R:
        """Run IR with concrete runtime inputs.

        Use this after `trace(...)` has produced an `IR`. Pass values with the same
        pytree structure as `in_ir_tree`; the method executes the stored equations
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

    def walk(self, *args: *A) -> Generator[tuple[IREqn | None, Tree], Tree, None]:
        """Step through this IR one equation at a time.

        Manual control over IR execution. Start with `next(gen)` to receive `(ir_eqn, in_values)`,
        compute or override the equation output, using `ir_eqn.bind(in_values, **ir_eqn.params)`
        for synchronous execution or `await ir_eqn.abind(in_values, **ir_eqn.params)` for async
        execution, and send that output back with `gen.send(...)`. After the last equation,
        the generator yields `(None, out_tree)`.

        Example:
            >>> import autoform as af
            >>> def wrap(x):
            ...     punctuated = af.concat(x, "!")
            ...     return af.format("[{}]", punctuated)
            >>> ir = af.trace(wrap)("x")
            >>> gen = ir.walk("y")
            >>> ir_eqn, in_values = next(gen)
            >>> ir_eqn.prim.name
            'concat'
            >>> step = gen.send(ir_eqn.bind(in_values, **ir_eqn.params))
            >>> ir_eqn, in_values = step
            >>> ir_eqn.prim.name
            'format'
            >>> done, out = gen.send(ir_eqn.bind(in_values, **ir_eqn.params))
            >>> done is None, out
            (True, '[y!]')
        """
        return walk(self)(*args)


def generate_text_code(ir: IR, indent: int = 2, *, expand_ir: bool = False) -> str:
    assert isinstance(indent, int) and indent >= 0
    sp = " " * indent

    def format_ir_val(ir_val) -> str:
        if is_irvar(ir_val):
            var_type = type(ir_val).__name__
            aval_info = (
                ir_val.aval.type.__name__
                if isinstance(ir_val.aval, TypedAVal)
                else repr(ir_val.aval)
            )
            type_info = f"[{aval_info}]"
            return f"%{ir_val.id}:{var_type}{type_info}"
        val = ir_val
        if isinstance(val, IR):
            if expand_ir:
                sub_code = generate_text_code(val, indent, expand_ir=True)
                return f"<IR:{{\n{sub_code}\n}}>"
            else:
                prim_names = ",".join(e.prim.name for e in val.ir_eqns)
                if len(prim_names) > 20:
                    prim_names = prim_names[:17] + "..."
                return f"<IR:[{prim_names}]>"
        else:
            val_repr = repr(val)
            if len(val_repr) > 30:
                val_repr = val_repr[:27] + "..."
            return f"{val_repr}:Lit"

    def format_tree(tree: Tree) -> str:
        leaves = treelib.leaves(tree)
        return ", ".join(format_ir_val(leaf) for leaf in leaves) if leaves else "()"

    in_sig = format_tree(ir.in_ir_tree)
    out_sig = format_tree(ir.out_ir_tree)

    header = f"func({in_sig}) -> ({out_sig}) {{"
    lines = [header]

    for ir_eqn in ir.ir_eqns:
        lhs = format_tree(ir_eqn.out_ir_tree)
        rhs = format_tree(ir_eqn.in_ir_tree)
        params_str = ", ".join(f"{k}={ir_eqn.params[k]!r}" for k in (ir_eqn.params or {}))
        if params_str:
            lines.append(f"{sp}({lhs}) = {ir_eqn.prim.name}({rhs}, {params_str})")
        else:
            lines.append(f"{sp}({lhs}) = {ir_eqn.prim.name}({rhs})")

    lines.append("}")
    return "\n".join(lines)


# ==================================================================================================
# INTERPRETER
# ==================================================================================================


class Interpreter(ABC):
    @abstractmethod
    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Any: ...

    @abstractmethod
    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Any: ...


@contextmanager
def using_interpreter[T: Interpreter](interpreter: T) -> Generator[T, None, None]:
    token = active_interpreter.set(interpreter)
    try:
        yield interpreter
    finally:
        active_interpreter.reset(token)


# ==================================================================================================
# EVAL
# ==================================================================================================


class EvalInterpreter(Interpreter):
    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return impl_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return await impl_rules.aget(prim)(in_tree, **params)


active_interpreter = ContextVar[Interpreter]("active_interpreter", default=EvalInterpreter())


# ==================================================================================================
# TRACING
# ==================================================================================================


class TracingInterpreter(Interpreter):
    def __init__(self):
        self.ir_eqns: list[IREqn] = []

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        def to_in_ir_atom(value):
            if not is_irvar(value):
                hash(value)
            return value

        def to_concrete(leaf, value):
            assert not is_irvar(value), f"Unexpected variable at {'/'.join(map(str, leaf))}"
            return value

        params = treelib.map_with_path(to_concrete, params)

        in_ir_tree = treelib.map(to_in_ir_atom, in_tree)
        in_aval_tree = treelib.map(ir_aval, in_ir_tree)
        out_aval_tree = abstract_rules.get(prim)(in_aval_tree, **params)

        def to_out_ir_atom(x):
            # NOTE(asem): abstract rules return `AVal`/ python leaves.
            # `AVal` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return IRVar.fresh(aval=x) if is_aval(x) else x

        out_ir_tree = treelib.map(to_out_ir_atom, out_aval_tree)
        tags = active_tags.get()
        self.ir_eqns.append(IREqn(prim, in_ir_tree, out_ir_tree, params, tags=tags))
        return out_ir_tree

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return self.interpret(prim, in_tree, **params)


def trace[*A, R](
    func: Callable[[*A], R], /, *, static: Tree[bool] = False
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
        return to_ir_var(x)

    def to_ir_var(x, /) -> IRVar:
        assert not is_irvar(x), "Inputs to `trace` must be normal python types"
        assert is_val(x), f"Unsupported input leaf type for `trace`: {type(x).__name__}. "
        return IRVar.fresh(aval=TypedAVal(type(x)))

    @ft.wraps(func)
    def wrapper(*args: *A) -> IR[*A, R]:
        in_tree = args
        in_static_tree = treelib.broadcast_prefix(static, in_tree, is_leaf=is_static_spec)
        in_ir_tree = treelib.map(to_in_ir_atom, in_tree, in_static_tree, is_leaf=is_val)
        with using_interpreter(TracingInterpreter()) as tracer:
            out_prog_tree = func(*cast(tuple, in_ir_tree))
        return IR(ir_eqns=tracer.ir_eqns, in_ir_tree=in_ir_tree, out_ir_tree=out_prog_tree)

    return wrapper


# ==================================================================================================
# WALK
# ==================================================================================================

type GenStep = tuple[IREqn | None, Tree]


@ft.partial(lru_cache, maxsize=256)
def walk[*A, R](ir: IR[*A, R], /) -> Callable[[*A], Generator[GenStep, Tree, None]]:
    """Walk an IR one equation at a time."""

    def func(*args: *A) -> Generator[GenStep, Tree, None]:
        assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
        env: dict[IRVar, Any] = {}

        def read(ir_val) -> Any:
            return env[ir_val] if is_irvar(ir_val) else ir_val

        def check_input(ir_val, value: Any):
            if not is_irvar(ir_val):
                expected = ir_val
                msg = f"Static input mismatch: expected {expected!r}, got {value!r}"
                assert expected == value, msg

        def write(ir_val, value: Any):
            is_irvar(ir_val) and setitem(env, ir_val, value)

        treelib.map(check_input, ir.in_ir_tree, args)
        treelib.map(write, ir.in_ir_tree, args)

        for ir_eqn in ir.ir_eqns:
            in_values = treelib.map(read, ir_eqn.in_ir_tree)
            out_values = yield ir_eqn, in_values
            treelib.map(write, ir_eqn.out_ir_tree, out_values)

        yield None, treelib.map(read, ir.out_ir_tree)

    return func


# ==================================================================================================
# CALL
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def call[*A, R](ir: IR[*A, R], /) -> Callable[[*A], R]:
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def func(*args: *A) -> R:
        step = next(gen := walk(ir)(*args))
        for _ in ir.ir_eqns:
            ir_eqn, in_values = step
            assert ir_eqn is not None
            out_values = ir_eqn.bind(in_values, **ir_eqn.params)
            step = gen.send(out_values)
        ir_eqn, out = step
        assert ir_eqn is None
        return out

    return func


@ft.partial(lru_cache, maxsize=256)
def acall[*A, R](ir: IR[*A, R], /) -> Callable[[*A], Awaitable[R]]:
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    async def func(*args: *A) -> R:
        step = next(gen := walk(ir)(*args))
        for _ in ir.ir_eqns:
            ir_eqn, in_values = step
            assert ir_eqn is not None
            out_values = await ir_eqn.abind(in_values, **ir_eqn.params)
            step = gen.send(out_values)
        ir_eqn, out = step
        assert ir_eqn is None
        return out

    return func


# ==================================================================================================
# RULES
# ==================================================================================================


class InterpreterRuleMapping[R]:
    def __init__(self):
        self.map: dict[Prim, Callable[..., R]] = {}
        self.amap: dict[Prim, Callable[..., Awaitable[R]]] = {}
        self.lock = RLock()

    def set(self, prim: Prim, rule: Callable[..., R], /) -> Callable[..., R]:
        assert isinstance(prim, Prim), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert prim not in self.map, f"Rule for primitive {prim} already defined"

        with self.lock:
            self.map[prim] = rule
        return rule

    def aset(self, prim: Prim, rule: Callable[..., Awaitable[R]], /) -> Callable[..., Awaitable[R]]:
        assert isinstance(prim, Prim), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert prim not in self.amap, f"Async rule for primitive {prim} already defined"

        with self.lock:
            self.amap[prim] = rule
        return rule

    def get(self, prim: Prim) -> Callable[..., R]:
        with self.lock:
            if prim not in self.map:
                raise KeyError(f"No {type(self).__name__} rule defined for primitive {prim}")
            return self.map[prim]

    def aget(self, prim: Prim) -> Callable[..., Awaitable[R]]:
        with self.lock:
            if prim not in self.amap:
                raise KeyError(f"No async {type(self).__name__} rule defined for primitive {prim}")
            return self.amap[prim]


impl_rules = InterpreterRuleMapping[Tree]()
batch_rules = InterpreterRuleMapping[tuple[Tree, Tree[bool] | bool]]()
push_rules = InterpreterRuleMapping[tuple[Tree, Tree]]()
pull_fwd_rules = InterpreterRuleMapping[tuple[Tree, Tree]]()
pull_bwd_rules = InterpreterRuleMapping[Tree]()
abstract_rules = InterpreterRuleMapping[Tree[EvalType]]()
