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
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from operator import setitem
from threading import RLock
from typing import Any, ClassVar, Self, TypeGuard, cast

from autoform.utils import Tree, lru_cache, pack_user_input, treelib

__all__ = [
    # base types
    "AVal",
    "val_types",
    # ir vals
    "IRVal",
    "IRVar",
    "IRLit",
    "is_irvar",
    "is_irlit",
    "is_irval",
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
    "aimpl_rules",
    "abatch_rules",
    "apush_rules",
    "apull_fwd_rules",
    "apull_bwd_rules",
    # ir structures
    "IREqn",
    "IR",
    # interpreters
    "Interpreter",
    "EvalInterpreter",
    "TracingInterpreter",
    "active_interpreter",
    "using_interpreter",
    # ir building and execution
    "trace",
    "call",
    "acall",
    # effects
    "Effect",
    "EffectInterpreter",
    "using_effect",
    "active_effect",
]

# ==================================================================================================
# BASE TYPES
# ==================================================================================================

val_types: set[type] = {str, int, float, bool}

type Val = str | int | float | bool


def is_val(x) -> bool:
    return isinstance(x, tuple(val_types))


class AVal:
    __slots__ = "type"

    def __init__(self, type: type):
        self.type = type


def is_var(x) -> TypeGuard[AVal]:
    return isinstance(x, AVal)


type EvalType = AVal | Val


def typeof(x, /) -> type:
    return x.type if is_var(x) else type(x)


# ==================================================================================================
# IR VALS
# ==================================================================================================


# NOTE(asem): leaf IR nodes either variables (placeholders) for user inputs
# or literals (constants) baked in the IR
class IRVal(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def aval(self) -> Any: ...


class IRVar[T: type](IRVal):
    __slots__ = ("id", "source", "type")
    counter: ClassVar[it.count[int]] = it.count(0)
    lock: ClassVar[RLock] = RLock()

    def __init__(self, /, *, type: T, source: IRVar | None = None):
        self.id = next(self.counter)
        assert is_irvar(source) or source is None
        self.source = source
        self.type = type

    @classmethod
    def fresh(cls, *, type: T, source: IRVar | None = None) -> Self:
        with cls.lock:
            return cls(source=source, type=type)

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source else ""
        return f"{type(self).__name__}[{self.type.__name__}](id={self.id}{source})"

    @property
    def aval(self) -> AVal:
        return AVal(self.type)


def is_irvar(x) -> TypeGuard[IRVar]:
    return isinstance(x, IRVar)


def is_irval(x) -> TypeGuard[IRVal]:
    return isinstance(x, IRVal)


class IRLit[T](IRVal):
    # NOTE(asem): IRLit wraps leaf-level values in pytrees. so non-hashable mutable structures

    __slots__ = "value"

    def __init__(self, value: T, /):
        assert not is_irval(value)
        assert hash(value) is not None
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, IRLit) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def aval(self) -> T:
        return self.value


def is_irlit(x) -> TypeGuard[IRLit]:
    return isinstance(x, IRLit)


# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


# NOTE(asem): tags are used to group primitives into categories
# to be later targeted by effect handlers, ...
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
# IR
# ==================================================================================================


class IREqn:
    __slots__ = ("prim", "effect", "in_ir_tree", "out_ir_tree", "params")
    __match_args__ = ("prim", "effect", "in_ir_tree", "out_ir_tree", "params")

    def __init__(
        self,
        prim: Prim,
        effect: Effect | None,
        in_ir_tree: Tree[IRVal],
        out_ir_tree: Tree[IRVal],
        params: dict | None = None,
    ):
        assert isinstance(prim, Prim)
        assert isinstance(effect, Effect) or effect is None
        assert treelib.all(treelib.map(is_irval, (in_ir_tree, out_ir_tree)))
        assert isinstance(params, dict) or params is None
        self.prim = prim
        self.effect = effect
        self.in_ir_tree = in_ir_tree
        self.out_ir_tree = out_ir_tree
        self.params = params if params is not None else {}

    def bind(self, in_tree: Tree, /, **params):
        with using_effect(self.effect):
            return self.prim.bind(in_tree, **params)

    async def abind(self, in_tree: Tree, /, **params):
        with using_effect(self.effect):
            return await self.prim.abind(in_tree, **params)

    def using(self, **kwargs) -> IREqn:
        return IREqn(
            self.prim, self.effect, self.in_ir_tree, self.out_ir_tree, self.params | kwargs
        )


class IR[**P, R]:
    __slots__ = ("ir_eqns", "in_ir_tree", "out_ir_tree")
    __match_args__ = ("ir_eqns", "in_ir_tree", "out_ir_tree")

    def __init__(self, ir_eqns: list[IREqn], in_ir_tree: Tree[IRVal], out_ir_tree: Tree[IRVal]):
        assert isinstance(ir_eqns, list)
        assert all(isinstance(ir_eqn, IREqn) for ir_eqn in ir_eqns)
        assert treelib.all(treelib.map(is_irval, (in_ir_tree, out_ir_tree)))
        self.ir_eqns = tuple(ir_eqns)
        self.in_ir_tree = in_ir_tree
        self.out_ir_tree = out_ir_tree

    def __repr__(self) -> str:
        return generate_text_code(ir=self, expand_ir=True)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return call(self)(*args, **kwargs)

    async def acall(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await acall(self)(*args, **kwargs)


def generate_text_code(ir: IR, indent: int = 2, *, expand_ir: bool = False) -> str:
    assert isinstance(indent, int) and indent >= 0
    sp = " " * indent

    def format_irval(irval: IRVal) -> str:
        assert isinstance(irval, IRVal)
        if is_irvar(irval):
            var_type = type(irval).__name__
            type_info = f"[{irval.type.__name__}]" if irval.type is not None else ""
            return f"%{irval.id}:{var_type}{type_info}"
        assert is_irlit(irval)
        val = irval.value
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
        return ", ".join(format_irval(leaf) for leaf in leaves) if leaves else "()"

    in_sig = format_tree(ir.in_ir_tree)
    out_sig = format_tree(ir.out_ir_tree)

    header = f"func({in_sig}) -> ({out_sig}) {{"
    lines = [header]

    for ir_eqn in ir.ir_eqns:
        lhs = format_tree(ir_eqn.out_ir_tree)
        rhs = format_tree(ir_eqn.in_ir_tree)
        params_str = ", ".join(f"{k}={ir_eqn.params[k]!r}" for k in (ir_eqn.params or {}))
        effect_str = f" @{ir_eqn.effect!r}" if ir_eqn.effect else ""
        if params_str:
            lines.append(f"{sp}({lhs}) = {ir_eqn.prim.name}({rhs}, {params_str}){effect_str}")
        else:
            lines.append(f"{sp}({lhs}) = {ir_eqn.prim.name}({rhs}){effect_str}")

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
# EFFECT
# ==================================================================================================


class Effect:
    __slots__ = ()


active_effect: ContextVar[Effect | None] = ContextVar("active_effect", default=None)


@contextmanager
def using_effect[T: Effect](effect: T | None) -> Generator[T | None, None, None]:
    if effect is None:
        yield effect
        return
    assert isinstance(effect, Effect), f"Expected Effect, got {type(effect)}"
    token = active_effect.set(effect)
    try:
        yield effect
    finally:
        active_effect.reset(token)


type Handler = Callable[..., Generator[Any, Any, Any]]


class EffectInterpreter(Interpreter, ABC):
    # NOTE(asem): handler patterns (callable-based, not methods)
    # handler signature: handler(prim, effect, in_tree, /, **params)
    #
    # skip (replace value, no continuation)
    # >>> def handler(prim, effect, in_tree, /):
    # ...     return replacement
    # ...     yield
    #
    # pass-through (observe only)
    # >>> def handler(prim, effect, in_tree, /):
    # ...     return (yield in_tree)
    #
    # pre-process input
    # >>> def handler(prim, effect, in_tree, /):
    # ...     return (yield transform(in_tree))
    #
    # post-process output
    # >>> def handler(prim, effect, in_tree, /, **params):
    # ...     result = yield in_tree
    # ...     return transform(result)
    def __init__(self, *handlers: tuple[type[Effect], Handler]):
        for handler in handlers:
            msg = "handlers must be (EffectType, handler) pairs"
            assert isinstance(handler, Sequence) and len(handler) == 2, msg
            eff_type, _ = handler
            assert issubclass(eff_type, Effect), f"Invalid effect type: {eff_type}"
        self.parent = active_interpreter.get()
        self.handlers: dict[type[Effect], Handler] = dict(handlers)

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        effect = active_effect.get()

        if (handler := self.handlers.get(type(effect))) is None:
            return self.parent.interpret(prim, in_tree, **params)

        gen = handler(prim, effect, in_tree, **params)
        result = None

        # NOTE(asem):
        # the handler can yield multiple times, each yield invokes the continuation.
        # >>> def handler(prim, effect, in_tree, /, **params):
        # ...     results = []
        # ...     for v in (...):
        # ...         result = yield v
        # ...         results.append(result)
        # ...     return best(results)
        # ...     yield
        while True:
            try:
                modified_input = next(gen) if result is None else gen.send(result)
            except StopIteration as e:
                return e.value

            result = self.parent.interpret(prim, modified_input, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        effect = active_effect.get()

        if (handler := self.handlers.get(type(effect))) is None:
            return await self.parent.ainterpret(prim, in_tree, **params)

        gen = handler(prim, effect, in_tree, **params)
        result = None

        # NOTE(asem): same generator protocol as sync, but uses ainterpret for continuation
        while True:
            try:
                modified_input = next(gen) if result is None else gen.send(result)
            except StopIteration as e:
                return e.value

            result = await self.parent.ainterpret(prim, modified_input, **params)


# ==================================================================================================
# TRACING
# ==================================================================================================


class TracingInterpreter(Interpreter):
    def __init__(self):
        self.ir_eqns: list[IREqn] = []

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree[IRVal]:
        def to_in_ir_val(x) -> IRVal:
            # NOTE(asem): input can be IRVal in 3 cases:
            # 1. function inputs wrapped by trace
            # 2. output from previous prim calls
            # 3. raw python values needed to be wrapped by IRLit (local consts, closed-over values)
            return x if is_irval(x) else IRLit(x)

        in_ir_tree = treelib.map(to_in_ir_val, in_tree)
        in_aval_tree = treelib.map(lambda x: x.aval, in_ir_tree)
        out_aval_tree = abstract_rules.get(prim)(in_aval_tree, **params)

        def to_out_ir_val(x) -> IRVal:
            # NOTE(asem): abstract rules return `AVal`/ python types.
            # `AVal` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return IRVar.fresh(type=x.type) if is_var(x) else IRLit(x)

        out_ir_tree = treelib.map(to_out_ir_val, out_aval_tree)
        effect = active_effect.get()
        self.ir_eqns.append(IREqn(prim, effect, in_ir_tree, out_ir_tree, params))
        return out_ir_tree

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree[IRVal]:
        return self.interpret(prim, in_tree, **params)


def trace[**P, R](func: Callable[P, R], /, *, static: Tree[bool] = False) -> Callable[P, IR[P, R]]:
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
        >>> af.call(ir)(True)
        'error'
    """

    def is_static_spec(x) -> bool:
        return isinstance(x, bool)

    def to_ir_var(x, /) -> IRVar:
        assert not is_irval(x), "Inputs to `trace` must be normal python types"
        assert is_val(x), f"Unsupported input leaf type for `trace`: {type(x).__name__}. "
        return IRVar.fresh(type=type(x))

    def to_in_ir_val(x, is_static: bool) -> IRVal:
        return IRLit(x) if is_static else to_ir_var(x)

    def to_out_ir_val(x) -> IRVal:
        return x if is_irval(x) else IRLit(x)

    @ft.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> IR[P, R]:
        assert not kwargs, "`trace` does not support keyword arguments"
        in_tree = pack_user_input(*args)
        in_static_tree = treelib.broadcast_prefix(static, in_tree, is_leaf=is_static_spec)
        in_ir_tree = treelib.map(to_in_ir_val, in_tree, in_static_tree, is_leaf=is_val)
        with using_interpreter(TracingInterpreter()) as tracer:
            in_prog_tree = (in_ir_tree,) if len(args) == 1 else cast(tuple, in_ir_tree)
            out_prog_tree = func(*in_prog_tree)
        out_ir_tree = treelib.map(to_out_ir_val, out_prog_tree)
        return IR(ir_eqns=tracer.ir_eqns, in_ir_tree=in_ir_tree, out_ir_tree=out_ir_tree)

    return wrapper


# ==================================================================================================
# CALL
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def call[**P, R](ir: IR[P, R], /) -> Callable[P, R]:
    """Call an IR.

    Args:
        ir: The IR to run.

    Returns:
        A callable that runs the IR with the provided arguments.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: af.format("Hello {}", x))("world")
        >>> af.call(ir)("Alice")
        'Hello Alice'
    """

    def func(*args: P.args, **kwargs: P.kwargs) -> R:
        assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
        assert not kwargs, "`call` does not support keyword arguments"
        in_tree = pack_user_input(*args)
        env: dict[IRVar, Any] = {}

        def read(irval: IRVal) -> Any:
            return env[irval] if is_irvar(irval) else cast(IRLit, irval).value

        def check_input(irval: IRVal, value: Any):
            if is_irlit(irval):
                msg = f"Static input mismatch: expected {irval.value!r}, got {value!r}"
                assert irval.value == value, msg

        def write(irval: IRVal, value):
            is_irvar(irval) and setitem(env, irval, value)

        treelib.map(check_input, ir.in_ir_tree, in_tree)
        treelib.map(write, ir.in_ir_tree, in_tree)
        for ir_eqn in ir.ir_eqns:
            in_values = treelib.map(read, ir_eqn.in_ir_tree)
            out_values = ir_eqn.bind(in_values, **ir_eqn.params)
            treelib.map(write, ir_eqn.out_ir_tree, out_values)
        return treelib.map(read, ir.out_ir_tree)

    return func


@ft.partial(lru_cache, maxsize=256)
def acall[**P, R](ir: IR[P, R], /) -> Callable[P, Awaitable[R]]:
    """Async call an IR.

    Args:
        ir: The IR to run.

    Returns:
        A callable that returns a coroutine running the IR with the provided arguments.

    Example:
        >>> import autoform as af
        >>> import asyncio
        >>> ir = af.trace(lambda x: af.format("Hello {}", x))("world")
        >>> asyncio.run(af.acall(ir)("Alice"))
        'Hello Alice'
    """

    async def func(*args: P.args, **kwargs: P.kwargs) -> R:
        assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
        assert not kwargs, "`acall` does not support keyword arguments"
        in_tree = pack_user_input(*args)
        env: dict[IRVar, Any] = {}

        def read(irval: IRVal) -> Any:
            return env[irval] if is_irvar(irval) else cast(IRLit, irval).value

        def check_input(irval: IRVal, value: Any):
            if is_irlit(irval):
                msg = f"Static input mismatch: expected {irval.value!r}, got {value!r}"
                assert irval.value == value, msg

        def write(irval: IRVal, value):
            is_irvar(irval) and setitem(env, irval, value)

        treelib.map(check_input, ir.in_ir_tree, in_tree)
        treelib.map(write, ir.in_ir_tree, in_tree)
        for ir_eqn in ir.ir_eqns:
            in_values = treelib.map(read, ir_eqn.in_ir_tree)
            out_values = await ir_eqn.abind(in_values, **ir_eqn.params)
            treelib.map(write, ir_eqn.out_ir_tree, out_values)
        return treelib.map(read, ir.out_ir_tree)

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
