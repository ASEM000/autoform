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
    "Var",
    "user_types",
    # ir atoms
    "IRAtom",
    "IRVar",
    "IRLit",
    "is_irvar",
    "is_irlit",
    "is_iratom",
    "iratom_to_evaltype",
    # tags
    "PrimitiveTag",
    "TransformationTag",
    # primitive
    "Primitive",
    # rule registries
    "impl_rules",
    "eval_rules",
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

user_types: set[type] = {str}

type UserType = str
type Value = str


def is_user_type(x) -> bool:
    return isinstance(x, tuple(user_types))


class Var:
    __slots__ = "type"

    def __init__(self, type: type):
        self.type = type


def is_var(x) -> TypeGuard[Var]:
    return isinstance(x, Var)


type EvalType = Var | UserType


# ==================================================================================================
# IR ATOMS
# ==================================================================================================


# NOTE(asem): atomic IR nodes either variables (placeholders) for user inputs
# or literals (constants) baked in the IR
class IRAtom:
    __slots__ = ()


class IRVar[T: type](IRAtom):
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


def is_irvar(x) -> TypeGuard[IRVar]:
    return isinstance(x, IRVar)


def is_iratom(x) -> TypeGuard[IRAtom]:
    return isinstance(x, IRAtom)


def iratom_to_evaltype(x: IRAtom) -> EvalType:
    return Var(x.type) if is_irvar(x) else x.value


class IRLit[T](IRAtom):
    # NOTE(asem): IRLit wraps leaf-level values in pytrees. so non-hashable mutable structures

    __slots__ = "value"

    def __init__(self, value: T, /):
        assert not is_iratom(value)
        assert hash(value) is not None
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __hash__(self) -> int:
        return hash(self.value)


def is_irlit(x) -> TypeGuard[IRLit]:
    return isinstance(x, IRLit)


# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


# NOTE(asem): tags are used to group primitives into categories
# to be later targeted by effect handlers, ...
class PrimitiveTag: ...


class TransformationTag(PrimitiveTag): ...


class Primitive:
    # NOTE(asem): primitive is a key used for matching against rules
    # defined in ``InterpreterRuleMapping``
    __slots__ = ("name", "tag")
    __match_args__ = ("name", "tag")

    def __init__(self, name: str, tag: set[type[PrimitiveTag]] | None = None):
        assert isinstance(name, str), f"Invalid name type: {type(name)=}"
        assert tag is None or all(issubclass(t, PrimitiveTag) for t in tag), f"Invalid tag: {tag=}"
        self.name = name
        self.tag: frozenset[type[PrimitiveTag]] = frozenset(tag) if tag else frozenset()

    def __repr__(self) -> str:
        return self.name

    def bind(self, value: Tree, /, **params):
        return active_interpreter.get().interpret(self, value, **params)

    async def abind(self, value: Tree, /, **params):
        return await active_interpreter.get().ainterpret(self, value, **params)


# ==================================================================================================
# IR EQUATIONS AND PROGRAMS
# ==================================================================================================


class IREqn:
    __slots__ = ("prim", "in_irtree", "out_irtree", "effect", "params")
    __match_args__ = ("prim", "in_irtree", "out_irtree", "effect", "params")

    def __init__(
        self,
        prim: Primitive,
        in_irtree: Tree[IRAtom],
        out_irtree: Tree[IRAtom],
        effect: Effect | None = None,
        params: dict | None = None,
    ):
        assert isinstance(prim, Primitive)
        assert treelib.all(treelib.map(is_iratom, (in_irtree, out_irtree)))
        assert isinstance(effect, Effect) or effect is None
        assert isinstance(params, dict) or params is None
        self.prim = prim
        self.in_irtree = in_irtree
        self.out_irtree = out_irtree
        self.effect = effect
        self.params = params if params is not None else {}

    def bind(self, in_tree: Tree, /, **params):
        with using_effect(self.effect):
            return self.prim.bind(in_tree, **params)

    async def abind(self, in_tree: Tree, /, **params):
        with using_effect(self.effect):
            return await self.prim.abind(in_tree, **params)

    def using(self, **kwargs) -> IREqn:
        return IREqn(self.prim, self.in_irtree, self.out_irtree, self.effect, self.params | kwargs)


class IR[**P, R]:
    __slots__ = ("ireqns", "in_irtree", "out_irtree")
    __match_args__ = ("ireqns", "in_irtree", "out_irtree")

    def __init__(self, ireqns: list[IREqn], in_irtree: Tree[IRAtom], out_irtree: Tree[IRAtom]):
        assert isinstance(ireqns, list)
        assert all(isinstance(ireqn, IREqn) for ireqn in ireqns)
        assert treelib.all(treelib.map(is_iratom, (in_irtree, out_irtree)))
        self.ireqns = tuple(ireqns)
        self.in_irtree = in_irtree
        self.out_irtree = out_irtree

    def __repr__(self) -> str:
        return generate_text_code(ir=self, expand_ir=True)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return call(self)(*args, **kwargs)

    async def acall(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await acall(self)(*args, **kwargs)


def generate_text_code(ir: IR, indent: int = 2, *, expand_ir: bool = False) -> str:
    assert isinstance(indent, int) and indent >= 0
    sp = " " * indent

    def format_atom(atom: IRAtom) -> str:
        assert isinstance(atom, IRAtom)
        if is_irvar(atom):
            var_type = type(atom).__name__
            type_info = f"[{atom.type.__name__}]" if atom.type is not None else ""
            return f"%{atom.id}:{var_type}{type_info}"
        assert is_irlit(atom)
        val = atom.value
        if isinstance(val, IR):
            if expand_ir:
                sub_code = generate_text_code(val, indent, expand_ir=True)
                return f"<IR:{{\n{sub_code}\n}}>"
            else:
                prim_names = ",".join(e.prim.name for e in val.ireqns)
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
        return ", ".join(format_atom(leaf) for leaf in leaves) if leaves else "()"

    in_sig = format_tree(ir.in_irtree)
    out_sig = format_tree(ir.out_irtree)

    header = f"func({in_sig}) -> ({out_sig}) {{"
    lines = [header]

    for ireqn in ir.ireqns:
        lhs = format_tree(ireqn.out_irtree)
        rhs = format_tree(ireqn.in_irtree)
        params_str = ", ".join(f"{k}={v!r}" for k, v in (ireqn.params or {}).items())
        effect_str = f" @{ireqn.effect!r}" if ireqn.effect else ""
        if params_str:
            lines.append(f"{sp}({lhs}) = {ireqn.prim.name}({rhs}, {params_str}){effect_str}")
        else:
            lines.append(f"{sp}({lhs}) = {ireqn.prim.name}({rhs}){effect_str}")

    lines.append("}")
    return "\n".join(lines)


# ==================================================================================================
# INTERPRETER
# ==================================================================================================


class Interpreter(ABC):
    @abstractmethod
    def interpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Any: ...

    @abstractmethod
    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Any: ...


class EvalInterpreter(Interpreter):
    def interpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree:
        return impl_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree:
        return await impl_rules.aget(prim)(in_tree, **params)


active_interpreter = ContextVar[Interpreter]("active_interpreter", default=EvalInterpreter())


@contextmanager
def using_interpreter[T: Interpreter](interpreter: T) -> Generator[T, None, None]:
    token = active_interpreter.set(interpreter)
    try:
        yield interpreter
    finally:
        active_interpreter.reset(token)


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
    # handler signature mirrors interpret: handler(effect, prim, in_tree, /, **params)
    #
    # skip (replace value, no continuation)
    # >>> def handler(effect, prim, in_tree, /):
    # ...     return replacement
    # ...     yield
    #
    # pass-through (observe only)
    # >>> def handler(effect, prim, in_tree, /):
    # ...     return (yield in_tree)
    #
    # pre-process input
    # >>> def handler(effect, prim, in_tree, /):
    # ...     return (yield transform(in_tree))
    #
    # post-process output
    # >>> def handler(effect, prim, in_tree, /, **params):
    # ...     result = yield in_tree
    # ...     return transform(result)
    def __init__(self, *handlers: tuple[type[Effect], Handler]):
        for handler in handlers:
            msg = "handlers must be (EffectType, handler) pairs"
            assert isinstance(handler, Sequence) and len(handler) == 2, msg
            eff_type, _ = handler
            assert issubclass(eff_type, Effect), f"Invalid effect type: {eff_type}"
        self.parent = active_interpreter.get()
        self.handlers = dict(handlers)

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree:
        effect = active_effect.get()
        handler = self.handlers.get(type(effect)) if effect else None

        if handler is None:
            return self.parent.interpret(prim, in_tree, **params)

        gen = handler(effect, prim, in_tree, **params)
        result = None

        # NOTE(asem):
        # the handler can yield multiple times, each yield invokes the continuation.
        # >>> def handler(effect, prim, in_tree, /, **params):
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

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree:
        effect = active_effect.get()
        handler = self.handlers.get(type(effect)) if effect else None

        if handler is None:
            return await self.parent.ainterpret(prim, in_tree, **params)

        gen = handler(effect, prim, in_tree, **params)
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
        self.ireqns: list[IREqn] = []

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree[IRAtom]:
        def to_in_iratom(x) -> IRAtom:
            # NOTE(asem): input can be IRAtom in 3 cases:
            # 1. function inputs wrapped by trace
            # 2. output from previous prim calls
            # 3. raw python values needed to be wrapped by IRLit (local consts, closed-over values)
            return x if is_iratom(x) else IRLit(x)

        in_irtree = treelib.map(to_in_iratom, in_tree)
        in_evaltree = treelib.map(iratom_to_evaltype, in_irtree)
        out_evaltree = eval_rules.get(prim)(in_evaltree, **params)

        def to_out_iratom(x) -> IRAtom:
            # NOTE(asem): eval rules return `Var`/ python types.
            # `Var` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return IRVar.fresh(type=x.type) if is_var(x) else IRLit(x)

        out_irtree = treelib.map(to_out_iratom, out_evaltree)
        effect = active_effect.get()
        self.ireqns.append(IREqn(prim, in_irtree, out_irtree, effect, params))
        return out_irtree

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree[IRAtom]:
        return self.interpret(prim, in_tree, **params)


def trace[**P, R](func: Callable[P, R], /) -> Callable[P, IR[P, R]]:
    """Build an IR from a sync function by tracing its execution.

    Args:
        func: A callable that uses autoform primitives (format, concat, lm_call, etc.).

    Returns:
        A tracer callable that takes ``(*args, **kwargs)`` and returns an IR.

    Example:
        >>> import autoform as af
        >>> def greet(name, punctuation):
        ...     return af.format("Hello, {}{}!", name, punctuation)
        >>> ir = af.trace(greet)("World", "?")
        >>> af.call(ir)("Alice", "!")
        'Hello, Alice!!'
    """

    def assert_usertype(x):
        assert not is_iratom(x), "Inputs to `trace` must be normal python types"

    def to_in_iratom(x):
        return IRVar.fresh(type=type(x)) if is_user_type(x) else IRLit(x)

    def to_out_iratom(x):
        return x if is_iratom(x) else IRLit(x)

    @ft.wraps(func)
    def trace(*args: P.args, **kwargs: P.kwargs) -> IR[P, R]:
        treelib.map(assert_usertype, (args, kwargs), is_leaf=is_user_type)
        in_irtree = treelib.map(to_in_iratom, (args, kwargs), is_leaf=is_user_type)
        in_irargs, in_irkwargs = in_irtree
        with using_interpreter(TracingInterpreter()) as tracer:
            out_irtree = func(*in_irargs, **in_irkwargs)
        in_irtree = pack_user_input(*in_irargs, **in_irkwargs)
        out_irtree = treelib.map(to_out_iratom, out_irtree)
        return IR(ireqns=tracer.ireqns, in_irtree=in_irtree, out_irtree=out_irtree)

    return trace


def atrace[**P, R](func: Callable[P, Awaitable[R]], /) -> Callable[P, Awaitable[IR[P, R]]]:
    """Build an IR from an async function by tracing its execution.

    Args:
        func: An async callable that uses autoform primitives (format, concat, lm_call, etc.).

    Returns:
        A tracer callable that takes ``(*args, **kwargs)`` and returns a coroutine yielding an IR.

    Example:
        >>> import autoform as af
        >>> import asyncio
        >>> async def greet(name, punctuation):
        ...     return af.format("Hello, {}{}!", name, punctuation)
        >>> ir = asyncio.run(af.atrace(greet)("World", "?"))
        >>> af.call(ir)("Alice", "!")
        'Hello, Alice!!'
    """

    def assert_usertype(x):
        assert not is_iratom(x), "Inputs to `trace` must be normal python types"

    def to_in_iratom(x):
        return IRVar.fresh(type=type(x)) if is_user_type(x) else IRLit(x)

    def to_out_iratom(x):
        return x if is_iratom(x) else IRLit(x)

    @ft.wraps(func)
    async def trace(*args: P.args, **kwargs: P.kwargs) -> IR[P, R]:
        treelib.map(assert_usertype, (args, kwargs), is_leaf=is_user_type)
        in_irtree = treelib.map(to_in_iratom, (args, kwargs), is_leaf=is_user_type)
        in_irargs, in_irkwargs = in_irtree
        with using_interpreter(TracingInterpreter()) as tracer:
            out_irtree = await func(*in_irargs, **in_irkwargs)
        in_irtree = pack_user_input(*in_irargs, **in_irkwargs)
        out_irtree = treelib.map(to_out_iratom, out_irtree)
        return IR(ireqns=tracer.ireqns, in_irtree=in_irtree, out_irtree=out_irtree)

    return trace


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
        in_tree = pack_user_input(*args, **kwargs)
        env: dict[IRVar, Value] = {}

        def read(atom: IRAtom) -> Value:
            return env[atom] if is_irvar(atom) else cast(IRLit, atom).value

        def write(atom: IRAtom, value):
            is_irvar(atom) and setitem(env, atom, value)

        treelib.map(write, ir.in_irtree, in_tree)
        for ireqn in ir.ireqns:
            in_values = treelib.map(read, ireqn.in_irtree)
            out_values = ireqn.bind(in_values, **ireqn.params)
            treelib.map(write, ireqn.out_irtree, out_values)
        return treelib.map(read, ir.out_irtree)

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
        in_tree = pack_user_input(*args, **kwargs)
        env: dict[IRVar, Value] = {}

        def read(atom: IRAtom) -> Value:
            return env[atom] if is_irvar(atom) else cast(IRLit, atom).value

        def write(atom: IRAtom, value):
            is_irvar(atom) and setitem(env, atom, value)

        treelib.map(write, ir.in_irtree, in_tree)
        for ireqn in ir.ireqns:
            in_values = treelib.map(read, ireqn.in_irtree)
            out_values = await ireqn.abind(in_values, **ireqn.params)
            treelib.map(write, ireqn.out_irtree, out_values)
        return treelib.map(read, ir.out_irtree)

    return func


# ==================================================================================================
# RULES
# ==================================================================================================


class InterpreterRuleMapping[R]:
    def __init__(self):
        self.map: dict[Primitive, Callable[..., R]] = {}
        self.amap: dict[Primitive, Callable[..., Awaitable[R]]] = {}
        self.lock = RLock()

    def set(self, prim: Primitive, rule: Callable[..., R], /) -> Callable[..., R]:
        assert isinstance(prim, Primitive), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert prim not in self.map, f"Rule for primitive {prim} already defined"

        with self.lock:
            self.map[prim] = rule
        return rule

    def aset(
        self, prim: Primitive, rule: Callable[..., Awaitable[R]], /
    ) -> Callable[..., Awaitable[R]]:
        assert isinstance(prim, Primitive), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert prim not in self.amap, f"Async rule for primitive {prim} already defined"

        with self.lock:
            self.amap[prim] = rule
        return rule

    def get(self, prim: Primitive) -> Callable[..., R]:
        with self.lock:
            if prim not in self.map:
                raise KeyError(f"No {type(self).__name__} rule defined for primitive {prim}")
            return self.map[prim]

    def aget(self, prim: Primitive) -> Callable[..., Awaitable[R]]:
        with self.lock:
            if prim not in self.amap:
                raise KeyError(f"No async {type(self).__name__} rule defined for primitive {prim}")
            return self.amap[prim]


impl_rules = InterpreterRuleMapping[Tree]()
batch_rules = InterpreterRuleMapping[tuple[Tree, Tree[bool] | bool]]()
push_rules = InterpreterRuleMapping[tuple[Tree, Tree]]()
pull_fwd_rules = InterpreterRuleMapping[tuple[Tree, Tree]]()
pull_bwd_rules = InterpreterRuleMapping[Tree]()
eval_rules = InterpreterRuleMapping[Tree[EvalType]]()
