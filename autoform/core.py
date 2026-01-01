"""IR data structures, primitives, interpreters, and IR building"""

from __future__ import annotations

import functools as ft
import itertools as it
import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from contextvars import ContextVar
from operator import setitem
from threading import RLock

from autoform.utils import Tree, lru_cache, pack_user_input, treelib

__all__ = [
    # base types
    "Var",
    "user_types",
    # ir atoms
    "IRAtom",
    "IRVar",
    "IRLit",
    "IRZero",
    "is_irvar",
    "is_irlit",
    "is_iratom",
    "iratom_to_evaltype",
    # primitive
    "Primitive",
    # rule registries
    "impl_rules",
    "eval_rules",
    "batch_rules",
    "push_rules",
    "pull_fwd_rules",
    "pull_bwd_rules",
    "iter_rules",
    "async_rules",
    "dce_rules",
    # default rules
    "default_impl",
    "default_eval",
    "default_push",
    "default_pull_fwd",
    "default_pull_bwd",
    "default_batch",
    "default_dce",
    # ir structures
    "IREqn",
    "IR",
    # interpreters
    "Interpreter",
    "EvalInterpreter",
    "TracingInterpreter",
    "get_interpreter",
    "using_interpreter",
    # ir building and execution
    "build_ir",
    "call",
    "icall",
    "acall",
]

# ==================================================================================================
# BASE TYPES
# ==================================================================================================

user_types: set[type] = {str}

type UserType = str
type Value = str


def is_user_type(x) -> bool:
    return isinstance(x, tuple(user_types))


class Var: ...


def is_var(x) -> tp.TypeIs[Var]:
    return isinstance(x, Var)


type EvalType = Var | UserType


# ==================================================================================================
# IR ATOMS
# ==================================================================================================


# NOTE(asem): atomic IR nodes either variables (placeholders) for user inputs
# or literals (constants) baked in the IR
class IRAtom: ...


class IRVar(IRAtom):
    __slots__ = ("id", "source")
    # NOTE(asem): global counter for all IRVars (and neighbors)
    counter: tp.ClassVar[it.count[int]] = it.count(0)
    lock: tp.ClassVar[RLock] = RLock()

    def __init__(self, /, *, source: IRVar | None = None):
        self.id = next(self.counter)
        assert is_irvar(source) or source is None
        self.source = source

    @classmethod
    def fresh(cls, source: IRVar | None = None) -> tp.Self:
        with cls.lock:
            return cls(source=source)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id})"


def is_irvar(x) -> tp.TypeGuard[IRVar]:
    return isinstance(x, IRVar)


def is_iratom(x) -> tp.TypeGuard[IRAtom]:
    return isinstance(x, IRAtom)


def iratom_to_evaltype(x: IRAtom) -> EvalType:
    return Var() if is_irvar(x) else x.value


class IRLit[T](IRAtom):
    def __init__(self, value: T, /, **meta):
        assert not is_iratom(value)
        assert hash(value) is not None
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __hash__(self) -> int:
        return hash(self.value)


class IRZero[T](IRLit[T]): ...


def is_irlit(x) -> tp.TypeGuard[IRLit]:
    return isinstance(x, IRLit)


# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


class Primitive:
    # NOTE(asem): primitive is a key used for matching against rules
    # defined in ``InterpreterRuleMapping``
    __slots__ = ("name", "tag")
    __match_args__ = ("name", "tag")

    def __init__(self, name: str, tag: tp.Hashable | None = None):
        self.name = name
        self.tag = tag

    def __repr__(self) -> str:
        return self.name

    def bind(self, in_tree: Tree, **params):
        return get_interpreter().interpret(self, in_tree, **params)


# ==================================================================================================
# RULE MAPPING
# ==================================================================================================


class InterpreterRuleMapping[T: Callable]:
    def __init__(self):
        self.map: dict[Primitive, T] = {}
        self.lock = RLock()

    def def_rule(self, prim: Primitive, rule: T) -> T:
        assert isinstance(prim, Primitive), f"Expected primitive, got {prim}"
        assert isinstance(rule, Callable), f"Expected callable, got {rule}"
        assert prim not in self.map, f"Rule for primitive {prim} already defined"

        with self.lock:
            self.map[prim] = rule
        return rule

    def __getitem__(self, prim: Primitive) -> T:
        with self.lock:
            if prim not in self.map:
                raise KeyError(f"No {type(self).__name__} rule defined for primitive {prim}")
            return self.map[prim]

    def __iter__(self):
        with self.lock:
            items = list(self.map.items())
        for prim, rule in items:
            yield prim, rule

    def __contains__(self, prim: Primitive) -> bool:
        with self.lock:
            return prim in self.map


# ==================================================================================================
# RULE TYPE ALIASES
# ==================================================================================================

type ImplRule = Callable[..., Tree]
type EvalRule = Callable[..., Tree[EvalType]]
type BatchRule = Callable[..., tuple[Tree, Tree[bool] | bool]]
type PushforwardRule = Callable[..., tuple[Tree, Tree]]
type PullbackFwdRule = Callable[..., tuple[Tree, Tree]]
type PullbackBwdRule = Callable[..., Tree]
type IterRule = Callable[..., tp.Iterator[Tree]]
type AsyncRule = Callable[..., Coroutine[tp.Any, tp.Any, Tree]]
type DCERule = Callable[[IREqn, set[IRVar]], tuple[bool, set[IRVar], IREqn]]

# ==================================================================================================
# RULE REGISTRIES
# ==================================================================================================


def default_impl(x, **_):
    return x


def default_eval(x, **_):
    return x


def default_push(func, primal, tangent, **params):
    p = func(primal, **params)
    t = func(tangent, **params)
    return p, t


def default_pull_fwd(func, x, **params):
    out = func(x, **params)
    return out, None


def default_pull_bwd(func, residuals, cotangent, **params):
    del residuals
    return func(cotangent, **params)


def default_batch(func, batch_size, in_batched, x, **params):
    del batch_size
    return func(x, **params), in_batched


def default_dce(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    out_vars = set(v for v in treelib.leaves(ireqn.out_irtree) if is_irvar(v))
    if out_vars.isdisjoint(active_irvars):
        return True, set(), ireqn  # axe (equation returned but unused)
    in_vars = set(v for v in treelib.leaves(ireqn.in_irtree) if is_irvar(v))
    return False, in_vars, ireqn  # keep (equation unchanged)


impl_rules = InterpreterRuleMapping[ImplRule]()
eval_rules = InterpreterRuleMapping[EvalRule]()
batch_rules = InterpreterRuleMapping[BatchRule]()
push_rules = InterpreterRuleMapping[PushforwardRule]()
pull_fwd_rules = InterpreterRuleMapping[PullbackFwdRule]()
pull_bwd_rules = InterpreterRuleMapping[PullbackBwdRule]()
iter_rules = InterpreterRuleMapping[IterRule]()
async_rules = InterpreterRuleMapping[AsyncRule]()
dce_rules = InterpreterRuleMapping[DCERule]()


# ==================================================================================================
# IR EQUATIONS AND PROGRAMS
# ==================================================================================================


class IREqn:
    __slots__ = ("prim", "in_irtree", "out_irtree", "params")
    __match_args__ = ("prim", "in_irtree", "out_irtree", "params")

    def __init__(
        self,
        prim: Primitive,
        in_irtree: Tree[IRAtom],
        out_irtree: Tree[IRAtom],
        params: dict | None = None,
    ):
        self.params = params if params is not None else {}
        self.prim = prim
        self.in_irtree = in_irtree
        self.out_irtree = out_irtree

    def __setitem__(self, _, __):
        raise TypeError("IREqn is immutable")

    def using(self, **kwargs) -> IREqn:
        return IREqn(self.prim, self.in_irtree, self.out_irtree, self.params | kwargs)


class IR[**P, R]:
    __slots__ = ("ireqns", "in_irtree", "out_irtree")
    __match_args__ = ("ireqns", "in_irtree", "out_irtree")

    def __init__(
        self,
        ireqns: tp.Sequence[IREqn],
        in_irtree: Tree[IRAtom],
        out_irtree: Tree[IRAtom],
    ):
        self.ireqns = tuple(ireqns)
        self.in_irtree = in_irtree
        self.out_irtree = out_irtree

    def __setitem__(self, _, __):
        raise TypeError("IR is immutable")

    def __repr__(self) -> str:
        return generate_text_code(ir=self, expand_ir=True)


def generate_text_code(ir: IR, indent: int = 2, *, expand_ir: bool = False) -> str:
    assert isinstance(indent, int) and indent >= 0
    sp = " " * indent

    def format_atom(atom: IRAtom) -> str:
        assert isinstance(atom, IRAtom)
        if is_irvar(atom):
            var_type = type(atom).__name__
            return f"%{atom.id}:{var_type}"
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
        if params_str:
            lines.append(f"{sp}({lhs}) = {ireqn.prim.name}({rhs}, {params_str})")
        else:
            lines.append(f"{sp}({lhs}) = {ireqn.prim.name}({rhs})")

    lines.append("}")
    return "\n".join(lines)


# ==================================================================================================
# INTERPRETER BASE
# ==================================================================================================


class Interpreter(ABC):
    @abstractmethod
    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> tp.Any: ...


class EvalInterpreter(Interpreter):
    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        return impl_rules[prim](in_tree, **params)


# ==================================================================================================
# INTERPRETER CONTEXT
# ==================================================================================================

active_interpreter = ContextVar[Interpreter]("active_interpreter", default=EvalInterpreter())


@contextmanager
def using_interpreter[T: Interpreter](interpreter: T) -> tp.Generator[T, None, None]:
    token = active_interpreter.set(interpreter)
    try:
        yield interpreter
    finally:
        active_interpreter.reset(token)


def get_interpreter() -> Interpreter:
    return active_interpreter.get()


# ==================================================================================================
# TRACING INTERPRETER
# ==================================================================================================


class TracingInterpreter(Interpreter):
    def __init__(self):
        self.ireqns: list[IREqn] = []

    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree[IRAtom]:
        def to_in_iratom(x) -> IRAtom:
            # NOTE(asem): function inputs are injected with IRVar/IRLit by `build_ir`
            # however, a function can take a constant value as input that is not an input
            # thus we need to wrap it here
            # >>> def f(x):
            # ...     const = "..."
            # ...     return some_user_func(const)
            # here const is not reachable by `build_ir` wrapping mechanism and thus
            # needs to be handled here
            return x if is_iratom(x) else IRLit(x)

        in_irtree = treelib.map(to_in_iratom, in_tree)

        def to_in_evaltype(x):
            # NOTE(asem): eval rules accept `Var`/ python types.
            # `Var` simply denotes a placeholder for a value that will be computed later
            return Var() if is_irvar(x) else x.value

        in_evaltree = treelib.map(to_in_evaltype, in_irtree)
        eval_rule = eval_rules[prim] if prim in eval_rules else eval_rules.fallback
        out_evaltree = eval_rule(in_evaltree, **params)

        def to_out_iratom(x) -> IRAtom:
            # NOTE(asem): eval rules return `Var`/ python types.
            # `Var` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return IRVar.fresh() if is_var(x) else IRLit(x)

        out_irtree = treelib.map(to_out_iratom, out_evaltree)
        self.ireqns.append(IREqn(prim, in_irtree, out_irtree, params))
        return out_irtree


# ==================================================================================================
# IR BUILDING
# ==================================================================================================


def build_ir[**P, R](func: Callable[P, R]) -> Callable[P, IR[P, R]]:
    """Build an IR from a function by tracing its execution.

    Args:
        func: A callable that uses autoform primitives (format, concat, lm_call, etc.).

    Returns:
        A tracer callable that takes ``(*args, **kwargs)`` and returns an IR.

    Example:
        >>> import autoform as af
        >>> def greet(name, punctuation):
        ...     return af.format("Hello, {}{}!", name, punctuation)
        >>> ir = af.build_ir(greet)("World", "?")
        >>> af.call(ir)("Alice", "!")
        'Hello, Alice!!'
    """

    def assert_usertype(x):
        assert not is_iratom(x), "Inputs to `build_ir` must be normal python types"

    def to_in_iratom(x):
        # NOTE(asem): user types are converted to IRVar/IRLit
        # to prepare for tracing.
        return IRVar.fresh() if is_user_type(x) else IRLit(x)

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


# ==================================================================================================
# IR EXECUTION
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def call[**P, R](ir: IR[P, R]) -> tp.Callable[P, R]:
    """Call an IR.

    Args:
        ir: The IR to run.

    Returns:
        A callable that runs the IR with the provided arguments.

    Example:
        >>> import autoform as af
        >>> ir = af.build_ir(lambda x: af.format("Hello {}", x))("world")
        >>> af.call(ir)("Alice")
        'Hello Alice'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        in_tree = pack_user_input(*args, **kwargs)
        env: dict[IRVar, Value] = {}

        def write(atom: IRVar, value):
            is_irvar(atom) and setitem(env, atom, value)

        def read(atom) -> Value:
            return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

        treelib.map(write, ir.in_irtree, in_tree)

        for ireqn in ir.ireqns:
            in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
            out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
            treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
        return treelib.map(read, ir.out_irtree)

    return execute


def accumulate_chunks(chunks: list[tp.Any]) -> tp.Any:
    # TODO(asem): open it up for users
    if not chunks:
        return None
    head = chunks[0]
    if isinstance(head, str):
        return "".join(chunks)
    if isinstance(head, list):
        return list(it.chain.from_iterable(chunks))
    try:
        return ft.reduce(lambda a, b: a + b, chunks)
    except TypeError:
        return chunks


@ft.partial(lru_cache, maxsize=256)
def icall[**P, R](ir: IR[P, R]) -> tp.Callable[P, tp.Iterator[R]]:
    """Create an iterator executor for an IR.

    Args:
        ir: The IR to execute.

    Returns:
        A callable that executes the IR and yields intermediate results.

    Example:
        >>> import autoform as af
        >>> ir = af.build_ir(lambda x: af.format("Hello {}", x))("world")
        >>> list(af.icall(ir)("Alice"))
        ['Hello Alice']
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> tp.Iterator[R]:
        in_tree = pack_user_input(*args, **kwargs)
        env: dict[IRVar, Value] = {}

        def write(atom: IRVar, value: Value):
            is_irvar(atom) and setitem(env, atom, value)

        def read(atom):
            return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

        treelib.map(write, ir.in_irtree, in_tree)

        for ireqn in ir.ireqns:
            in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
            if ireqn.prim in iter_rules:
                iter_rule = iter_rules[ireqn.prim]
                out_treespec = treelib.structure(ireqn.out_irtree)
                acc = [[] for _ in range(out_treespec.num_leaves)]
                for chunk in iter_rule(in_ireqn_tree, **ireqn.params):
                    for i, leaf in enumerate(out_treespec.flatten_up_to(chunk)):
                        acc[i].append(leaf)
                    yield chunk
                out_ireqn_tree = out_treespec.unflatten(map(accumulate_chunks, acc))
            else:
                out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
            treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
        yield treelib.map(read, ir.out_irtree)

    return execute


@ft.partial(lru_cache, maxsize=256)
def acall[**P, R](ir: IR[P, R]) -> tp.Callable[P, tp.Coroutine[tp.Any, tp.Any, R]]:
    """Create an async executor for an IR.

    Args:
        ir: The IR to execute.

    Returns:
        A callable that executes the IR asynchronously.

    Example:
        >>> import autoform as af
        >>> ir = af.build_ir(lambda x: af.format("Hello {}", x))("world")
        >>> import asyncio
        >>> asyncio.run(af.acall(ir)("Alice"))
        'Hello Alice'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    async def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        in_tree = pack_user_input(*args, **kwargs)
        env: dict[IRVar, Value] = {}

        def write(atom: IRVar, value):
            is_irvar(atom) and setitem(env, atom, value)

        def read(atom) -> Value:
            return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

        treelib.map(write, ir.in_irtree, in_tree)

        for ireqn in ir.ireqns:
            in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
            if ireqn.prim in async_rules:
                async_rule = async_rules[ireqn.prim]
                out_ireqn_tree = await async_rule(in_ireqn_tree, **ireqn.params)
            else:
                out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
            treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
        return treelib.map(read, ir.out_irtree)

    return execute
