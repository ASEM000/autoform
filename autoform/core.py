"""IR data structures, primitives, interpreters, and IR building"""

from __future__ import annotations

import itertools as it
import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

from autoform.utils import Tree, pack_user_input, treelib

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


class IRAtom: ...


class IRVar(IRAtom):
    counter = it.count(0)
    lock = RLock()

    def __init__(self, id: int, meta: dict | None = None):
        self.id = id
        self.meta = meta or {}

    @classmethod
    def fresh(cls, **meta) -> tp.Self:
        with cls.lock:
            return cls(next(cls.counter), meta=meta or None)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id})"


def is_irvar(x) -> tp.TypeIs[IRVar]:
    return isinstance(x, IRVar)


def is_iratom(x) -> tp.TypeIs[IRAtom]:
    return isinstance(x, IRAtom)


class IRLit[T](IRAtom):
    def __init__(self, value: T, /, **meta):
        assert not is_iratom(value)
        assert hash(value) is not None  # NOTE(asem): for CSE
        self.value = value
        self.meta = meta

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __hash__(self) -> int:
        return hash((self.value, frozenset(self.meta.items()) if self.meta else None))


class IRZero[T](IRLit[T]): ...


def is_irlit(x) -> tp.TypeIs[IRLit]:
    return isinstance(x, IRLit)


# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


class Primitive:
    __slots__ = ("name", "tag")
    __match_args__ = ("name", "tag")

    def __init__(self, name: str, tag: tp.Hashable | None = None):
        self.name = name
        self.tag = tag

    def __repr__(self) -> str:
        return self.name

    def bind(self, in_tree: Tree, **params):
        return get_interp().process(self, in_tree, **params)


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
            assert prim in self.map, f"No rule found for primitive {prim}"
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

    def __setitem__(self, key, value):
        raise TypeError("IREqn is immutable")

    def using(self, **kwargs) -> IREqn:
        """Return new IREqn with kwargs merged into params."""
        return IREqn(self.prim, self.in_irtree, self.out_irtree, self.params | kwargs)


class IR:
    __slots__ = ("ireqns", "in_irtree", "out_irtree")
    __match_args__ = ("ireqns", "in_irtree", "out_irtree")

    def __init__(
        self,
        ireqns: list[IREqn],
        in_irtree: Tree[IRAtom],
        out_irtree: Tree[IRAtom],
    ):
        self.ireqns = ireqns
        self.in_irtree = in_irtree
        self.out_irtree = out_irtree

    def __setitem__(self, key, value):
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
    def process(self, prim: Primitive, in_tree: Tree, **params) -> tp.Any: ...


class EvalInterpreter(Interpreter):
    def process(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        return impl_rules[prim](in_tree, **params)


# ==================================================================================================
# INTERPRETER CONTEXT
# ==================================================================================================

active_interpreter = ContextVar[Interpreter]("active_interpreter", default=EvalInterpreter())


@contextmanager
def using_interp(interpreter: Interpreter):
    token = active_interpreter.set(interpreter)
    try:
        yield interpreter
    finally:
        active_interpreter.reset(token)


def get_interp() -> Interpreter:
    return active_interpreter.get()


# ==================================================================================================
# TRACING INTERPRETER
# ==================================================================================================


class TracingInterpreter(Interpreter):
    def __init__(self):
        self.ireqns: list[IREqn] = []

    def process(self, prim: Primitive, in_tree: Tree, **params) -> list[IRVar]:
        def to_ir_atom(x):
            return x if is_iratom(x) else IRLit(x)

        in_irtree = treelib.map(to_ir_atom, in_tree)
        assert prim in eval_rules, f"Primitive {prim.name} has no `eval_rule` defined"

        def to_eval(x):
            return Var() if is_irvar(x) else x.value

        in_eval_tree = treelib.map(to_eval, in_irtree)
        out_tree = eval_rules[prim](in_eval_tree, **params)

        def from_eval(x):
            return IRVar.fresh() if is_var(x) else IRLit(x)

        out_irtree = treelib.map(from_eval, out_tree, is_leaf=is_var)
        self.ireqns.append(IREqn(prim, in_irtree, out_irtree, params))
        return out_irtree


# ==================================================================================================
# IR BUILDING
# ==================================================================================================


def build_ir(func: Callable[..., Tree], *args, **kwargs) -> IR:
    """Build an IR from a function by tracing its execution.

    Traces `func` with example inputs to capture the computation graph as an
    intermediate representation (IR).

    Args:
        func: A callable that uses autoform primitives (format, concat, lm_call, etc.).
        *args: Example positional arguments to trace with.
        **kwargs: Example keyword arguments to trace with.

    Returns:
        An IR representing the traced computation graph.

    Example:
        >>> import autoform as af
        >>> def greet(name, punctuation):
        ...     return af.format("Hello, {}{}!", name, punctuation)
        >>> ir = af.build_ir(greet, "World", "?")
        >>> af.run_ir(ir, "Alice", "!")
        'Hello, Alice!!'
    """

    def assert_no_iratom(x):
        assert not is_iratom(x)
        return x

    treelib.map(assert_no_iratom, (args, kwargs), is_leaf=is_user_type)

    def populate(x):
        return IRVar.fresh() if is_user_type(x) else IRLit(x)

    in_ir_args, in_ir_kwargs = treelib.map(populate, (args, kwargs), is_leaf=is_user_type)

    def assert_ir(x):
        assert is_iratom(x)
        return x

    with using_interp(TracingInterpreter()) as tracer:
        out_irtree = func(*in_ir_args, **in_ir_kwargs)
        in_irtree = pack_user_input(*in_ir_args, **in_ir_kwargs)
        out_irtree = treelib.map(assert_ir, out_irtree)
        return IR(ireqns=tracer.ireqns, in_irtree=in_irtree, out_irtree=out_irtree)
