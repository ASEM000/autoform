from __future__ import annotations

import asyncio
from operator import setitem
from contextvars import ContextVar
import functools as ft
import itertools as it
import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
import optree
from threading import RLock
import optree.pytree
from collections import defaultdict, deque
from litellm import completion, acompletion, batch_completion
import pydantic

# ==================================================================================================
# UTILS
# ==================================================================================================

PYTREE_NAMESPACE = "AUTOFORM"
treelib = optree.pytree.reexport(namespace=PYTREE_NAMESPACE)
type Tree[T] = tp.Any


def lru_cache[**P, R](func: Callable[P, R], maxsize: int = 256) -> Callable[P, R]:
    return tp.cast(Callable[P, R], ft.lru_cache(maxsize=maxsize)(func))


def index(tree: Tree, mask: Tree, i: int) -> Tree:
    # NOTE(asem): indexing a tree according to a mask tree
    # mask is not necessarily same structure as tree but a compatible one.

    # example
    # >>> tree, mask = [[1, 2, 3]], True
    # >>> index(tree, mask, 0)
    # 1
    spec = treelib.structure(mask)
    up_to_tree = spec.flatten_up_to(tree)
    flat_mask = treelib.leaves(mask)

    def select(leaf, take: bool):
        return leaf[i] if take else leaf

    selected = [select(leaf, take) for leaf, take in zip(up_to_tree, flat_mask, strict=True)]
    return spec.unflatten(selected)


def pack_user_input(*args, **kwargs) -> Tree:
    # NOTE(asem): used at the interface of user-bind (i.e. args,kwargs => in_tree, **params)
    if kwargs:
        return (*args, kwargs)
    if len(args) == 1:
        return args[0]
    return args


# ==================================================================================================
# BASE INTERPRETER
# ==================================================================================================


user_types = {str}

type UserType = str


def is_user_type(x) -> bool:
    return isinstance(x, tuple(user_types))


class Var:
    """A symbolic user-facing variable used for `eval_rule` definitions."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "Var"


def is_var(x) -> tp.TypeIs[Var]:
    return isinstance(x, Var)


type EvalType = Var | UserType


class Interpreter(ABC):
    @abstractmethod
    def process(self, prim: Primitive, in_tree: Tree, **params) -> tp.Any:
        # NOTE(asem): divide inputs into traced values (in_tree) and static configuration (params)
        # traced valued can be args, kwargs, or combination of both packed as a tree.
        # the design choice to simplify having to deal with both args and kwargs separately.
        # in every equation execution. instead the args/kwargs are packed once.
        ...


class EvalInterpreter(Interpreter):
    def process(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        return impl_rules.get(prim)(in_tree, **params)


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

type Value = str


# ==================================================================================================
# IR
# ==================================================================================================


class IRAtom: ...


class IRVar(IRAtom):
    def __init__(self, id: int, meta: dict | None = None):
        self.id = id
        self.meta = meta or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id})"


def is_irvar(x) -> tp.TypeIs[IRVar]:
    return isinstance(x, IRVar)


def is_iratom(x) -> tp.TypeIs[IRAtom]:
    return isinstance(x, IRAtom)


class IRPVar(IRVar): ...


class IRTVar(IRVar): ...


class IRCVar(IRVar): ...


class IRBVar(IRVar): ...


class IRVarCounter:
    def __init__(self, start_id: int = 0):
        self.counter = it.count(start_id)

    def __next__(self):
        return next(self.counter)


class IRLit[T: Value](IRAtom):
    def __init__(self, value: T, /, **meta):
        assert not is_iratom(value)
        assert hash(value) is not None  # NOTE(asem): for CSE
        self.value = value
        self.meta = meta

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __hash__(self) -> int:
        return hash((self.value, frozenset(self.meta.items()) if self.meta else None))


class IRZero[T](IRLit[T]):
    def __init__(self, value: T, /, **meta):
        super().__init__(value, **meta)


def is_irlit(x) -> tp.TypeIs[IRLit]:
    return isinstance(x, IRLit)


class IREqn:
    """An equation in the intermediate representation (IR).

    Args:
        prim: The primitive that the equation represents.
        in_irtree: The input to the primitive in the IR.
        out_irtree: The output of the primitive in the IR.
        params: The parameters of the primitive.
    """

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
    """The intermediate representation (IR) of a program.

    Args:
        ireqns: The equations in the IR.
        in_irtree: The input to the IR.
        out_irtree: The output of the IR.
    """

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
                # show summary: <IR:prim_names>
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
# PRIMITIVES
# ==================================================================================================


class Primitive:
    """A primitive operation in the IR.

    Primitives are the building blocks of IR programs. Each primitive has a name
    and an optional tag for categorization. Tags enable pattern matching on IR
    equations by semantic category rather than individual primitive names.

    Args:
        name: Unique identifier for the primitive (e.g., "concat", "lm_call").
        tag: Optional hashable value for categorization (e.g., "string", "lm").

    Example:
        >>> string_concat_p = Primitive("concat", tag="string")
        >>> lm_call_p = Primitive("lm_call", tag="lm")

        Pattern matching on equations by tag:

        >>> from autoform import build_ir, concat, format
        >>> def program(x):
        ...     return concat(format("{}", x), x)
        >>> ir = build_ir(program, "test")
        >>> for ireqn in ir.ireqns:
        ...     match ireqn.prim:
        ...         case Primitive(_, tag="string"):
        ...             print(f"String primitive: {ireqn.prim.name}")
        ...         case Primitive(name, _):
        ...             print(f"Other: {name}")
        String primitive: format
        String primitive: concat

        Filtering equations by tag:

        >>> lm_eqns = [ireqn for ireqn in ir.ireqns if ireqn.prim.tag == "lm"]
    """

    __slots__ = ("name", "tag")
    __match_args__ = ("name", "tag")

    def __init__(self, name: str, tag: tp.Hashable | None = None):
        self.name = name
        self.tag = tag

    def __repr__(self) -> str:
        return self.name

    def bind(self, in_tree: Tree, **params):
        return get_interp().process(self, in_tree, **params)


class InterpreterRuleMapping[T: Callable]:
    def __init__(self):
        self.map: dict[Primitive, T] = {}
        self.lock = RLock()

    def set(self, prim: Primitive, rule: T) -> T:
        assert isinstance(prim, Primitive)
        assert isinstance(rule, Callable)
        assert prim not in self.map

        with self.lock:
            self.map[prim] = rule
        return rule

    def get(self, prim: Primitive) -> T:
        with self.lock:
            return self.map[prim]

    def __iter__(self):
        with self.lock:
            items = list(self.map.items())
        for prim, rule in items:
            yield prim, rule

    def __contains__(self, prim: Primitive) -> bool:
        with self.lock:
            return prim in self.map


type ImplRule = Callable[..., Tree]
type EvalRule = Callable[..., Tree[EvalType]]
type BatchRule = Callable[[int, Tree[bool], Tree], tuple[Tree, Tree[bool]]]
type PushforwardRule = Callable[[Tree, Tree], tuple[Tree, Tree]]
type PullbackFwdRule = Callable[..., tuple[Tree, Tree]]
type PullbackBwdRule = Callable[[Tree, Tree], Tree]
type IterRule = Callable[..., tp.Iterator[Tree]]
type AsyncRule = Callable[..., Coroutine[tp.Any, tp.Any, Tree]]
type DCERule = Callable[[IREqn, set[IRVar]], tuple[bool, set[IRVar], IREqn]]

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
# IR BUILD AND EVALUATION
# ==================================================================================================


class TracingInterpreter(Interpreter):
    def __init__(self, *, counter: IRVarCounter):
        self.ireqns: list[IREqn] = []
        self.counter = counter

    def process(self, prim: Primitive, in_tree: Tree, **params) -> list[IRVar]:
        # NOTE(asem): convert inputs to IR-level atoms
        # example
        # >>> def progam(name: str):
        # >>>   return concat("Hello, ", name)
        # here concat receives ("Hello, ", IRVar(0)) as inputs, where
        # iRVar(0) is created by `build_ir` with the tracing interpreter.
        # but "Hello" must be converted to IRLit("Hello") before passing to concat.
        def to_ir_atom(x):
            return x if is_iratom(x) else IRLit(x)

        in_irtree = treelib.map(to_ir_atom, in_tree)

        assert prim in eval_rules, f"Primitive {prim.name} has no `eval_rule` defined"

        # NOTE(asem): `eval_rule` objective is to get the output structure in terms
        # of `irvar` and `irlit` instances. however, we do not want the user to interact
        # with IR-level constructs inside `eval_rule`, hence we expose `Var` and raw values only.
        # where `Var` -> `IRVar` and raw values -> `IRLit`.
        def to_eval(x):
            return Var() if is_irvar(x) else x.value

        in_eval_tree = treelib.map(to_eval, in_irtree)
        out_tree = eval_rules.get(prim)(in_eval_tree, **params)

        def from_eval(x):
            return IRVar(next(self.counter)) if is_var(x) else IRLit(x)

        out_irtree = treelib.map(from_eval, out_tree, is_leaf=is_var)
        self.ireqns.append(IREqn(prim, in_irtree, out_irtree, params))
        return out_irtree


def build_ir(func: Callable[..., Tree], *args, **kwargs) -> IR:
    """Build an intermediate representation (IR) of the given function.

    Args:
        func: The function to build the IR for.
        *args: Positional arguments to provide as inputs to the function.
        **kwargs: Keyword arguments to provide as inputs to the function.

    Returns:
        An intermediate representation (IR) of the function.

    Example:
        >>> import autoform as af
        >>> def ir(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.build_ir(ir, "Hello, ", "World!")
        >>> print(ir)
        func(%0:IRVar, %1:IRVar) -> (%2:IRVar) {
          (%2:IRVar) = concat(%0:IRVar, %1:IRVar)
        }
    """

    counter = IRVarCounter()

    def assert_no_iratom(x):
        assert not is_iratom(x)
        return x

    treelib.map(assert_no_iratom, (args, kwargs), is_leaf=is_user_type)

    def assert_ir(x):
        assert is_iratom(x)
        return x

    def populate(x):
        return IRVar(next(counter)) if is_user_type(x) else IRLit(x)

    in_ir_args, in_ir_kwargs = treelib.map(populate, (args, kwargs), is_leaf=is_user_type)

    with using_interp(TracingInterpreter(counter=counter)) as tracer:
        out_irtree = func(*in_ir_args, **in_ir_kwargs)
        in_irtree = pack_user_input(*in_ir_args, **in_ir_kwargs)
        out_irtree = treelib.map(assert_ir, out_irtree)
        return IR(ireqns=tracer.ireqns, in_irtree=in_irtree, out_irtree=out_irtree)


def default_dce(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    out_vars = set(x for x in treelib.leaves(ireqn.out_irtree) if is_irvar(x))
    if out_vars.isdisjoint(active_irvars):
        return True, set(), ireqn  # axe (equation returned but unused)
    in_vars = set(x for x in treelib.leaves(ireqn.in_irtree) if is_irvar(x))
    return False, in_vars, ireqn  # (equation unchanged)


def dce_ir(ir: IR) -> IR:
    """Remove code paths that are not executed."""
    # NOTE(asem): simple axe/no axe is used for now.
    # in futrue maybe we can do partial elimination.
    active_irvars: set[IRVar] = set(x for x in treelib.leaves(ir.out_irtree) if is_irvar(x))
    active_ireqns: deque[IREqn] = deque()

    for ireqn in reversed(ir.ireqns):
        dce_rule = dce_rules.get(ireqn.prim) if ireqn.prim in dce_rules else default_dce
        can_axe, cur_active, new_eqn = dce_rule(ireqn, active_irvars)

        if not can_axe:
            active_ireqns.appendleft(new_eqn)
            active_irvars |= cur_active

    return IR(list(active_ireqns), in_irtree=ir.in_irtree, out_irtree=ir.out_irtree)


def fold_ir(ir: IR) -> IR:
    """Evaluate constant IR subexpressions.

    Args:
        ir: The intermediate representation to fold.

    Returns:
        The folded intermediate representation.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     constant = af.format("{}, {}", "hello", "world")
        ...     return af.concat(x, constant)
        >>> ir = af.build_ir(program, "input")
        >>> print(ir)
        func(%0:IRVar) -> (%2:IRVar) {
          (%1:IRVar) = format('hello':Lit, 'world':Lit, template='{}, {}')
          (%2:IRVar) = concat(%0:IRVar, %1:IRVar)
        }
        >>> folded = af.fold_ir(ir)
        >>> print(folded)
        func(%0:IRVar) -> (%2:IRVar) {
          (%2:IRVar) = concat(%0:IRVar, 'hello, world':Lit)
        }
    """

    # TODO(asem): fold nested IRs

    def is_const_irtree(irtree: Tree[IRAtom]) -> bool:
        leaves = treelib.leaves(irtree)
        return all(isinstance(leaf, IRLit) for leaf in leaves)

    def run_const_eqn(ireqn: IREqn, in_irtree: Tree[IRAtom]):
        in_ireqn_tree = treelib.map(lambda x: x.value, in_irtree)
        out_ireqn_tree = impl_rules.get(ireqn.prim)(in_ireqn_tree, **ireqn.params)
        return treelib.map(IRLit, out_ireqn_tree)

    env: dict[IRVar, IRVar | IRLit] = {}
    eqns = []

    def write(atom: IRAtom, value):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom: IRAtom):
        return env[atom] if is_irvar(atom) else atom

    treelib.map(write, ir.in_irtree, ir.in_irtree)

    for ireqn in ir.ireqns:
        # NOTE(asem): read the input IR tree might have folded values
        in_irtree = treelib.map(read, ireqn.in_irtree)
        if is_const_irtree(in_irtree):
            out_irtree = run_const_eqn(ireqn, in_irtree)
            treelib.map(write, ireqn.out_irtree, out_irtree)
        else:
            treelib.map(write, ireqn.out_irtree, ireqn.out_irtree)
            # NOTE(asem): use the read in_irtree (might have folded values)
            # example:
            # >>> def program(x):
            # ...     a = af.format("{}, {}", "a", "b")
            # ...     return af.concat(a, x)
            # >>> print(ir)
            # func(%0:IRVar) -> (%2:IRVar) {
            #   (%1:IRVar) = format('a':Lit, 'b':Lit, template='{}, {}')
            #   (%2:IRVar) = concat(%1:IRVar, %0:IRVar)
            # }
            # after folding, %1 is replaced with IRLit("a, b") in the concat equation:
            # >>> print(folded)
            # func(%0:IRVar) -> (%2:IRVar) {
            #   (%2:IRVar) = concat('a, b':Lit, %0:IRVar)
            # }
            eqns.append(IREqn(ireqn.prim, in_irtree, ireqn.out_irtree, ireqn.params))

    out_irtree = treelib.map(read, ir.out_irtree)

    return IR(eqns, ir.in_irtree, out_irtree)


def run_ir(ir: IR, *args, **kwargs) -> Tree:
    """Run the given intermediate representation (IR) with the provided inputs.

    Args:
        ir: The intermediate representation to run.
        *args: Positional arguments to provide as inputs to the IR.
        **kwargs: Keyword arguments to provide as inputs to the IR.

    Returns:
        The output tree produced by running the IR.

    Example:
        >>> import autoform as af
        >>> def ir(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.build_ir(ir, "Hello, ", "World!")
        >>> af.run_ir(ir, "Hello, ", "World!")
        'Hello, World!'
    """
    assert isinstance(ir, IR), f"{type(ir)=} is not an IR instance."

    # NOTE(asem): traverse the IR using the active interpreter (via bind)
    # in case of EvalInterpreter:
    # >>> def ir(x): return af.concat("Hello, ", x)
    # >>> ir = af.build_ir(ir, "World")
    # >>> af.run_ir(ir, "World")
    # this simply executes the ir.
    # whereas, in case of TracingInterpreter:
    # >>> def ir(x):
    # ...    def inner_ir(y):
    # ...        return af.concat("Hello, ", y)
    # ...    inner_ir = af.build_ir(inner_ir, "world")
    # ...    return af.run_ir(inner_ir, x)
    # >>> outer_ir = af.build_ir(ir, "test")
    # the inner `run_ir` with the inner IR will traverse the inner IR equations
    # and record them into the outer IR (via bind).

    # NOTE(asem): use pack_user_input for user-facing API,
    # however, inside the IR, the signature (in_tree, **params) is used.
    # where in_tree is combination of args and kwargs that are to be traced.

    in_tree = pack_user_input(*args, **kwargs)
    env: dict[IRVar, Value] = {}

    def write(atom: IRVar, value):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom: IRAtom) -> Value:
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    treelib.map(write, ir.in_irtree, in_tree)

    for ireqn in ir.ireqns:
        in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
        out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
        treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
    return treelib.map(read, ir.out_irtree)


def accumulate_chunks(chunks: list[tp.Any]) -> tp.Any:
    # TODO(asem): open it for user rules.
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


def iter_ir(ir: IR, *args, **kwargs):
    in_tree = pack_user_input(*args, **kwargs)
    env: dict[IRVar, Value] = {}

    def write(atom: IRVar, value: Value):
        is_irvar(atom) and setitem(env, atom, value)

    treelib.map(write, ir.in_irtree, in_tree)

    def read(atom: IRAtom):
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    for ireqn in ir.ireqns:
        in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
        if ireqn.prim in iter_rules:
            iter_rule = iter_rules.get(ireqn.prim)
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


async def arun_ir(ir: IR, *args, **kwargs):
    in_tree = pack_user_input(*args, **kwargs)
    env: dict[IRVar, Value] = {}

    def write(atom: IRVar, value):
        is_irvar(atom) and setitem(env, atom, value)

    treelib.map(write, ir.in_irtree, in_tree)

    def read(atom: IRAtom) -> Value:
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    for ireqn in ir.ireqns:
        in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
        if ireqn.prim in async_rules:
            async_rule = async_rules.get(ireqn.prim)
            out_ireqn_tree = await async_rule(in_ireqn_tree, **ireqn.params)
        else:
            out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
        treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
    return treelib.map(read, ir.out_irtree)


# ==================================================================================================
# HIGHER-ORDER PRIMITIVES
# ==================================================================================================

# higher-order primitives denotes primitives that operates on IRs

# PUSHFORWARD CALL =================================================================================

pushforward_call_p = Primitive("pushforward_call", tag="transformation")


class PushforwardInterpreter(Interpreter):
    def __init__(self):
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params):
        in_primal, in_tangent = in_tree
        with using_interp(self.parent):
            return push_rules.get(prim)(in_primal, in_tangent, **params)


@ft.partial(impl_rules.set, pushforward_call_p)
def impl_pushforward_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, t_in_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    t_env: dict[IRVar, Value] = {}

    def write_p(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def write_t(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(t_env, atom, value)

    def read_p(atom: IRAtom) -> Value:
        return p_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    def read_t(atom: IRAtom) -> Value:
        return t_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    treelib.map(write_p, ir.in_irtree, p_in_tree)
    treelib.map(write_t, ir.in_irtree, t_in_tree)

    with using_interp(PushforwardInterpreter()):
        for ireqn in ir.ireqns:
            p_in_ireqn = treelib.map(read_p, ireqn.in_irtree)
            t_in_ireqn = treelib.map(read_t, ireqn.in_irtree)
            in_tree = (p_in_ireqn, t_in_ireqn)
            p_out_ireqn, t_out_ireqn = ireqn.prim.bind(in_tree, **ireqn.params)
            treelib.map(write_p, ireqn.out_irtree, p_out_ireqn)
            treelib.map(write_t, ireqn.out_irtree, t_out_ireqn)

    p_out_tree = treelib.map(read_p, ir.out_irtree)
    t_out_tree = treelib.map(read_t, ir.out_irtree)
    return p_out_tree, t_out_tree


@ft.partial(eval_rules.set, pushforward_call_p)
def eval_pushforward_call(in_tree: Tree, *, ir: IR) -> tuple[Tree[EvalType], Tree[EvalType]]:
    del ir
    p_tree, t_tree = in_tree
    p_out = treelib.map(lambda _: Var(), p_tree, is_leaf=is_var)
    t_out = treelib.map(lambda _: Var(), t_tree, is_leaf=is_var)
    return p_out, t_out


@ft.partial(push_rules.set, pushforward_call_p)
def pushforward_pushforward_call(primals: Tree, tangents: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    # NOTE(asem): nesting happens on the IR level.
    # here the inner call is pushed forward first via run_ir on the pushforward_ir
    (p_in, t_in), (p_in_t, t_in_t) = primals, tangents
    pf_ir = pushforward_ir(ir)
    p_out = run_ir(pf_ir, (p_in, t_in))
    t_out = run_ir(pf_ir, (p_in_t, t_in_t))
    return p_out, t_out


@ft.partial(pull_fwd_rules.set, pushforward_call_p)
def pullback_fwd_pushforward_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    # NOTE(asem): forward pass for pullback of pushforward_call.
    (p_in, t_in) = in_tree
    pf_ir = pushforward_ir(ir)
    p_out, t_out = run_ir(pf_ir, (p_in, t_in))
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


@ft.partial(pull_bwd_rules.set, pushforward_call_p)
def pullback_bwd_pushforward_call(residuals: Tree, cotangent_out: Tree, *, ir: IR) -> Tree:
    # NOTE(asem): backward pass for pullback of pushforward_call.
    # here the inner call is pulled back first via run_ir on the pullback_ir
    p_in, t_in = residuals
    c_p_out, c_t_out = cotangent_out
    pb_ir = pullback_ir(ir)
    _, c_p_in = run_ir(pb_ir, (p_in, c_p_out))
    _, c_t_in = run_ir(pb_ir, (t_in, c_t_out))
    return (c_p_in, c_t_in)


@ft.partial(batch_rules.set, pushforward_call_p)
def batch_pushforward_call(
    batch_size: int,
    in_batched: Tree[bool],
    in_tree: Tree,
    *,
    ir: IR,
) -> tuple[Tree, Tree]:

    (p_cols, t_cols), (p_batched, t_batched) = in_tree, in_batched
    index_p = ft.partial(index, p_cols, p_batched)
    index_t = ft.partial(index, t_cols, t_batched)
    pf_ir = pushforward_ir(ir)
    results = [run_ir(pf_ir, (index_p(b), index_t(b))) for b in range(batch_size)]
    out_spec = treelib.structure(pf_ir.out_irtree)
    leaves_bi = [out_spec.flatten_up_to(r) for r in results]
    stacked = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(out_spec.num_leaves)]
    out_cols = out_spec.unflatten(stacked)
    out_batched = treelib.map(lambda _: True, pf_ir.out_irtree)
    return out_cols, out_batched


@ft.partial(dce_rules.set, pushforward_call_p)
def dce_pushforward_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    dced_ir = dce_ir(ireqn.params["ir"])
    new_eqn = ireqn.using(ir=dced_ir)
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn


# PULLBACK CALL ====================================================================================

pullback_call_p = Primitive("pullback_call", tag="transformation")


class PullbackFwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params):
        with using_interp(self.parent):
            return pull_fwd_rules.get(prim)(in_tree, **params)


class PullbackBwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params):
        in_residual, out_cotangent = in_tree
        with using_interp(self.parent):
            return pull_bwd_rules.get(prim)(in_residual, out_cotangent, **params)


zero_cotangents_map: dict[type, Callable[[], tp.Any]] = {}
cotangent_accumulators: dict[type, Callable[[list], tp.Any]] = {}


def register_cotangent_rules(
    typ: type,
    zero: Callable[[], tp.Any],
    accumulator: Callable[[list], tp.Any],
):
    zero_cotangents_map[typ] = zero
    cotangent_accumulators[typ] = accumulator


register_cotangent_rules(str, lambda: "", lambda cs: "".join(cs))


def zero_cotangent(example: tp.Any = None) -> tp.Any:
    if example is not None and (zero_fn := zero_cotangents_map.get(type(example))):
        return zero_fn()
    return ""


def accumulate_cotangents(cotangents: list) -> tp.Any:
    if not cotangents:
        return zero_cotangent()
    if len(cotangents) == 1:
        return cotangents[0]
    first, *_ = cotangents
    for typ, acc in cotangent_accumulators.items():
        if isinstance(first, typ):
            return acc(cotangents)
    return sum(cotangents[1:], cotangents[0])


@ft.partial(impl_rules.set, pullback_call_p)
def impl_pullback_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    # NOTE(asem): residuals-based pullback implementation.
    # forward pass computes outputs AND saves residuals for backward.
    # backward pass uses residuals, eliminating forced replay.
    (p_in_tree, c_out_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    res_env: dict[int, Tree] = {}  # eqn index -> residuals
    c_env: defaultdict[IRVar, list] = defaultdict(list)

    def write_p(atom: IRAtom, value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def read_p(atom: IRAtom):
        return p_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    def write_c(atom: IRAtom, value):
        if is_irvar(atom):
            value = zero_cotangent(value.value) if isinstance(value, IRZero) else value
            c_env[atom].append(value)

    def read_c(atom: IRAtom):
        return accumulate_cotangents(c_env[atom]) if is_irvar(atom) else ""

    treelib.map(write_p, ir.in_irtree, p_in_tree)

    with using_interp(PullbackFwdInterpreter()):
        for i, eqn in enumerate(ir.ireqns):
            p_in_ireqn = treelib.map(read_p, eqn.in_irtree)
            p_out_ireqn, residuals = eqn.prim.bind(p_in_ireqn, **eqn.params)
            res_env[i] = residuals
            treelib.map(write_p, eqn.out_irtree, p_out_ireqn)

    treelib.map(write_c, ir.out_irtree, c_out_tree)

    with using_interp(PullbackBwdInterpreter()):
        for i, eqn in enumerate(reversed(ir.ireqns)):
            idx = len(ir.ireqns) - 1 - i
            residuals = res_env[idx]
            c_out_ireqn = treelib.map(read_c, eqn.out_irtree)
            c_in_ireqn = eqn.prim.bind((residuals, c_out_ireqn), **eqn.params)
            treelib.map(write_c, eqn.in_irtree, c_in_ireqn)

    p_out_tree = treelib.map(read_p, ir.out_irtree)
    c_in_tree = treelib.map(read_c, ir.in_irtree)
    return p_out_tree, c_in_tree


@ft.partial(eval_rules.set, pullback_call_p)
def eval_pullback_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    del ir
    p_in, c_out = in_tree
    p_out = treelib.map(lambda _: Var(), c_out, is_leaf=is_var)
    c_in = treelib.map(lambda _: Var(), p_in, is_leaf=is_var)
    return p_out, c_in


@ft.partial(push_rules.set, pullback_call_p)
def pushforward_pullback_call(primals: Tree, tangents: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out), (t_p_in, t_c_out) = primals, tangents
    pb_ir = pullback_ir(ir)
    p_result = run_ir(pb_ir, (p_in, c_out))
    t_result = run_ir(pb_ir, (t_p_in, t_c_out))
    return p_result, t_result


@ft.partial(pull_fwd_rules.set, pullback_call_p)
def pullback_fwd_pullback_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    # NOTE(asem): forward pass for pullback of pullback_call.
    # save the inputs and outputs as residuals.
    (p_in, c_out) = in_tree
    pb_ir = pullback_ir(ir)
    p_out, c_in = run_ir(pb_ir, (p_in, c_out))
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


@ft.partial(pull_bwd_rules.set, pullback_call_p)
def pullback_bwd_pullback_call(residuals: Tree, cotangent_out: Tree, *, ir: IR) -> Tree:
    # NOTE(asem): backward pass for pullback of pullback_call.
    p_in, c_out, _, _ = residuals
    c_p_out, c_c_in = cotangent_out
    pb_ir = pullback_ir(ir)
    _, c_p_in = run_ir(pb_ir, (p_in, c_p_out))
    _, c_c_out = run_ir(pb_ir, (p_in, c_c_in))
    return (c_p_in, c_c_out)


@ft.partial(batch_rules.set, pullback_call_p)
def batch_pullback_call(size: int, in_batched: Tree, in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_cols, c_out_cols) = in_tree
    (p_batched, c_batched) = in_batched
    p_index = ft.partial(index, p_cols, p_batched)
    c_index = ft.partial(index, c_out_cols, c_batched)
    pb_ir = pullback_ir(ir)
    results = [run_ir(pb_ir, (p_index(b), c_index(b))) for b in range(size)]
    out_spec = treelib.structure(pb_ir.out_irtree)
    leaves_bi = [out_spec.flatten_up_to(r) for r in results]
    stacked = [[leaves_bi[b][i] for b in range(size)] for i in range(out_spec.num_leaves)]
    out_cols = out_spec.unflatten(stacked)
    out_batched = treelib.map(lambda _: True, pb_ir.out_irtree)
    return out_cols, out_batched


@ft.partial(dce_rules.set, pullback_call_p)
def dce_pullback_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    dced_ir = dce_ir(ireqn.params["ir"])
    new_eqn = ireqn.using(ir=dced_ir)
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn


# BATCH CALL =======================================================================================

batch_call_p = Primitive("batch_call", tag="transformation")


def is_axis_spec(x) -> bool:
    # NOTE(asem): axis spec defines how to batch by passing the container type if the leaf is to be
    # batched, otherwise pass `None` to indicate no batching on that leaf.
    # the rational behind passing the type is that fundamentally ff has no registered type of
    # container type, as a container type like a list can be a leaf.
    # another approach is to implement a custom Batch container, however this would require the
    # users to wraptheir inputs with Batch which is not user-friendly.
    return x is None or isinstance(x, type)


def make_is_batch_leaf(in_axes: Tree) -> Callable[[Tree], bool]:
    # NOTE(asem): get all types leaves and create a function to check if a leaf is of any types
    batch_types = tuple(treelib.leaves(in_axes, is_leaf=lambda x: isinstance(x, type)))
    return (lambda x: isinstance(x, batch_types)) if batch_types else (lambda _: False)


def infer_batch_size(tree: Tree, in_axes: Tree) -> int:
    is_batch_leaf = make_is_batch_leaf(in_axes)
    axes_leaves = treelib.leaves(in_axes, is_leaf=is_axis_spec)
    col_leaves = treelib.leaves(tree, is_leaf=is_batch_leaf)
    return next((len(v) for v, a in zip(col_leaves, axes_leaves) if a is not None), 0)


def broadcast_in_axes_prefix(in_axes: Tree, tree: Tree) -> Tree:
    # NOTE(asem): broadcast in_axes spec to match the structure of tree
    # >>> in_axes = (list, None)
    # >>> tree = (["a", "b", "c"], {"x": 1, "y": 2})
    # >>> broadcasted_in_axes = (list, {"x": None, "y": None})
    is_batch_leaf = make_is_batch_leaf(in_axes)
    is_leaf = lambda x: is_axis_spec(x) or is_batch_leaf(x)
    return treelib.broadcast_prefix(in_axes, tree, is_leaf=is_leaf)


def in_axes_to_batch_tree(in_axes: Tree) -> Tree[bool]:
    # NOTE(asem): convert in_axes spec to a tree of booleans indicating which leaves are batched
    return treelib.map(lambda ax: ax is not None, in_axes, is_leaf=is_axis_spec)


def assert_trees(batch_tree: Tree, irtree: Tree, prim_name: str) -> Tree:
    # NOTE(asem): ensure batch_tree (of booleans) matches the structure of irtree exactly
    # convert irtree to a tree of bools for structure comparison
    expected_batch_tree = treelib.map(lambda _: False, irtree)
    is_bool_leaf = lambda x: isinstance(x, bool)
    batch_spec = treelib.structure(batch_tree, is_leaf=is_bool_leaf)
    expected_spec = treelib.structure(expected_batch_tree, is_leaf=is_bool_leaf)
    if batch_spec != expected_spec:
        raise ValueError(
            f"Primitive '{prim_name}' batch_rule returned out_batched with structure {batch_spec}, "
            f"but expected structure {expected_spec} to match output. "
            f"out_batched must match the structure of the output exactly."
        )
    return batch_tree


class BatchInterpreter(Interpreter):
    def __init__(self, *, batch_size: int):
        self.parent = get_interp()
        self.batch_size = batch_size

    def process(self, prim: Primitive, in_tree: Tree, **params):
        batch_size, in_batched, in_values = in_tree
        with using_interp(self.parent):
            return batch_rules.get(prim)(batch_size, in_batched, in_values, **params)


@ft.partial(impl_rules.set, batch_call_p)
def impl_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    col_tree = in_tree
    axes_tree = broadcast_in_axes_prefix(in_axes, col_tree)
    in_batched_tree: Tree[bool] = in_axes_to_batch_tree(axes_tree)
    in_batched_tree = treelib.broadcast_prefix(in_batched_tree, ir.in_irtree)
    batch_size = infer_batch_size(col_tree, in_axes)
    v_env: dict[IRVar, Value | list[Value]] = {}
    axes_tree = broadcast_in_axes_prefix(in_axes, col_tree)
    in_batched_tree: Tree[bool] = in_axes_to_batch_tree(axes_tree)
    in_batched_tree = treelib.broadcast_prefix(in_batched_tree, ir.in_irtree)
    batch_size = infer_batch_size(col_tree, in_axes)

    v_env: dict[IRVar, Value | list[Value]] = {}
    b_env: dict[IRVar, bool] = {}

    def write_v(atom: IRAtom, value: Value | list[Value]):
        is_irvar(atom) and setitem(v_env, atom, value)

    def write_b(atom: IRAtom, is_batched: bool):
        is_irvar(atom) and setitem(b_env, atom, is_batched)

    def read_v(atom: IRAtom) -> Value | list[Value]:
        return v_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    def read_b(atom: IRAtom) -> bool:
        return b_env[atom] if is_irvar(atom) else False

    treelib.map(write_v, ir.in_irtree, col_tree)
    treelib.map(write_b, ir.in_irtree, in_batched_tree)

    with using_interp(BatchInterpreter(batch_size=batch_size)):
        for ireqn in ir.ireqns:
            in_vals = treelib.map(read_v, ireqn.in_irtree)
            in_batched = treelib.map(read_b, ireqn.in_irtree)
            in_tree = (batch_size, in_batched, in_vals)
            out_vals, out_batched = ireqn.prim.bind(in_tree, **ireqn.params)
            treelib.map(write_v, ireqn.out_irtree, out_vals)
            out_batched = assert_trees(out_batched, ireqn.out_irtree, ireqn.prim.name)
            treelib.map(write_b, ireqn.out_irtree, out_batched)

    return treelib.map(read_v, ir.out_irtree)


@ft.partial(eval_rules.set, batch_call_p)
def eval_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    del ir, in_axes
    return treelib.map(lambda _: Var(), in_tree, is_leaf=is_var)


@ft.partial(push_rules.set, batch_call_p)
def pushforward_batch_call(
    primals: Tree,
    tangents: Tree,
    *,
    ir: IR,
    in_axes: Tree,
) -> tuple[Tree, Tree]:
    p_cols, t_cols = primals, tangents
    pf_ir = pushforward_ir(ir)
    batch_pf_ir = batch_ir(pf_ir, in_axes=(in_axes, in_axes))
    return run_ir(batch_pf_ir, (p_cols, t_cols))


@ft.partial(pull_fwd_rules.set, batch_call_p)
def pullback_fwd_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    # NOTE(asem): forward pass for pullback of batch_call.
    col_tree = in_tree
    batched_ir = batch_ir(ir, in_axes=in_axes)
    out_cols = run_ir(batched_ir, col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


@ft.partial(pull_bwd_rules.set, batch_call_p)
def pullback_bwd_batch_call(residuals: Tree, cotangent_out: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    # NOTE(asem): backward pass for pullback of batch_call.
    p_cols, _ = residuals
    c_out_cols = cotangent_out
    pb_ir = pullback_ir(ir)
    batch_pb_ir = batch_ir(pb_ir, in_axes=(in_axes, list))
    _, c_in_cols = run_ir(batch_pb_ir, (p_cols, c_out_cols))
    return c_in_cols


@ft.partial(batch_rules.set, batch_call_p)
def batch_batch_call(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    ir: IR,
    in_axes: Tree,
) -> tuple[Tree, Tree]:
    col_cols = in_tree

    in_axes_tree = broadcast_in_axes_prefix(in_axes, col_cols)
    get_is_leaf = make_is_batch_leaf(in_axes_tree)
    batched_ir = batch_ir(ir, in_axes=in_axes)

    def get(b):
        get_at_b = lambda v, a: v if a is None else v[b]
        return treelib.map(get_at_b, col_cols, in_axes_tree, is_leaf=get_is_leaf)

    results = [run_ir(batched_ir, get(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, ir.out_irtree)
    out_spec = treelib.structure(ir.out_irtree)
    leaves_bi = [out_spec.flatten_up_to(r) for r in results]
    num_leaves = out_spec.num_leaves
    stacked = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(num_leaves)]
    out_cols = out_spec.unflatten(stacked)
    return out_cols, out_batched


@ft.partial(async_rules.set, batch_call_p)
async def async_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    col_tree = in_tree

    axes_tree = broadcast_in_axes_prefix(in_axes, col_tree)
    run_is_leaf = make_is_batch_leaf(axes_tree)
    batch_size = infer_batch_size(col_tree, in_axes)

    async def run_item(b: int):
        def get(v, a):
            return v if a is None else v[b]

        value = treelib.map(get, col_tree, axes_tree, is_leaf=run_is_leaf)
        return await arun_ir(ir, value)

    results = await asyncio.gather(*[run_item(b) for b in range(batch_size)])
    out_spec = treelib.structure(ir.out_irtree)
    leaves_bi = [out_spec.flatten_up_to(r) for r in results]
    stacked = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(out_spec.num_leaves)]
    return out_spec.unflatten(stacked)


@ft.partial(dce_rules.set, batch_call_p)
def dce_batch_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    new_eqn = ireqn.using(ir=dce_ir(ireqn.params["ir"]))
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn


# ==================================================================================================
# IR -> IR TRANSFORMS
# ==================================================================================================

# PUSHFORWARD IR ===================================================================================


@ft.partial(lru_cache, maxsize=256)
def pushforward_ir(ir: IR) -> IR:
    counter = IRVarCounter()

    def make_p(atom: IRAtom):
        return IRPVar(next(counter), dict(source=atom)) if is_irvar(atom) else atom

    def make_t(atom: IRAtom):
        return IRTVar(next(counter), dict(source=atom)) if is_irvar(atom) else IRZero(atom)

    p_in_irtree = treelib.map(make_p, ir.in_irtree)
    t_in_irtree = treelib.map(make_t, ir.in_irtree)
    in_irtree = (p_in_irtree, t_in_irtree)
    p_out_irtree = treelib.map(make_p, ir.out_irtree)
    t_out_irtree = treelib.map(make_t, ir.out_irtree)
    out_irtree = (p_out_irtree, t_out_irtree)
    eqn = IREqn(pushforward_call_p, in_irtree, out_irtree, dict(ir=ir))
    return IR([eqn], in_irtree, out_irtree)


# PULLBACK IR ======================================================================================


@ft.partial(lru_cache, maxsize=256)
def pullback_ir(ir: IR) -> IR:
    counter = IRVarCounter()

    def make_p(atom: IRAtom):
        return IRPVar(next(counter), dict(source=atom)) if is_irvar(atom) else atom

    def make_c(atom: IRAtom):
        return IRCVar(next(counter), dict(source=atom)) if is_irvar(atom) else IRZero(atom)

    p_in = treelib.map(make_p, ir.in_irtree)
    c_out = treelib.map(make_c, ir.out_irtree)
    in_irtree = (p_in, c_out)
    p_out = treelib.map(make_p, ir.out_irtree)
    c_in = treelib.map(make_c, ir.in_irtree)
    out_irtree = (p_out, c_in)
    eqn = IREqn(pullback_call_p, in_irtree, out_irtree, dict(ir=ir))
    return IR([eqn], in_irtree, out_irtree)


# BATCH IR =========================================================================================


@ft.partial(lru_cache, maxsize=256)
def batch_ir(ir: IR, in_axes: Tree[type | None] = list) -> IR:
    counter = IRVarCounter()

    def make_b(atom: IRAtom):
        return IRBVar(next(counter), dict(source=atom)) if is_irvar(atom) else atom

    b_in_irtree = treelib.map(make_b, ir.in_irtree)
    b_out_irtree = treelib.map(make_b, ir.out_irtree)
    eqn = IREqn(batch_call_p, b_in_irtree, b_out_irtree, dict(ir=ir, in_axes=in_axes))
    return IR([eqn], b_in_irtree, b_out_irtree)


# FORMAT ===========================================================================================

format_p = Primitive("format", tag="string")


def format(template: str, *args) -> str:
    return format_p.bind(args, template=template)


@ft.partial(impl_rules.set, format_p)
def impl_format(in_tree: Tree, *, template: str) -> str:
    return template.format(*in_tree)


@ft.partial(eval_rules.set, format_p)
def eval_format(_: Tree, *, template: str) -> Var:
    del template
    return Var()


@ft.partial(pull_fwd_rules.set, format_p)
def pullback_fwd_format(in_tree: Tree, *, template: str) -> tuple[Tree, Tree]:
    out = template.format(*in_tree)
    return out, len(in_tree)


@ft.partial(pull_bwd_rules.set, format_p)
def pullback_bwd_format(residuals: Tree, cotangent_out: Tree, *, template: str) -> Tree:
    del template
    n = residuals
    return tuple([cotangent_out] * n)


@ft.partial(batch_rules.set, format_p)
def batch_format(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    template: str,
) -> tuple[Tree, Tree]:
    args = tuple(in_tree)
    args_batched = tuple(in_batched)

    def get(i, b):
        return args[i][b] if args_batched[i] else args[i]

    result = [template.format(*[get(i, b) for i in range(len(args))]) for b in range(batch_size)]
    return result, True


@ft.partial(push_rules.set, format_p)
def pushforward_format(primals: Tree, tangents: Tree, *, template: str) -> tuple[Tree, Tree]:
    primal_out = format(template, *primals)
    tangent_out = format(template, *tangents)
    return primal_out, tangent_out


# LM CALL ==========================================================================================

lm_call_p = Primitive("lm_call", tag="lm")


def lm_call(messages: list[dict[str, str]], *, model: str) -> str:
    """Calls a language model with the given messages and model name using Litellm.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use (e.g., "gpt-3.5-turbo").

    Returns:
        The content of the model's response as a string.

    Example:
        >>> import autoform as af
        >>> def ir(name: str) -> str:
        ...     greeting = af.format("Hello, {}!", name)
        ...     system_message = dict(role="system", content="translate the greeting to Korean")
        ...     user_message = dict(role="user", content=greeting)
        ...     greeting = af.lm_call([system_message, user_message], model="gpt-3.5-turbo")
        ...     return greeting
        >>> ir = af.build_ir(ir, "World") # doctest: +SKIP
        >>> result = af.run_ir(ir, "Alice") # doctest: +SKIP
    """
    # NOTE(asem): separate roles (static) from contents (traced) at bind time
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m.keys()=}"
        assert "content" in m, f"message must have a 'content' key, got {m.keys()=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return lm_call_p.bind(contents, roles=roles, model=model)


@ft.partial(impl_rules.set, lm_call_p)
def impl_lm_call(contents: tuple, *, roles: tuple[str, ...], model: str) -> str:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = completion(messages=messages, model=model)
    return resp.choices[0].message.content


@ft.partial(eval_rules.set, lm_call_p)
def eval_lm_call(in_tree: Tree, **params) -> Var:
    return Var()


@ft.partial(push_rules.set, lm_call_p)
def pushforward_lm_call(
    primals: tuple, tangents: tuple, *, roles: tuple, model: str
) -> tuple[Tree, Tree]:
    p_messages = [dict(role=r, content=c) for r, c in zip(roles, primals, strict=True)]
    t_messages = [dict(role=r, content=c) for r, c in zip(roles, tangents, strict=True)]
    p_resp = completion(messages=p_messages, model=model)
    t_resp = completion(messages=t_messages, model=model)
    return p_resp.choices[0].message.content, t_resp.choices[0].message.content


@ft.partial(pull_fwd_rules.set, lm_call_p)
def pullback_fwd_lm_call(contents: tuple, *, roles: tuple, model: str) -> tuple[Tree, Tree]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model)
    out = resp.choices[0].message.content
    residuals = (contents, out)  # save for backward pass
    return out, residuals


@ft.partial(pull_bwd_rules.set, lm_call_p)
def pullback_bwd_lm_call(
    residuals: tuple, cotangent_out: Tree, *, roles: tuple, model: str
) -> tuple:
    contents, output = residuals
    grads = []
    for content in contents:
        grad_prompt = f"""Given this LLM interaction:

INPUT: {content}
OUTPUT: {output}
FEEDBACK ON OUTPUT: {cotangent_out}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""
        resp = completion(messages=[dict(role="user", content=grad_prompt)], model=model)
        grads.append(resp.choices[0].message.content)
    return tuple(grads)


@ft.partial(batch_rules.set, lm_call_p)
def batch_lm_call(
    batch_size: int, in_batched: Tree, contents: tuple, *, roles: tuple, model: str
) -> tuple[Tree, Tree]:
    batched_messages = []
    for b in range(batch_size):
        msgs = [
            dict(role=r, content=contents[i][b] if in_batched[i] else contents[i])
            for i, r in enumerate(roles)
        ]
        batched_messages.append(msgs)
    responses = batch_completion(messages=batched_messages, model=model)
    return [resp.choices[0].message.content for resp in responses], True


@ft.partial(iter_rules.set, lm_call_p)
def iter_lm_call(contents: tuple, *, roles: tuple, model: str) -> tp.Iterator[str]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        yield delta


@ft.partial(async_rules.set, lm_call_p)
async def async_lm_call(contents: tuple, *, roles: tuple, model: str) -> str:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = await acompletion(messages=messages, model=model)
    return resp.choices[0].message.content


# STRUCT LM CALL ===================================================================================


class Struct(pydantic.BaseModel):
    """Pydantic BaseModel that is also a PyTree.

    Auto-registers subclasses as pytrees
    Uses ``model_construct`` in unflatten to skip validation.

    Example:
        >>> class Answer(Struct):
        ...     reasoning: str
        ...     answer: int
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def flatten(model):
            values = tuple(getattr(model, k) for k in cls.model_fields)
            return values, tuple(cls.model_fields.keys())

        def unflatten(keys, children):
            return cls.model_construct(**dict(zip(keys, children)))

        treelib.register_node(cls, flatten, unflatten)

    def __hash__(self):
        # NOTE(asem): to use for `in_axes` in batch rules
        return hash(tuple(getattr(self, k) for k in type(self).model_fields))


struct_lm_call_p = Primitive("struct_lm_call", tag="lm")


def struct_lm_call(messages: list[dict[str, str]], *, model: str, struct: type[Struct]) -> str:
    """Calls a language model with structured output using response_format.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use.
        struct: A Pydantic model or type for structured output.

    Returns:
        The structured response as a JSON string.
    """
    assert issubclass(struct, Struct), "struct must be a subclass of ``Struct``"
    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return struct_lm_call_p.bind(contents, roles=roles, model=model, struct=struct)


@ft.partial(impl_rules.set, struct_lm_call_p)
def impl_struct_lm_call(
    contents: tuple,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> Struct:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct)
    return struct.model_validate_json(resp.choices[0].message.content)


@ft.partial(eval_rules.set, struct_lm_call_p)
def eval_struct_lm_call(in_tree: Tree, *, struct: type[Struct], **params) -> Tree:
    return struct.model_construct(**{k: Var() for k in struct.model_fields})


@ft.partial(pull_fwd_rules.set, struct_lm_call_p)
def pullback_fwd_struct_lm_call(
    contents: tuple,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct)
    out = struct.model_validate_json(resp.choices[0].message.content)
    residuals = (contents, out)
    return out, residuals


@ft.partial(pull_bwd_rules.set, struct_lm_call_p)
def pullback_bwd_struct_lm_call(
    residuals: tuple,
    cotangent_out: Tree,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> tuple:
    contents, output = residuals
    grads = []
    for content in contents:
        # TODO(asem): remove this
        grad_prompt = f"""Given this LLM interaction:

INPUT: {content}
OUTPUT: {output}
FEEDBACK ON OUTPUT: {cotangent_out}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""
        resp = completion(messages=[dict(role="user", content=grad_prompt)], model=model)
        grads.append(resp.choices[0].message.content)
    return tuple(grads)


@ft.partial(iter_rules.set, struct_lm_call_p)
def iter_struct_lm_call(
    contents: tuple,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> tp.Iterator[str]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        yield delta


@ft.partial(async_rules.set, struct_lm_call_p)
async def async_struct_lm_call(
    contents: tuple,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> Struct:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = await acompletion(messages=messages, model=model, response_format=struct)
    return resp.choices[0].message.content


# CONCAT ===========================================================================================

concat_p = Primitive("concat", tag="string")


def concat(*args) -> str:
    """Concatenates multiple strings into a single string.

    Args:
        *args: A variable number of string arguments to concatenate.

    Returns:
        A single string that is the concatenation of all input strings.

    Example:
        >>> import autoform as af
        >>> result = af.concat("Hello, ", "world", "!")
        >>> print(result)
        Hello, world!
    """
    return concat_p.bind(args)


@ft.partial(impl_rules.set, concat_p)
def impl_concat(in_tree: Tree) -> str:
    return "".join(in_tree)


@ft.partial(eval_rules.set, concat_p)
def eval_concat(in_tree: Tree, **params):
    del in_tree
    return Var()


@ft.partial(push_rules.set, concat_p)
def pushforward_concat(primals: Tree, tangents: Tree) -> tuple[Tree, Tree]:
    return concat(*primals), concat(*tangents)


@ft.partial(pull_fwd_rules.set, concat_p)
def pullback_fwd_concat(in_tree: Tree) -> tuple[Tree, Tree]:
    out = "".join(in_tree)
    return out, len(in_tree)


@ft.partial(pull_bwd_rules.set, concat_p)
def pullback_bwd_concat(residuals: Tree, cotangent_out: Tree) -> Tree:
    n = residuals
    return tuple([cotangent_out] * n)


@ft.partial(batch_rules.set, concat_p)
def batch_concat(batch_size: int, in_batched: Tree, in_tree: Tree) -> tuple[Tree, Tree]:
    cols = tuple(in_tree)
    batched = tuple(in_batched)
    if batch_size == 0:
        return [], True

    def get(i, b):
        return cols[i][b] if batched[i] else cols[i]

    result = ["".join(get(i, b) for i in range(len(cols))) for b in range(batch_size)]
    return result, True


# STOP GRADIENT ====================================================================================

stop_gradient_p = Primitive("stop_gradient", tag="control")


def stop_gradient(x: Tree) -> Tree:
    """Stops the gradient flow through the input during backpropagation.

    Args:
        x: The input tree (e.g., a string, number, or nested structure)

    Returns:
        The same input tree with gradients stopped.

    Example:
        >>> import autoform as af
        >>> def ir(x, y):
        ...     stopped = af.stop_gradient(x)
        ...     return af.concat(stopped, y)
        >>> ir = af.build_ir(ir, "a", "b")
        >>> pb_ir = af.pullback_ir(ir)
        >>> _, (cotangent_x, cotangent_y) = af.run_ir(pb_ir, (("a", "b"), "grad"))
        >>> cotangent_x
        ''
        >>> cotangent_y
        'grad'
    """
    return stop_gradient_p.bind(x)


@ft.partial(impl_rules.set, stop_gradient_p)
def impl_stop_gradient(x: Tree) -> Tree:
    return x


@ft.partial(eval_rules.set, stop_gradient_p)
def eval_stop_gradient(x: Tree[EvalType]) -> Tree[EvalType]:
    return x


@ft.partial(push_rules.set, stop_gradient_p)
def pushforward_stop_gradient(primal: Tree, tangent: Tree) -> tuple[Tree, Tree]:
    zero_tangent = treelib.map(zero_cotangent, primal)
    return primal, zero_tangent


@ft.partial(pull_fwd_rules.set, stop_gradient_p)
def pullback_fwd_stop_gradient(x: Tree) -> tuple[Tree, Tree]:
    residuals = x  # need to know structure for zeroing
    return x, residuals


@ft.partial(pull_bwd_rules.set, stop_gradient_p)
def pullback_bwd_stop_gradient(residuals: Tree, cotangent_out: Tree) -> Tree:
    del cotangent_out
    return treelib.map(zero_cotangent, residuals)


@ft.partial(batch_rules.set, stop_gradient_p)
def batch_stop_gradient(batch_size: int, in_batched: Tree, x: Tree) -> tuple[Tree, Tree]:
    del batch_size
    return x, in_batched


# MARK =============================================================================================

mark_p = Primitive("mark", tag="core")


def mark(x: Tree, *, tag: tp.Hashable) -> Tree:
    """Identity operation with a tag.

    Args:
        x: The input tree (e.g., a string, number, or nested structure)
        tag: A hashable value to identify this mark in the IR.

    Returns:
        The same input tree, unchanged.

    Example:
        >>> import autoform as af
        >>> def ir(x):
        ...     marked = af.mark(x, tag="my_tag")
        ...     return af.concat("Result: ", marked)
        >>> ir = af.build_ir(ir, "hello")
        >>> af.run_ir(ir, "world")
        'Result: world'
    """
    assert hash(tag) is not None, "Tag must be hashable"
    return mark_p.bind(x, tag=tag)


@ft.partial(impl_rules.set, mark_p)
def impl_mark(x: Tree, *, tag: tp.Hashable) -> Tree:
    del tag
    return x


@ft.partial(eval_rules.set, mark_p)
def eval_mark(x: Tree[EvalType], *, tag: tp.Hashable) -> Tree[EvalType]:
    del tag
    return x


@ft.partial(push_rules.set, mark_p)
def pushforward_mark(primal: Tree, tangent: Tree, *, tag: tp.Hashable) -> tuple[Tree, Tree]:
    del tag
    return primal, tangent


@ft.partial(pull_fwd_rules.set, mark_p)
def pullback_fwd_mark(x: Tree, *, tag: tp.Hashable) -> tuple[Tree, Tree]:
    del tag
    return x, x  # residuals = input for structure


@ft.partial(pull_bwd_rules.set, mark_p)
def pullback_bwd_mark(residuals: Tree, cotangent_out: Tree, *, tag: tp.Hashable) -> Tree:
    del residuals, tag
    return cotangent_out  # identity backward pass


@ft.partial(batch_rules.set, mark_p)
def batch_mark(_: int, in_batched: Tree, x: Tree, *, tag: tp.Hashable) -> tuple[Tree, Tree]:
    del tag
    return x, in_batched


# IR CALL ==========================================================================================

ir_call_p = Primitive("ir_call", tag="call")

# NOTE(asem): IR is a first-class citizen in autoform IRs
# in essence, IR itself can be differentiated, batched, ...
user_types.add(IR)


def ir_call(ir: IR, *args, **kwargs) -> Tree:
    """Call a ir as a differentiable operation.

    Args:
        ir: The IR ir to execute.
        *args: Positional arguments to pass to the ir.
        **kwargs: Keyword arguments to pass to the ir.

    Returns:
        The result of running the ir.
    """
    # NOTE(asem): key idea here is that IR is being traced
    # and not a param. Use pack_user_input for consistent structure.
    return ir_call_p.bind((ir, pack_user_input(*args, **kwargs)))


@ft.partial(impl_rules.set, ir_call_p)
def impl_ir_call(in_tree):
    ir, operands = in_tree
    # NOTE(asem): key idea here is that IR is being traced
    # and not a param
    return run_ir(ir, operands)


@ft.partial(eval_rules.set, ir_call_p)
def eval_ir_call(in_tree, **params):
    ir, operands = in_tree

    # NOTE(asem): when IR itself is being traced, ir is a Var.
    # in this case, we can't introspect the output structure, so return Var.
    if is_var(ir):
        return Var()

    # NOTE(asem): convert IR atoms to eval-level values
    # IRVar -> Var, IRLit -> value (matching TracingInterpreter.to_eval)
    def to_eval(atom):
        return Var() if is_irvar(atom) else atom.value

    return treelib.map(to_eval, ir.out_irtree)


@ft.partial(push_rules.set, ir_call_p)
def pushforward_ir_call(primals, tangents):
    ir, p_operands = primals
    _, t_operands = tangents
    primal_out = run_ir(ir, p_operands)
    tangent_out = run_ir(ir, t_operands)
    return primal_out, tangent_out


@ft.partial(pull_fwd_rules.set, ir_call_p)
def pullback_fwd_ir_call(in_tree):
    ir, operands = in_tree
    out = run_ir(ir, operands)
    residuals = (ir, operands)
    return out, residuals


@ft.partial(pull_bwd_rules.set, ir_call_p)
def pullback_bwd_ir_call(residuals, cotangent_out):
    # NOTE(asem): ir_call differentiates through operands by running
    # the pullback of the inner IR. The IR itself is treated as a constant
    # (zero gradient).
    ir, operands = residuals
    pb_ir = pullback_ir(ir)
    _, c_operands = run_ir(pb_ir, (operands, cotangent_out))
    zero_ir = zero_cotangent(ir)
    return (zero_ir, c_operands)


@ft.partial(batch_rules.set, ir_call_p)
def batch_ir_call(batch_size, in_batched, in_tree):
    irs, operands = in_tree
    prog_batched, operands_batched = in_batched

    results = []
    for b in range(batch_size):
        prog = irs[b] if prog_batched else irs
        batch_operands = index(operands, operands_batched, b)
        results.append(run_ir(prog, batch_operands))

    return results, True


@ft.partial(iter_rules.set, ir_call_p)
def iter_ir_call(in_tree):
    # NOTE(asem): streaming support for ir_call.
    # when `iter_ir` encounters a `ir_call` equation, it uses this `iter_rule`
    # to delegate streaming to the inner ir's `iter_ir`.
    # this enables nested streaming: an outer `iter_ir` can stream through
    # an inner ir that itself contains streaming primitives (e.g., `lm_call`).
    #
    # example:
    # >>> inner_ir = af.build_ir(lambda x: af.lm_call([...], model="..."), "input")
    # >>> def outer(x):
    # ...     return af.ir_call(inner_ir, x)
    # >>> outer_ir = af.build_ir(outer, "input")
    # >>> for chunk in af.iter_ir(outer_ir, "hello"):
    # ...     print(chunk)  # streams tokens from inner lm_call
    #
    # NOTE(asem): iter_ir yields chunks AND the final result.
    # iter_rule should only yield chunks (the final is computed by accumulation).
    # so we yield all but the last item from iter_ir.
    ir, operands = in_tree
    # NOTE(asem): final result is the accumulated output after all chunks
    *chunks, _ = iter_ir(ir, operands)
    for chunk in chunks:
        yield chunk


# SWITCH ===========================================================================================

switch_p = Primitive("switch", tag="control")


def switch(key: str, branches: dict[str, IR], *operands, **kw_operands) -> Tree:
    """Select and execute one of multiple IR branches based on a string key.

    Args:
        key: String key selecting which branch to execute.
        branches: Dict mapping string keys to IR irs, each with compatible input signature.
        *args: Positional arguments passed to the selected branch.
        **kwargs: Keyword arguments passed to the selected branch.

    Returns:
        Result of run_ir(branches[key], *args, **kwargs)

    Raises:
        KeyError: If key is not in branches.

    Example:
        >>> import autoform as af
        >>> branches = {
        ...     "zero": af.build_ir(lambda x: af.concat("zero: ", x), "X"),
        ...     "one": af.build_ir(lambda x: af.concat("one: ", x), "X"),
        ...     "two": af.build_ir(lambda x: af.concat("two: ", x), "X"),
        ... }
        >>> def ir(key, x):
        ...     return af.switch(key, branches, x)
        >>> ir = af.build_ir(ir, "one", "hello")
        >>> af.run_ir(ir, "one", "hello")
        'one: hello'
        >>> af.run_ir(ir, "zero", "hello")
        'zero: hello'
    """
    assert is_user_type(key) or is_iratom(key), "key must be a user-type (traceable) value"
    assert all(isinstance(branches[k], IR) for k in branches)
    tree_struct0 = treelib.structure(branches[next(iter(branches))].in_irtree)
    assert all(treelib.structure(branches[key].in_irtree) == tree_struct0 for key in branches)
    tree_struct0 = treelib.structure(branches[next(iter(branches))].out_irtree)
    assert all(treelib.structure(branches[key].out_irtree) == tree_struct0 for key in branches)
    # NOTE(asem): always use pack_user_input to at the interface of user-bind
    return switch_p.bind((key, pack_user_input(*operands, **kw_operands)), branches=branches)


@ft.partial(impl_rules.set, switch_p)
def impl_switch(in_tree, *, branches: dict[str, IR]):
    key, operands = in_tree
    return run_ir(branches[key], operands)


@ft.partial(eval_rules.set, switch_p)
def eval_switch(in_tree, *, branches: dict[str, IR]) -> Tree[EvalType]:
    # NOTE(asem): all inputs/outputs structure are be the same
    del in_tree
    key0 = next(iter(branches))
    branch0 = branches[key0]
    return treelib.map(lambda atom: Var() if is_irvar(atom) else atom.value, branch0.out_irtree)


@ft.partial(push_rules.set, switch_p)
def pushforward_switch(primals, tangents, *, branches: dict[str, IR]):
    (key, p_operands), (_, t_operands) = primals, tangents
    pf_ir = pushforward_ir(branches[key])
    return run_ir(pf_ir, (p_operands, t_operands))


@ft.partial(pull_fwd_rules.set, switch_p)
def pullback_fwd_switch(in_tree, *, branches: dict[str, IR]) -> tuple[Tree, Tree]:
    key, operands = in_tree
    out = run_ir(branches[key], operands)
    residuals = (key, operands)
    return out, residuals


@ft.partial(pull_bwd_rules.set, switch_p)
def pullback_bwd_switch(residuals, cotangent_out, *, branches: dict[str, IR]):
    key, operands = residuals
    pb_ir = pullback_ir(branches[key])
    _, c_operands = run_ir(pb_ir, (operands, cotangent_out))
    return (zero_cotangent(key), c_operands)


@ft.partial(batch_rules.set, switch_p)
def batch_switch(
    batch_size: int,
    in_batched,
    in_tree,
    *,
    branches: dict[str, IR],
) -> tuple[Tree, bool]:
    key_col, operands_col = in_tree
    key_batched, operands_batched = in_batched
    index_at = ft.partial(index, operands_col, operands_batched)

    def run_ir_at(b):
        return run_ir(branches[key_col[b] if key_batched else key_col], index_at(b))

    return [run_ir_at(b) for b in range(batch_size)], True


@ft.partial(iter_rules.set, switch_p)
def iter_switch(in_tree, *, branches: dict[str, IR]):
    key, operands = in_tree
    # NOTE(asem): final result is the accumulated output after all chunks
    *chunks, _ = iter_ir(branches[key], operands)
    for chunk in chunks:
        yield chunk


@ft.partial(async_rules.set, switch_p)
async def async_switch(in_tree, *, branches: dict[str, IR]) -> Tree:
    key, operands = in_tree
    return await arun_ir(branches[key], operands)


@ft.partial(dce_rules.set, switch_p)
def dce_switch(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    # >>> import autoform as af
    # >>> # branch A has dead code
    # >>> def branch_a_fn(x):
    # ...     dead = af.concat(x, " DEAD")  # unused
    # ...     live = af.concat(x, " LIVE")  # returned
    # ...     return live
    # >>> branch_a = af.build_ir(branch_a_fn, "test")
    # >>> branch_b = af.build_ir(lambda x: af.concat(x, " B"), "test")
    # >>> len(branch_a.ireqns)
    # 2
    # >>> # outer program using switch
    # >>> def program(key, x):
    # ...     return af.switch(key, {"a": branch_a, "b": branch_b}, x)
    # >>> ir = af.build_ir(program, "a", "input")
    # >>> len(ir.ireqns[0].params["branches"]["a"].ireqns)
    # 2
    # >>> # After DCE, nested IR has dead code eliminated
    # >>> dced_ir = af.dce_ir(ir)
    # >>> len(dced_ir.ireqns[0].params["branches"]["a"].ireqns)
    # 1
    # >>> af.run_ir(dced_ir, "a", "hello")
    # 'hello LIVE'

    # NOTE(asem): the key idea here is that dce rules need to return equations
    # to allow for recursive pruning. otherwise, we do not have any
    # mechanism to dce higher order primitives.

    for k in (branches := dict(ireqn.params["branches"])):
        branches[k] = dce_ir(branches[k])

    new_eqn = ireqn.using(branches=branches)
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn
