"""Harvest primitives"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections import defaultdict
from collections.abc import Callable
from autoform.core import Interpreter, get_interpreter, using_interpreter
from autoform.core import IR, EvalType
from autoform.core import (
    Primitive,
    batch_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, treelib
from autoform.core import (
    call,
    IREqn,
    IRVar,
    IRLit,
    Var,
    is_iratom,
    is_irvar,
    is_var,
    IRAtom,
    pack_user_input,
    is_user_type,
)

# ==================================================================================================
# MARK
# ==================================================================================================

mark_p = Primitive("mark", tag="core")


def mark(in_tree: Tree, /, *, collection: tp.Hashable, name: tp.Hashable) -> Tree:
    """Tag a value with a collection and name for later collection.

    `mark` marks a value with a `collection` and `name` (unique identifier)
    that can be collected by `collect`. It acts as an identity operation in
    normal execution.

    Args:
        in_tree: the value to mark (returned unchanged).
        collection: collection for filtering (e.g., "debug", "cache", "metrics").
        name: unique identifier within the collection namespace.

    Returns:
        the input value unchanged.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.mark(af.format("Q: {}", x), collection="debug", name="prompt")
        ...     response = af.concat(prompt, " A: 42")
        ...     return af.mark(response, collection="debug", name="response")
        >>> ir = af.build_ir(program)("test")
        >>> result, collected = af.collect(ir, collection="debug")("What is 6*7?")
        >>> result
        'Q: What is 6*7? A: 42'
        >>> collected["prompt"]
        ['Q: What is 6*7?']
        >>> collected["response"]
        ['Q: What is 6*7? A: 42']
    """
    assert hash(collection) is not None, "Collection must be hashable"
    assert hash(name) is not None, "Name must be hashable"
    return mark_p.bind(in_tree, collection=collection, name=name)


@ft.partial(impl_rules.def_rule, mark_p)
def impl_mark(in_tree: Tree, *, collection: tp.Hashable, name: tp.Hashable) -> Tree:
    del collection, name
    return in_tree


@ft.partial(eval_rules.def_rule, mark_p)
def eval_mark(
    in_tree: Tree[EvalType], *, collection: tp.Hashable, name: tp.Hashable
) -> Tree[EvalType]:
    del collection, name
    return in_tree


@ft.partial(push_rules.def_rule, mark_p)
def pushforward_mark(
    primal: Tree, tangent: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    p = mark(primal, collection=(collection, "primal"), name=name)
    t = mark(tangent, collection=(collection, "tangent"), name=name)
    return p, t


@ft.partial(pull_fwd_rules.def_rule, mark_p)
def pullback_fwd_mark(
    in_tree: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    out = mark(in_tree, collection=(collection, "primal"), name=name)
    return out, out


@ft.partial(pull_bwd_rules.def_rule, mark_p)
def pullback_bwd_mark(
    in_residuals: Tree, out_cotangent: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> Tree:
    del in_residuals
    return mark(out_cotangent, collection=(collection, "cotangent"), name=name)


@ft.partial(batch_rules.def_rule, mark_p)
def batch_mark(
    _: int, in_batched: Tree, x: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    return mark(x, collection=(collection, "batch"), name=name), in_batched


# ==================================================================================================
# COLLECT
# ==================================================================================================

type Collected = dict[tp.Hashable, list[Tree]]


class CollectInterpreter(Interpreter):
    def __init__(self, *, collection: tp.Hashable):
        self.collection = collection
        # NOTE(asem): collect into a defaultdict of lists for situations where
        # a value is marked multiple times (e.g. a value in a loop)
        self.collected: Collected = defaultdict(list)
        self.parent = get_interpreter()

    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        # NOTE(asem): No context switch here. We call parent.interpret() directly
        # Example: lets say we have a push rule that marks stuff inside it
        # >>> def pushforward_mark(primal, tangent, ...):
        # ...     p = mark(primal, ...)    # <- calls mark_p.bind()
        # ...     t = mark(tangent, ...)   # <- calls mark_p.bind()
        # ...     return p, t
        # here we have an interplay between 3 interpreters (collect, eval, and push)
        # within the rule we have parent=eval (push_impl -> induced PushInterp -> called this rule)
        # if we do not call collect.
        # now lets say we call collect, then we induce CollectInterp with parent=eval
        # now lets say we switch to parent, then we move to eval->push_call.impl->
        # induce a PushInterp with parent=eval. **at this point collect is lost**.
        # however, if we simply call previous interpreter (parent) then we will
        # keep CollectInterpreter as the active interpreter (from using_interpreter at collect).
        # when push rule calls mark(), it goes through prim.bind() -> get_interp()
        # -> CollectInterpreter.interpret() and we can observe those nested calls.
        # this observation applies to other observer interpreters as well.
        result = self.parent.interpret(prim, in_tree, **params)
        if prim == mark_p and params.get("collection") == self.collection:
            self.collected[params["name"]].append(result)
        return result


def collect[**P, R](ir: IR, *, collection: tp.Hashable) -> tp.Callable[P, tuple[R, Collected]]:
    """Collect marked values from an IR.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter marked values by.

    Returns:
        A callable that executes the IR and returns (result, collected_dict).
        The collected_dict maps names to lists of values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.mark(af.format("Q: {}", x), collection="debug", name="prompt")
        ...     return af.concat(prompt, " A: 42")
        >>> ir = af.build_ir(program)("test")
        >>> result, collected = af.collect(ir, collection="debug")("What?")
        >>> result
        'Q: What? A: 42'
        >>> collected["prompt"]
        ['Q: What?']
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> tuple[R, Collected]:
        with using_interpreter(CollectInterpreter(collection=collection)) as collector:
            result = call(ir)(*args, **kwargs)
        return result, collector.collected

    return execute


# ==================================================================================================
# INJECT
# ==================================================================================================


class InjectInterpreter(Interpreter):
    def __init__(self, *, collection: tp.Hashable, values: dict[tp.Hashable, list[Tree]]):
        self.collection = collection
        # NOTE(asem): reverse the lists to pop from the end to match collect's list output
        self.values = {k: list(reversed(values[k])) for k in values}
        self.parent = get_interpreter()

    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        if (
            prim == mark_p
            and params.get("collection") == self.collection
            and (name := params.get("name")) in self.values
            # NOTE(asem): allow empty values to allow round-tripping
            and self.values[name]
        ):
            return self.values[name].pop()
        # NOTE(asem): check the note in CollectInterpreter.interpret for explanation
        # on why no context switch here.
        return self.parent.interpret(prim, in_tree, **params)


def inject[**P, R](ir: IR, *, collection: tp.Hashable, values: Collected) -> tp.Callable[P, R]:
    """Create an injecting executor for an IR.

    Values are consumed from lists in order (matching collect's list output).
    This allows round-tripping: collect values, then inject them back.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter mark locations by.
        values: Dictionary mapping mark names to lists of values to inject.
            The lists are consumed from left to right (matching collect's list output).

    Returns:
        A callable that executes the IR with injected values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.mark(af.concat("Hello, ", x), collection="cache", name="greeting")
        >>> ir = af.build_ir(program)("test")
        >>> af.inject(ir, collection="cache", values={"greeting": ["CACHED"]})("World")
        'CACHED'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
    assert isinstance(values, dict), f"Expected dict, got {type(values)}"
    for key in values:
        assert isinstance(values[key], list), f"Expected list, got {type(values[key])} for {key=}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        with using_interpreter(InjectInterpreter(collection=collection, values=values)):
            return call(ir)(*args, **kwargs)

    return execute


# ==================================================================================================
# SPLIT
# ==================================================================================================


class SplitInterpreter(Interpreter):
    def __init__(self, name: tp.Hashable):
        # NOTE(asem): trace and split interpreter
        # mostly similar to TraceInterpreter but splits the IR into two parts
        # at the mark with the given name
        self.lhs_ireqns: list[IREqn] = []
        self.rhs_ireqns: list[IREqn] = []
        self.name = name
        self.split: bool = False

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

        assert prim in eval_rules, f"Primitive {prim.name} has no `eval_rule` defined"

        def to_in_evaltype(x):
            # NOTE(asem): eval rules accept `Var`/ python types.
            # `Var` simply denotes a placeholder for a value that will be computed later
            return Var() if is_irvar(x) else x.value

        in_evaltree = treelib.map(to_in_evaltype, in_irtree)
        out_evaltree = eval_rules[prim](in_evaltree, **params)

        def to_out_iratom(x) -> IRAtom:
            # NOTE(asem): eval rules return `Var`/ python types.
            # `Var` simply denotes a placeholder for a value that will be computed later
            # this is basically delegated to the user to handle
            return IRVar.fresh() if is_var(x) else IRLit(x)

        out_irtree = treelib.map(to_out_iratom, out_evaltree)

        ireqns = self.rhs_ireqns if self.split else self.lhs_ireqns
        ireqns.append(IREqn(prim, in_irtree, out_irtree, params))

        if prim == mark_p and params.get("name") == self.name:
            # NOTE(asem): mark belongs to the LHS
            assert self.split is False, "Cannot split multiple times"
            self.split = True

        return out_irtree


def split[**P, R](func: Callable[P, R], name: tp.Hashable) -> tuple[IR[P, R], IR[P, R]]:
    """Split a function into left and right IRs at marked name.

    Args:
        func: A callable that uses autoform primitives (format, concat, lm_call, etc.).
        name: A unique hashable value.

    Returns:
        A tracer callable that takes ``(*args, **kwargs)`` and returns a pair of IR
    """
    # NOTE(asem): calling split inside a traced function will inline the splitted IRs
    # >>> def outer(x):
    # ...     lhs, rhs = split(inner, name="mid")("...")
    # ...     mid = af.call(lhs)(x)    # lhs equations inline into outer
    # ...     return af.call(rhs)(mid) # rhs equations inline into outer
    # >>> ir = af.build_ir(outer)("...")
    # result is a single flat IR with all equations from lhs and rhs

    def assert_usertype(x):
        assert not is_iratom(x), "Inputs to `build_ir` must be normal python types"

    def to_in_iratom(x):
        # NOTE(asem): user types are converted to IRVar/IRLit
        # to prepare for tracing.
        return IRVar.fresh() if is_user_type(x) else IRLit(x)

    def to_out_iratom(x):
        return x if is_iratom(x) else IRLit(x)

    @ft.wraps(func)
    def trace(*args: P.args, **kwargs: P.kwargs) -> tuple[IR, IR]:
        treelib.map(assert_usertype, (args, kwargs), is_leaf=is_user_type)
        in_irtree = treelib.map(to_in_iratom, (args, kwargs), is_leaf=is_user_type)
        in_irargs, in_irkwargs = in_irtree

        with using_interpreter(SplitInterpreter(name=name)) as tracer:
            out_irtree = func(*in_irargs, **in_irkwargs)

        assert tracer.split is True, f"`split` could not find mark matches {name=}"

        lhs_ireqns = tracer.lhs_ireqns
        lhs_in_irtree = pack_user_input(*in_irargs, **in_irkwargs)
        lhs_out_irtree = lhs_ireqns[-1].out_irtree
        lhs = IR(ireqns=lhs_ireqns, in_irtree=lhs_in_irtree, out_irtree=lhs_out_irtree)

        # NOTE(asem): rhs takes mark output as input, wrapped in call-compatible structure
        # The mark output becomes the new "input" for rhs
        rhs_ireqns = tracer.rhs_ireqns
        rhs_in_irtree = pack_user_input(lhs_ireqns[-1].out_irtree)
        rhs_out_irtree = treelib.map(to_out_iratom, out_irtree)
        rhs = IR(ireqns=rhs_ireqns, in_irtree=rhs_in_irtree, out_irtree=rhs_out_irtree)

        return lhs, rhs

    return trace
