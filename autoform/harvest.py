"""Harvest primitives"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections import defaultdict
from collections.abc import Callable

from autoform.core import (
    IR,
    EvalType,
    Interpreter,
    IRAtom,
    IREqn,
    IRLit,
    IRVar,
    Primitive,
    Var,
    batch_rules,
    call,
    dce_rules,
    default_batch,
    default_dce,
    default_eval,
    default_impl,
    default_pull_bwd,
    default_pull_fwd,
    default_push,
    eval_rules,
    get_interpreter,
    impl_rules,
    is_iratom,
    is_irvar,
    is_user_type,
    is_var,
    pack_user_input,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.utils import Tree, treelib

# ==================================================================================================
# CHECKPOINT
# ==================================================================================================

checkpoint_p = Primitive("checkpoint", tag="core")


def checkpoint(in_tree: Tree, *, key: tp.Hashable, collection: tp.Hashable | None = None) -> Tree:
    """Tag a value with a collection and key for later collection.

    `checkpoint` marks a value with a `collection` and `key` (unique identifier)
    that can be collected by `collect`. It acts as an identity operation in
    normal execution.

    Args:
        in_tree: the value to mark (returned unchanged).
        key: unique identifier within the collection namespace.
        collection: optional collection for filtering (e.g., "debug", "cache", "metrics").

    Returns:
        the input value unchanged.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.checkpoint(af.format("Q: {}", x), key="prompt", collection="debug")
        ...     response = af.concat(prompt, " A: 42")
        ...     return af.checkpoint(response, key="response", collection="debug")
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
    assert hash(key) is not None, "Key must be hashable"
    return checkpoint_p.bind(in_tree, collection=collection, key=key)


@ft.partial(impl_rules.def_rule, checkpoint_p)
def impl_checkpoint(
    in_tree: Tree, *, key: tp.Hashable, collection: tp.Hashable | None = None
) -> Tree:
    del collection, key
    return in_tree


@ft.partial(eval_rules.def_rule, checkpoint_p)
def eval_checkpoint(
    in_tree: Tree[EvalType], *, key: tp.Hashable, collection: tp.Hashable | None = None
) -> Tree[EvalType]:
    del collection, key
    return in_tree


@ft.partial(push_rules.def_rule, checkpoint_p)
def pushforward_checkpoint(
    primal: Tree, tangent: Tree, *, key: tp.Hashable, collection: tp.Hashable | None = None
) -> tuple[Tree, Tree]:
    p = checkpoint(primal, key=key, collection=(collection, "primal"))
    t = checkpoint(tangent, key=key, collection=(collection, "tangent"))
    return p, t


@ft.partial(pull_fwd_rules.def_rule, checkpoint_p)
def pullback_fwd_checkpoint(
    in_tree: Tree, *, key: tp.Hashable, collection: tp.Hashable | None = None
) -> tuple[Tree, Tree]:
    out = checkpoint(in_tree, key=key, collection=(collection, "primal"))
    return out, out


@ft.partial(pull_bwd_rules.def_rule, checkpoint_p)
def pullback_bwd_checkpoint(
    in_residuals: Tree,
    out_cotangent: Tree,
    *,
    key: tp.Hashable,
    collection: tp.Hashable | None = None,
) -> Tree:
    del in_residuals
    return checkpoint(out_cotangent, key=key, collection=(collection, "cotangent"))


@ft.partial(batch_rules.def_rule, checkpoint_p)
def batch_checkpoint(
    batch_size: int,
    in_batched: Tree,
    x: Tree,
    *,
    key: tp.Hashable,
    collection: tp.Hashable | None = None,
) -> tuple[Tree, Tree]:
    del batch_size
    return checkpoint(x, key=key, collection=(collection, "batch")), in_batched


dce_rules.def_rule(checkpoint_p, default_dce)


# ==================================================================================================
# SPLITPOINT
# ==================================================================================================

splitpoint_p = Primitive("splitpoint", tag="core")


def splitpoint(in_tree: Tree, /, *, name: tp.Hashable) -> Tree:
    """Mark a split point for IR splitting.

    Used with `split()` to divide an IR into left and right portions.

    Args:
        in_tree: the value to mark as split point (returned unchanged).
        name: unique identifier for the split point.

    Returns:
        the input value unchanged.

    Example:
        >>> import autoform as af
        >>> def pipeline(x):
        ...     step1 = af.concat(x, "!")
        ...     step1 = af.splitpoint(step1, name="mid")
        ...     step2 = af.concat(step1, "?")
        ...     return step2
        >>> lhs, rhs = af.split(pipeline, name="mid")("hello")
    """
    assert hash(name) is not None, "Name must be hashable"
    return splitpoint_p.bind(in_tree, name=name)


impl_rules.def_rule(splitpoint_p, default_impl)
eval_rules.def_rule(splitpoint_p, default_eval)
push_rules.def_rule(splitpoint_p, default_push)
pull_fwd_rules.def_rule(splitpoint_p, default_pull_fwd)
pull_bwd_rules.def_rule(splitpoint_p, default_pull_bwd)
batch_rules.def_rule(splitpoint_p, default_batch)
dce_rules.def_rule(splitpoint_p, default_dce)


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
        # >>> def pushforward_checkpoint(primal, tangent, ...):
        # ...     p = checkpoint(primal, ...)    # <- calls checkpoint_p.bind()
        # ...     t = checkpoint(tangent, ...)   # <- calls checkpoint_p.bind()
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
        if prim == checkpoint_p and params.get("collection") == self.collection:
            self.collected[params["key"]].append(result)
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
            prim == checkpoint_p
            and params.get("collection") == self.collection
            and (key := params.get("key")) in self.values
            # NOTE(asem): allow empty values to allow round-tripping
            and self.values[key]
        ):
            return self.values[key].pop()
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

        if prim == splitpoint_p and params.get("name") == self.name:
            # NOTE(asem): splitpoint belongs to the LHS
            assert self.split is False, "Cannot split multiple times"
            self.split = True

        return out_irtree


def split[**P, R](ir: IR, *, name: tp.Hashable) -> tuple[IR, IR]:
    """Split an IR into left and right IRs at the splitpoint with given name.

    Args:
        ir: The intermediate representation to split.
        name: The name of the splitpoint to split at.

    Returns:
        A tuple (lhs_ir, rhs_ir) of two IR objects. The lhs_ir contains all
        equations up to and including the splitpoint. The rhs_ir contains
        all equations after the splitpoint.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     y = af.splitpoint(af.concat(x, "!"), name="mid")
        ...     return af.concat(y, "?")
        >>> ir = af.build_ir(program)("x")
        >>> lhs, rhs = af.split(ir, name="mid")
        >>> af.call(lhs)("hello")
        'hello!'
        >>> af.call(rhs)("hello!")
        'hello!?'
    """

    with using_interpreter(SplitInterpreter(name=name)) as tracer:
        call(ir)(ir.in_irtree)

    assert tracer.split is True, f"`split` could not find splitpoint with {name=}"

    lhs_ireqns = tracer.lhs_ireqns
    lhs_in_irtree = ir.in_irtree
    lhs_out_irtree = lhs_ireqns[-1].out_irtree
    lhs = IR(ireqns=lhs_ireqns, in_irtree=lhs_in_irtree, out_irtree=lhs_out_irtree)

    rhs_ireqns = tracer.rhs_ireqns
    rhs_in_irtree = pack_user_input(lhs_ireqns[-1].out_irtree)
    rhs_out_irtree = tracer.rhs_ireqns[-1].out_irtree if rhs_ireqns else lhs_out_irtree
    rhs = IR(ireqns=rhs_ireqns, in_irtree=rhs_in_irtree, out_irtree=rhs_out_irtree)

    return lhs, rhs
