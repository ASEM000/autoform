"""Harvest primitives"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections import defaultdict
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
from autoform.utils import Tree
from autoform.core import call

# ==================================================================================================
# CHECKPOINT
# ==================================================================================================

checkpoint_p = Primitive("checkpoint", tag="core")


def checkpoint(in_tree: Tree, /, *, collection: tp.Hashable, name: tp.Hashable) -> Tree:
    """Tag a value with a collection and name for later collection.

    `checkpoint` marks a value with a `collection` and `name` (unique identifier)
    that can be collected by `collect`. It acts as an identity operation in
    normal execution.

    Args:
        in_tree: the value to checkpoint (returned unchanged).
        collection: collection for filtering (e.g., "debug", "cache", "metrics").
        name: unique identifier within the collection namespace.

    Returns:
        the input value unchanged.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.checkpoint(af.format("Q: {}", x), collection="debug", name="prompt")
        ...     response = af.concat(prompt, " A: 42")
        ...     return af.checkpoint(response, collection="debug", name="response")
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
    return checkpoint_p.bind(in_tree, collection=collection, name=name)


@ft.partial(impl_rules.def_rule, checkpoint_p)
def impl_checkpoint(in_tree: Tree, *, collection: tp.Hashable, name: tp.Hashable) -> Tree:
    del collection, name
    return in_tree


@ft.partial(eval_rules.def_rule, checkpoint_p)
def eval_checkpoint(
    in_tree: Tree[EvalType], *, collection: tp.Hashable, name: tp.Hashable
) -> Tree[EvalType]:
    del collection, name
    return in_tree


@ft.partial(push_rules.def_rule, checkpoint_p)
def pushforward_checkpoint(
    primal: Tree, tangent: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    p = checkpoint(primal, collection=(collection, "primal"), name=name)
    t = checkpoint(tangent, collection=(collection, "tangent"), name=name)
    return p, t


@ft.partial(pull_fwd_rules.def_rule, checkpoint_p)
def pullback_fwd_checkpoint(
    in_tree: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    out = checkpoint(in_tree, collection=(collection, "primal"), name=name)
    return out, out


@ft.partial(pull_bwd_rules.def_rule, checkpoint_p)
def pullback_bwd_checkpoint(
    in_residuals: Tree, out_cotangent: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> Tree:
    del in_residuals
    return checkpoint(out_cotangent, collection=(collection, "cotangent"), name=name)


@ft.partial(batch_rules.def_rule, checkpoint_p)
def batch_checkpoint(
    _: int, in_batched: Tree, x: Tree, *, collection: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    return checkpoint(x, collection=(collection, "batch"), name=name), in_batched


# ==================================================================================================
# COLLECT
# ==================================================================================================

type Collected = dict[tp.Hashable, list[Tree]]


class CollectInterpreter(Interpreter):
    def __init__(self, *, collection: tp.Hashable):
        self.collection = collection
        # NOTE(asem): collect into a defaultdict of lists for situations where
        # a value is checkpointed multiple times (e.g. a value in a loop)
        self.collected: Collected = defaultdict(list)
        self.parent = get_interpreter()

    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        # NOTE(asem): No context switch here. We call parent.interpret() directly
        # Example: lets say we have a push rule that checkpoints stuff inside it
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
        # when push rule calls checkpoint(), it goes through prim.bind() -> get_interp()
        # -> CollectInterpreter.interpret() and we can observe those nested calls.
        # this observation applies to other observer interpreters as well.
        result = self.parent.interpret(prim, in_tree, **params)
        if prim == checkpoint_p and params.get("collection") == self.collection:
            self.collected[params["name"]].append(result)
        return result


def collect[**P, R](ir: IR, *, collection: tp.Hashable) -> tp.Callable[P, tuple[R, Collected]]:
    """Collect checkpointed values from an IR.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter checkpointed values by.

    Returns:
        A callable that executes the IR and returns (result, collected_dict).
        The collected_dict maps names to lists of values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.checkpoint(af.format("Q: {}", x), collection="debug", name="prompt")
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
        collection: The collection to filter checkpoint locations by.
        values: Dictionary mapping checkpoint names to lists of values to inject.
            The lists are consumed from left to right (matching collect's list output).

    Returns:
        A callable that executes the IR with injected values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.checkpoint(af.concat("Hello, ", x), collection="cache", name="greeting")
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
