"""Harvest primitives"""

from __future__ import annotations

import functools as ft
import typing as tp
from autoform.core import Interpreter, get_interp, using_interp
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
        'Q: What is 6*7?'
        >>> collected["response"]
        'Q: What is 6*7? A: 42'
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

type Collected = dict[tp.Hashable, Tree]


class CollectInterpreter(Interpreter):
    def __init__(self, *, collection: tp.Hashable):
        self.collection = collection
        self.collected: Collected = {}
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        # NOTE(asem): no context switch for interception interpreter
        result = self.parent.process(prim, in_tree, **params)
        if prim == checkpoint_p and params.get("collection") == self.collection:
            self.collected[params["name"]] = result
        return result


def collect[**P, R](ir: IR, *, collection: tp.Hashable) -> tp.Callable[P, tuple[R, Collected]]:
    """Collect checkpointed values from an IR.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter checkpointed values by.

    Returns:
        A callable that executes the IR and returns (result, collected_dict).

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.checkpoint(af.format("Q: {}", x), collection="debug", name="prompt")
        ...     return af.concat(prompt, " A: 42")
        >>> ir = af.build_ir(program)("test")
        >>> result, collected = af.collect(ir, collection="debug")("What?")
        >>> result
        'Q: What? A: 42'
        >>> collected
        {'prompt': 'Q: What?'}
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> tuple[R, Collected]:
        with using_interp(CollectInterpreter(collection=collection)) as collector:
            result = call(ir)(*args, **kwargs)
        return result, collector.collected

    return execute


# ==================================================================================================
# INJECT
# ==================================================================================================


class InjectInterpreter(Interpreter):
    def __init__(self, *, collection: tp.Hashable, values: dict[tp.Hashable, Tree]):
        self.collection = collection
        self.values = values
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        if (
            prim == checkpoint_p
            and params.get("collection") == self.collection
            and (name := params.get("name")) in self.values
        ):
            return self.values[name]
        # NOTE(asem): no context switch for interception interpreter
        return self.parent.process(prim, in_tree, **params)


def inject[**P, R](ir: IR, *, collection: tp.Hashable, values: Collected) -> tp.Callable[P, R]:
    """Create an injecting executor for an IR.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter checkpoint locations by.
        values: Dictionary mapping checkpoint names to values to inject.

    Returns:
        A callable that executes the IR with injected values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.checkpoint(af.concat("Hello, ", x), collection="cache", name="greeting")
        >>> ir = af.build_ir(program)("test")
        >>> af.inject(ir, collection="cache", values={"greeting": "CACHED"})("World")
        'CACHED'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        with using_interp(InjectInterpreter(collection=collection, values=values)):
            return call(ir)(*args, **kwargs)

    return execute
