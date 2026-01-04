"""Harvest"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections import defaultdict

from autoform.core import IR, Effect, call, using_effect, using_effect_handler
from autoform.effects import effect_p
from autoform.utils import Tree, lru_cache

# ==================================================================================================
# CHECKPOINT EFFECT
# ==================================================================================================


class CheckpointEffect(Effect):
    __slots__ = ("key", "collection")
    __match_args__ = ("key", "collection")

    def __init__(self, *, key: tp.Hashable, collection: tp.Hashable | None = None):
        self.key = key
        self.collection = collection


def checkpoint(value: Tree, /, *, key: tp.Hashable, collection: tp.Hashable | None = None) -> Tree:
    """Tag a value with a collection and key for later collection.

    `checkpoint` marks a value with a `collection` and `key` (unique identifier)
    that can be collected by `collect`. It acts as an identity operation in
    normal execution.

    Args:
        value: the value to mark (returned unchanged).
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
    """
    with using_effect(CheckpointEffect(key=key, collection=collection)):
        return effect_p.bind(value)


# ==================================================================================================
# COLLECT
# ==================================================================================================


type Collected = dict[tp.Hashable, list[Tree]]


class CollectHandler:
    def __init__(self, *, collection: tp.Hashable):
        self.collection = collection
        self.collected: dict[tp.Hashable, list[tp.Any]] = defaultdict(list)

    def __call__(self, effect: CheckpointEffect, in_tree: tp.Any):
        result = yield in_tree
        if self.collection is ... or effect.collection == self.collection:
            self.collected[effect.key].append(result)
        return result
        yield


@ft.partial(lru_cache, maxsize=256)
def collect[**P, R](ir: IR, *, collection: tp.Hashable) -> tp.Callable[P, tuple[R, Collected]]:
    """Collect marked values from an IR.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter marked values by. If `...`, collect all values.

    Returns:
        A callable that executes the IR and returns (result, collected_dict).
        The collected_dict maps keys to lists of values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.checkpoint(af.format("Q: {}", x), key="prompt", collection="debug")
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
        handler = CollectHandler(collection=collection)
        with using_effect_handler({CheckpointEffect: handler}):
            result = call(ir)(*args, **kwargs)
        return result, handler.collected

    return execute


# ==================================================================================================
# INJECT
# ==================================================================================================


class InjectHandler:
    def __init__(self, *, collection: tp.Hashable, values: Collected):
        self.collection = collection
        self.cache = {k: list(reversed(v)) for k, v in values.items()}

    def __call__(self, effect: CheckpointEffect, in_tree: tp.Any):
        if effect.collection == self.collection:
            if effect.key in self.cache and self.cache[effect.key]:
                return self.cache[effect.key].pop()
        return (yield in_tree)


def inject[**P, R](ir: IR, *, collection: tp.Hashable, values: Collected) -> tp.Callable[P, R]:
    """Create an injecting executor for an IR.

    Values are consumed from lists in order (matching collect's list output).
    This allows round-tripping: collect values, then inject them back.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter checkpoint locations by.
        values: Dictionary mapping checkpoint keys to lists of values to inject.

    Returns:
        A callable that executes the IR with injected values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")
        >>> ir = af.build_ir(program)("test")
        >>> af.inject(ir, collection="cache", values={"greeting": ["CACHED"]})("World")
        'CACHED'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
    assert isinstance(values, dict)
    for key in values:
        assert isinstance(values[key], list), f"{type(values[key])} for key {key} is not a list."

    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        handler = InjectHandler(collection=collection, values=values)
        with using_effect_handler({CheckpointEffect: handler}):
            return call(ir)(*args, **kwargs)

    return execute
