"""Harvest"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from typing import Any

from autoform.core import Effect, EffectInterpreter, using_effect, using_interpreter
from autoform.effects import effect_p
from autoform.utils import Tree

# ==================================================================================================
# CHECKPOINT EFFECT
# ==================================================================================================


class CheckpointEffect(Effect):
    __slots__ = ("key", "collection")

    def __init__(self, *, key: Hashable, collection: Hashable | None = None):
        self.key = key
        self.collection = collection


def checkpoint(value: Tree, /, *, key: Hashable, collection: Hashable | None = None) -> Tree:
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
        >>> ir = af.trace(program)("test")
        >>> with af.collect(collection="debug") as collected:
        ...     result = af.call(ir)("What is 6*7?")
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


type Collected = dict[Hashable, list[Tree]]


@contextmanager
def collect(*, collection: Hashable) -> Generator[Collected, None, None]:
    """Collect marked values within the context.

    Args:
        collection: The collection to filter marked values by. If `...`, collect all values.

    Yields:
        A dict that maps keys to lists of collected values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.checkpoint(af.format("Q: {}", x), key="prompt", collection="debug")
        ...     return af.concat(prompt, " A: 42")
        >>> ir = af.trace(program)("test")
        >>> with af.collect(collection="debug") as collected:
        ...     result = af.call(ir)("What?")
        >>> result
        'Q: What? A: 42'
        >>> collected["prompt"]
        ['Q: What?']
    """
    collected: Collected = defaultdict(list)

    def collector(prim, effect: CheckpointEffect, in_tree: Any, /):
        result = yield in_tree
        if collection is ... or effect.collection == collection:
            collected[effect.key].append(result)
        return result

    with using_interpreter(EffectInterpreter((CheckpointEffect, collector))):
        yield collected


# ==================================================================================================
# INJECT
# ==================================================================================================


@contextmanager
def inject(*, collection: Hashable, values: Collected) -> Generator[None, None, None]:
    """Inject values for checkpoints within the context.

    Values are consumed from lists in order (matching collect's list output).
    This allows round-tripping: collect values, then inject them back.

    Args:
        collection: The collection to filter checkpoint locations by.
        values: Dictionary mapping checkpoint keys to lists of values to inject.

    Yields:
        None.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")
        >>> ir = af.trace(program)("test")
        >>> with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
        ...     af.call(ir)("World")
        'CACHED'
    """
    assert isinstance(values, dict)
    for key in values:
        assert isinstance(values[key], list), f"{type(values[key])} for key {key} is not a list."

    cache = {k: deque(values[k]) for k in values}

    def injector(prim, effect: CheckpointEffect, in_tree: Any, /):
        if effect.collection == collection:
            if effect.key in cache and cache[effect.key]:
                return cache[effect.key].popleft()
        out_tree = yield in_tree
        return out_tree

    with using_interpreter(EffectInterpreter((CheckpointEffect, injector))):
        yield
