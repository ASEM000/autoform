"""Harvest"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections import defaultdict

from autoform.core import IR, call
from autoform.effects import Effect, EffectHandler, effect_p, using_handler
from autoform.utils import Tree, lru_cache

# ==================================================================================================
# CHECKPOINT EFFECT
# ==================================================================================================


class Checkpoint(Effect):
    __slots__ = "collection"
    __match_args__ = ("key", "collection")

    def __init__(self, *, key: tp.Hashable, collection: tp.Hashable | None = None):
        super().__init__(key=key)
        self.collection = collection


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
    """
    effect = Checkpoint(key=key, collection=collection)
    return effect_p.bind(in_tree, effect=effect)


# ==================================================================================================
# HANDLERS
# ==================================================================================================


class CollectHandler(EffectHandler):
    handles = (Checkpoint,)

    def __init__(self, *, collection: tp.Hashable | None = None):
        super().__init__()
        self.collection = collection
        self.collected: dict[tp.Hashable, list[tp.Any]] = defaultdict(list)

    def handle(self, effect: Checkpoint, value: tp.Any) -> tp.Any:
        if self.collection is None or effect.collection == self.collection:
            self.collected[effect.key].append(value)
        return value


class InjectHandler(EffectHandler):
    handles = (Checkpoint,)

    def __init__(
        self, *, collection: tp.Hashable | None = None, values: dict[tp.Hashable, list[tp.Any]]
    ):
        super().__init__()
        self.collection = collection
        self.values = {k: list(reversed(v)) for k, v in values.items()}

    def handle(self, effect: Checkpoint, value: tp.Any) -> tp.Any:
        if self.collection is None or effect.collection == self.collection:
            if effect.key in self.values and self.values[effect.key]:
                return self.values[effect.key].pop()
        return value


# ==================================================================================================
# COLLECT
# ==================================================================================================

type Collected = dict[tp.Hashable, list[Tree]]


@ft.partial(lru_cache, maxsize=256)
def collect[**P, R](ir: IR, *, collection: tp.Hashable) -> tp.Callable[P, tuple[R, Collected]]:
    """Collect marked values from an IR.

    Args:
        ir: The intermediate representation to run.
        collection: The collection to filter marked values by.

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
        with using_handler(CollectHandler(collection=collection)) as handler:
            result = call(ir)(*args, **kwargs)
        return result, handler.collected

    return execute


# ==================================================================================================
# INJECT
# ==================================================================================================


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
    assert isinstance(values, dict), f"Expected dict, got {type(values)}"
    for key in values:
        assert isinstance(values[key], list), f"Expected list, got {type(values[key])} for {key=}"

    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        with using_handler(InjectHandler(collection=collection, values=values)):
            return call(ir)(*args, **kwargs)

    return execute
