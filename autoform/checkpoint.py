# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from typing import Any

from autoform.core import Interpreter, Prim, active_interpreter, using_interpreter
from autoform.intercepts import intercept_p
from autoform.utils import Tree

# ==================================================================================================
# CHECKPOINT
# ==================================================================================================


def is_checkpoint_call(prim: Prim, params: dict[str, Any], /) -> bool:
    return prim is intercept_p and "key" in params and "collection" in params


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
        ...     result = ir.call("What is 6*7?")
        >>> result
        'Q: What is 6*7? A: 42'
        >>> collected["prompt"]
        ['Q: What is 6*7?']
    """
    return intercept_p.bind(value, key=key, collection=collection)


# ==================================================================================================
# COLLECT
# ==================================================================================================


type Collected = dict[Hashable, list[Tree]]


class CollectingInterpreter(Interpreter):
    def __init__(self, *, collection: Hashable):
        self.parent = active_interpreter.get()
        self.collection = collection
        self.collected: Collected = defaultdict(list)

    def interpret(self, prim: Prim, in_tree: Any, /, **params):
        result = self.parent.interpret(prim, in_tree, **params)
        if is_checkpoint_call(prim, params):
            if self.collection is ... or params["collection"] == self.collection:
                self.collected[params["key"]].append(result)
        return result

    async def ainterpret(self, prim: Prim, in_tree: Any, /, **params):
        result = await self.parent.ainterpret(prim, in_tree, **params)
        if is_checkpoint_call(prim, params):
            if self.collection is ... or params["collection"] == self.collection:
                self.collected[params["key"]].append(result)
        return result


class InjectingInterpreter(Interpreter):
    def __init__(self, *, collection: Hashable, values: Collected):
        self.parent = active_interpreter.get()
        self.collection = collection
        self.cache = {k: deque(values[k]) for k in values}

    def interpret(self, prim: Prim, in_tree: Any, /, **params):
        if is_checkpoint_call(prim, params) and params["collection"] == self.collection:
            if params["key"] in self.cache and self.cache[params["key"]]:
                return self.cache[params["key"]].popleft()
        return self.parent.interpret(prim, in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Any, /, **params):
        if is_checkpoint_call(prim, params) and params["collection"] == self.collection:
            if params["key"] in self.cache and self.cache[params["key"]]:
                return self.cache[params["key"]].popleft()
        return await self.parent.ainterpret(prim, in_tree, **params)


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
        ...     result = ir.call("What?")
        >>> result
        'Q: What? A: 42'
        >>> collected["prompt"]
        ['Q: What?']
    """
    with using_interpreter(CollectingInterpreter(collection=collection)) as interpreter:
        yield interpreter.collected


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
        ...     ir.call("World")
        'CACHED'
    """
    assert isinstance(values, dict)
    for key in values:
        assert isinstance(values[key], list), f"{type(values[key])} for key {key} is not a list."

    with using_interpreter(InjectingInterpreter(collection=collection, values=values)):
        yield
