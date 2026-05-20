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

import functools as ft
from collections import defaultdict, deque
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from typing import Any

import autoform.core as core
import autoform.dce as dce
import autoform.utils as utils

type Tree[T] = utils.Tree[T]

# ==================================================================================================
# CHECKPOINT
# ==================================================================================================


checkpoint_p = core.Prim("checkpoint")
dce.non_dce_primitives.add(checkpoint_p)


def impl_checkpoint(x, /, *, key: Hashable, collection: Hashable | None):
    del key, collection
    return x


def abstract_checkpoint(x, /, *, key: Hashable, collection: Hashable | None):
    del key, collection
    return x


def push_checkpoint(in_tree, /, *, key: Hashable, collection: Hashable | None):
    primal, tangent = in_tree
    out_p = checkpoint_p.bind(primal, key=key, collection=collection)
    out_t = checkpoint_p.bind(tangent, key=key, collection=collection)
    return out_p, out_t


def pull_fwd_checkpoint(x, /, *, key: Hashable, collection: Hashable | None):
    return checkpoint_p.bind(x, key=key, collection=collection), None


def pull_bwd_checkpoint(in_tree, /, *, key: Hashable, collection: Hashable | None):
    _, cotangent = in_tree
    return checkpoint_p.bind(cotangent, key=key, collection=collection)


def batch_checkpoint(in_tree, /, *, key: Hashable, collection: Hashable | None):
    b_sz, in_batched, x = in_tree

    if utils.batch_spec(x, in_batched) is None:
        return checkpoint_p.bind(x, key=key, collection=collection), False

    batch_x = ft.partial(utils.batch_index, x, in_batched)

    out_bi = [checkpoint_p.bind(batch_x(b), key=key, collection=collection) for b in range(b_sz)]
    out_batched = in_batched
    out_ib = utils.batch_transpose(b_sz, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_checkpoint(in_tree, /, *, key: Hashable, collection: Hashable | None):
    b_sz, in_batched, x = in_tree

    if utils.batch_spec(x, in_batched) is None:
        return await checkpoint_p.abind(x, key=key, collection=collection), False

    batch_at = ft.partial(utils.batch_index, x, in_batched)

    async def abind(b):
        return await checkpoint_p.abind(batch_at(b), key=key, collection=collection)

    out_bi = [await abind(b) for b in range(b_sz)]
    out_batched = in_batched
    out_ib = utils.batch_transpose(b_sz, out_batched, out_bi)
    return out_ib, out_batched


core.impl_rules.set(checkpoint_p, impl_checkpoint)
core.impl_rules.aset(checkpoint_p, utils.asyncify(impl_checkpoint))
core.abstract_rules.set(checkpoint_p, abstract_checkpoint)
core.push_rules.set(checkpoint_p, push_checkpoint)
core.push_rules.aset(checkpoint_p, utils.asyncify(push_checkpoint))
core.pull_fwd_rules.set(checkpoint_p, pull_fwd_checkpoint)
core.pull_fwd_rules.aset(checkpoint_p, utils.asyncify(pull_fwd_checkpoint))
core.pull_bwd_rules.set(checkpoint_p, pull_bwd_checkpoint)
core.pull_bwd_rules.aset(checkpoint_p, utils.asyncify(pull_bwd_checkpoint))
core.batch_rules.set(checkpoint_p, batch_checkpoint)
core.batch_rules.aset(checkpoint_p, abatch_checkpoint)


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
    return checkpoint_p.bind(value, key=key, collection=collection)


# ==================================================================================================
# COLLECT
# ==================================================================================================


type Collected = dict[Hashable, list[Tree]]


class CollectingInterpreter(core.Interpreter):
    __slots__ = ["parent", "collection", "collected"]

    def __init__(self, *, collection: Hashable):
        self.parent = core.active_interpreter.get()
        self.collection = collection
        self.collected: Collected = defaultdict(list)

    def interpret(self, prim: core.Prim, in_tree: Any, /, **params):
        result = self.parent.interpret(prim, in_tree, **params)
        if prim is checkpoint_p:
            if self.collection is ... or params["collection"] == self.collection:
                self.collected[params["key"]].append(result)
        return result

    async def ainterpret(self, prim: core.Prim, in_tree: Any, /, **params):
        result = await self.parent.ainterpret(prim, in_tree, **params)
        if prim is checkpoint_p:
            if self.collection is ... or params["collection"] == self.collection:
                self.collected[params["key"]].append(result)
        return result


class InjectingInterpreter(core.Interpreter):
    __slots__ = ["parent", "collection", "cache"]

    def __init__(self, *, collection: Hashable, values: Collected):
        self.parent = core.active_interpreter.get()
        self.collection = collection
        self.cache = {k: deque(values[k]) for k in values}

    def interpret(self, prim: core.Prim, in_tree: Any, /, **params):
        if prim is checkpoint_p and params["collection"] == self.collection:
            if params["key"] in self.cache and self.cache[params["key"]]:
                return self.cache[params["key"]].popleft()
        return self.parent.interpret(prim, in_tree, **params)

    async def ainterpret(self, prim: core.Prim, in_tree: Any, /, **params):
        if prim is checkpoint_p and params["collection"] == self.collection:
            if params["key"] in self.cache and self.cache[params["key"]]:
                return self.cache[params["key"]].popleft()
        return await self.parent.ainterpret(prim, in_tree, **params)


@contextmanager
def collect(*, collection: Hashable) -> Generator[Collected, None, None]:
    """Collect checkpoint values produced during IR execution.

    ``collect`` is an execution-time context. Trace the program first, then
    place ``collect`` around ``ir.call(...)`` or ``ir.acall(...)``. Values are
    appended when executed :func:`autoform.checkpoint` primitives run.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     normalized = af.format("item: {}", x)
        ...     normalized = af.checkpoint(normalized, key="normalized", collection="debug")
        ...     return af.concat(normalized, "!")
        >>> ir = af.trace(program)("test")
        >>> with af.collect(collection="debug") as collected:
        ...     result = ir.call("alpha")
        >>> result
        'item: alpha!'
        >>> collected["normalized"]
        ['item: alpha']

    Transformed IR execution is also execution, so collection works there too.

    Example:
        >>> batched = af.batch(ir)
        >>> with af.collect(collection="debug") as collected:
        ...     result = batched.call(["alpha", "beta"])
        >>> result
        ['item: alpha!', 'item: beta!']
        >>> collected["normalized"]
        ['item: alpha', 'item: beta']

    Do not wrap trace construction with ``collect``. Tracing builds IR equations;
    it does not produce concrete runtime checkpoint values. Do not use
    ``collect`` inside the function being traced either; during tracing, dynamic
    values are placeholders, not runtime values.

    Args:
        collection: The collection to filter marked values by. If `...`, collect all values.

    Yields:
        A dict that maps keys to lists of collected values.
    """
    with core.using_interpreter(CollectingInterpreter(collection=collection)) as interpreter:
        yield interpreter.collected


# ==================================================================================================
# INJECT
# ==================================================================================================


@contextmanager
def inject(*, collection: Hashable, values: Collected) -> Generator[None, None, None]:
    """Inject values for checkpoints within the context.

    Values are consumed from lists in encounter order for each key. A dictionary
    produced by :func:`autoform.collect` can be supplied here to reproduce or
    modify checkpointed intermediates.

    Around ``ir.call(...)`` or ``ir.acall(...)``, ``inject`` performs runtime
    checkpoint substitution by replacing matching checkpoint values as the IR
    executes.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     normalized = af.format("item: {}", x)
        ...     normalized = af.checkpoint(normalized, key="normalized", collection="cache")
        ...     return af.concat(normalized, "!")
        >>> ir = af.trace(program)("test")
        >>> with af.inject(collection="cache", values={"normalized": ["cached item"]}):
        ...     ir.call("alpha")
        'cached item!'

    Inside the function being traced, ``inject`` performs trace-time
    specialization. A matching checkpoint is replaced by the injected literal
    while the IR is being built, so later calls to the traced IR reuse that
    specialized value.

    Example:
        >>> def program(x):
        ...     normalized = af.format("item: {}", x)
        ...     with af.inject(collection="cache", values={"normalized": ["cached item"]}):
        ...         normalized = af.checkpoint(normalized, key="normalized", collection="cache")
        ...     return af.concat(normalized, "!")
        >>> ir = af.trace(program)("test")
        >>> [ir_eqn.prim.name for ir_eqn in ir.ir_eqns]
        ['format', 'concat']
        >>> ir.call("alpha")
        'cached item!'
        >>> ir.call("beta")
        'cached item!'

    Args:
        collection: The collection to filter checkpoint locations by.
        values: Dictionary mapping checkpoint keys to lists of values to inject.

    Yields:
        None.
    """
    assert isinstance(values, dict)
    for key in values:
        assert isinstance(values[key], list), f"{type(values[key])} for key {key} is not a list."

    with core.using_interpreter(InjectingInterpreter(collection=collection, values=values)):
        yield
