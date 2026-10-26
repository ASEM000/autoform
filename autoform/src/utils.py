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

"""Utility functions for autoform"""

from __future__ import annotations

import builtins
import functools as ft
from collections.abc import Awaitable, Callable
from typing import Any, cast

import autoform.src.tree as tree

type Tree[T] = Any


def strict_zip(*iterables):
    return builtins.zip(*iterables, strict=True)


zip = strict_zip


# ==================================================================================================
# ASYNC UTILITIES
# ==================================================================================================


def asyncify[**P, R](func: Callable[P, R], /) -> Callable[P, Awaitable[R]]:
    @ft.wraps(func)
    async def afunc(*args, **kwargs):
        return func(*args, **kwargs)

    return afunc


def dispatch[R](default: Callable[..., R] | None = None):
    # NOTE(asem): dispatch using optree tree spec with leaves exact type. unlike single dispatch
    # this dispatch on exact type, allow pytrees. however its a bit slower.
    rules: dict[tuple[tree.PyTreeSpec, tuple[type, ...]], Callable[..., R]] = {}

    def register[**P](types: Tree[type], rule: Callable[P, R], /) -> Callable[P, R]:
        leaves, spec = tree.flatten(types)
        rules[spec, tuple(leaves)] = rule
        return rule

    def call(values: Tree, /, **params: Any) -> R:
        leaves, spec = tree.flatten(values)
        key = spec, tuple(type(x) for x in leaves)
        if (rule := rules.get(key, default)) is None:
            raise TypeError(f"No dispatch rule for {key!r}")
        return rule(values, **params)

    if default is not None:
        call = ft.wraps(default)(call)
    call.register = register
    call.rules = rules
    return call


def lru_cache[**P, R](func: Callable[P, R], maxsize: int = 256) -> Callable[P, R]:
    return cast(Callable[P, R], ft.lru_cache(maxsize=maxsize)(func))


def index(node: Tree, b: int, /) -> Tree:
    # NOTE(asem): index a struct without indexing support
    # useful to deal with arbitrary pytrees
    children = tree.leaves(node, is_leaf=lambda x: x is None or x is not node)
    return children[b]


# ==================================================================================================
# BATCH UTILITIES
# ==================================================================================================


def batch_index(in_tree: Tree, in_batched: Tree[bool], b: int, /) -> Tree:
    """Extract one batch item from a tree.

    Batched leaves are indexed at ``b``. Non-batched leaves are broadcast by
    returning the original leaf unchanged.

    Args:
        in_tree: Tree containing batched and non-batched leaves.
        in_batched: Tree of booleans with the same outer structure as
            ``in_tree``. ``True`` marks a batched leaf.
        b: Batch index to extract.

    Returns:
        A tree with the same structure as ``in_batched`` containing one batch
        item.

    Example:
        >>> batch_index(([1, 2], "x"), (True, False), 0)
        (1, 'x')
    """
    # Extract item at index b from batched leaves, broadcast non-batched.
    # Inverse of transpose_batch: extracts a single item from each batched leaf
    # while keeping non-batched leaves unchanged.

    # Args:
    #     in_tree: tree with batched leaves (index-batch order, each leaf is a list).
    #     in_batched: tree of bools indicating which leaves are batched.
    #     b: index to extract from batched leaves.

    # >>> in_tree, in_batched = [[1, 2, 3], "constant"], [True, False]
    # >>> batch_index(in_tree, in_batched, 0)
    # [1, 'constant']
    spec = tree.structure(in_batched)
    # NOTE(asem): flatten in_tree to match in_batched structure
    # >>> spec = tree.structure([1, 2, 3])
    # >>> spec.flatten_up_to([1, [2, 3]])
    # [[1, 2, 3]]
    in_flat_tree = spec.flatten_up_to(in_tree)
    in_flat_batched = tree.leaves(in_batched)
    # NOTE(asem): iterate over the flat version and index iff its batched
    # and broadcast otherwise
    zipped = zip(in_flat_tree, in_flat_batched)
    leaves_i = (index(leaf, b) if is_batched else leaf for leaf, is_batched in zipped)
    return spec.unflatten(leaves_i)


def batch_spec(in_tree: Tree, in_batched: Tree[bool], /) -> tree.PyTreeSpec | None:
    """Return the common output container spec for batched leaves.

    Batch rules use this to decide whether an operation has any batched inputs
    and how per-example results should be repacked. If no input leaf is batched,
    ``None`` is returned.

    Args:
        in_tree: Tree containing batched and non-batched leaves.
        in_batched: Tree of booleans with the same outer structure as
            ``in_tree``. ``True`` marks a batched leaf.

    Returns:
        The common pytree spec of all batched leaves, or ``None`` if there are no batched leaves.

    Raises:
        AssertionError: If batched leaves do not share the same container structure.
    """
    # NOTE(asem): return the common container pytreespec of batched leaves.
    # returns None if no leaves are batched.
    # >>> in_tree = ("a", "b", "c")
    # >>> in_batched = True
    # >>> batch_repack(in_tree, in_batched, ["x", "y", "z"])
    # ('x', 'y', 'z')

    # NOTE(asem): this function will error if the batch continer types are mismatched
    # this design choice is ensure that a \optimes b over a consistent batch container type
    # will be wrapped in the same container type. otherwise, we have to pick one container type
    # arbitrarily which can lead to confusing behavior.
    is_axis_spec = lambda x: isinstance(x, bool)
    spec = tree.structure(in_batched, is_leaf=is_axis_spec)
    batched_leaves = tree.leaves(in_batched, is_leaf=is_axis_spec)
    tree_leaves = spec.flatten_up_to(in_tree)
    specs = []
    for v, b in zip(tree_leaves, batched_leaves):
        b and specs.append(tree.structure(v, is_leaf=lambda x: x is not v))
    if not specs:
        return None
    s0, *rest = specs
    # NOTE(asem): ensure all batched leaves have the same container type
    # this avoids ambiguity during batch repack of the output.
    assert all(s0 == s for s in rest), "Mismatched container types among batched leaves."
    return s0


def batch_transpose(batch_size: int, in_batched: Tree[bool], in_tree: Tree, /) -> Tree:
    assert batch_size, f"{batch_size=} must be > 0"
    spec = tree.structure(in_tree, is_leaf=lambda x: x is not in_tree)
    rows = spec.flatten_up_to(in_tree)
    if len(rows) != batch_size:
        raise ValueError(f"Expected {batch_size} rows, got {len(rows)}")
    return tree.stack(spec, in_batched, rows)
