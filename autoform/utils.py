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

import functools as ft
from collections.abc import Awaitable, Callable
from typing import Any, cast

import optree.pytree
from optree import PyTreeSpec

__all__ = ["PYTREE_NAMESPACE", "tree", "tree_equal", "batch_index", "batch_spec", "batch_transpose"]

# ==================================================================================================
# ASYNC UTILITIES
# ==================================================================================================


def asyncify[**P, R](func: Callable[P, R], /) -> Callable[P, Awaitable[R]]:
    @ft.wraps(func)
    async def afunc(*args, **kwargs):
        return func(*args, **kwargs)

    return afunc


# ==================================================================================================
# PYTREE UTILITIES
# ==================================================================================================

PYTREE_NAMESPACE = "OPTREE_AUTOFORM_NAMESPACE"
tree = optree.pytree.reexport(namespace=PYTREE_NAMESPACE)
type Tree[T] = Any


def lru_cache[**P, R](func: Callable[P, R], maxsize: int = 256) -> Callable[P, R]:
    return cast(Callable[P, R], ft.lru_cache(maxsize=maxsize)(func))


def index(node: Tree, b: int, /) -> Tree:
    # NOTE(asem): index a struct without indexing support
    # useful to deal with arbitrary pytrees
    children = tree.leaves(node, is_leaf=lambda x: id(x) != id(node), none_is_leaf=True)
    return children[b]


def mask(in_tree: Tree, use: Tree[bool], replace: Any = None) -> Tree:
    def func(node: Tree, use_node: bool):
        assert isinstance(use_node, bool)
        return node if use_node else replace

    return tree.map(func, in_tree, use)


def tree_equal(lhs: Tree, rhs: Tree, /) -> bool:
    """Return whether two pytrees have equal structure and leaves."""
    if tree.structure(lhs) != tree.structure(rhs):
        return False
    lhs_leaves = tree.leaves(lhs)
    rhs_leaves = tree.leaves(rhs)
    return all(
        lhs_leaf == rhs_leaf for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves, strict=True)
    )


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
    flat_in_tree = spec.flatten_up_to(in_tree)
    flat_in_batched = tree.leaves(in_batched)
    # NOTE(asem): iterate over the flat version and index iff its batched
    # and broadcast otherwise
    zipped = zip(flat_in_tree, flat_in_batched, strict=True)
    leaves_i = (index(leaf, b) if is_batched else leaf for leaf, is_batched in zipped)
    return spec.unflatten(leaves_i)


def batch_spec(in_tree: Tree, in_batched: Tree[bool], /) -> PyTreeSpec | None:
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
    for v, b in zip(tree_leaves, batched_leaves, strict=True):
        b and specs.append(tree.structure(v, is_leaf=lambda x: x is not v))
    if not specs:
        return None
    s0, *rest = specs
    # NOTE(asem): ensure all batched leaves have the same container type
    # this avoids ambiguity during batch repack of the output.
    assert all(s0 == s for s in rest), "Mismatched container types among batched leaves."
    return s0


def batch_transpose(batch_size: int, in_batched: Tree[bool], in_tree: Tree, /) -> Tree:
    # NOTE(asem): AoS -> SoA
    # Example (used throughout):
    #   in_tree    = [Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)]
    #   in_batched = Point(x=True, y=True)
    #   batch_size = 3
    #   desired    = Point(x=[1, 3, 5], y=[2, 4, 6])

    # inner_spec = Point(x=*, y=*)
    # outer_spec = [*, *, *]
    # items = [Point(1,2), Point(3,4), Point(5,6)]
    # leaves_bi = [[1, 2], [3, 4], [5, 6]]  (batch, leaves)
    # leaves_ib = [[1, 3, 5], [2, 4, 6]]  (leaves, batch)
    # leaf_batches = [[1, 3, 5], [2, 4, 6]]
    # result = Point(x=[1, 3, 5], y=[2, 4, 6])
    assert batch_size, f"{batch_size=} must be > 0"
    inner_spec = tree.structure(in_batched, is_leaf=lambda x: isinstance(x, bool))
    outer_spec = tree.structure(in_tree, is_leaf=lambda x: x is not in_tree)
    first_level_leaves = tree.leaves(in_tree, is_leaf=lambda x: x is not in_tree)

    # NOTE(asem): flatten_up_to (not leaves) because we want to stop at the boundary
    # defined by inner_spec. if leaves are containers (e.g., lists), regular leaves()
    # would descend into them. flatten_up_to respects the spec and stops at each leaf slot.
    # >>> item = Batch(items=[1, 2, 3], mask=[True, False])
    # >>> inner_spec = Batch(items=*, mask=*)
    # >>> tree.leaves(item) -> [1, 2, 3, True, False]
    # >>> inner_spec.flatten_up_to(item) -> [[1, 2, 3], [True, False]]
    leaves_bi = [inner_spec.flatten_up_to(item) for item in first_level_leaves]

    # NOTE(asem): transpose without indexing
    # leaf_ids prevents transpose from descending INTO the leaf values.
    # >>> leaves_bi = [[[1,2], [3,4]], [[5,6], [7,8]]]  # leaves are lists
    # >>> transpose(leaves_bi) without is_leaf descends into [1,2]
    # >>> transpose(leaves_bi) with is_leaf stops at [1,2]
    ids = {id(leaf) for row in leaves_bi for leaf in row}

    leaves_ib = tree.transpose(
        tree.structure([object()] * batch_size),
        tree.structure([object()] * inner_spec.num_leaves),
        leaves_bi,
        is_leaf=lambda x: id(x) in ids,
    )

    leaf_batches = [outer_spec.unflatten(col) for col in leaves_ib]
    result = inner_spec.unflatten(leaf_batches)
    return result
