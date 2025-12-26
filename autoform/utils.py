"""Utility functions for autoform"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections.abc import Callable

import optree.pytree


# ==================================================================================================
# PYTREE UTILITIES
# ==================================================================================================

PYTREE_NAMESPACE = "AUTOFORM"
treelib = optree.pytree.reexport(namespace=PYTREE_NAMESPACE)
type Tree[T] = tp.Any


def lru_cache[**P, R](func: Callable[P, R], maxsize: int = 256) -> Callable[P, R]:
    return tp.cast(Callable[P, R], ft.lru_cache(maxsize=maxsize)(func))


def unbatch_at(in_tree: Tree, in_batched: Tree[bool], b: int) -> Tree:
    # Extract item at index b from batched leaves, broadcast non-batched.
    # Inverse of transpose_batch: extracts a single item from each batched leaf
    # while keeping non-batched leaves unchanged.

    # Args:
    #     tree_ib: tree with batched leaves (index-batch order, each leaf is a list).
    #     batched: tree of bools indicating which leaves are batched.
    #     b: index to extract from batched leaves.

    # Example:
    #     >>> tree_ib, batched = [[1, 2, 3], "constant"], [True, False]
    #     >>> unbatch_at(tree_ib, batched, 0)
    #     [1, 'constant']
    spec = treelib.structure(in_batched)
    leaves_ib = spec.flatten_up_to(in_tree)
    flat_batched = treelib.leaves(in_batched)
    leaves_i = [
        leaf[b] if is_batched else leaf
        for leaf, is_batched in zip(leaves_ib, flat_batched, strict=True)
    ]
    return spec.unflatten(leaves_i)


def pack_user_input(*args, **kwargs) -> Tree:
    # Pack args/kwargs into a single tree for user-bind interface.
    if kwargs:
        return (*args, kwargs)
    if len(args) == 1:
        return args[0]
    return args


def transpose_batch(batch_size: int, in_batched: Tree[bool], in_tree: list[Tree]) -> Tree:
    # Transpose outer(inner) => inner(outer).
    # Example:
    #     >>> import typing as tp
    #     >>> class Point(tp.NamedTuple):
    #     ...     x: int
    #     ...     y: int
    #     >>> batch_size = 3
    #     >>> in_batched = Point(x=True, y=True)
    #     >>> in_tree = [Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)]
    #     >>> desired = Point(x=[1,3,5], y=[2,4,6])
    #     >>> transposed = transpose_batch(batch_size, in_batched, in_tree)
    #     >>> transposed == desired
    #     True
    # get spec from in_batched -> Point(*, *)
    inner_spec = treelib.structure(in_batched, is_leaf=lambda x: isinstance(x, bool))
    # flatten each result -> [[1, 2], [3, 4], [5, 6]]
    # make each inner result (e.g. [1, 2]) match inner_spec (Point(*, *))
    leaves_bi = [inner_spec.flatten_up_to(r) for r in in_tree]
    # transpose leaves -> [[1, 3, 5], [2, 4, 6]]
    # note that in case batch_size=0 this will still work
    # it will produce [[], []] which is valid (zip(*...) is invalid here)
    leaves_ib = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(inner_spec.num_leaves)]
    return inner_spec.unflatten(leaves_ib)
