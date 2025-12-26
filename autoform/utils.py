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


def index(tree: Tree, mask: Tree, i: int) -> Tree:
    """Index a tree according to a mask tree.

    Example:
        >>> tree, mask = [[1, 2, 3]], [True]
        >>> index(tree, mask, 0)
        [1]
    """
    spec = treelib.structure(mask)
    up_to_tree = spec.flatten_up_to(tree)
    flat_mask = treelib.leaves(mask)

    def select(leaf, take: bool):
        return leaf[i] if take else leaf

    selected = [select(leaf, take) for leaf, take in zip(up_to_tree, flat_mask, strict=True)]
    return spec.unflatten(selected)


def pack_user_input(*args, **kwargs) -> Tree:
    """Pack args/kwargs into a single tree for user-bind interface."""
    if kwargs:
        return (*args, kwargs)
    if len(args) == 1:
        return args[0]
    return args


def transpose_batch(batch_size: int, in_batched: Tree[bool], results: list[Tree]) -> Tree:
    """Transpose outer(inner) => inner(outer).

    Example:
        >>> import typing as tp
        >>> class Point(tp.NamedTuple):
        ...     x: int
        ...     y: int
        >>> batch_size = 3
        >>> in_batched = Point(x=True, y=True)
        >>> results = [Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)]
        >>> desired = Point(x=[1,3,5], y=[2,4,6])
        >>> transposed = transpose_batch(batch_size, in_batched, results)
        >>> transposed == desired
        True
    """
    # get spec from in_batched -> Point(*, *)
    inner_spec = treelib.structure(in_batched, is_leaf=lambda x: isinstance(x, bool))
    # flatten each result -> [[1, 2], [3, 4], [5, 6]]
    # make each inner result (e.g. [1, 2]) match inner_spec (Point(*, *))
    leaves_bi = [inner_spec.flatten_up_to(r) for r in results]
    # transpose leaves -> [[1, 3, 5], [2, 4, 6]]
    # note that in case batch_size=0 this will still work
    # it will produce [[], []] which is valid (zip(*...) is invalid here)
    leaves_ib = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(inner_spec.num_leaves)]
    return inner_spec.unflatten(leaves_ib)
