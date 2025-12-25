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
