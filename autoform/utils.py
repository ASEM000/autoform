"""Utility functions for autoform"""

from __future__ import annotations

import functools as ft
from collections.abc import Awaitable, Callable
from typing import Any, cast

import optree.pytree
import pydantic

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

PYTREE_NAMESPACE = "AUTOFORM"
treelib = optree.pytree.reexport(namespace=PYTREE_NAMESPACE)
type Tree[T] = Any


def lru_cache[**P, R](func: Callable[P, R], maxsize: int = 256) -> Callable[P, R]:
    return cast(Callable[P, R], ft.lru_cache(maxsize=maxsize)(func))


def index_tree_at(node: Tree, b: int, /) -> Tree:
    # NOTE(asem): index a struct without indexing support
    # useful to deal with arbitrary pytrees
    children, *_ = treelib.flatten_one_level(node)
    return children[b]


def unbatch_at(in_tree: Tree, in_batched: Tree[bool], b: int, /) -> Tree:
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
    # NOTE(asem): flatten in_tree to match in_batched structure
    # Example:
    #     >>> spec = treelib.structure([1, 2, 3])
    #     >>> spec.flatten_up_to([1, [2, 3]])
    #     [[1, 2, 3]]
    flat_in_tree = spec.flatten_up_to(in_tree)
    flat_in_batched = treelib.leaves(in_batched)
    # NOTE(asem): iterate over the flat version and index iff its batched
    # and broadcast otherwise
    zipped = zip(flat_in_tree, flat_in_batched, strict=True)
    leaves_i = (index_tree_at(leaf, b) if is_batched else leaf for leaf, is_batched in zipped)
    return spec.unflatten(leaves_i)


def pack_user_input(*args, **kwargs) -> Tree:
    # NOTE(asem): pack args/kwargs into a single tree for user-bind interface.
    # useful to avoid dealing with args/kwargs unpacking at the IR level.
    if kwargs:
        return (*args, kwargs)
    if len(args) == 1:
        return args[0]
    return args


def rebatch(in_tree: Tree, in_batched: Tree[bool], out_flat: list, /) -> Tree:
    # NOTE(asem): wrap results in the container type inferred from the first batched input.
    # Example:
    #     >>> in_tree = ("a", "b", "c")
    #     >>> in_batched = True
    #     >>> rebatch(in_tree, in_batched, ["x", "y", "z"])
    #     ('x', 'y', 'z')
    is_bool = lambda x: isinstance(x, bool)
    spec = treelib.structure(in_batched, is_leaf=is_bool)
    batched_leaves = treelib.leaves(in_batched, is_leaf=is_bool)
    tree_leaves = spec.flatten_up_to(in_tree)
    for v, b in zip(tree_leaves, batched_leaves, strict=True):
        if b:
            container_spec = treelib.structure(v, is_leaf=lambda x: x is not v)
            return container_spec.unflatten(out_flat)
    return out_flat


def transpose_batch(batch_size: int, in_batched: Tree[bool], in_tree: Tree, /) -> Tree:
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
    inner_spec = treelib.structure(in_batched, is_leaf=lambda x: isinstance(x, bool))
    outer_spec = treelib.structure(in_tree, is_leaf=lambda x: x is not in_tree)
    first_level_leaves = treelib.leaves(in_tree, is_leaf=lambda x: x is not in_tree)

    # NOTE(asem): flatten_up_to (not leaves) because we want to stop at the boundary
    # defined by inner_spec. if leaves are containers (e.g., lists), regular leaves()
    # would descend into them. flatten_up_to respects the spec and stops at each leaf slot.
    # >>> item = Batch(items=[1, 2, 3], mask=[True, False])
    # >>> inner_spec = Batch(items=*, mask=*)
    # >>> treelib.leaves(item) -> [1, 2, 3, True, False]
    # >>> inner_spec.flatten_up_to(item) -> [[1, 2, 3], [True, False]]
    leaves_bi = [inner_spec.flatten_up_to(item) for item in first_level_leaves]

    # NOTE(asem): transpose without indexing
    # leaf_ids prevents transpose from descending INTO the leaf values.
    # >>> leaves_bi = [[[1,2], [3,4]], [[5,6], [7,8]]]  # leaves are lists
    # >>> transpose(leaves_bi) without is_leaf descends into [1,2]
    # >>> transpose(leaves_bi) with is_leaf stops at [1,2]
    ids = {id(leaf) for row in leaves_bi for leaf in row}

    leaves_ib = treelib.transpose(
        treelib.structure([object()] * batch_size),
        treelib.structure([object()] * inner_spec.num_leaves),
        leaves_bi,
        is_leaf=lambda x: id(x) in ids,
    )

    leaf_batches = [outer_spec.unflatten(col) for col in leaves_ib]
    result = inner_spec.unflatten(leaf_batches)
    return result


# ==================================================================================================
# STRUCT
# ==================================================================================================


def flatten_struct(obj: pydantic.BaseModel) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    assert isinstance(obj, pydantic.BaseModel)
    fields = type(obj).model_fields
    return (tuple(getattr(obj, k) for k in fields), tuple(fields))


def unflatten_struct[T: type[pydantic.BaseModel]](cls: T, keys: tuple[str, ...], children) -> T:
    assert isinstance(keys, tuple)
    assert isinstance(children, tuple)
    return cls.model_construct(**dict(zip(keys, children, strict=True)))


class Struct(pydantic.BaseModel):
    """Pydantic BaseModel that is also a PyTree.

    Auto-sets subclasses as pytrees.
    Uses ``model_construct`` in unflatten to skip validation.

    Example:
        >>> class Answer(Struct):
        ...     reasoning: str
        ...     answer: int
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        treelib.register_node(cls, flatten_struct, ft.partial(unflatten_struct, cls))

    def __hash__(self):
        structure = treelib.structure(self)
        leaves = tuple(treelib.leaves(self))
        return hash((structure, leaves))

    def __eq__(self, other):
        if not isinstance(other, Struct):
            return False
        lhs_structure = treelib.structure(self)
        rhs_structure = treelib.structure(other)
        if lhs_structure != rhs_structure:
            return False
        lhs_flat = treelib.leaves(self)
        rhs_flat = treelib.leaves(other)
        return lhs_flat == rhs_flat
