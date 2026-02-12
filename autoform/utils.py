"""Utility functions for autoform"""

from __future__ import annotations

import functools as ft
from collections.abc import Awaitable, Callable
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import optree.pytree
from annotated_types import Len
from optree import PyTreeSpec
from pydantic import BaseModel, ConfigDict

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


def tree_index(node: Tree, b: int, /) -> Tree:
    # NOTE(asem): index a struct without indexing support
    # useful to deal with arbitrary pytrees
    children, *_ = treelib.flatten_one_level(node)
    return children[b]


def pack_user_input(*args, **kwargs) -> Tree:
    # NOTE(asem): pack args/kwargs into a single tree for user-bind interface.
    # useful to avoid dealing with args/kwargs unpacking at the IR level.
    if kwargs:
        return (*args, kwargs)
    if len(args) == 1:
        return args[0]
    return args


# ==================================================================================================
# STRUCT
# ==================================================================================================


def check_struct_field_type(cls_name: str, field_name: str, tp: type) -> None:

    leaf_types: set[type] = {int, bool, float, str}
    container_types: set[type] = {list, tuple}

    def check(tp: type) -> None:
        # NOTE(asem): scalar types case
        # >>> class MyStruct(Struct):
        # ...     a: int
        # ...     b: str
        if tp in leaf_types:
            return
        # NOTE(asem): nested Struct case
        # >>> class Nested(Struct):
        # ...     x: int
        # >>> class MyStruct(Struct):
        # ...     n: Nested
        if isinstance(tp, type) and issubclass(tp, Struct):
            return
        origin = get_origin(tp)
        # NOTE(asem): Literal[v, ...] case
        # >>> from typing import Literal
        # >>> class MyStruct(Struct):
        # ...     choice: Literal[1, 2, 3]  # all values share one type of leaf type
        if origin is Literal:
            args = get_args(tp)
            if len(args) == 0:
                raise TypeError(f"{tp!r}: Literal must have at least one value")
            types = set(type(v) for v in args)
            if len(types) != 1:
                raise TypeError(f"{tp!r}: Literal values must share one type, got {types}")
            if not isinstance(args[0], tuple(leaf_types)):
                raise TypeError(f"{tp!r}: Literal base type must be str/int/float/bool")
            return
        # NOTE(asem): Annotated[list[E], Len(n, n)] — fixed-size list case
        # >>> from typing import Annotated
        # >>> from annotated_types import Len
        # >>> class MyStruct(Struct):
        # ...     scores: Annotated[list[int], Len(3, 3)]
        if origin is Annotated:
            inner, *extras = get_args(tp)
            if get_origin(inner) in container_types:
                lens = [e for e in extras if isinstance(e, Len)]
                if len(lens) != 1:
                    raise TypeError(f"{tp!r}: No ``Len`` constraint for containers")
                if lens[0].min_length != lens[0].max_length:
                    raise TypeError(f"{tp!r}: ``Len`` must be fixed size (min == max)")
                inner_args = get_args(inner)
                inner_origin = get_origin(inner)
                if inner_origin is list:
                    if len(inner_args) != 1:
                        raise TypeError(f"{tp!r}: list must have exactly one type argument")
                    elem_type = inner_args[0]
                else:
                    # tuple[T, ...] form
                    if len(inner_args) != 2 or inner_args[1] is not Ellipsis:
                        raise TypeError(f"{tp!r}: tuple must be tuple[T, ...] form")
                    elem_type = inner_args[0]
                check(elem_type)
                return
        raise TypeError(
            f"Struct field '{cls_name}.{field_name}' has invalid type {tp!r}. Allowed:\n"
            "  1. str, int, float, bool\n"
            "  2. Struct subclass\n"
            "  3. Literal[v, ...] (single type)\n"
            "  4. Annotated[list[T], Len(n, n)] (T satisfies 1-3)"
        )

    check(tp)


@ft.partial(lru_cache, maxsize=1024)
def struct_type_tree[T: type[BaseModel]](tp: T) -> T:
    # NOTE(asem): construct a pytree type tree for Struct types.
    # >>> class Inner(af.Struct):
    # ...     x: str
    # ...     y: Annotated[list[int], Len(2, 2)]
    # >>> class MyStruct(af.Struct):
    # ...     a: int
    # ...     b: Inner
    # ...     c: Literal["option1", "option2"]
    # >>> struct_type_tree(MyStruct)
    # MyStruct(
    #     a=int,
    #     b=Inner(
    #         x=str,
    #         y=(int, int),
    #     ),
    #     c=str,
    # )
    if isinstance(tp, type) and issubclass(tp, Struct):
        hints = get_type_hints(tp, include_extras=True)
        names, children = [], []
        for name in tp.__annotations__:
            if not (get_origin(t := hints[name]) is ClassVar):
                names.append(name)
                children.append(struct_type_tree(t))
        return unflatten_struct(tp, tuple(names), tuple(children))
    if (origin := get_origin(tp)) is Literal:
        tp0, *_ = get_args(tp)
        return type(tp0)
    if origin is Annotated:
        inner, *extras = get_args(tp)
        lens = [e for e in extras if isinstance(e, Len)]
        if lens and get_origin(inner) in (list, tuple):
            n = lens[0].min_length
            elem_type, *_ = get_args(inner)
            output = (struct_type_tree(elem_type) for _ in range(n))
            container_types = get_origin(inner)
            return container_types(output)
    return tp


def flatten_struct(obj: Struct) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    fields = type(obj).model_fields
    return (tuple(getattr(obj, k) for k in fields), tuple(fields))


def unflatten_struct(cls: type[Struct], keys: tuple[str, ...], children: tuple) -> Struct:
    obj = object.__new__(cls)
    for k, v in zip(keys, children, strict=True):
        object.__setattr__(obj, k, v)
    return obj


class Struct(BaseModel):
    """Pytree-compatible Pydantic model with typed fields.

    No validation during ``model_construct()`` — accepts any values
    (including IRVar during tracing).

    Allowed field types:

        ╔══════╦═════════════════════════════════════╦══════════════════════════════════════════╗
        ║ Rule ║ Form                                ║ Example                                  ║
        ╠══════╬═════════════════════════════════════╬══════════════════════════════════════════╣
        ║  1   ║ str, int, float, bool               ║ name: str                                ║
        ║  2   ║ ``Struct`` subclass                 ║ inner: Inner                             ║
        ║  3   ║ Literal[v, ...] (single type)       ║ status: Literal["a", "b"]                ║
        ║  4   ║ Annotated[list[E], Len(n, n)]       ║ scores: Annotated[list[int], Len(3, 3)]  ║
        ║  5   ║ Annotated[tuple[E, ...], Len(n, n)] ║ scores: Annotated[list[int], Len(3, 3)]  ║
        ╚══════╩═════════════════════════════════════╩══════════════════════════════════════════╝

        E must satisfy rule 1, 2, or 3.

    Note:
        Container types must use Len(n, n) from annotated_types because at trace time the list is
        unrolled into n individual IRVar leaves. A variable-length list would make the pytree
        structure unknown until runtime.

    Example:

        >>> import autoform as af
        >>> from typing import Literal
        >>> from annotated_types import Len
        >>> class Inner(af.Struct):
        ...     x: str
        >>> class Answer(af.Struct):
        ...     reasoning: str
        ...     answer: int
        ...     status: Literal["correct", "incorrect"]
        ...     scores: Annotated[list[int], Len(3, 3)]
        ...     inner: Inner

    """

    model_config = ConfigDict(frozen=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        hints = get_type_hints(cls, include_extras=True)
        for name in cls.__annotations__:
            if not (get_origin(tp := hints[name]) is ClassVar):
                check_struct_field_type(cls.__name__, name, tp)
        treelib.register_node(cls, flatten_struct, ft.partial(unflatten_struct, cls))

    def __hash__(self):
        leaves, struct = treelib.flatten(self)
        return hash((struct, tuple(leaves)))

    def __eq__(self, other):
        if not isinstance(other, Struct):
            return False
        lhs_leaves, lhs_struct = treelib.flatten(self)
        rhs_leaves, rhs_struct = treelib.flatten(other)
        if lhs_struct != rhs_struct:
            return False
        return lhs_leaves == rhs_leaves


# ==================================================================================================
# BATCH UTILITIES
# ==================================================================================================


def batch_index(in_tree: Tree, in_batched: Tree[bool], b: int, /) -> Tree:
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
    spec = treelib.structure(in_batched)
    # NOTE(asem): flatten in_tree to match in_batched structure
    # >>> spec = treelib.structure([1, 2, 3])
    # >>> spec.flatten_up_to([1, [2, 3]])
    # [[1, 2, 3]]
    flat_in_tree = spec.flatten_up_to(in_tree)
    flat_in_batched = treelib.leaves(in_batched)
    # NOTE(asem): iterate over the flat version and index iff its batched
    # and broadcast otherwise
    zipped = zip(flat_in_tree, flat_in_batched, strict=True)
    leaves_i = (tree_index(leaf, b) if is_batched else leaf for leaf, is_batched in zipped)
    return spec.unflatten(leaves_i)


def batch_spec(in_tree: Tree, in_batched: Tree[bool], /) -> PyTreeSpec | None:
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
    spec = treelib.structure(in_batched, is_leaf=is_axis_spec)
    batched_leaves = treelib.leaves(in_batched, is_leaf=is_axis_spec)
    tree_leaves = spec.flatten_up_to(in_tree)
    specs = []
    for v, b in zip(tree_leaves, batched_leaves, strict=True):
        b and specs.append(treelib.structure(v, is_leaf=lambda x: x is not v))
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
