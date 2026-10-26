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

"""Pytree utilities"""

import builtins
from collections.abc import Callable, Iterable
from typing import Any, dataclass_transform, overload

import optree
import optree.dataclasses
from optree import PyTreeSpec
from optree.dataclasses import field as field

NAMESPACE = "OPTREE_AUTOFORM_NAMESPACE"
type Tree[T] = Any
type Predicate = Callable[[Any], bool] | None
type Children = Iterable[Any]
type FlattenResult = tuple[Children, Any] | tuple[Children, Any, Children]
type Flatten[T] = Callable[[T], FlattenResult]
type Unflatten[T] = Callable[[Any, Children], T]


def flatten[T](x: Tree[T], /, *, is_leaf: Predicate = None) -> tuple[list[T], PyTreeSpec]:
    with optree.dict_insertion_ordered(True, namespace=NAMESPACE):
        return optree.tree_flatten(x, is_leaf=is_leaf, namespace=NAMESPACE)


def leaves[T](x: Tree[T], /, *, is_leaf: Predicate = None) -> list[T]:
    leaves, _ = flatten(x, is_leaf=is_leaf)
    return leaves


def structure(x: Tree, /, *, is_leaf: Predicate = None) -> PyTreeSpec:
    _, spec = flatten(x, is_leaf=is_leaf)
    return spec


def is_leaf(x: Tree, /) -> bool:
    return optree.tree_is_leaf(x, namespace=NAMESPACE)


def map[R](func: Callable[..., R], x: Tree, /, *xs: Tree, is_leaf: Predicate = None) -> Tree[R]:
    leaves, spec = flatten(x, is_leaf=is_leaf)
    flat_args = [leaves] + [spec.flatten_up_to(r) for r in xs]
    return spec.unflatten(builtins.map(func, *flat_args))


def broadcast_prefix(x: Tree, y: Tree, /, *, is_leaf: Predicate = None) -> Tree:
    def broadcast(x, y):
        spec = structure(y, is_leaf=is_leaf)
        return spec.unflatten([x] * spec.num_leaves)

    return map(broadcast, x, y, is_leaf=is_leaf)


def register_node[T](
    cls: type[T],
    flatten_func: Flatten[T],
    unflatten_func: Unflatten[T],
    /,
    *,
    path_entry_type: type[optree.PyTreeEntry] = optree.AutoEntry,
) -> type[T]:
    return optree.register_pytree_node(
        cls,
        flatten_func,
        unflatten_func,
        path_entry_type=path_entry_type,
        namespace=NAMESPACE,
    )


@overload
def dataclass[T](cls: type[T], /, **kwargs: bool) -> type[T]: ...


@overload
def dataclass[T](cls: None = None, /, **kwargs: bool) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(field_specifiers=(field,))
def dataclass[T](cls: type[T] | None = None, /, **kwargs: bool):
    return optree.dataclasses.dataclass(cls, namespace=NAMESPACE, **kwargs)


def unstack(spec: PyTreeSpec, axes: Tree[bool], x: Tree, /) -> list[Tree]:
    """Return one tree per item, indexing selected fields and retaining shared fields."""
    return [map(lambda u, x: get(x) if u else x, axes, x) for get in spec.accessors()]


def stack(spec: PyTreeSpec, shape: Tree, rows: Iterable[Tree], /) -> Tree:
    """Collect corresponding fields from rows into spec's container; shape defines the fields."""
    rows = list(rows)
    if len(rows) != spec.num_leaves:
        raise ValueError(f"Expected {spec.num_leaves} rows, got {len(rows)}")
    if not spec.num_leaves:
        raise ValueError("Cannot stack an empty container")
    return map(lambda _, *xs: spec.unflatten(xs), shape, *rows)


def mask(in_tree: Tree, use: Tree[bool], replace: Any = None) -> Tree:
    def func(node: Tree, use_node: bool):
        assert isinstance(use_node, bool)
        return node if use_node else replace

    return map(func, in_tree, use)


def partition(
    predicate: Callable[[Any], bool],
    in_tree: Tree,
    /,
    *,
    is_leaf: Predicate = None,
    fillvalue: Any = None,
) -> tuple[Tree, Tree]:
    flat, spec = flatten(in_tree, is_leaf=is_leaf)
    selected = [predicate(x) for x in flat]
    lhs = [x if keep else fillvalue for x, keep in zip(flat, selected, strict=True)]
    rhs = [fillvalue if keep else x for x, keep in zip(flat, selected, strict=True)]
    return spec.unflatten(lhs), spec.unflatten(rhs)


def equal(lhs: Tree, rhs: Tree, /) -> bool:
    """Return whether two pytrees have equal structure and leaves."""
    if structure(lhs) != structure(rhs):
        return False
    lhs_leaves = leaves(lhs)
    rhs_leaves = leaves(rhs)
    return all(l == r for l, r in zip(lhs_leaves, rhs_leaves, strict=True))
