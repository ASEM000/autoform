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

"""Named-axis primitives for batched programs."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from autoform.ad import Zero
from autoform.core import (
    AVal,
    AxisFrame,
    Prim,
    TypedAVal,
    abstract_rules,
    batch_rules,
    get_axis,
    impl_rules,
    ir_aval,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    require_current_axis,
)
from autoform.utils import Tree, asyncify, treelib

__all__ = [
    "GatheredAVal",
    "axis_gather",
    "axis_index",
    "axis_size",
]


class GatheredAVal(AVal):
    """Abstract value for a gathered runtime batch container."""

    __slots__ = ["elem_aval"]

    def __init__(self, elem_aval: AVal | Any):
        self.elem_aval = elem_aval

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.elem_aval!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, GatheredAVal) and self.elem_aval == other.elem_aval

    def __hash__(self) -> int:
        return hash((type(self), self.elem_aval))


def tree_zero_like(tree: Tree) -> Tree:
    return treelib.map(lambda leaf: Zero(type(leaf)), tree)


def map_batched_leaves(func, values: Tree, batched: Tree[bool]) -> Tree:
    is_axis_spec = lambda x: isinstance(x, bool)
    spec = treelib.structure(batched, is_leaf=is_axis_spec)
    flat_values = spec.flatten_up_to(values)
    flat_batched = treelib.leaves(batched, is_leaf=is_axis_spec)
    mapped = (func(v, b) for v, b in zip(flat_values, flat_batched, strict=True))
    return spec.unflatten(mapped)


def false_tree_like_batched(batched: Tree[bool]) -> Tree[bool]:
    is_axis_spec = lambda x: isinstance(x, bool)
    spec = treelib.structure(batched, is_leaf=is_axis_spec)
    return spec.unflatten(False for _ in treelib.leaves(batched, is_leaf=is_axis_spec))


# ==================================================================================================
# AXIS SIZE
# ==================================================================================================


axis_size_p = Prim("axis_size")


def axis_size(*, axis_name: Hashable) -> int:
    """Return the size of a named batch axis."""

    return axis_size_p.bind((), axis_name=axis_name)


def impl_axis_size(in_tree: Tree, /, *, axis_name: Hashable) -> int:
    del in_tree
    return get_axis(axis_name).size


def abstract_axis_size(in_tree: Tree, /, *, axis_name: Hashable) -> TypedAVal:
    del in_tree, axis_name
    return TypedAVal(int)


def push_axis_size(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[int, Zero]:
    del in_tree
    return axis_size_p.bind((), axis_name=axis_name), Zero(int)


def pull_fwd_axis_size(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[int, None]:
    del in_tree
    return axis_size_p.bind((), axis_name=axis_name), None


def pull_bwd_axis_size(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[()]:
    del in_tree, axis_name
    return ()


def batch_axis_size(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[int, bool]:
    batch_size, _, _ = in_tree
    require_current_axis(axis_name)
    return batch_size, False


impl_rules.set(axis_size_p, impl_axis_size)
impl_rules.aset(axis_size_p, asyncify(impl_axis_size))
abstract_rules.set(axis_size_p, abstract_axis_size)
push_rules.set(axis_size_p, push_axis_size)
push_rules.aset(axis_size_p, asyncify(push_axis_size))
pull_fwd_rules.set(axis_size_p, pull_fwd_axis_size)
pull_fwd_rules.aset(axis_size_p, asyncify(pull_fwd_axis_size))
pull_bwd_rules.set(axis_size_p, pull_bwd_axis_size)
pull_bwd_rules.aset(axis_size_p, asyncify(pull_bwd_axis_size))
batch_rules.set(axis_size_p, batch_axis_size)
batch_rules.aset(axis_size_p, asyncify(batch_axis_size))


# ==================================================================================================
# AXIS INDEX
# ==================================================================================================


axis_index_p = Prim("axis_index")


def axis_index(*, axis_name: Hashable) -> int:
    """Return each element's index along a named batch axis."""

    return axis_index_p.bind((), axis_name=axis_name)


def impl_axis_index(in_tree: Tree, /, *, axis_name: Hashable) -> int:
    del in_tree, axis_name
    raise RuntimeError("axis_index requires batched execution")


def abstract_axis_index(in_tree: Tree, /, *, axis_name: Hashable) -> TypedAVal:
    del in_tree, axis_name
    return TypedAVal(int)


def push_axis_index(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[int, Zero]:
    del in_tree
    return axis_index_p.bind((), axis_name=axis_name), Zero(int)


def pull_fwd_axis_index(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[int, None]:
    del in_tree
    return axis_index_p.bind((), axis_name=axis_name), None


def pull_bwd_axis_index(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[()]:
    del in_tree, axis_name
    return ()


def batch_axis_index(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[list[int], bool]:
    batch_size, _, _ = in_tree
    require_current_axis(axis_name)
    return list(range(batch_size)), True


impl_rules.set(axis_index_p, impl_axis_index)
impl_rules.aset(axis_index_p, asyncify(impl_axis_index))
abstract_rules.set(axis_index_p, abstract_axis_index)
push_rules.set(axis_index_p, push_axis_index)
push_rules.aset(axis_index_p, asyncify(push_axis_index))
pull_fwd_rules.set(axis_index_p, pull_fwd_axis_index)
pull_fwd_rules.aset(axis_index_p, asyncify(pull_fwd_axis_index))
pull_bwd_rules.set(axis_index_p, pull_bwd_axis_index)
pull_bwd_rules.aset(axis_index_p, asyncify(pull_bwd_axis_index))
batch_rules.set(axis_index_p, batch_axis_index)
batch_rules.aset(axis_index_p, asyncify(batch_axis_index))


# ==================================================================================================
# AXIS GATHER
# ==================================================================================================


axis_gather_p = Prim("axis_gather")


def axis_gather(x: Tree, *, axis_name: Hashable) -> Tree:
    """Gather ``x`` across every element of a named batch axis.

    Inside a named ``batch``, ``axis_gather`` makes the current axis values
    available to each mapped element. The gathered value is marked unbatched by
    the batch interpreter, so returning it from the batched program broadcasts
    the same gathered value once per mapped element.

    Example:
        >>> import autoform as af
        >>> def program(item):
        ...     return af.axis_gather(item, axis_name="items")
        >>> ir = af.trace(program)("item")
        >>> batched = af.batch(ir, axis_name="items")
        >>> batched.call(["a", "b"])
        [['a', 'b'], ['a', 'b']]
    """

    return axis_gather_p.bind(x, axis_name=axis_name)


def impl_axis_gather(x: Tree, /, *, axis_name: Hashable) -> Tree:
    require_current_axis(axis_name)
    raise RuntimeError("axis_gather requires batched execution")


def abstract_axis_gather(x: Tree, /, *, axis_name: Hashable) -> Tree:
    del axis_name
    return treelib.map(lambda leaf: GatheredAVal(ir_aval(leaf)), x)


def push_axis_gather(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    return (
        axis_gather_p.bind(primals, axis_name=axis_name),
        axis_gather_p.bind(tangents, axis_name=axis_name),
    )


def pull_fwd_axis_gather(x: Tree, /, *, axis_name: Hashable) -> tuple[Tree, None]:
    return axis_gather_p.bind(x, axis_name=axis_name), x


def pull_bwd_axis_gather(in_tree: Tree, /, *, axis_name: Hashable) -> Tree:
    del axis_name
    residuals, _ = in_tree
    return tree_zero_like(residuals)


def gather_leaf(value: Any, is_batched: bool, frame: AxisFrame) -> Any:
    return value if is_batched else frame.spec.unflatten([value] * frame.size)


def batch_axis_gather(in_tree: Tree, /, *, axis_name: Hashable) -> tuple[Tree, Tree]:
    _, in_batched, x = in_tree
    frame = require_current_axis(axis_name)
    out = map_batched_leaves(
        lambda leaf, batched: gather_leaf(leaf, batched, frame),
        x,
        in_batched,
    )
    return out, false_tree_like_batched(in_batched)


impl_rules.set(axis_gather_p, impl_axis_gather)
impl_rules.aset(axis_gather_p, asyncify(impl_axis_gather))
abstract_rules.set(axis_gather_p, abstract_axis_gather)
push_rules.set(axis_gather_p, push_axis_gather)
push_rules.aset(axis_gather_p, asyncify(push_axis_gather))
pull_fwd_rules.set(axis_gather_p, pull_fwd_axis_gather)
pull_fwd_rules.aset(axis_gather_p, asyncify(pull_fwd_axis_gather))
pull_bwd_rules.set(axis_gather_p, pull_bwd_axis_gather)
pull_bwd_rules.aset(axis_gather_p, asyncify(pull_bwd_axis_gather))
batch_rules.set(axis_gather_p, batch_axis_gather)
batch_rules.aset(axis_gather_p, asyncify(batch_axis_gather))
