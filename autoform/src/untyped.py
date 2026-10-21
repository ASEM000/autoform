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

"""Untyped values."""

from __future__ import annotations

import functools as ft
from typing import Any, Self

import autoform.src.core as core
import autoform.src.pp as pp
import autoform.src.stage as stage
import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]


class AUntyped(core.AVal):
    __slots__ = []

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def join(self, other, /) -> Self:
        match other:
            case AUntyped() | core.Zero(aval=AUntyped()) | Untyped():
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Untyped:
    __slots__ = ["value"]

    def __init__(self, value: Any, /):
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


utils.tree.register_node(
    Untyped,
    lambda untyped: ((untyped.value,), None, ("value",)),
    lambda _, children: Untyped(*children),
)
stage.register_trace_type(Untyped, lambda _: AUntyped())


@ft.partial(pp.register, AUntyped)
def pretty_auntyped(aval):
    return pp.style(type(aval).__name__, pp.Theme.TYPE), "()"


core.s_primal.set(AUntyped, lambda aval: aval)
core.s_tangent.set(AUntyped, lambda aval: aval)
core.s_cotangent.set(AUntyped, lambda aval: aval)


# ==================================================================================================
# UNTYPED
# ==================================================================================================


p_untyped = core.Prim("untyped")


def untyped(tree: Tree, /) -> Untyped:
    """Wrap input as a single untyped value.

    Useful when ``tree`` is an arbitrary Python object or its structure is only known at run time.

    Args:
        tree: any Python value.

    Returns:
        An :class:`Untyped` holding ``tree``.

    Example:
        >>> import autoform as af
        >>> handle = af.untyped({"draft": ["hello"]})
        >>> handle
        Untyped()
        >>> handle.value
        {'draft': ['hello']}
        >>> af.trace(lambda x, h: x)("seed", handle).in_tree[1].aval
        AUntyped()
    """
    return p_untyped.bind(tree)


def impl_untyped(in_tree: Tree, /) -> Untyped | core.Zero:
    match in_tree:
        case core.Zero():
            return core.Zero(AUntyped())
        case Untyped():
            raise TypeError(f"Expected a typed value, got {in_tree!r}")
    return Untyped(in_tree)


def abstract_untyped(in_tree: Tree, /) -> AUntyped:
    match in_tree:
        case AUntyped() | Untyped():
            raise TypeError(f"Expected a typed value, got {in_tree!r}")
    return AUntyped()


def push_untyped(in_tree: Tree, /) -> tuple[Untyped, Untyped]:
    p, t = in_tree
    return untyped(p), untyped(t)


def pull_fwd_untyped(in_tree: Tree, /) -> tuple[Untyped, None]:
    return untyped(in_tree), None


def pull_bwd_untyped(in_tree: Tree, /) -> Tree:
    _, c = in_tree
    return core.Zero(AUntyped()) if isinstance(c, core.Zero) else c.value


def batch_untyped(in_tree: Tree, /) -> tuple[Tree, bool]:
    size, batched, x = in_tree
    if not batched:
        return untyped(x), False
    spec = utils.batch_spec(x, batched)
    return spec.unflatten([untyped(utils.batch_index(x, batched, b)) for b in range(size)]), True


core.r_impl.set(p_untyped, impl_untyped)
core.r_aimpl.set(p_untyped, utils.asyncify(impl_untyped))
core.r_abstract.set(p_untyped, abstract_untyped)
core.r_push.set(p_untyped, push_untyped)
core.r_apush.set(p_untyped, utils.asyncify(push_untyped))
core.r_pull_fwd.set(p_untyped, pull_fwd_untyped)
core.r_apull_fwd.set(p_untyped, utils.asyncify(pull_fwd_untyped))
core.r_pull_bwd.set(p_untyped, pull_bwd_untyped)
core.r_apull_bwd.set(p_untyped, utils.asyncify(pull_bwd_untyped))
core.r_batch.set(p_untyped, batch_untyped)
core.r_abatch.set(p_untyped, utils.asyncify(batch_untyped))
