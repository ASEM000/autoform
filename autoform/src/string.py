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

"""String primitives."""

from __future__ import annotations

import functools as ft
import string as stringlib
from typing import Any

import autoform.src.ad as ad
import autoform.src.core as core
import autoform.src.pp as pp
import autoform.src.stage as stage
import autoform.src.tree as tree
import autoform.src.utils as utils

type Tree[T] = Any


# ==================================================================================================
# TYPES
# ==================================================================================================


class AStr(core.AVal):
    """Abstract value for ``str`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)("x")
        >>> (x,) = ir.in_tree
        >>> x.aval
        AStr()
    """

    __slots__ = []

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def zero(self) -> str:
        return ""

    def add(self, left, right, /):
        return left + right

    def join(self, other, /):
        match other:
            case str() | AStr() | core.Zero(aval=AStr()):
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")


@ft.partial(pp.register, AStr)
def pretty_astr(aval):
    return pp.style(type(aval).__name__, pp.Theme.TYPE), "()"


stage.register_trace_type(str, lambda value: AStr())
stage.register_trace_type(AStr, lambda aval: aval)
core.s_primal.set(AStr, lambda aval: aval)
core.s_tangent.set(AStr, lambda aval: aval)
core.s_cotangent.set(AStr, lambda aval: aval)

# ==================================================================================================
# RENDER
# ==================================================================================================


def abstract_render(_, /, *, width: int = 80):
    assert type(width) is int and width > 0, "width must be a positive integer"
    return AStr()


p_render = core.Prim("render")


def render(doc, /, *, width: int = 80) -> str:
    """Render a pytree document"""
    assert type(width) is int and width > 0, "width must be a positive integer"
    doc = tree.map(ft.partial(pp.pretty, default=lambda value: value), doc)
    return p_render.bind(doc, width=width)


def push_render(in_tree, /, **params):
    primal, tangent = in_tree
    return p_render.bind(primal, **params), p_render.bind(tangent, **params)


def pull_fwd_render(doc, /, **params):
    return p_render.bind(doc, **params), doc


def pull_bwd_render(in_tree, /, **params):
    del params
    doc, feedback = in_tree

    def cotangent(value):
        if isinstance(core.avalof(value), AStr):
            return feedback
        return core.s_cotangent.zeros_like(value)

    return tree.map(cotangent, doc)


def batch_render(in_tree, /, **params):
    size, batched, doc = in_tree
    if (spec := utils.batch_spec(doc, batched)) is None:
        return p_render.bind(doc, **params), False
    outputs = [p_render.bind(utils.batch_index(doc, batched, i), **params) for i in range(size)]
    return spec.unflatten(outputs), True


core.r_impl.set(p_render, pp.layout)
core.r_aimpl.set(p_render, utils.asyncify(pp.layout))
core.r_abstract.set(p_render, abstract_render)
core.r_batch.set(p_render, batch_render)
core.r_abatch.set(p_render, utils.asyncify(batch_render))
core.r_push.set(p_render, ad.concrete_zeros(push_render))
core.r_apush.set(p_render, utils.asyncify(ad.concrete_zeros(push_render)))
core.r_pull_fwd.set(p_render, pull_fwd_render)
core.r_apull_fwd.set(p_render, utils.asyncify(pull_fwd_render))
core.r_pull_bwd.set(p_render, pull_bwd_render)
core.r_apull_bwd.set(p_render, utils.asyncify(pull_bwd_render))


# ==================================================================================================
# CONCAT
# ==================================================================================================


def concat(*args) -> str:
    """Render document parts in sequence as a single string."""
    return render(args)


stage.register_dunder(stage.Dunder.ADD, AStr, concat)


# ==================================================================================================
# MATCH
# ==================================================================================================

p_match = core.Prim("match")


def match(a: str, b: str, /) -> bool:
    """Check if two strings are equal.

    This is a traceable version of `==` that works correctly during tracing.

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal, False otherwise.

    Example:
        >>> import autoform as af
        >>> "yes" == "yes"
        True
        >>> "yes" == "no"
        False
    """
    return p_match.bind((a, b))


def impl_match(in_tree: Tree, /) -> bool:
    a, b = in_tree
    return a == b


def abstract_bool(values: Tree, /) -> Any:
    return core.avalof(False)


@utils.dispatch
def abstract_match(values: Tree, /) -> Any:
    assert False, f"Expected strings: {values!r}"


abstract_match.register((str, str), abstract_bool)
abstract_match.register((str, AStr), abstract_bool)
abstract_match.register((AStr, str), abstract_bool)
abstract_match.register((AStr, AStr), abstract_bool)


def pushforward_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    p, t = in_tree
    out_p = p_match.bind(p)
    return out_p, core.s_tangent.zeros_like(out_p)


def pullback_fwd_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    out = p_match.bind(in_tree)
    residuals = in_tree
    return out, residuals


def pullback_bwd_match(in_tree: Tree, /) -> Tree:
    residuals, out_c = in_tree
    del out_c
    return core.s_cotangent.zeros_like(residuals)


def batch_match(in_tree: Tree, /) -> tuple[list[bool], bool]:
    batch_size, in_batched, in_values = in_tree
    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return p_match.bind(in_values), False
    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    result = [p_match.bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


core.r_impl.set(p_match, impl_match)
core.r_aimpl.set(p_match, utils.asyncify(impl_match))
core.r_abstract.set(p_match, abstract_match)
core.r_batch.set(p_match, batch_match)
core.r_abatch.set(p_match, utils.asyncify(batch_match))
core.r_push.set(p_match, pushforward_match)
core.r_apush.set(p_match, utils.asyncify(pushforward_match))
core.r_pull_fwd.set(p_match, pullback_fwd_match)
core.r_apull_fwd.set(p_match, utils.asyncify(pullback_fwd_match))
core.r_pull_bwd.set(p_match, pullback_bwd_match)
core.r_apull_bwd.set(p_match, utils.asyncify(pullback_bwd_match))


stage.register_dunder(stage.Dunder.EQ, AStr, match)


# ==================================================================================================
# FORMAT
# ==================================================================================================


def format(template: str, **kwargs) -> str:
    """Format a string using named document parts as keyword arguments.

    Note:
        - Template fields are keyword only.
        - Conversion and formatting specs are not allowed.

    Example:
        >>> import autoform as af
        >>> af.text.format("Hello {name}!", name="World")
        'Hello World!'
    """
    unused = set(kwargs)
    parts = []
    for literal, field, format_spec, conversion in stringlib.Formatter().parse(template):
        if literal:
            parts.append(literal)
        if field is None:
            continue

        assert field in kwargs, f"Unknown format field: {field!r}"
        assert conversion is None, "`format` does not support conversions."
        assert not format_spec, "`format` does not support format specifications."

        parts.append(kwargs[field])
        unused.discard(field)
    assert not unused, f"Unused format arguments: {unused}"
    return concat(*parts)
