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

"""String primitives"""

from __future__ import annotations

import functools as ft
import string as stringlib
from typing import Any

import autoform.src.core as core
import autoform.src.stage as stage
import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# TYPES
# ==================================================================================================


class AValStr(core.AVal):
    """Abstract value for ``str`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)("x")
        >>> (x,) = ir.in_tree
        >>> x.aval
        AValStr()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def zero(self) -> str:
        return ""

    def add(self, left, right, /):
        return left + right

    def join(self, other, /):
        match self, other:
            case AValStr(), str() | AValStr() | core.Zero(aval=AValStr()):
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")


stage.register_trace_type(str, lambda value: AValStr())
core.aval_types[AValStr] = lambda aval: aval
core.s_primal.set(AValStr, lambda aval: aval)
core.s_tangent.set(AValStr, lambda aval: aval)
core.s_cotangent.set(AValStr, lambda aval: aval)

# ==================================================================================================
# CONCAT
# ==================================================================================================

p_concat = core.Prim("concat")


def concat(*args) -> str:
    """Concatenates multiple strings into a single string.

    Args:
        *args: A variable number of string arguments to concatenate.

    Returns:
        A single string that is the concatenation of all input strings.

    Example:
        >>> import autoform as af
        >>> result = "Hello, " + "world" + "!"
        >>> print(result)
        Hello, world!
    """
    return p_concat.bind(args)


def impl_concat(in_tree: Tree, /) -> str:
    return "".join(in_tree)


def abstract_concat(in_tree: Tree, /) -> Any:
    assert all(type(x) in (str, AValStr) for x in in_tree), f"Expected strings: {in_tree!r}"
    return AValStr()


def pushforward_concat(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    zero = lambda x: x.aval.zero() if isinstance(x, core.Zero) else x
    t = utils.tree.map(zero, t)
    return p_concat.bind(p), p_concat.bind(t)


def pullback_fwd_concat(in_tree: Tree, /) -> TreePair:
    out = p_concat.bind(in_tree)
    return out, len(in_tree)


def pullback_bwd_concat(in_tree: Tree, /) -> Tree:
    residuals, out_c = in_tree
    n = residuals
    return tuple([out_c] * n)


def batch_concat(in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, in_values = in_tree
    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return p_concat.bind(in_values), False
    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    result = [p_concat.bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


core.r_impl.set(p_concat, impl_concat)
core.r_aimpl.set(p_concat, utils.asyncify(impl_concat))
core.r_abstract.set(p_concat, abstract_concat)
core.r_batch.set(p_concat, batch_concat)
core.r_abatch.set(p_concat, utils.asyncify(batch_concat))
core.r_push.set(p_concat, pushforward_concat)
core.r_apush.set(p_concat, utils.asyncify(pushforward_concat))
core.r_pull_fwd.set(p_concat, pullback_fwd_concat)
core.r_apull_fwd.set(p_concat, utils.asyncify(pullback_fwd_concat))
core.r_pull_bwd.set(p_concat, pullback_bwd_concat)
core.r_apull_bwd.set(p_concat, utils.asyncify(pullback_bwd_concat))


stage.register_dunder(stage.Dunder.ADD, AValStr, concat)


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


def abstract_match(in_tree: Tree, /) -> Any:
    assert all(type(x) in (str, AValStr) for x in in_tree), f"Expected strings: {in_tree!r}"
    return core.avalof(False)


def pushforward_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    p, t = in_tree
    out_p = p_match.bind(p)
    return out_p, core.Zero(core.s_tangent.map(core.avalof(False)))


def pullback_fwd_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    out = p_match.bind(in_tree)
    residuals = in_tree
    return out, residuals


def pullback_bwd_match(in_tree: Tree, /) -> Tree:
    def make_c(x):
        if isinstance(x, core.Zero):
            return x
        return core.Zero(core.s_cotangent.map(core.avalof(x)))

    residuals, out_c = in_tree
    del out_c
    return utils.tree.map(make_c, residuals)


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


stage.register_dunder(stage.Dunder.EQ, AValStr, match)


# ==================================================================================================
# FORMAT
# ==================================================================================================


def format(template: str, **kwargs) -> str:
    """Format a string using named keyword arguments.

    Note:
        - Template fields are keyword only.
        - Conversion and formatting specs are not allowed.

    Example:
        >>> import autoform as af
        >>> af.string.format("Hello {name}!", name="World")
        'Hello World!'
    """
    unused = set(kwargs)
    parts = []
    for literal, field, spec, conversion in stringlib.Formatter().parse(template):
        if literal:
            parts.append(literal)
        if field is None:
            continue
        assert conversion is None, "`format` does not support conversions."
        assert not spec, "`format` does not support format specifications."
        assert field in kwargs, "Template field name is not found in keyword arguments ."
        parts.append(kwargs[field])
        # NOTE(asem): in case of multiple ref to same kw discard does not raise error
        unused.discard(field)
    assert not unused, f"Unused format arguments: {unused}"
    return concat(*parts)
