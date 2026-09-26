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

import autoform.core as core
import autoform.utils as utils

__all__ = ["StrAVal", "format", "concat", "match"]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# TYPES
# ==================================================================================================


class StrAVal(core.AVal):
    """Abstract value for ``str`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)("x")
        >>> (x,) = ir.in_tree
        >>> x.aval
        StrAVal()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def zero(self) -> str:
        return ""

    def accumulate(self, cotangents: list[str], /) -> str:
        return "".join(cotangents)


core.trace_types.add(str)
core.primal_s.set(str, lambda _: StrAVal())
core.tangent_s.set(StrAVal, lambda aval: aval)
core.cotangent_s.set(StrAVal, lambda aval: aval)

# ==================================================================================================
# CONCAT
# ==================================================================================================

concat_p = core.Prim("concat")


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
    return concat_p.bind(args)


def impl_concat(in_tree: Tree, /) -> str:
    return "".join(in_tree)


def abstract_concat(in_tree: Tree, /) -> core.EvalType:
    assert all(type(x) in (str, StrAVal) for x in in_tree), f"Expected strings: {in_tree!r}"
    return StrAVal()


def pushforward_concat(in_tree: Tree, /) -> TreePair:
    primals, tangents = in_tree
    tangents = core.materialize_zeros(tangents)
    return concat_p.bind(primals), concat_p.bind(tangents)


def pullback_fwd_concat(in_tree: Tree, /) -> TreePair:
    out = concat_p.bind(in_tree)
    return out, len(in_tree)


def pullback_bwd_concat(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    n = residuals
    return tuple([out_cotangent] * n)


def batch_concat(in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, in_values = in_tree
    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return concat_p.bind(in_values), False
    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    result = [concat_p.bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


core.impl_rules.set(concat_p, impl_concat)
core.impl_rules.aset(concat_p, utils.asyncify(impl_concat))
core.abstract_rules.set(concat_p, abstract_concat)
core.push_rules.set(concat_p, pushforward_concat)
core.push_rules.aset(concat_p, utils.asyncify(pushforward_concat))
core.pull_fwd_rules.set(concat_p, pullback_fwd_concat)
core.pull_fwd_rules.aset(concat_p, utils.asyncify(pullback_fwd_concat))
core.pull_bwd_rules.set(concat_p, pullback_bwd_concat)
core.pull_bwd_rules.aset(concat_p, utils.asyncify(pullback_bwd_concat))
core.batch_rules.set(concat_p, batch_concat)
core.batch_rules.aset(concat_p, utils.asyncify(batch_concat))


core.dunder_rules[core.Dunder.ADD, StrAVal] = concat


# ==================================================================================================
# MATCH
# ==================================================================================================

match_p = core.Prim("match")


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
    return match_p.bind((a, b))


def impl_match(in_tree: Tree, /) -> bool:
    a, b = in_tree
    return a == b


def abstract_match(in_tree: Tree, /) -> core.EvalType:
    assert all(type(x) in (str, StrAVal) for x in in_tree), f"Expected strings: {in_tree!r}"
    return core.primal_s.avalof(False)


def pushforward_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    primals, tangents = in_tree
    out_primal = match_p.bind(primals)
    return out_primal, core.tangent_s.zeroof(core.primal_s.avalof(False))


def pullback_fwd_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    out = match_p.bind(in_tree)
    residuals = in_tree
    return out, residuals


def pullback_bwd_match(in_tree: Tree, /) -> Tree:
    def make_c(x):
        if isinstance(x, core.Zero):
            return x
        return core.cotangent_s.zeroof(core.primal_s.avalof(x))

    residuals, out_cotangent = in_tree
    del out_cotangent
    return utils.tree.map(make_c, residuals)


def batch_match(in_tree: Tree, /) -> tuple[list[bool], bool]:
    batch_size, in_batched, in_values = in_tree
    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return match_p.bind(in_values), False
    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    result = [match_p.bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


core.impl_rules.set(match_p, impl_match)
core.impl_rules.aset(match_p, utils.asyncify(impl_match))
core.abstract_rules.set(match_p, abstract_match)
core.push_rules.set(match_p, pushforward_match)
core.push_rules.aset(match_p, utils.asyncify(pushforward_match))
core.pull_fwd_rules.set(match_p, pullback_fwd_match)
core.pull_fwd_rules.aset(match_p, utils.asyncify(pullback_fwd_match))
core.pull_bwd_rules.set(match_p, pullback_bwd_match)
core.pull_bwd_rules.aset(match_p, utils.asyncify(pullback_bwd_match))
core.batch_rules.set(match_p, batch_match)
core.batch_rules.aset(match_p, utils.asyncify(batch_match))


core.dunder_rules[core.Dunder.EQ, StrAVal] = match


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
