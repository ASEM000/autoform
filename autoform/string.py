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

import autoform.ad as ad
import autoform.core as core
import autoform.utils as utils

__all__ = ["format", "concat", "match"]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]

# ==================================================================================================
# FORMAT
# ==================================================================================================

format_p = core.Prim("format")


def format(template: str, *args, **kwargs) -> str:
    """Format a string template with positional and/or keyword arguments.

    Example:
        >>> import autoform as af
        >>> af.format("Hello, {}!", "World")
        'Hello, World!'
        >>> af.format("Hello, {name}!", name="World")
        'Hello, World!'
        >>> af.format("{0}, {name}!", "Hi", name="World")
        'Hi, World!'
    """
    in_tree = (args, tuple(kwargs.values()))
    return format_p.bind(in_tree, template=template, keys=tuple(kwargs))


def impl_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> str:
    args, kwargs_values = in_tree
    kwargs = dict(zip(keys, kwargs_values))
    return template.format(*args, **kwargs)


def abstract_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> core.EvalType:
    return core.StrAVal()


def pushforward_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> TreePair:
    primals, tangents = in_tree
    p_out = format_p.bind(primals, template=template, keys=keys)
    tangents = ad.materialize(tangents)
    t_out = format_p.bind(tangents, template=template, keys=keys)
    return p_out, t_out


def pullback_fwd_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> TreePair:
    args, kwargs_values = in_tree
    out = format_p.bind(in_tree, template=template, keys=keys)
    residuals = (len(args), len(kwargs_values))
    return out, residuals


def pullback_bwd_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> Tree:
    del template, keys
    (n_args, n_kwargs), out_cotangent = in_tree
    args_cotangent = tuple([out_cotangent] * n_args)
    kwargs_cotangent = tuple([out_cotangent] * n_kwargs)
    return (args_cotangent, kwargs_cotangent)


def batch_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return format_p.bind(in_values, template=template, keys=keys), False

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    bind = ft.partial(format_p.bind, template=template, keys=keys)
    result = [bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


core.impl_rules.set(format_p, impl_format)
core.impl_rules.aset(format_p, utils.asyncify(impl_format))
core.abstract_rules.set(format_p, abstract_format)
core.push_rules.set(format_p, pushforward_format)
core.push_rules.aset(format_p, utils.asyncify(pushforward_format))
core.pull_fwd_rules.set(format_p, pullback_fwd_format)
core.pull_fwd_rules.aset(format_p, utils.asyncify(pullback_fwd_format))
core.pull_bwd_rules.set(format_p, pullback_bwd_format)
core.pull_bwd_rules.aset(format_p, utils.asyncify(pullback_bwd_format))
core.batch_rules.set(format_p, batch_format)
core.batch_rules.aset(format_p, utils.asyncify(batch_format))

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
        >>> result = af.concat("Hello, ", "world", "!")
        >>> print(result)
        Hello, world!
    """
    return concat_p.bind(args)


def impl_concat(in_tree: Tree, /) -> str:
    return "".join(in_tree)


def abstract_concat(in_tree: Tree, /) -> core.EvalType:
    assert all(type(x) in (str, core.StrAVal) for x in in_tree), f"Expected strings: {in_tree!r}"
    return core.StrAVal()


def pushforward_concat(in_tree: Tree, /) -> TreePair:
    primals, tangents = in_tree
    tangents = ad.materialize(tangents)
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


core.trace_add_rules[core.StrAVal] = concat


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
        >>> af.match("yes", "yes")
        True
        >>> af.match("yes", "no")
        False
    """
    return match_p.bind((a, b))


def impl_match(in_tree: Tree, /) -> bool:
    a, b = in_tree
    return a == b


def abstract_match(in_tree: Tree, /) -> core.EvalType:
    assert all(type(x) in (str, core.StrAVal) for x in in_tree), f"Expected strings: {in_tree!r}"
    return core.BoolAVal()


def pushforward_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    primals, tangents = in_tree
    out_primal = match_p.bind(primals)
    return out_primal, ad.tangent_zeroof(core.BoolAVal())


def pullback_fwd_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    out = match_p.bind(in_tree)
    residuals = in_tree
    return out, residuals


def pullback_bwd_match(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    del out_cotangent
    return utils.tree.map(lambda r: r if ad.is_zero(r) else ad.cotangent_zeroof(r), residuals)


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


ad.zero_rules[core.StrAVal] = lambda _: ""
ad.cot_acc_rules[core.StrAVal] = lambda cs, _: "".join(cs)

core.trace_eq_rules[core.StrAVal] = match
