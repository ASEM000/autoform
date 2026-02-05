"""String primitives"""

from __future__ import annotations

import functools as ft

from autoform.ad import zero_cotangent
from autoform.core import (
    EvalType,
    Primitive,
    PrimitiveTag,
    Var,
    batch_rules,
    eval_rules,
    impl_rules,
    is_var,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, asyncify, batch_index, batch_spec, treelib


class StringTag(PrimitiveTag): ...


# ==================================================================================================
# FORMAT
# ==================================================================================================

format_p = Primitive("format", tag={StringTag})


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


def eval_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> EvalType:
    return Var(str)


def pushforward_format(
    in_tree: Tree, /, *, template: str, keys: tuple[str, ...]
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_out = format_p.bind(primals, template=template, keys=keys)
    t_out = format_p.bind(tangents, template=template, keys=keys)
    return p_out, t_out


def pullback_fwd_format(
    in_tree: Tree, /, *, template: str, keys: tuple[str, ...]
) -> tuple[Tree, Tree]:
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


def batch_format(in_tree: Tree, /, *, template: str, keys: tuple[str, ...]) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if (spec := batch_spec(in_values, in_batched)) is None:
        return format_p.bind(in_values, template=template, keys=keys), False

    unbatch = ft.partial(batch_index, in_values, in_batched)
    bind = ft.partial(format_p.bind, template=template, keys=keys)
    result = [bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


impl_rules.set(format_p, impl_format)
impl_rules.aset(format_p, asyncify(impl_format))
eval_rules.set(format_p, eval_format)
push_rules.set(format_p, pushforward_format)
push_rules.aset(format_p, asyncify(pushforward_format))
pull_fwd_rules.set(format_p, pullback_fwd_format)
pull_fwd_rules.aset(format_p, asyncify(pullback_fwd_format))
pull_bwd_rules.set(format_p, pullback_bwd_format)
pull_bwd_rules.aset(format_p, asyncify(pullback_bwd_format))
batch_rules.set(format_p, batch_format)
batch_rules.aset(format_p, asyncify(batch_format))

# ==================================================================================================
# CONCAT
# ==================================================================================================

concat_p = Primitive("concat", tag={StringTag})


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


def eval_concat(in_tree: Tree, /) -> EvalType:
    return Var(str)


def pushforward_concat(in_tree: Tree, /) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    return concat_p.bind(primals), concat_p.bind(tangents)


def pullback_fwd_concat(in_tree: Tree, /) -> tuple[Tree, Tree]:
    out = concat_p.bind(in_tree)
    return out, len(in_tree)


def pullback_bwd_concat(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    n = residuals
    return tuple([out_cotangent] * n)


def batch_concat(in_tree: Tree, /) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree
    if (spec := batch_spec(in_values, in_batched)) is None:
        return concat_p.bind(in_values), False
    unbatch = ft.partial(batch_index, in_values, in_batched)
    result = [concat_p.bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


impl_rules.set(concat_p, impl_concat)
impl_rules.aset(concat_p, asyncify(impl_concat))
eval_rules.set(concat_p, eval_concat)
push_rules.set(concat_p, pushforward_concat)
push_rules.aset(concat_p, asyncify(pushforward_concat))
pull_fwd_rules.set(concat_p, pullback_fwd_concat)
pull_fwd_rules.aset(concat_p, asyncify(pullback_fwd_concat))
pull_bwd_rules.set(concat_p, pullback_bwd_concat)
pull_bwd_rules.aset(concat_p, asyncify(pullback_bwd_concat))
batch_rules.set(concat_p, batch_concat)
batch_rules.aset(concat_p, asyncify(batch_concat))


# ==================================================================================================
# MATCH
# ==================================================================================================

match_p = Primitive("match", tag={StringTag})


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


def eval_match(in_tree: Tree, /) -> EvalType:
    a, b = in_tree
    if is_var(a) or is_var(b):
        return Var(bool)
    return a == b


def pushforward_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    primals, tangents = in_tree
    out_primal = match_p.bind(primals)
    out_tangent = treelib.map(zero_cotangent, primals)
    return out_primal, out_tangent


def pullback_fwd_match(in_tree: Tree, /) -> tuple[bool, Tree]:
    out = match_p.bind(in_tree)
    residuals = in_tree
    return out, residuals


def pullback_bwd_match(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    del out_cotangent
    return treelib.map(zero_cotangent, residuals)


def batch_match(in_tree: Tree, /) -> tuple[list[bool], bool]:
    batch_size, in_batched, in_values = in_tree
    if (spec := batch_spec(in_values, in_batched)) is None:
        return match_p.bind(in_values), False
    unbatch = ft.partial(batch_index, in_values, in_batched)
    result = [match_p.bind(unbatch(b)) for b in range(batch_size)]
    return spec.unflatten(result), True


impl_rules.set(match_p, impl_match)
impl_rules.aset(match_p, asyncify(impl_match))
eval_rules.set(match_p, eval_match)
push_rules.set(match_p, pushforward_match)
push_rules.aset(match_p, asyncify(pushforward_match))
pull_fwd_rules.set(match_p, pullback_fwd_match)
pull_fwd_rules.aset(match_p, asyncify(pullback_fwd_match))
pull_bwd_rules.set(match_p, pullback_bwd_match)
pull_bwd_rules.aset(match_p, asyncify(pullback_bwd_match))
batch_rules.set(match_p, batch_match)
batch_rules.aset(match_p, asyncify(batch_match))
