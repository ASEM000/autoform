"""String primitives"""

from __future__ import annotations

import functools as ft

from autoform.core import Var, is_var, EvalType
from autoform.core import (
    Primitive,
    batch_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, treelib
from autoform.ad import zero_cotangent

# ==================================================================================================
# FORMAT
# ==================================================================================================

format_p = Primitive("format", tag="string")


def format(template: str, *args) -> str:
    """Format a string template with arguments.

    Example:
        >>> import autoform as af
        >>> af.format("Hello, {}!", "World")
        'Hello, World!'
    """
    return format_p.bind(args, template=template)


@ft.partial(impl_rules.def_rule, format_p)
def impl_format(in_tree: Tree, *, template: str) -> str:
    return template.format(*in_tree)


@ft.partial(eval_rules.def_rule, format_p)
def eval_format(in_tree: Tree, *, template: str) -> EvalType:
    return Var() if any(is_var(x) for x in in_tree) else impl_format(in_tree, template=template)


@ft.partial(pull_fwd_rules.def_rule, format_p)
def pullback_fwd_format(in_tree: Tree, *, template: str) -> tuple[Tree, Tree]:
    out = template.format(*in_tree)
    return out, len(in_tree)


@ft.partial(pull_bwd_rules.def_rule, format_p)
def pullback_bwd_format(residuals: Tree, out_cotangent: Tree, *, template: str) -> Tree:
    del template
    n = residuals
    return tuple([out_cotangent] * n)


@ft.partial(batch_rules.def_rule, format_p)
def batch_format(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    template: str,
) -> tuple[Tree, Tree]:
    args = tuple(in_tree)
    args_batched = tuple(in_batched)

    def get(i, b):
        return args[i][b] if args_batched[i] else args[i]

    result = [template.format(*[get(i, b) for i in range(len(args))]) for b in range(batch_size)]
    return result, True


@ft.partial(push_rules.def_rule, format_p)
def pushforward_format(primals: Tree, tangents: Tree, *, template: str) -> tuple[Tree, Tree]:
    out_primal = format(template, *primals)
    out_tangent = format(template, *tangents)
    return out_primal, out_tangent


# ==================================================================================================
# CONCAT
# ==================================================================================================

concat_p = Primitive("concat", tag="string")


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


@ft.partial(impl_rules.def_rule, concat_p)
def impl_concat(in_tree: Tree) -> str:
    return "".join(in_tree)


@ft.partial(eval_rules.def_rule, concat_p)
def eval_concat(in_tree: Tree, **params) -> EvalType:
    return Var() if any(is_var(x) for x in in_tree) else impl_concat(in_tree)


@ft.partial(push_rules.def_rule, concat_p)
def pushforward_concat(primals: Tree, tangents: Tree) -> tuple[Tree, Tree]:
    return concat(*primals), concat(*tangents)


@ft.partial(pull_fwd_rules.def_rule, concat_p)
def pullback_fwd_concat(in_tree: Tree) -> tuple[Tree, Tree]:
    out = concat(*in_tree)
    return out, len(in_tree)


@ft.partial(pull_bwd_rules.def_rule, concat_p)
def pullback_bwd_concat(residuals: Tree, out_cotangent: Tree) -> Tree:
    n = residuals
    return tuple([out_cotangent] * n)


@ft.partial(batch_rules.def_rule, concat_p)
def batch_concat(batch_size: int, in_batched: Tree, in_tree: Tree) -> tuple[Tree, Tree]:
    cols = tuple(in_tree)
    batched = tuple(in_batched)
    if batch_size == 0:
        return [], True

    def get(i, b):
        return cols[i][b] if batched[i] else cols[i]

    result = [concat(*[get(i, b) for i in range(len(cols))]) for b in range(batch_size)]
    return result, True


# ==================================================================================================
# MATCH
# ==================================================================================================

match_p = Primitive("match", tag="string")


def match(a: str, b: str) -> bool:
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


@ft.partial(impl_rules.def_rule, match_p)
def impl_match(in_tree: Tree) -> bool:
    a, b = in_tree
    return a == b


@ft.partial(eval_rules.def_rule, match_p)
def eval_match(in_tree: Tree) -> EvalType:
    a, b = in_tree
    if is_var(a) or is_var(b):
        return Var()
    return a == b


@ft.partial(batch_rules.def_rule, match_p)
def batch_match(batch_size: int, in_batched: Tree, in_tree: Tree) -> tuple[list[bool], bool]:
    a_col, b_col = in_tree
    a_batched, b_batched = in_batched

    def get_a(b):
        return a_col[b] if a_batched else a_col

    def get_b(b):
        return b_col[b] if b_batched else b_col

    result = [get_a(b) == get_b(b) for b in range(batch_size)]
    return result, True


@ft.partial(push_rules.def_rule, match_p)
def pushforward_match(primals: Tree, tangents: Tree) -> tuple[bool, Tree]:
    out_primal = match(*primals)
    out_tangent = treelib.map(zero_cotangent, primals)
    return out_primal, out_tangent


@ft.partial(pull_fwd_rules.def_rule, match_p)
def pullback_fwd_match(in_tree: Tree) -> tuple[bool, Tree]:
    a, b = in_tree
    out = a == b
    residuals = (a, b)
    return out, residuals


@ft.partial(pull_bwd_rules.def_rule, match_p)
def pullback_bwd_match(residuals: Tree, out_cotangent: Tree) -> Tree:
    del out_cotangent
    return treelib.map(zero_cotangent, residuals)
