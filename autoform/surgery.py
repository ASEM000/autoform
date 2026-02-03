"""IR surgery functionality."""

from __future__ import annotations

import functools as ft
from collections.abc import Hashable

from autoform.core import (
    IR,
    IREqn,
    Primitive,
    PrimitiveTag,
    batch_rules,
    eval_rules,
    impl_rules,
    pack_user_input,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, asyncify, lru_cache


class SurgeryTag(PrimitiveTag): ...


# ==================================================================================================
# SPLITPOINT
# ==================================================================================================

splitpoint_p = Primitive("splitpoint", tag={SurgeryTag})


def splitpoint(in_tree: Tree, /, *, key: Hashable) -> Tree:
    """Mark a split point for IR splitting.

    Used with `split()` to divide an IR into left and right portions.

    Args:
        in_tree: the value to mark as split point (returned unchanged).
        key: unique identifier for the split point.

    Returns:
        the input value unchanged.

    Example:
        >>> import autoform as af
        >>> def pipeline(x):
        ...     step1 = af.concat(x, "!")
        ...     step1 = af.splitpoint(step1, key="mid")
        ...     step2 = af.concat(step1, "?")
        ...     return step2
        >>> ir = af.trace(pipeline)("x")
        >>> lhs, rhs = af.split(ir, key="mid")
    """
    assert hash(key) is not None, "Key must be hashable"
    return splitpoint_p.bind(in_tree, key=key)


def impl_splitpoint(x, /, **_):
    return x


def eval_splitpoint(x, /, **_):
    return x


def push_splitpoint(in_tree, /, **params):
    primal, tangent = in_tree
    return splitpoint(primal, **params), splitpoint(tangent, **params)


def pull_fwd_splitpoint(x, /, **params):
    return splitpoint(x, **params), None


def pull_bwd_splitpoint(in_tree, /, **params):
    _, cotangent = in_tree
    return splitpoint(cotangent, **params)


def batch_splitpoint(in_tree, /, **params):
    _, in_batched, x = in_tree
    return splitpoint(x, **params), in_batched


impl_rules.set(splitpoint_p, impl_splitpoint)
impl_rules.aset(splitpoint_p, asyncify(impl_splitpoint))
eval_rules.set(splitpoint_p, eval_splitpoint)
push_rules.set(splitpoint_p, push_splitpoint)
push_rules.aset(splitpoint_p, asyncify(push_splitpoint))
pull_fwd_rules.set(splitpoint_p, pull_fwd_splitpoint)
pull_fwd_rules.aset(splitpoint_p, asyncify(pull_fwd_splitpoint))
pull_bwd_rules.set(splitpoint_p, pull_bwd_splitpoint)
pull_bwd_rules.aset(splitpoint_p, asyncify(pull_bwd_splitpoint))
batch_rules.set(splitpoint_p, batch_splitpoint)
batch_rules.aset(splitpoint_p, asyncify(batch_splitpoint))


# ==================================================================================================
# SPLIT
# ==================================================================================================


def maybe_split(ir: IR, splitpoint_key: Hashable) -> tuple[IR, IR] | None:
    for idx, ireqn in enumerate(ir.ireqns):
        if ireqn.prim == splitpoint_p and ireqn.params.get("key") == splitpoint_key:
            # NOTE(asem): splitpoint at top level. split ir.ireqns at idx.
            # def program(x):
            #     y = splitpoint(concat(x, "!"), key="mid")
            #     return concat(y, "?")
            # split(ir, key="mid")
            #   lhs: [concat], rhs: [concat]  (splitpoint stripped)
            #   boundary = splitpoint's OUTPUT (y), so rhs equations still reference it
            lhs_ireqns = list(ir.ireqns[:idx])  # exclude splitpoint
            rhs_ireqns = list(ir.ireqns[idx + 1 :])

            # NOTE(asem): use splitpoint's output as the boundary
            # This works because rhs equations reference splitpoint's output variable
            lhs_out = lhs_ireqns[-1].out_irtree if lhs_ireqns else ir.in_irtree
            lhs = IR(lhs_ireqns, ir.in_irtree, lhs_out)

            rhs_in_irtree = pack_user_input(ireqn.out_irtree)
            # NOTE(asem): in case rhs_ireqn is empty
            # >>> def program(x):
            # ...   y = concat(x, x)
            # ...   z = splitpoint(y, key="mid")
            # >>> split(ir, key="mid")
            #   lhs: [concat], rhs: []
            # in here rhs in_irtree = splitpoint out_irtree
            # and out_irtree = splitpoint out_irtree still
            rhs_out_irtree = ([ireqn] + rhs_ireqns)[-1].out_irtree
            rhs = IR(rhs_ireqns, rhs_in_irtree, rhs_out_irtree)
            return lhs, rhs

        # NOTE(asem): check if splitpoint is inside a nested IR (HOP).
        for irkey in filter(lambda k: isinstance(ireqn.params[k], IR), ireqn.params):
            result = maybe_split(ireqn.params[irkey], splitpoint_key)

            if result is None:
                continue

            inner_lhs, inner_rhs = result
            # NOTE(asem): recursively split nested_ir, wrap each half in HOP.
            # def program(x):
            #     y = splitpoint(concat(x, "!"), key="mid")
            #     return concat(y, "?")
            # split(pushforward(ir), key="mid")
            #   lhs: [pushforward_call(nested_lhs)]
            #   rhs: [pushforward_call(nested_rhs)]
            lhs_hop = ireqn.using(**{irkey: inner_lhs})
            rhs_hop = ireqn.using(**{irkey: inner_rhs})

            before: list[IREqn] = list(ir.ireqns[:idx])
            after: list[IREqn] = list(ir.ireqns[idx + 1 :])

            lhs = IR(before + [lhs_hop], ir.in_irtree, lhs_hop.out_irtree)

            # NOTE(asem): if rhs is empty, splitpoint at the end of ir.
            rhs_in_irtree = pack_user_input(lhs.out_irtree)
            rhs_out_irtree = ([rhs_hop] + after)[-1].out_irtree
            rhs = IR([rhs_hop] + after, rhs_in_irtree, rhs_out_irtree)
            return lhs, rhs

    return None


@ft.partial(lru_cache, maxsize=256)
def split[**P, R](ir: IR, /, *, key: Hashable) -> tuple[IR, IR]:
    """Split an IR into left and right IRs at the splitpoint with given key.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     y = af.splitpoint(af.concat(x, "!"), key="mid")
        ...     return af.concat(y, "?")
        >>> ir = af.trace(program)("x")
        >>> lhs, rhs = af.split(ir, key="mid")
        >>> af.call(lhs)("hello")
        'hello!'
        >>> af.call(rhs)("hello!")
        'hello!?'

    To merge the split IRs back together, retrace with splitpoint::

        >>> def merged(x):
        ...     y = lhs.call(x)
        ...     z = af.splitpoint(y, key="mid")
        ...     return rhs.call(z)
        >>> merged_ir = af.trace(merged)("...")
    """
    assert isinstance(ir, IR), f"`split` expected an IR, got {type(ir)}"
    result = maybe_split(ir, key)
    assert result is not None, f"`split` could not find splitpoint with {key=}"
    return result
