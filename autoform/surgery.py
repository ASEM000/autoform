"""IR surgery functionality."""

from __future__ import annotations

import functools as ft
import typing as tp

from autoform.core import (
    IR,
    IREqn,
    Primitive,
    PrimitiveTag,
    batch_rules,
    dce_rules,
    default_batch,
    default_dce,
    default_eval,
    default_impl,
    default_pull_bwd,
    default_pull_fwd,
    default_push,
    eval_rules,
    impl_rules,
    pack_user_input,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import lru_cache


class SurgeryTag(PrimitiveTag): ...


# ==================================================================================================
# SPLITPOINT
# ==================================================================================================

splitpoint_p = Primitive("splitpoint", tag={SurgeryTag})


def splitpoint(in_tree: Tree, /, *, key: tp.Hashable) -> Tree:
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
        >>> ir = af.build_ir(pipeline)("x")
        >>> lhs, rhs = af.split(ir, key="mid")
    """
    assert hash(key) is not None, "Key must be hashable"
    return splitpoint_p.bind(in_tree, key=key)


impl_rules.def_rule(splitpoint_p, default_impl)
eval_rules.def_rule(splitpoint_p, default_eval)
push_rules.def_rule(splitpoint_p, ft.partial(default_push, splitpoint))
pull_fwd_rules.def_rule(splitpoint_p, ft.partial(default_pull_fwd, splitpoint))
pull_bwd_rules.def_rule(splitpoint_p, ft.partial(default_pull_bwd, splitpoint))
batch_rules.def_rule(splitpoint_p, ft.partial(default_batch, splitpoint))
dce_rules.def_rule(splitpoint_p, default_dce)


# ==================================================================================================
# SPLIT
# ==================================================================================================


def maybe_split(ir: IR, key: tp.Hashable) -> tuple[IR, IR] | None:
    for idx, eqn in enumerate(ir.ireqns):
        if eqn.prim == splitpoint_p and eqn.params.get("key") == key:
            # NOTE(asem): splitpoint at top level. split ir.ireqns at idx.
            # def program(x):
            #     y = splitpoint(concat(x, "!"), key="mid")
            #     return concat(y, "?")
            # split(ir, key="mid")
            #   lhs: [concat, splitpoint], rhs: [concat]
            lhs_eqns = list(ir.ireqns[: idx + 1])
            rhs_eqns = list(ir.ireqns[idx + 1 :])
            lhs = IR(lhs_eqns, ir.in_irtree, lhs_eqns[-1].out_irtree)

            # NOTE(asem): if rhs is empty, splitpoint at the end of ir (use lhs.out_irtree)
            rhs_in_irtree = pack_user_input(lhs.out_irtree)
            rhs_out_irtree = ([lhs] + rhs_eqns)[-1].out_irtree
            rhs = IR(rhs_eqns, rhs_in_irtree, rhs_out_irtree)
            return lhs, rhs

        # NOTE(asem): check if splitpoint is inside a nested IR (HOP).
        for irkey in filter(lambda k: isinstance(eqn.params[k], IR), eqn.params):
            result = maybe_split(eqn.params[irkey], key)

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
            lhs_hop = eqn.using(**{irkey: inner_lhs})
            rhs_hop = eqn.using(**{irkey: inner_rhs})

            before: list[IREqn] = list(ir.ireqns[:idx])
            after: list[IREqn] = list(ir.ireqns[idx + 1 :])

            lhs = IR(before + [lhs_hop], ir.in_irtree, lhs_hop.out_irtree)

            # NOTE(asem): if rhs is empty, splitpoint at the end of ir.
            rhs_in_irtree = pack_user_input(lhs.out_irtree)
            rhs_out_irtree = ([rhs_hop] + after)[-1].out_irtree if after else lhs.out_irtree
            rhs = IR([rhs_hop] + after, rhs_in_irtree, rhs_out_irtree)
            return lhs, rhs

    return None


@ft.partial(lru_cache, maxsize=256)
def split[**P, R](ir: IR, /, *, key: tp.Hashable) -> tuple[IR, IR]:
    """Split an IR into left and right IRs at the splitpoint with given key.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     y = af.splitpoint(af.concat(x, "!"), key="mid")
        ...     return af.concat(y, "?")
        >>> ir = af.build_ir(program)("x")
        >>> lhs, rhs = af.split(ir, key="mid")
        >>> af.call(lhs)("hello")
        'hello!'
        >>> af.call(rhs)("hello!")
        'hello!?'
    """
    assert isinstance(ir, IR), f"`split` expected an IR, got {type(ir)}"
    result = maybe_split(ir, key)
    assert result is not None, f"`split` could not find splitpoint with {key=}"
    return result
