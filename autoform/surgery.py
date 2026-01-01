"""IR surgery functionality."""

from __future__ import annotations

import functools as ft
import typing as tp

from autoform.core import (
    IR,
    Interpreter,
    IRAtom,
    IREqn,
    IRLit,
    IRVar,
    Primitive,
    Tree,
    Var,
    batch_rules,
    call,
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
    is_iratom,
    is_irvar,
    is_var,
    pack_user_input,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    treelib,
    using_interpreter,
)
from autoform.utils import lru_cache

# ==================================================================================================
# SPLITPOINT
# ==================================================================================================

splitpoint_p = Primitive("splitpoint", tag="core")


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


class SplitInterpreter(Interpreter):
    def __init__(self, key: tp.Hashable):
        self.lhs_ireqns: list[IREqn] = []
        self.rhs_ireqns: list[IREqn] = []
        self.key = key
        self.split: bool = False

    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree[IRAtom]:
        def to_in_iratom(x) -> IRAtom:
            return x if is_iratom(x) else IRLit(x)

        in_irtree = treelib.map(to_in_iratom, in_tree)

        # Recursively process nested IRs in params (HOPs like pushforward_call)
        for value in params.values():
            if isinstance(value, IR):
                for eqn in value.ireqns:
                    self.interpret(eqn.prim, eqn.in_irtree, **eqn.params)

        def to_in_evaltype(x):
            return Var() if is_irvar(x) else x.value

        in_evaltree = treelib.map(to_in_evaltype, in_irtree)
        out_evaltree = eval_rules[prim](in_evaltree, **params)

        def to_out_iratom(x) -> IRAtom:
            return IRVar.fresh() if is_var(x) else IRLit(x)

        out_irtree = treelib.map(to_out_iratom, out_evaltree)

        ireqns = self.rhs_ireqns if self.split else self.lhs_ireqns
        ireqns.append(IREqn(prim, in_irtree, out_irtree, params))

        if prim == splitpoint_p and params.get("key") == self.key:
            assert self.split is False, "Cannot split multiple times"
            self.split = True

        return out_irtree


@ft.partial(lru_cache, maxsize=256)
def split[**P, R](ir: IR, *, key: tp.Hashable) -> tuple[IR, IR]:
    """Split an IR into left and right IRs at the splitpoint with given key.

    Args:
        ir: The intermediate representation to split.
        key: The key of the splitpoint to split at.

    Returns:
        A tuple (lhs_ir, rhs_ir) of two IR objects. The lhs_ir contains all
        equations up to and including the splitpoint. The rhs_ir contains
        all equations after the splitpoint.

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

    with using_interpreter(SplitInterpreter(key=key)) as tracer:
        call(ir)(ir.in_irtree)

    assert tracer.split is True, f"`split` could not find splitpoint with {key=}"

    lhs_ireqns = tracer.lhs_ireqns
    lhs_in_irtree = ir.in_irtree
    lhs_out_irtree = lhs_ireqns[-1].out_irtree
    lhs = IR(ireqns=lhs_ireqns, in_irtree=lhs_in_irtree, out_irtree=lhs_out_irtree)

    rhs_ireqns = tracer.rhs_ireqns
    rhs_in_irtree = pack_user_input(lhs_ireqns[-1].out_irtree)
    rhs_out_irtree = tracer.rhs_ireqns[-1].out_irtree if rhs_ireqns else lhs_out_irtree
    rhs = IR(ireqns=rhs_ireqns, in_irtree=rhs_in_irtree, out_irtree=rhs_out_irtree)

    return lhs, rhs
