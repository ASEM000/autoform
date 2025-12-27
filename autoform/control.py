"""Control flow primitives"""

from __future__ import annotations

import functools as ft
from autoform.optims import default_dce, dce
from autoform.core import call, icall, acall
from autoform.core import IR, Var, is_irvar, is_user_type, is_iratom
from autoform.core import (
    Primitive,
    async_rules,
    batch_rules,
    dce_rules,
    eval_rules,
    impl_rules,
    iter_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, unbatch_at, pack_user_input, treelib
from autoform.ad import pullback, zero_cotangent, pushforward

# ==================================================================================================
# STOP GRADIENT
# ==================================================================================================

stop_gradient_p = Primitive("stop_gradient", tag="control")


def stop_gradient(x: Tree) -> Tree:
    """Stops the gradient flow through the input during backpropagation.

    Args:
        x: The input tree (e.g., a string, number, or nested structure)

    Returns:
        The same input tree with gradients stopped.

    Example:
        >>> import autoform as af
        >>> def ir(x, y):
        ...     stopped = af.stop_gradient(x)
        ...     return af.concat(stopped, y)
        >>> ir = af.build_ir(ir)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> _, (cotangent_x, cotangent_y) = call(pb_ir)((("a", "b"), "grad"))
        >>> cotangent_x
        ''
        >>> cotangent_y
        'grad'
    """
    return stop_gradient_p.bind(x)


@ft.partial(impl_rules.def_rule, stop_gradient_p)
def impl_stop_gradient(x: Tree) -> Tree:
    return x


@ft.partial(eval_rules.def_rule, stop_gradient_p)
def eval_stop_gradient(x: Tree) -> Tree:
    return x


@ft.partial(push_rules.def_rule, stop_gradient_p)
def pushforward_stop_gradient(primal: Tree, tangent: Tree) -> tuple[Tree, Tree]:
    zero_tangent = treelib.map(zero_cotangent, primal)
    return primal, zero_tangent


@ft.partial(pull_fwd_rules.def_rule, stop_gradient_p)
def pullback_fwd_stop_gradient(x: Tree) -> tuple[Tree, Tree]:
    residuals = x
    return x, residuals


@ft.partial(pull_bwd_rules.def_rule, stop_gradient_p)
def pullback_bwd_stop_gradient(residuals: Tree, out_cotangent: Tree) -> Tree:
    del out_cotangent
    return treelib.map(zero_cotangent, residuals)


@ft.partial(batch_rules.def_rule, stop_gradient_p)
def batch_stop_gradient(batch_size: int, in_batched: Tree, x: Tree) -> tuple[Tree, Tree]:
    del batch_size
    return x, in_batched


# ==================================================================================================
# SWITCH
# ==================================================================================================

switch_p = Primitive("switch", tag="control")


def switch(key: str, branches: dict[str, IR], *operands, **kw_operands) -> Tree:
    """Select and execute one of multiple IR branches based on a string key.

    Args:
        key: String key selecting which branch to execute.
        branches: Dict mapping string keys to IR irs, each with compatible input signature.
        *args: Positional arguments passed to the selected branch.
        **kwargs: Keyword arguments passed to the selected branch.

    Returns:
        Result of run_ir(branches[key], *args, **kwargs)

    Raises:
        KeyError: If key is not in branches.

    Example:
        >>> import autoform as af
        >>> branches = {
        ...     "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
        ...     "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        ...     "two": af.build_ir(lambda x: af.concat("two: ", x))("X"),
        ... }
        >>> def ir(key, x):
        ...     return af.switch(key, branches, x)
        >>> ir = af.build_ir(ir)("one", "hello")
        >>> call(ir)("one", "hello")
        'one: hello'
        >>> call(ir)("zero", "hello")
        'zero: hello'
    """
    assert is_user_type(key) or is_iratom(key), "key must be a user-type (traceable) value"
    assert all(isinstance(branches[k], IR) for k in branches)
    tree_struct0 = treelib.structure(branches[next(iter(branches))].in_irtree)
    assert all(treelib.structure(branches[key].in_irtree) == tree_struct0 for key in branches)
    tree_struct0 = treelib.structure(branches[next(iter(branches))].out_irtree)
    assert all(treelib.structure(branches[key].out_irtree) == tree_struct0 for key in branches)
    return switch_p.bind((key, pack_user_input(*operands, **kw_operands)), branches=branches)


@ft.partial(impl_rules.def_rule, switch_p)
def impl_switch(in_tree, *, branches: dict[str, IR]):
    key, operands = in_tree
    return call(branches[key])(operands)


@ft.partial(eval_rules.def_rule, switch_p)
def eval_switch(in_tree, *, branches: dict[str, IR]) -> Tree:
    del in_tree
    key0 = next(iter(branches))
    branch0 = branches[key0]
    return treelib.map(lambda atom: Var() if is_irvar(atom) else atom.value, branch0.out_irtree)


@ft.partial(push_rules.def_rule, switch_p)
def pushforward_switch(primals, tangents, *, branches: dict[str, IR]):
    (key, p_operands), (_, t_operands) = primals, tangents
    pf_ir = pushforward(branches[key])
    return call(pf_ir)((p_operands, t_operands))


@ft.partial(pull_fwd_rules.def_rule, switch_p)
def pullback_fwd_switch(in_tree, *, branches: dict[str, IR]) -> tuple[Tree, Tree]:
    key, operands = in_tree
    out = call(branches[key])(operands)
    residuals = (key, operands)
    return out, residuals


@ft.partial(pull_bwd_rules.def_rule, switch_p)
def pullback_bwd_switch(residuals, out_cotangent, *, branches: dict[str, IR]):
    key, operands = residuals
    pb_ir = pullback(branches[key])
    _, c_operands = call(pb_ir)((operands, out_cotangent))
    return (zero_cotangent(key), c_operands)


@ft.partial(batch_rules.def_rule, switch_p)
def batch_switch(
    batch_size: int,
    in_batched,
    in_tree,
    *,
    branches: dict[str, IR],
) -> tuple[Tree, bool]:
    key_col, operands_col = in_tree
    key_batched, operands_batched = in_batched
    unbatch_operands = ft.partial(unbatch_at, operands_col, operands_batched)

    def run_ir_at(b):
        return call(branches[key_col[b] if key_batched else key_col])(unbatch_operands(b))

    return [run_ir_at(b) for b in range(batch_size)], True


@ft.partial(iter_rules.def_rule, switch_p)
def iter_switch(in_tree, *, branches: dict[str, IR]):
    key, operands = in_tree
    *chunks, _ = icall(branches[key])(operands)
    for chunk in chunks:
        yield chunk


@ft.partial(async_rules.def_rule, switch_p)
async def async_switch(in_tree, *, branches: dict[str, IR]) -> Tree:
    key, operands = in_tree
    return await acall(branches[key])(operands)


@ft.partial(dce_rules.def_rule, switch_p)
def dce_switch(ireqn, active_irvars) -> tuple[bool, set, object]:
    for k in (branches := dict(ireqn.params["branches"])):
        branches[k] = dce(branches[k])

    new_eqn = ireqn.using(branches=branches)
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn
