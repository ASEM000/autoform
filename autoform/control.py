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
from autoform.utils import Tree, unbatch_at, pack_user_input, treelib, transpose_batch
from autoform.ad import pullback, zero_cotangent, pushforward
from autoform.batch import batch

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


# ==================================================================================================
# WHILE LOOP
# ==================================================================================================

while_loop_p = Primitive("while_loop", tag="control")


def while_loop(cond_func: IR, body_func: IR, init_val: Tree, *, max_iters: int) -> Tree:
    """Repeatedly apply ``body_func`` while ``cond_func`` returns True.

    - Loop continues while cond_func(state) returns True
    - body_func is applied each iteration
    - Returns final state when cond_func returns False or max_iters reached

    Args:
        cond_func: IR that returns bool. Loop continues while True.
        body_func: IR that transforms state, f: State -> State
        init_val: Initial state
        max_iters: Maximum iterations

    Returns:
        Final state when cond_func returns False or max_iters reached.

    Example:
        >>> import autoform as af
        >>> def cond(x):
        ...     return False  # Exit immediately
        >>> def body(x):
        ...     return af.concat(x, "x")
        >>> cond_ir = af.build_ir(cond)("x")
        >>> body_ir = af.build_ir(body)("x")
        >>> result = af.while_loop(cond_ir, body_ir, "a", max_iters=10)
        >>> result
        'a'
    """
    assert isinstance(cond_func, IR), f"cond_func must be an IR, got {type(cond_func)}"
    assert isinstance(body_func, IR), f"body_func must be an IR, got {type(body_func)}"

    in_struct = treelib.structure(body_func.in_irtree)
    out_struct = treelib.structure(body_func.out_irtree)
    assert in_struct == out_struct, (
        f"body_func must have identical input/output structure (f: State -> State).\n"
        f"in_struct:  {in_struct}\n"
        f"out_struct: {out_struct}"
    )
    return while_loop_p.bind(
        init_val,
        cond_func=cond_func,
        body_func=body_func,
        max_iters=max_iters,
    )


@ft.partial(impl_rules.def_rule, while_loop_p)
def impl_while_loop(in_tree: Tree, *, cond_func: IR, body_func: IR, max_iters: int) -> Tree:
    state = in_tree
    for _ in range(max_iters):
        if not call(cond_func)(state):
            break
        state = call(body_func)(state)
    return state


@ft.partial(eval_rules.def_rule, while_loop_p)
def eval_while_loop(in_tree: Tree, *, cond_func: IR, body_func: IR, max_iters: int) -> Tree:
    del cond_func, max_iters
    return treelib.map(lambda _: Var(), body_func.out_irtree)


@ft.partial(batch_rules.def_rule, while_loop_p)
def batch_while_loop(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    cond_func: IR,
    body_func: IR,
    max_iters: int,
) -> tuple[Tree, Tree]:
    unbatch_state = ft.partial(unbatch_at, in_tree, in_batched)
    states = [unbatch_state(b) for b in range(batch_size)]

    alive = [True] * batch_size
    cond_in_axes = treelib.map(lambda _: list, cond_func.in_irtree)
    body_in_axes = treelib.map(lambda _: list, body_func.in_irtree)
    batched_cond = batch(cond_func, in_axes=cond_in_axes)
    batched_body = batch(body_func, in_axes=body_in_axes)

    for _ in range(max_iters):
        alive_idx = [i for i in range(batch_size) if alive[i]]
        if not alive_idx:
            break

        alive_states = [states[i] for i in alive_idx]
        n_alive = len(alive_states)
        cond_batched_in = treelib.map(lambda _: True, cond_func.in_irtree)
        transposed_cond_in = transpose_batch(n_alive, cond_batched_in, alive_states)
        conds_batched = call(batched_cond)(transposed_cond_in)
        cond_out_batched = isinstance(conds_batched, list)
        conds = [unbatch_at(conds_batched, cond_out_batched, b) for b in range(n_alive)]

        for i, c in zip(alive_idx, conds, strict=True):
            if not c:
                alive[i] = False

        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            n_still_alive = len(still_alive_states)
            batched_in = treelib.map(lambda _: True, body_func.in_irtree)
            transposed_in = transpose_batch(n_still_alive, batched_in, still_alive_states)
            transposed_out = call(batched_body)(transposed_in)
            batched_out = treelib.map(lambda _: True, body_func.out_irtree)
            new_states = [unbatch_at(transposed_out, batched_out, b) for b in range(n_still_alive)]
            for i, s in zip(still_alive, new_states, strict=True):
                states[i] = s

    out_batched = treelib.map(lambda _: True, body_func.out_irtree)
    out_tree = transpose_batch(batch_size, out_batched, states)
    return out_tree, out_batched


@ft.partial(pull_fwd_rules.def_rule, while_loop_p)
def pullback_fwd_while_loop(
    in_tree: Tree, *, cond_func: IR, body_func: IR, max_iters: int
) -> tuple[Tree, Tree]:
    state = in_tree
    trajectory = [state]

    for _ in range(max_iters):
        if not call(cond_func)(state):
            break
        state = call(body_func)(state)
        trajectory.append(state)

    residuals = (trajectory, body_func)
    return state, residuals


@ft.partial(pull_bwd_rules.def_rule, while_loop_p)
def pullback_bwd_while_loop(
    residuals: Tree, out_cotangent: Tree, *, cond_func: IR, body_func: IR, max_iters: int
) -> Tree:
    del cond_func, max_iters
    trajectory, _ = residuals
    n_iters = len(trajectory) - 1

    cotangent = out_cotangent
    pb_body = pullback(body_func)

    for t in reversed(range(n_iters)):
        state_t = trajectory[t]
        _, cotangent = call(pb_body)((state_t, cotangent))

    return cotangent
