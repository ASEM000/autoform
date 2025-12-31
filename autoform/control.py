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


def switch(key: str, branches: dict[str, IR], *args, **kwargs) -> Tree:
    """Select and execute one of multiple IR branches based on a string key.

    Args:
        key: String key selecting which branch to execute.
        branches: Dict mapping string keys to IR irs, each with compatible input signature.
        *args: Positional arguments passed to the selected branch.
        **kwargs: Keyword arguments passed to the selected branch.

    Returns:
        Result of ``run_ir(branches[key], *args, **kwargs)``

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
    return switch_p.bind((key, pack_user_input(*args, **kwargs)), branches=branches)


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
    # NOTE(asem): in_tree is a SoA object, however we need to pass in only parts of the SoA
    # that are alive (:= still needs some work). so we need to convert from SoA to AoS
    # filter out dead items then convert back to SoA for the batched cond/body to work.
    # finally convert back to SoA for the output. the following code does exactly this with
    # some bookkeeping to handle divergence.

    # NOTE(asem): batched while loop with early exit. each item exits independently when
    # cond returns False, saving LLM calls on items that finish early.
    # example: Struct(text=batched, note=broadcast) with 3 items
    # >>> in_tree = Struct(text=["A","B","C"], note="v1")  # text batched, note broadcast
    # >>> in_batched = Struct(text=True, note=False)
    # >>> cond_func: s.note != "done"
    # >>> body_func: Struct(text=refine(s.text), note="done" if good else s.note)
    #
    # unbatch SoA -> AoS (broadcast note is replicated):
    # >>> states = [Struct(A,v1), Struct(B,v1), Struct(C,v1)]
    #
    # iter 0: conds=[T,F,T] -> B exits -> body on [A,C]
    # >>> in_transposed  = Struct(text=["A","C"], note=["v1","v1"])
    # >>> out_transposed = Struct(text=["A'","C'"], note=["v2","v1"])
    # >>> states = [Struct(A',v2), Struct(B,v1), Struct(C',v1)]
    #
    # iter 1: conds=[F,T] -> A' exits -> body on [C']
    # >>> states = [Struct(A',v2), Struct(B,v1), Struct(C'',done)]
    #
    # iter 2: conds=[F] -> C'' exits -> done
    #
    # transpose AoS -> SoA (note becomes batched in output):
    # >>> out_tree = Struct(text=["A'","B","C''"], note=["v2","v1","done"])
    # >>> out_batched = Struct(text=True, note=True)

    # NOTE(asem): unbatch SoA -> AoS so each state can be tracked independently
    # and keep track of which items are not done. initially everything is alive
    states = [unbatch_at(in_tree, in_batched, b) for b in range(batch_size)]
    alive = [True] * batch_size

    # NOTE(asem): pre-batch cond and body IRs. list is used as in_axes
    # because moving from AoS to SoA a list is used (see states) also
    # list is used rather than the origina container structure
    cond_in_axes = treelib.map(lambda _: list, cond_func.in_irtree)
    body_in_axes = treelib.map(lambda _: list, body_func.in_irtree)
    batched_cond = batch(cond_func, in_axes=cond_in_axes)
    batched_body = batch(body_func, in_axes=body_in_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(batch_size) if alive[i]]):
            break

        # NOTE(asem): check conditions only for alive items (transpose AoS -> SoA for call)
        alive_states = [states[i] for i in alive_idx]
        n_alive = len(alive_states)
        in_batched_cond = treelib.map(lambda _: True, cond_func.in_irtree)
        # NOTE(asem): move from AoS to SoA for alive states
        in_transposed_cond = transpose_batch(n_alive, in_batched_cond, alive_states)
        conds_result = call(batched_cond)(in_transposed_cond)
        # NOTE(asem): cond returns scalar bool, batched -> list. use unbatch for consistency.
        out_batched_cond = isinstance(conds_result, list)
        conds = [unbatch_at(conds_result, out_batched_cond, b) for b in range(n_alive)]

        # NOTE(asem): mark items as dead if cond returned False
        for idx, c in zip(alive_idx, conds, strict=True):
            alive[idx] = c

        # NOTE(asem): run body ONLY on still-alive items
        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            n_still_alive = len(still_alive_states)
            in_batched = treelib.map(lambda _: True, body_func.in_irtree)
            in_transposed = transpose_batch(n_still_alive, in_batched, still_alive_states)
            out_transposed = call(batched_body)(in_transposed)
            out_batched = treelib.map(lambda _: True, body_func.out_irtree)

            for local_idx, batch_idx in enumerate(still_alive):
                states[batch_idx] = unbatch_at(out_transposed, out_batched, local_idx)

    # NOTE(asem): transpose final states AoS -> SoA for batched output
    out_batched = treelib.map(lambda _: True, body_func.out_irtree)

    # NOTE(asem): wrap back the state in their original container
    spec = treelib.structure(in_tree, is_leaf=lambda x: x is not in_tree)
    states = spec.unflatten(states)
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
