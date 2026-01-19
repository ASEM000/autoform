"""Control flow primitives"""

from __future__ import annotations

import asyncio
import functools as ft

from autoform.ad import pullback, pushforward, zero_cotangent
from autoform.batch import batch
from autoform.core import (
    IR,
    IREqn,
    Primitive,
    PrimitiveTag,
    acall,
    batch_rules,
    call,
    eval_rules,
    impl_rules,
    iratom_to_evaltype,
    is_iratom,
    is_irvar,
    is_user_type,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.optims import dce, dce_rules, default_dce
from autoform.utils import (
    Tree,
    asyncify,
    batch_index,
    batch_spec,
    batch_transpose,
    pack_user_input,
    treelib,
)


class ControlTag(PrimitiveTag): ...


# ==================================================================================================
# STOP GRADIENT
# ==================================================================================================

stop_gradient_p = Primitive("stop_gradient", tag={ControlTag})


def stop_gradient(x: Tree, /) -> Tree:
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
        >>> ir = af.trace(ir)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> _, (cotangent_x, cotangent_y) = call(pb_ir)((("a", "b"), "grad"))
        >>> cotangent_x
        ''
        >>> cotangent_y
        'grad'
    """
    return stop_gradient_p.bind(x)


def impl_stop_gradient(x: Tree, /) -> Tree:
    return x


def eval_stop_gradient(x: Tree, /) -> Tree:
    return x


def pushforward_stop_gradient(in_tree: Tree, /) -> tuple[Tree, Tree]:
    primal, tangent = in_tree
    zero_tangent = treelib.map(zero_cotangent, primal)
    return primal, zero_tangent


def pullback_fwd_stop_gradient(x: Tree, /) -> tuple[Tree, Tree]:
    residuals = x
    return x, residuals


def pullback_bwd_stop_gradient(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    del out_cotangent
    return treelib.map(zero_cotangent, residuals)


def batch_stop_gradient(in_tree: Tree, /) -> tuple[Tree, Tree]:
    batch_size, in_batched, x = in_tree
    del batch_size
    return x, in_batched


impl_rules.set(stop_gradient_p, impl_stop_gradient)
impl_rules.aset(stop_gradient_p, asyncify(impl_stop_gradient))
eval_rules.set(stop_gradient_p, eval_stop_gradient)
push_rules.set(stop_gradient_p, pushforward_stop_gradient)
push_rules.aset(stop_gradient_p, asyncify(pushforward_stop_gradient))
pull_fwd_rules.set(stop_gradient_p, pullback_fwd_stop_gradient)
pull_fwd_rules.aset(stop_gradient_p, asyncify(pullback_fwd_stop_gradient))
pull_bwd_rules.set(stop_gradient_p, pullback_bwd_stop_gradient)
pull_bwd_rules.aset(stop_gradient_p, asyncify(pullback_bwd_stop_gradient))
batch_rules.set(stop_gradient_p, batch_stop_gradient)
batch_rules.aset(stop_gradient_p, asyncify(batch_stop_gradient))


# ==================================================================================================
# SWITCH
# ==================================================================================================

switch_p = Primitive("switch", tag={ControlTag})


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
        ...     "zero": af.trace(lambda x: af.concat("zero: ", x))("X"),
        ...     "one": af.trace(lambda x: af.concat("one: ", x))("X"),
        ...     "two": af.trace(lambda x: af.concat("two: ", x))("X"),
        ... }
        >>> def ir(key, x):
        ...     return af.switch(key, branches, x)
        >>> ir = af.trace(ir)("one", "hello")
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


def impl_switch(in_tree, /, *, branches: dict[str, IR]):
    key, operands = in_tree
    return call(branches[key])(operands)


async def aimpl_switch(in_tree, /, *, branches: dict[str, IR]):
    key, operands = in_tree
    return await acall(branches[key])(operands)


def eval_switch(in_tree, /, *, branches: dict[str, IR]) -> Tree:
    del in_tree
    key0 = next(iter(branches))
    branch0 = branches[key0]
    return treelib.map(iratom_to_evaltype, branch0.out_irtree)


def pushforward_switch(in_tree, /, *, branches: dict[str, IR]):
    primals, tangents = in_tree
    (key, p_operands), (_, t_operands) = primals, tangents
    pf_ir = pushforward(branches[key])
    return call(pf_ir)((p_operands, t_operands))


async def apush_switch(in_tree, /, *, branches: dict[str, IR]):
    primals, tangents = in_tree
    (key, p_operands), (_, t_operands) = primals, tangents
    pf_ir = pushforward(branches[key])
    return await acall(pf_ir)((p_operands, t_operands))


def pullback_fwd_switch(in_tree, /, *, branches: dict[str, IR]) -> tuple[Tree, Tree]:
    key, operands = in_tree
    out = call(branches[key])(operands)
    residuals = (key, operands)
    return out, residuals


async def apull_fwd_switch(in_tree, /, *, branches: dict[str, IR]) -> tuple[Tree, Tree]:
    key, operands = in_tree
    out = await acall(branches[key])(operands)
    residuals = (key, operands)
    return out, residuals


def pullback_bwd_switch(in_tree, /, *, branches: dict[str, IR]):
    residuals, out_cotangent = in_tree
    key, operands = residuals
    pb_ir = pullback(branches[key])
    _, c_operands = call(pb_ir)((operands, out_cotangent))
    return (zero_cotangent(key), c_operands)


async def apull_bwd_switch(in_tree, /, *, branches: dict[str, IR]):
    residuals, out_cotangent = in_tree
    key, operands = residuals
    pb_ir = pullback(branches[key])
    _, c_operands = await acall(pb_ir)((operands, out_cotangent))
    return (zero_cotangent(key), c_operands)


def batch_switch(in_tree, /, *, branches: dict[str, IR]) -> tuple[Tree, bool]:
    batch_size, in_batched, in_values = in_tree
    key_col, operands_col = in_values
    key_batched, operands_batched = in_batched
    unbatch = ft.partial(batch_index, operands_col, operands_batched)

    def run_ir_at(b):
        return call(branches[key_col[b] if key_batched else key_col])(unbatch(b))

    result = [run_ir_at(b) for b in range(batch_size)]
    return batch_spec(in_values, in_batched).unflatten(result), True


async def abatch_switch(in_tree, /, *, branches: dict[str, IR]) -> tuple[Tree, bool]:
    batch_size, in_batched, in_values = in_tree
    key_col, operands_col = in_values
    key_batched, operands_batched = in_batched
    unbatch = ft.partial(batch_index, operands_col, operands_batched)

    async def run_ir_at(b):
        return await acall(branches[key_col[b] if key_batched else key_col])(unbatch(b))

    results = await asyncio.gather(*[run_ir_at(b) for b in range(batch_size)])
    return batch_spec(in_values, in_batched).unflatten(results), True


impl_rules.set(switch_p, impl_switch)
impl_rules.aset(switch_p, aimpl_switch)
eval_rules.set(switch_p, eval_switch)
push_rules.set(switch_p, pushforward_switch)
push_rules.aset(switch_p, apush_switch)
pull_fwd_rules.set(switch_p, pullback_fwd_switch)
pull_fwd_rules.aset(switch_p, apull_fwd_switch)
pull_bwd_rules.set(switch_p, pullback_bwd_switch)
pull_bwd_rules.aset(switch_p, apull_bwd_switch)
batch_rules.set(switch_p, batch_switch)
batch_rules.aset(switch_p, abatch_switch)


def dce_switch(ireqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    branches = {k: dce(v, out_used=out_used) for k, v in ireqn.params["branches"].items()}
    new_eqn = ireqn.using(branches=branches)
    return default_dce(new_eqn, out_used)


dce_rules[switch_p] = dce_switch


# ==================================================================================================
# WHILE LOOP
# ==================================================================================================

while_loop_p = Primitive("while_loop", tag={ControlTag})


def while_loop(cond_ir: IR, body_ir: IR, init_val: Tree, *, max_iters: int) -> Tree:
    """Repeatedly apply ``body_ir`` while ``cond_ir`` returns True.

    - Loop continues while cond_ir(state) returns True
    - body_ir is applied each iteration
    - Returns final state when cond_ir returns False or max_iters reached

    Args:
        cond_ir: IR that returns bool. Loop continues while True.
        body_ir: IR that transforms state, f: State -> State
        init_val: Initial state
        max_iters: Maximum iterations

    Returns:
        Final state when cond_ir returns False or max_iters reached.

    Example:
        >>> import autoform as af
        >>> def cond(x):
        ...     return False  # Exit immediately
        >>> def body(x):
        ...     return af.concat(x, "x")
        >>> cond_ir = af.trace(cond)("x")
        >>> body_ir = af.trace(body)("x")
        >>> result = af.while_loop(cond_ir, body_ir, "a", max_iters=10)
        >>> result
        'a'
    """
    assert isinstance(cond_ir, IR), f"cond_ir must be an IR, got {type(cond_ir)}"
    assert isinstance(body_ir, IR), f"body_ir must be an IR, got {type(body_ir)}"

    in_struct = treelib.structure(body_ir.in_irtree)
    out_struct = treelib.structure(body_ir.out_irtree)
    assert in_struct == out_struct, (
        f"body_ir must have identical input/output structure (f: State -> State).\n"
        f"in_struct:  {in_struct}\n"
        f"out_struct: {out_struct}"
    )
    return while_loop_p.bind(
        init_val,
        cond_ir=cond_ir,
        body_ir=body_ir,
        max_iters=max_iters,
    )


def impl_while_loop(in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int) -> Tree:
    state = in_tree
    for _ in range(max_iters):
        if not call(cond_ir)(state):
            break
        state = call(body_ir)(state)
    return state


async def aimpl_while_loop(in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int) -> Tree:
    state = in_tree
    for _ in range(max_iters):
        if not await acall(cond_ir)(state):
            break
        state = await acall(body_ir)(state)
    return state


def eval_while_loop(in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int) -> Tree:
    del cond_ir, max_iters
    return treelib.map(iratom_to_evaltype, body_ir.out_irtree)


def pullback_fwd_while_loop(
    in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int
) -> tuple[Tree, Tree]:
    state = in_tree
    trajectory = [state]

    for _ in range(max_iters):
        if not call(cond_ir)(state):
            break
        state = call(body_ir)(state)
        trajectory.append(state)

    residuals = (trajectory, body_ir)
    return state, residuals


async def apull_fwd_while_loop(
    in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int
) -> tuple[Tree, Tree]:
    state = in_tree
    trajectory = [state]

    for _ in range(max_iters):
        if not await acall(cond_ir)(state):
            break
        state = await acall(body_ir)(state)
        trajectory.append(state)

    residuals = (trajectory, body_ir)
    return state, residuals


def pullback_bwd_while_loop(in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int) -> Tree:
    residuals, out_cotangent = in_tree
    del cond_ir, max_iters
    trajectory, _ = residuals
    n_iters = len(trajectory) - 1

    cotangent = out_cotangent
    pb_body = pullback(body_ir)

    for t in reversed(range(n_iters)):
        state_t = trajectory[t]
        _, cotangent = call(pb_body)((state_t, cotangent))

    return cotangent


async def apull_bwd_while_loop(
    in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int
) -> Tree:
    residuals, out_cotangent = in_tree
    del cond_ir, max_iters
    trajectory, _ = residuals
    n_iters = len(trajectory) - 1

    cotangent = out_cotangent
    pb_body = pullback(body_ir)

    for t in reversed(range(n_iters)):
        state_t = trajectory[t]
        _, cotangent = await acall(pb_body)((state_t, cotangent))

    return cotangent


def batch_while_loop(
    in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int
) -> tuple[Tree, Tree]:
    batch_size, in_batched, init_val = in_tree
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
    # >>> cond_ir: s.note != "done"
    # >>> body_ir: Struct(text=refine(s.text), note="done" if good else s.note)
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
    states = [batch_index(init_val, in_batched, b) for b in range(batch_size)]
    alive = [True] * batch_size

    # NOTE(asem): pre-batch cond and body IRs. True marks all leaves as batched.
    cond_in_axes = treelib.map(lambda _: True, cond_ir.in_irtree)
    body_in_axes = treelib.map(lambda _: True, body_ir.in_irtree)
    batched_cond = batch(cond_ir, in_axes=cond_in_axes)
    batched_body = batch(body_ir, in_axes=body_in_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(batch_size) if alive[i]]):
            break

        # NOTE(asem): check conditions only for alive items (transpose AoS -> SoA for call)
        alive_states = [states[i] for i in alive_idx]
        n_alive = len(alive_states)
        in_batched_cond = treelib.map(lambda _: True, cond_ir.in_irtree)
        # NOTE(asem): move from AoS to SoA for alive states
        in_transposed_cond = batch_transpose(n_alive, in_batched_cond, alive_states)
        conds_result = call(batched_cond)(in_transposed_cond)
        # NOTE(asem): cond returns scalar bool, batched -> list. use unbatch for consistency.
        out_batched_cond = isinstance(conds_result, list)
        conds = [batch_index(conds_result, out_batched_cond, b) for b in range(n_alive)]
        # NOTE(asem): mark items as dead if cond returned False
        for idx, c in zip(alive_idx, conds, strict=True):
            alive[idx] = c
        # NOTE(asem): run body ONLY on still-alive items
        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            n_still_alive = len(still_alive_states)
            in_batched = treelib.map(lambda _: True, body_ir.in_irtree)
            in_transposed = batch_transpose(n_still_alive, in_batched, still_alive_states)
            out_transposed = call(batched_body)(in_transposed)
            out_batched = treelib.map(is_irvar, body_ir.out_irtree)

            for local_idx, batch_idx in enumerate(still_alive):
                states[batch_idx] = batch_index(out_transposed, out_batched, local_idx)
    # NOTE(asem): transpose final states AoS -> SoA for batched output
    # only IRVar positions are batched; IRLit positions stay scalar
    out_batched = treelib.map(is_irvar, body_ir.out_irtree)
    out_tree = batch_transpose(batch_size, out_batched, states)
    in_spec = treelib.structure(init_val, is_leaf=lambda x: x is not init_val)
    out_tree = in_spec.unflatten(treelib.leaves(out_tree, is_leaf=lambda x: x is not out_tree))

    return out_tree, out_batched


async def abatch_while_loop(
    in_tree: Tree, /, *, cond_ir: IR, body_ir: IR, max_iters: int
) -> tuple[Tree, Tree]:
    batch_size, in_batched, init_val = in_tree

    # NOTE(asem): unbatch SoA -> AoS so each state can be tracked independently
    states = [batch_index(init_val, in_batched, b) for b in range(batch_size)]
    alive = [True] * batch_size

    # NOTE(asem): pre-batch cond and body IRs
    cond_in_axes = treelib.map(lambda _: True, cond_ir.in_irtree)
    body_in_axes = treelib.map(lambda _: True, body_ir.in_irtree)
    batched_cond = batch(cond_ir, in_axes=cond_in_axes)
    batched_body = batch(body_ir, in_axes=body_in_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(batch_size) if alive[i]]):
            break

        alive_states = [states[i] for i in alive_idx]
        n_alive = len(alive_states)
        in_batched_cond = treelib.map(lambda _: True, cond_ir.in_irtree)
        in_transposed_cond = batch_transpose(n_alive, in_batched_cond, alive_states)
        conds_result = await acall(batched_cond)(in_transposed_cond)
        out_batched_cond = isinstance(conds_result, list)
        conds = [batch_index(conds_result, out_batched_cond, b) for b in range(n_alive)]

        for idx, c in zip(alive_idx, conds, strict=True):
            alive[idx] = c

        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            n_still_alive = len(still_alive_states)
            in_batched_body = treelib.map(lambda _: True, body_ir.in_irtree)
            in_transposed = batch_transpose(n_still_alive, in_batched_body, still_alive_states)
            out_transposed = await acall(batched_body)(in_transposed)
            out_batched_body = treelib.map(is_irvar, body_ir.out_irtree)

            for local_idx, batch_idx in enumerate(still_alive):
                states[batch_idx] = batch_index(out_transposed, out_batched_body, local_idx)

    out_batched = treelib.map(is_irvar, body_ir.out_irtree)
    out_tree = batch_transpose(batch_size, out_batched, states)
    in_spec = treelib.structure(init_val, is_leaf=lambda x: x is not init_val)
    out_tree = in_spec.unflatten(treelib.leaves(out_tree, is_leaf=lambda x: x is not out_tree))
    return out_tree, out_batched


impl_rules.set(while_loop_p, impl_while_loop)
impl_rules.aset(while_loop_p, aimpl_while_loop)
eval_rules.set(while_loop_p, eval_while_loop)
pull_fwd_rules.set(while_loop_p, pullback_fwd_while_loop)
pull_fwd_rules.aset(while_loop_p, apull_fwd_while_loop)
pull_bwd_rules.set(while_loop_p, pullback_bwd_while_loop)
pull_bwd_rules.aset(while_loop_p, apull_bwd_while_loop)
batch_rules.set(while_loop_p, batch_while_loop)
batch_rules.aset(while_loop_p, abatch_while_loop)
