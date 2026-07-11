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

"""Control flow primitives"""

from __future__ import annotations

import functools as ft

import autoform.ad as ad
import autoform.batch as batch
import autoform.core as core
import autoform.dce as dce
import autoform.scheduling as scheduling
import autoform.utils as utils

__all__ = ["stop_gradient", "switch", "while_loop", "fixpoint"]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type Branches = dict[str, core.IR]

# ==================================================================================================
# STOP GRADIENT
# ==================================================================================================

stop_gradient_p = core.Prim("stop_gradient")


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
        >>> _, (cotangent_x, cotangent_y) = pb_ir.call(("a", "b"), "grad")
        >>> cotangent_x
        Zero(StrAVal())
        >>> cotangent_y
        'grad'
    """
    return stop_gradient_p.bind(x)


def impl_stop_gradient(x: Tree, /) -> Tree:
    return x


def abstract_stop_gradient(x: Tree, /) -> Tree:
    return x


def pushforward_stop_gradient(in_tree: Tree, /) -> TreePair:
    primal, tangent = in_tree
    zero_t = utils.tree.map(lambda p: p if ad.is_zero(p) else ad.zeroof(p), primal)
    return primal, zero_t


def pullback_fwd_stop_gradient(x: Tree, /) -> TreePair:
    residuals = x
    return x, residuals


def pullback_bwd_stop_gradient(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    del out_cotangent
    return utils.tree.map(lambda r: r if ad.is_zero(r) else ad.zeroof(r), residuals)


def batch_stop_gradient(in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, x = in_tree
    del batch_size
    return x, in_batched


core.impl_rules.set(stop_gradient_p, impl_stop_gradient)
core.impl_rules.aset(stop_gradient_p, utils.asyncify(impl_stop_gradient))
core.abstract_rules.set(stop_gradient_p, abstract_stop_gradient)
core.push_rules.set(stop_gradient_p, pushforward_stop_gradient)
core.push_rules.aset(stop_gradient_p, utils.asyncify(pushforward_stop_gradient))
core.pull_fwd_rules.set(stop_gradient_p, pullback_fwd_stop_gradient)
core.pull_fwd_rules.aset(stop_gradient_p, utils.asyncify(pullback_fwd_stop_gradient))
core.pull_bwd_rules.set(stop_gradient_p, pullback_bwd_stop_gradient)
core.pull_bwd_rules.aset(stop_gradient_p, utils.asyncify(pullback_bwd_stop_gradient))
core.batch_rules.set(stop_gradient_p, batch_stop_gradient)
core.batch_rules.aset(stop_gradient_p, utils.asyncify(batch_stop_gradient))


# ==================================================================================================
# SWITCH
# ==================================================================================================

switch_p = core.Prim("switch")


def switch(key: str, branches: Branches, *args, **kwargs) -> Tree:
    """Select and execute one of multiple IR branches based on a string key.

    Args:
        key: String key selecting which branch to execute.
        branches: Dict mapping string keys to IR irs, each with compatible input signature.
        *args: Positional arguments passed to the selected branch.

    Returns:
        Result of ``branches[key].call(*args)``

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
        >>> ir.call("one", "hello")
        'one: hello'
        >>> ir.call("zero", "hello")
        'zero: hello'
    """
    assert not kwargs, "`switch` does not support keyword arguments"
    assert all(isinstance(branches[k], core.IR) for k in branches)
    tree_struct0 = utils.tree.structure(branches[next(iter(branches))].in_tree)
    assert all(utils.tree.structure(branches[key].in_tree) == tree_struct0 for key in branches)
    tree_struct0 = utils.tree.structure(branches[next(iter(branches))].out_tree)
    assert all(utils.tree.structure(branches[key].out_tree) == tree_struct0 for key in branches)
    return switch_p.bind((key, args), branches=branches)


def impl_switch(in_tree, /, *, branches: Branches):
    key, operands = in_tree
    return branches[key].call(*operands)


async def aimpl_switch(in_tree, /, *, branches: Branches):
    key, operands = in_tree
    return await branches[key].acall(*operands)


def abstract_switch(in_tree, /, *, branches: Branches) -> Tree:
    key, _ = in_tree
    assert type(key) in (str, core.StrAVal), f"`switch` expects string key: {key!r}"
    key0 = next(iter(branches))
    branch0 = branches[key0]
    return utils.tree.map(core.aval_if_var, branch0.out_tree)


def pushforward_switch(in_tree, /, *, branches: Branches):
    primals, tangents = in_tree
    (key, p_operands), (_, t_operands) = primals, tangents
    pf_ir = ad.pushforward(branches[key])
    return pf_ir.call(p_operands, t_operands)


async def apush_switch(in_tree, /, *, branches: Branches):
    primals, tangents = in_tree
    (key, p_operands), (_, t_operands) = primals, tangents
    pf_ir = ad.pushforward(branches[key])
    return await pf_ir.acall(p_operands, t_operands)


def pullback_fwd_switch(in_tree, /, *, branches: Branches) -> TreePair:
    key, operands = in_tree
    out = branches[key].call(*operands)
    residuals = (key, operands)
    return out, residuals


async def apull_fwd_switch(in_tree, /, *, branches: Branches) -> TreePair:
    key, operands = in_tree
    out = await branches[key].acall(*operands)
    residuals = (key, operands)
    return out, residuals


def pullback_bwd_switch(in_tree, /, *, branches: Branches):
    residuals, out_cotangent = in_tree
    key, operands = residuals
    pb_ir = ad.pullback(branches[key])
    _, c_operands = pb_ir.call(operands, out_cotangent)
    return (ad.zeroof(key), c_operands)


async def apull_bwd_switch(in_tree, /, *, branches: Branches):
    residuals, out_cotangent = in_tree
    key, operands = residuals
    pb_ir = ad.pullback(branches[key])
    _, c_operands = await pb_ir.acall(operands, out_cotangent)
    return (ad.zeroof(key), c_operands)


def batch_switch(in_tree, /, *, branches: Branches) -> core.BatchRuleResult:
    batch_size, in_batched, in_values = in_tree
    key_col, operands_col = in_values
    key_batched, operands_batched = in_batched

    if utils.batch_spec(in_values, in_batched) is None:
        return switch_p.bind(in_values, branches=branches), False

    unbatch = ft.partial(utils.batch_index, operands_col, operands_batched)

    def run_ir_at(b):
        return branches[key_col[b] if key_batched else key_col].call(*unbatch(b))

    results = [run_ir_at(b) for b in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_tree = utils.batch_transpose(batch_size, out_batched, results)
    return out_tree, out_batched


async def abatch_switch(in_tree, /, *, branches: Branches) -> core.BatchRuleResult:
    batch_size, in_batched, in_values = in_tree
    key_col, operands_col = in_values
    key_batched, operands_batched = in_batched

    if utils.batch_spec(in_values, in_batched) is None:
        return await switch_p.abind(in_values, branches=branches), False

    unbatch = ft.partial(utils.batch_index, operands_col, operands_batched)

    irs = [branches[key_col[b] if key_batched else key_col] for b in range(batch_size)]
    inputs = [unbatch(b) for b in range(batch_size)]
    results = await scheduling.gather_p.abind(inputs, irs=irs)
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_tree = utils.batch_transpose(batch_size, out_batched, results)
    return out_tree, out_batched


core.impl_rules.set(switch_p, impl_switch)
core.impl_rules.aset(switch_p, aimpl_switch)
core.abstract_rules.set(switch_p, abstract_switch)
core.push_rules.set(switch_p, pushforward_switch)
core.push_rules.aset(switch_p, apush_switch)
core.pull_fwd_rules.set(switch_p, pullback_fwd_switch)
core.pull_fwd_rules.aset(switch_p, apull_fwd_switch)
core.pull_bwd_rules.set(switch_p, pullback_bwd_switch)
core.pull_bwd_rules.aset(switch_p, apull_bwd_switch)
core.batch_rules.set(switch_p, batch_switch)
core.batch_rules.aset(switch_p, abatch_switch)


def dce_switch(eqn: core.Eqn, out_used: dce.UsedTree, /) -> dce.DCEResult:
    branches: Branches = eqn.params["branches"]
    branches = {k: dce.dce(branches[k], out_used=out_used) for k in branches}
    new_eqn = eqn.using(branches=branches)
    return dce.default_dce(new_eqn, out_used)


dce.dce_rules[switch_p] = dce_switch


# ==================================================================================================
# WHILE LOOP
# ==================================================================================================

while_loop_p = core.Prim("while_loop")


def while_loop(cond_ir: core.IR, body_ir: core.IR, init_val: Tree, *, max_iters: int) -> Tree:
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
        ...     return af.match(x, "go")
        >>> def body(x):
        ...     return "stop"
        >>> cond_ir = af.trace(cond)("...")
        >>> body_ir = af.trace(body)("...")
        >>> result = af.while_loop(cond_ir, body_ir, "go", max_iters=10)
        >>> result
        'stop'
    """
    assert isinstance(cond_ir, core.IR), f"cond_ir must be an IR, got {type(cond_ir)}"
    assert isinstance(body_ir, core.IR), f"body_ir must be an IR, got {type(body_ir)}"
    assert len(cond_ir.in_tree) == 1, "cond_ir must take exactly one positional argument"
    assert len(body_ir.in_tree) == 1, "body_ir must take exactly one positional argument"

    in_struct = utils.tree.structure(body_ir.in_tree[0])
    out_struct = utils.tree.structure(body_ir.out_tree)
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


def impl_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> Tree:
    state = (in_tree,)
    out = in_tree
    for _ in range(max_iters):
        if not cond_ir.call(*state):
            break
        out = body_ir.call(*state)
        state = (out,)
    return out


async def aimpl_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> Tree:
    state = (in_tree,)
    out = in_tree
    for _ in range(max_iters):
        if not await cond_ir.acall(*state):
            break
        out = await body_ir.acall(*state)
        state = (out,)
    return out


def abstract_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> Tree:
    del in_tree, cond_ir, max_iters
    return utils.tree.map(core.aval_if_var, body_ir.out_tree)


def pullback_fwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> TreePair:
    state = (in_tree,)
    out = in_tree
    trajectory = [state]

    for _ in range(max_iters):
        if not cond_ir.call(*state):
            break
        out = body_ir.call(*state)
        state = (out,)
        trajectory.append(state)

    residuals = (trajectory, body_ir)
    return out, residuals


async def apull_fwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> TreePair:
    state = (in_tree,)
    out = in_tree
    trajectory = [state]

    for _ in range(max_iters):
        if not await cond_ir.acall(*state):
            break
        out = await body_ir.acall(*state)
        state = (out,)
        trajectory.append(state)

    residuals = (trajectory, body_ir)
    return out, residuals


def pullback_bwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> Tree:
    residuals, out_cotangent = in_tree
    del cond_ir, max_iters
    trajectory, _ = residuals
    n_iters = len(trajectory) - 1

    cotangent = out_cotangent
    pb_body = ad.pullback(body_ir)

    for t in reversed(range(n_iters)):
        state_t = trajectory[t]
        _, cotangent = pb_body.call(state_t, cotangent)
        cotangent = cotangent[0]

    return cotangent


async def apull_bwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> Tree:
    residuals, out_cotangent = in_tree
    del cond_ir, max_iters
    trajectory, _ = residuals
    n_iters = len(trajectory) - 1

    cotangent = out_cotangent
    pb_body = ad.pullback(body_ir)

    for t in reversed(range(n_iters)):
        state_t = trajectory[t]
        _, cotangent = await pb_body.acall(state_t, cotangent)
        cotangent = cotangent[0]

    return cotangent


def batch_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> TreePair:
    b_sz, in_batched, init_val = in_tree
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
    init_at = ft.partial(utils.batch_index, init_val, in_batched)
    states = [(init_at(b),) for b in range(b_sz)]
    alive = [True] * b_sz

    # NOTE(asem): pre-batch cond and body IRs. True marks all leaves as batched.
    state_in_axes = utils.tree.map(lambda _: True, body_ir.in_tree)
    cond_in_axes = state_in_axes
    body_in_axes = state_in_axes
    batched_cond = batch.batch(cond_ir, in_axes=cond_in_axes)
    batched_body = batch.batch(body_ir, in_axes=body_in_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        # NOTE(asem): check conditions only for alive items (transpose AoS -> SoA for call)
        alive_states = [states[i] for i in alive_idx]
        n_alive = len(alive_states)
        in_batched_cond = state_in_axes
        # NOTE(asem): move from AoS to SoA for alive states
        in_transposed_cond = utils.batch_transpose(n_alive, in_batched_cond, alive_states)
        conds_result = batched_cond.call(*in_transposed_cond)
        # NOTE(asem): cond returns scalar bool, batched -> list. use unbatch for consistency.
        out_batched_cond = isinstance(conds_result, list)
        cond_at = ft.partial(utils.batch_index, conds_result, out_batched_cond)
        conds = [cond_at(b) for b in range(n_alive)]
        # NOTE(asem): mark items as dead if cond returned False
        for idx, c in zip(alive_idx, conds, strict=True):
            alive[idx] = c
        # NOTE(asem): run body ONLY on still-alive items
        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            n_body = len(still_alive_states)
            b_body = state_in_axes
            in_transposed = utils.batch_transpose(n_body, b_body, still_alive_states)
            out_transposed = batched_body.call(*in_transposed)
            out_batched = utils.tree.map(core.is_var, body_ir.out_tree)
            out_at = ft.partial(utils.batch_index, out_transposed, out_batched)

            for local_idx, batch_idx in enumerate(still_alive):
                states[batch_idx] = (out_at(local_idx),)
    # NOTE(asem): transpose final states AoS -> SoA for batched output
    # only Var positions are batched; literal positions stay scalar
    out_batched = utils.tree.map(core.is_var, body_ir.out_tree)
    out_tree = utils.batch_transpose(b_sz, out_batched, [state[0] for state in states])
    in_spec = utils.tree.structure(init_val, is_leaf=lambda x: x is not init_val)
    out_tree = in_spec.unflatten(utils.tree.leaves(out_tree, is_leaf=lambda x: x is not out_tree))
    return out_tree, out_batched


async def abatch_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: core.IR,
    body_ir: core.IR,
    max_iters: int,
) -> TreePair:
    b_sz, in_batched, init_val = in_tree

    # NOTE(asem): unbatch SoA -> AoS so each state can be tracked independently
    init_at = ft.partial(utils.batch_index, init_val, in_batched)
    states = [(init_at(b),) for b in range(b_sz)]
    alive = [True] * b_sz

    # NOTE(asem): pre-batch cond and body IRs
    state_in_axes = utils.tree.map(lambda _: True, body_ir.in_tree)
    cond_in_axes = state_in_axes
    body_in_axes = state_in_axes
    batched_cond = batch.batch(cond_ir, in_axes=cond_in_axes)
    batched_body = batch.batch(body_ir, in_axes=body_in_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        alive_states = [states[i] for i in alive_idx]
        n_alive = len(alive_states)
        in_batched_cond = state_in_axes
        in_transposed_cond = utils.batch_transpose(n_alive, in_batched_cond, alive_states)
        conds_result = await batched_cond.acall(*in_transposed_cond)
        out_batched_cond = isinstance(conds_result, list)
        cond_at = ft.partial(utils.batch_index, conds_result, out_batched_cond)
        conds = [cond_at(b) for b in range(n_alive)]

        for idx, c in zip(alive_idx, conds, strict=True):
            alive[idx] = c

        if still_alive := [i for i in alive_idx if alive[i]]:
            still_alive_states = [states[i] for i in still_alive]
            n_body = len(still_alive_states)
            b_body = state_in_axes
            in_transposed = utils.batch_transpose(n_body, b_body, still_alive_states)
            out_transposed = await batched_body.acall(*in_transposed)
            out_batched_body = utils.tree.map(core.is_var, body_ir.out_tree)
            out_at = ft.partial(utils.batch_index, out_transposed, out_batched_body)

            for local_idx, batch_idx in enumerate(still_alive):
                states[batch_idx] = (out_at(local_idx),)

    out_batched = utils.tree.map(core.is_var, body_ir.out_tree)
    out_tree = utils.batch_transpose(b_sz, out_batched, [state[0] for state in states])
    in_spec = utils.tree.structure(init_val, is_leaf=lambda x: x is not init_val)
    out_tree = in_spec.unflatten(utils.tree.leaves(out_tree, is_leaf=lambda x: x is not out_tree))
    return out_tree, out_batched


core.impl_rules.set(while_loop_p, impl_while_loop)
core.impl_rules.aset(while_loop_p, aimpl_while_loop)
core.abstract_rules.set(while_loop_p, abstract_while_loop)
core.pull_fwd_rules.set(while_loop_p, pullback_fwd_while_loop)
core.pull_fwd_rules.aset(while_loop_p, apull_fwd_while_loop)
core.pull_bwd_rules.set(while_loop_p, pullback_bwd_while_loop)
core.pull_bwd_rules.aset(while_loop_p, apull_bwd_while_loop)
core.batch_rules.set(while_loop_p, batch_while_loop)
core.batch_rules.aset(while_loop_p, abatch_while_loop)


def dce_while_loop(eqn: core.Eqn, out_used: dce.UsedTree, /) -> dce.DCEResult:
    cond_ir = eqn.params["cond_ir"]
    body_ir = eqn.params["body_ir"]
    # NOTE(asem): every state leaf is loop-carried into later condition/body calls,
    # even if the caller only uses part of the final state.
    state_used = utils.tree.map(lambda _: True, body_ir.out_tree)
    cond_ir = dce.dce(cond_ir)
    body_ir = dce.dce(body_ir, out_used=state_used)
    new_eqn = eqn.using(cond_ir=cond_ir, body_ir=body_ir)
    return dce.default_dce(new_eqn, out_used)


dce.dce_rules[while_loop_p] = dce_while_loop


# ==================================================================================================
# FIXPOINT
# ==================================================================================================

fixpoint_p = core.Prim("fixpoint")


def cot_tree_acc(lhs: Tree, rhs: Tree, /) -> Tree:
    return utils.tree.map(lambda l, r: ad.cot_acc([l, r]), lhs, rhs, is_leaf=ad.is_zero)


def fixpoint(
    step_ir: core.IR,
    init_val: Tree,
    theta: Tree = (),
    *,
    max_iters: int,
    adj_iters: int = 1,
    equiv_ir: core.IR | None = None,
) -> Tree:
    """Iterate ``step_ir`` until the state reaches a fixed point.

    ``step_ir`` must have shape ``(State, Theta) -> State``. The loop stops when
    the new state is equivalent to the previous state, or when ``max_iters``
    is reached. Equivalence defaults to structural equality of the state
    pytree; pass ``equiv_ir`` with shape ``(State, State) -> Bool`` to decide
    stability inside the program.
    """
    assert isinstance(step_ir, core.IR), f"step_ir must be an IR, got {type(step_ir)}"
    assert len(step_ir.in_tree) == 2, "step_ir must take exactly two positional arguments"

    in_struct = utils.tree.structure(step_ir.in_tree[0])
    out_struct = utils.tree.structure(step_ir.out_tree)
    assert in_struct == out_struct, (
        f"step_ir must have identical state input/output structure (step: (State, Theta) -> State).\n"
        f"in_struct:  {in_struct}\n"
        f"out_struct: {out_struct}"
    )
    assert isinstance(max_iters, int) and max_iters >= 1, f"max_iters must be >= 1: {max_iters!r}"
    assert isinstance(adj_iters, int) and adj_iters >= 0, f"adj_iters must be >= 0: {adj_iters!r}"
    if equiv_ir is not None:
        assert isinstance(equiv_ir, core.IR), f"equiv_ir must be an IR, got {type(equiv_ir)}"
        assert len(equiv_ir.in_tree) == 2, "equiv_ir must take two positional arguments"
        for side in equiv_ir.in_tree:
            assert utils.tree.structure(side) == in_struct, (
                "equiv_ir inputs must both match the state structure"
            )
    return fixpoint_p.bind(
        (init_val, theta),
        step_ir=step_ir,
        max_iters=max_iters,
        adj_iters=adj_iters,
        equiv_ir=equiv_ir,
    )


def impl_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> Tree:
    del adj_iters
    state, theta = in_tree

    for _ in range(max_iters):
        new_state = step_ir.call(state, theta)
        stable = (
            utils.tree_equal(state, new_state)
            if equiv_ir is None
            else equiv_ir.call(state, new_state)
        )
        if stable:
            return new_state
        state = new_state
    return state


async def aimpl_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> Tree:
    del adj_iters
    state, theta = in_tree

    for _ in range(max_iters):
        new_state = await step_ir.acall(state, theta)
        stable = (
            utils.tree_equal(state, new_state)
            if equiv_ir is None
            else await equiv_ir.acall(state, new_state)
        )
        if stable:
            return new_state
        state = new_state
    return state


def abstract_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> Tree:
    del in_tree, max_iters, adj_iters, equiv_ir
    return utils.tree.map(core.aval_if_var, step_ir.out_tree)


def pullback_fwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> TreePair:
    out = fixpoint_p.bind(
        in_tree, step_ir=step_ir, max_iters=max_iters, adj_iters=adj_iters, equiv_ir=equiv_ir
    )
    _, theta = in_tree
    return out, (out, theta)


async def apull_fwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> TreePair:
    out = await fixpoint_p.abind(
        in_tree, step_ir=step_ir, max_iters=max_iters, adj_iters=adj_iters, equiv_ir=equiv_ir
    )
    _, theta = in_tree
    return out, (out, theta)


def pullback_bwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> Tree:
    del max_iters, equiv_ir
    residuals, g = in_tree
    x_star, theta = residuals
    dx0 = utils.tree.map(ad.zeroof, x_star)

    if ad.all_zero(g):
        return dx0, utils.tree.map(ad.zeroof, theta)

    res: dict[core.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = ad.PullbackFwdInterpreter(parent=parent)

    with core.using_interpreter(fwd):

        def custom_bind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            boxed_out, residuals = eqn.bind(boxed_in, **eqn.params)
            res[eqn] = residuals
            return boxed_out

        eqn, boxed_in = next(gen := step_ir.walk(*utils.tree.map(fwd.box, (x_star, theta))))
        while eqn:
            eqn, boxed_in = gen.send(custom_bind(eqn, boxed_in))

    def transpose_eq(cot: Tree, /) -> Tree:
        bwd = ad.PullbackBwdInterpreter(parent=parent)
        with core.using_interpreter(bwd):

            def custom_bind(eqn: core.Eqn, c_out: Tree, /) -> Tree:
                residuals = res[eqn]
                boxed_c_out = utils.tree.map(bwd.box, c_out)
                boxed_c_in = eqn.bind((residuals, boxed_c_out), **eqn.params)
                return bwd.unbox(boxed_c_in)

            eqn, c_out = next(gen := ad.transpose_walk(step_ir, cot))
            while eqn:
                eqn, c_out = gen.send(custom_bind(eqn, c_out))
        return c_out

    # g is the output cotangent at x*.  transpose_eq(u) returns cotangents for (x*, theta).
    u = g

    for _ in range(adj_iters):
        x_bar, theta_bar = transpose_eq(u)
        next_u = cot_tree_acc(g, x_bar)
        if utils.tree_equal(next_u, u):
            dtheta = theta_bar
            break
        u = next_u
    else:
        _, dtheta = transpose_eq(u)

    return dx0, dtheta


async def apull_bwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> Tree:
    del max_iters, equiv_ir
    residuals, g = in_tree
    x_star, theta = residuals
    dx0 = utils.tree.map(ad.zeroof, x_star)

    if ad.all_zero(g):
        return dx0, utils.tree.map(ad.zeroof, theta)

    res: dict[core.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = ad.PullbackFwdInterpreter(parent=parent)

    with core.using_interpreter(fwd):

        async def custom_abind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            boxed_out, residuals = await eqn.abind(boxed_in, **eqn.params)
            res[eqn] = residuals
            return boxed_out

        eqn, boxed_in = next(gen := step_ir.walk(*utils.tree.map(fwd.box, (x_star, theta))))
        while eqn:
            eqn, boxed_in = gen.send(await custom_abind(eqn, boxed_in))

    async def atranspose_eq(cot: Tree, /) -> Tree:
        bwd = ad.PullbackBwdInterpreter(parent=parent)
        with core.using_interpreter(bwd):

            async def custom_abind(eqn: core.Eqn, c_out: Tree, /) -> Tree:
                residuals = res[eqn]
                boxed_c_out = utils.tree.map(bwd.box, c_out)
                boxed_c_in = await eqn.abind((residuals, boxed_c_out), **eqn.params)
                return bwd.unbox(boxed_c_in)

            eqn, c_out = next(gen := ad.transpose_walk(step_ir, cot))
            while eqn:
                eqn, c_out = gen.send(await custom_abind(eqn, c_out))
        return c_out

    # g is the output cotangent at x*.  atranspose_eq(u) returns cotangents for (x*, theta).
    u = g

    for _ in range(adj_iters):
        x_bar, theta_bar = await atranspose_eq(u)
        next_u = cot_tree_acc(g, x_bar)
        if utils.tree_equal(next_u, u):
            dtheta = theta_bar
            break
        u = next_u
    else:
        _, dtheta = await atranspose_eq(u)

    return dx0, dtheta


def batch_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> TreePair:
    b_sz, in_batched, in_values = in_tree
    params = dict(step_ir=step_ir, max_iters=max_iters, adj_iters=adj_iters, equiv_ir=equiv_ir)

    if utils.batch_spec(in_values, in_batched) is None:
        out = fixpoint_p.bind(in_values, **params)
        return out, utils.tree.map(lambda _: False, out)

    init_val, theta = in_values
    init_batched, theta_batched = in_batched
    init_at = ft.partial(utils.batch_index, init_val, init_batched)
    theta_at = ft.partial(utils.batch_index, theta, theta_batched)
    states = [init_at(b) for b in range(b_sz)]
    thetas = [theta_at(b) for b in range(b_sz)]
    alive = [True] * b_sz

    state_in_axes = utils.tree.map(lambda _: True, step_ir.in_tree[0])
    theta_in_axes = utils.tree.map(lambda _: True, step_ir.in_tree[1])
    in_axes = (state_in_axes, theta_in_axes)
    batched_step = batch.batch(step_ir, in_axes=in_axes)
    if equiv_ir is not None:
        equiv_axes = (state_in_axes, state_in_axes)
        batched_equiv = batch.batch(equiv_ir, in_axes=equiv_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        alive_in = [(states[i], thetas[i]) for i in alive_idx]
        n_alive = len(alive_in)
        in_transposed = utils.batch_transpose(n_alive, in_axes, alive_in)
        out_transposed = batched_step.call(*in_transposed)
        out_batched = utils.tree.map(core.is_var, step_ir.out_tree)
        out_at = ft.partial(utils.batch_index, out_transposed, out_batched)

        new_states = [out_at(i) for i in range(n_alive)]
        if equiv_ir is None:
            flags = [
                utils.tree_equal(states[b], ns) for b, ns in zip(alive_idx, new_states, strict=True)
            ]
        else:
            pairs = [(states[b], ns) for b, ns in zip(alive_idx, new_states, strict=True)]
            equiv_in = utils.batch_transpose(n_alive, equiv_axes, pairs)
            flags = batched_equiv.call(*equiv_in)

        for flag, batch_idx, new_state in zip(flags, alive_idx, new_states, strict=True):
            if flag:
                alive[batch_idx] = False
            states[batch_idx] = new_state

    out_batched = utils.tree.map(core.is_var, step_ir.out_tree)
    out_tree = utils.batch_transpose(b_sz, out_batched, states)
    in_spec = utils.tree.structure(init_val, is_leaf=lambda x: x is not init_val)
    out_tree = in_spec.unflatten(utils.tree.leaves(out_tree, is_leaf=lambda x: x is not out_tree))
    return out_tree, out_batched


async def abatch_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: core.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: core.IR | None,
) -> TreePair:
    b_sz, in_batched, in_values = in_tree
    params = dict(step_ir=step_ir, max_iters=max_iters, adj_iters=adj_iters, equiv_ir=equiv_ir)

    if utils.batch_spec(in_values, in_batched) is None:
        out = await fixpoint_p.abind(in_values, **params)
        return out, utils.tree.map(lambda _: False, out)

    init_val, theta = in_values
    init_batched, theta_batched = in_batched
    init_at = ft.partial(utils.batch_index, init_val, init_batched)
    theta_at = ft.partial(utils.batch_index, theta, theta_batched)
    states = [init_at(b) for b in range(b_sz)]
    thetas = [theta_at(b) for b in range(b_sz)]
    alive = [True] * b_sz

    state_in_axes = utils.tree.map(lambda _: True, step_ir.in_tree[0])
    theta_in_axes = utils.tree.map(lambda _: True, step_ir.in_tree[1])
    in_axes = (state_in_axes, theta_in_axes)
    batched_step = batch.batch(step_ir, in_axes=in_axes)
    if equiv_ir is not None:
        equiv_axes = (state_in_axes, state_in_axes)
        batched_equiv = batch.batch(equiv_ir, in_axes=equiv_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        alive_in = [(states[i], thetas[i]) for i in alive_idx]
        n_alive = len(alive_in)
        in_transposed = utils.batch_transpose(n_alive, in_axes, alive_in)
        out_transposed = await batched_step.acall(*in_transposed)
        out_batched = utils.tree.map(core.is_var, step_ir.out_tree)
        out_at = ft.partial(utils.batch_index, out_transposed, out_batched)

        new_states = [out_at(i) for i in range(n_alive)]
        if equiv_ir is None:
            flags = [
                utils.tree_equal(states[b], ns) for b, ns in zip(alive_idx, new_states, strict=True)
            ]
        else:
            pairs = [(states[b], ns) for b, ns in zip(alive_idx, new_states, strict=True)]
            equiv_in = utils.batch_transpose(n_alive, equiv_axes, pairs)
            flags = await batched_equiv.acall(*equiv_in)

        for flag, batch_idx, new_state in zip(flags, alive_idx, new_states, strict=True):
            if flag:
                alive[batch_idx] = False
            states[batch_idx] = new_state

    out_batched = utils.tree.map(core.is_var, step_ir.out_tree)
    out_tree = utils.batch_transpose(b_sz, out_batched, states)
    in_spec = utils.tree.structure(init_val, is_leaf=lambda x: x is not init_val)
    out_tree = in_spec.unflatten(utils.tree.leaves(out_tree, is_leaf=lambda x: x is not out_tree))
    return out_tree, out_batched


core.impl_rules.set(fixpoint_p, impl_fixpoint)
core.impl_rules.aset(fixpoint_p, aimpl_fixpoint)
core.abstract_rules.set(fixpoint_p, abstract_fixpoint)
core.pull_fwd_rules.set(fixpoint_p, pullback_fwd_fixpoint)
core.pull_fwd_rules.aset(fixpoint_p, apull_fwd_fixpoint)
core.pull_bwd_rules.set(fixpoint_p, pullback_bwd_fixpoint)
core.pull_bwd_rules.aset(fixpoint_p, apull_bwd_fixpoint)
core.batch_rules.set(fixpoint_p, batch_fixpoint)
core.batch_rules.aset(fixpoint_p, abatch_fixpoint)


def dce_fixpoint(eqn: core.Eqn, out_used: dce.UsedTree, /) -> dce.DCEResult:
    step_ir = eqn.params["step_ir"]
    equiv_ir = eqn.params["equiv_ir"]
    # NOTE(asem): every state leaf is loop-carried into the next step, even if
    # the caller only uses part of the final state.
    state_used = utils.tree.map(lambda _: True, step_ir.out_tree)
    equiv_ir = None if equiv_ir is None else dce.dce(equiv_ir)
    step_ir = dce.dce(step_ir, out_used=state_used)
    new_eqn = eqn.using(step_ir=step_ir, equiv_ir=equiv_ir)
    return dce.default_dce(new_eqn, out_used)


dce.dce_rules[fixpoint_p] = dce_fixpoint
