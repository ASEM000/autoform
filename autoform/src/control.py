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

from collections.abc import Hashable
from typing import Any

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.order as order
import autoform.src.stage as stage
import autoform.src.tree as tree
import autoform.src.utils as utils

zip = utils.strict_zip

type Tree[T] = Any
type TreePair = tuple[Tree, Tree]
type Branches = dict[Hashable, stage.IR]

# ==================================================================================================
# STOP GRADIENT
# ==================================================================================================

p_stop_gradient = core.Prim("stop_gradient")


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
        ...     return stopped + y
        >>> ir = af.trace(ir)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> _, (c_x, c_y) = pb_ir.call(("a", "b"), "grad")
        >>> c_x
        Zero(AStr())
        >>> c_y
        'grad'
    """
    return x if all(map(stage.is_traceable, tree.leaves(x))) else p_stop_gradient.bind(x)


def impl_stop_gradient(x: Tree, /) -> Tree:
    return x


def abstract_stop_gradient(x: Tree, /) -> Tree:
    return x


def pushforward_stop_gradient(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    zero_t = core.s_tangent.zeros_like(p)
    return p, zero_t


def pullback_fwd_stop_gradient(x: Tree, /) -> TreePair:
    residuals = x
    return x, residuals


def pullback_bwd_stop_gradient(in_tree: Tree, /) -> Tree:
    residuals, out_c = in_tree
    del out_c
    return core.s_cotangent.zeros_like(residuals)


def batch_stop_gradient(in_tree: Tree, /) -> TreePair:
    _, in_axes, x = in_tree
    return x, in_axes


core.r_impl.set(p_stop_gradient, impl_stop_gradient)
core.r_aimpl.set(p_stop_gradient, utils.asyncify(impl_stop_gradient))
core.r_abstract.set(p_stop_gradient, abstract_stop_gradient)
core.r_batch.set(p_stop_gradient, batch_stop_gradient)
core.r_abatch.set(p_stop_gradient, utils.asyncify(batch_stop_gradient))
core.r_push.set(p_stop_gradient, pushforward_stop_gradient)
core.r_apush.set(p_stop_gradient, utils.asyncify(pushforward_stop_gradient))
core.r_pull_fwd.set(p_stop_gradient, pullback_fwd_stop_gradient)
core.r_apull_fwd.set(p_stop_gradient, utils.asyncify(pullback_fwd_stop_gradient))
core.r_pull_bwd.set(p_stop_gradient, pullback_bwd_stop_gradient)
core.r_apull_bwd.set(p_stop_gradient, utils.asyncify(pullback_bwd_stop_gradient))


# ==================================================================================================
# SWITCH
# ==================================================================================================

p_switch = core.Prim("switch")


def switch(key: Hashable, branches: Branches, *args, **kwargs) -> Tree:
    """Select and execute one of multiple IR branches based on a traceable key.

    Args:
        key: Key selecting which branch to execute, with the same AVal as the branch keys.
        branches: Dict whose keys share one registered trace type and AVal mapping to IRs of
            matching input/output structures.
        *args: Positional arguments passed to the selected branch.

    Returns:
        Result of ``branches[key].call(*args)``

    Raises:
        KeyError: If key is not in branches.

    Example:
        >>> import autoform as af
        >>> branches = {
        ...     "zero": af.trace(lambda x: ("zero: " + x))("X"),
        ...     "one": af.trace(lambda x: ("one: " + x))("X"),
        ...     "two": af.trace(lambda x: ("two: " + x))("X"),
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
    assert all(isinstance(branches[k], stage.IR) for k in branches)
    key0 = next(iter(branches))
    assert stage.is_traceable(key0)
    assert all(type(k) is type(key0) for k in branches)
    key_aval = core.avalof(key0)
    assert all(core.avalof(k) == key_aval for k in branches)
    branch0 = branches[key0]
    assert all(stage.is_same_structure(branch0, branch) for branch in branches.values())
    return p_switch.bind((key, args), branches=branches)


def impl_switch(in_tree, /, *, branches: Branches):
    key, operands = in_tree
    return branches[key].call(*operands)


async def aimpl_switch(in_tree, /, *, branches: Branches):
    key, operands = in_tree
    return await branches[key].acall(*operands)


def abstract_switch(in_tree, /, *, branches: Branches) -> Tree:
    key, _ = in_tree
    key0 = next(iter(branches))
    key_aval = key if isinstance(key, core.AVal) else core.avalof(key)
    assert key_aval == core.avalof(key0)
    branch0 = branches[key0]
    return tree.map(stage.aval_if_var, branch0.out_tree)


def pushforward_switch(in_tree, /, *, branches: Branches):
    p, t = in_tree
    (key, p_operands), (_, t_operands) = p, t
    pf_ir = ad.pushforward(branches[key])
    return pf_ir.call(p_operands, t_operands)


async def apushforward_switch(in_tree, /, *, branches: Branches):
    p, t = in_tree
    (key, p_operands), (_, t_operands) = p, t
    pf_ir = ad.pushforward(branches[key])
    return await pf_ir.acall(p_operands, t_operands)


def pullback_fwd_switch(in_tree, /, *, branches: Branches) -> TreePair:
    key, operands = in_tree
    fwd, _ = ad.fwd_bwd(branches[key])
    out, residual = fwd.call(*operands)
    return out, (key, residual)


async def apull_fwd_switch(in_tree, /, *, branches: Branches) -> TreePair:
    key, operands = in_tree
    fwd, _ = ad.fwd_bwd(branches[key])
    out, residual = await fwd.acall(*operands)
    return out, (key, residual)


def pullback_bwd_switch(in_tree, /, *, branches: Branches):
    (key, residual), out_c = in_tree
    _, bwd = ad.fwd_bwd(branches[key])
    return (core.s_cotangent.zeros_like(key), bwd.call(residual, out_c))


async def apull_bwd_switch(in_tree, /, *, branches: Branches):
    (key, residual), out_c = in_tree
    _, bwd = ad.fwd_bwd(branches[key])
    return (core.s_cotangent.zeros_like(key), await bwd.acall(residual, out_c))


def batch_switch(in_tree, /, *, branches: Branches) -> tuple[Tree, Tree[bool]]:
    spec, in_axes, in_values = in_tree
    items = spec.unstack(in_axes, in_values)
    results = [branches[key].call(*args) for key, args in items]
    out_axes = tree.map(lambda _: True, results[0])
    return spec.stack(out_axes, results), out_axes


async def abatch_switch(in_tree, /, *, branches: Branches) -> tuple[Tree, Tree[bool]]:
    spec, in_axes, in_values = in_tree
    items = spec.unstack(in_axes, in_values)
    irs = [branches[key] for key, _ in items]
    inputs = [args for _, args in items]
    results = await order.p_fanout.abind(inputs, irs=irs)
    out_axes = tree.map(lambda _: True, results[0])
    return spec.stack(out_axes, results), out_axes


def dce_switch(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    branches: Branches = eqn.params["branches"]

    # NOTE(asem): a bit more care here, simply map DCE to each branch would not work
    # for example:
    # branch 1 => a(x) = (x, x)
    # branch 2 => b(x) = (x, concat(x, "-"))
    # after DCE with (True, False) mask
    # dce(a(x)) = (x, x)
    # dce(b(x)) = (x, None)
    # here, the two branches disagree on the output structure and shall fail.
    # the fix is to enforce the mask on each of IR's out_tree
    def func(ir: stage.IR):
        ir = stage.IR(list(ir.eqns), ir.in_tree, tree.mask(ir.out_tree, out_used))
        return dead.dce(ir)

    new_eqn = eqn.using(branches=tree.map(func, branches))
    new_eqn.out_tree = tree.mask(eqn.out_tree, out_used)
    return dead.default_dce(new_eqn, out_used)


core.r_impl.set(p_switch, impl_switch)
core.r_aimpl.set(p_switch, aimpl_switch)
core.r_abstract.set(p_switch, abstract_switch)
core.r_batch.set(p_switch, batch_switch)
core.r_abatch.set(p_switch, abatch_switch)
core.r_push.set(p_switch, pushforward_switch)
core.r_apush.set(p_switch, apushforward_switch)
core.r_pull_fwd.set(p_switch, pullback_fwd_switch)
core.r_apull_fwd.set(p_switch, apull_fwd_switch)
core.r_pull_bwd.set(p_switch, pullback_bwd_switch)
core.r_apull_bwd.set(p_switch, apull_bwd_switch)
dead.register_dce(p_switch, dce_switch)


# ==================================================================================================
# WHILE LOOP
# ==================================================================================================

p_while_loop = core.Prim("while_loop")


def while_loop(cond_ir: stage.IR, body_ir: stage.IR, init_val: Tree, *, max_iters: int) -> Tree:
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
        ...     return x == "go"
        >>> def body(x):
        ...     return "stop"
        >>> cond_ir = af.trace(cond)("...")
        >>> body_ir = af.trace(body)("...")
        >>> result = af.while_loop(cond_ir, body_ir, "go", max_iters=10)
        >>> result
        'stop'
    """
    assert isinstance(cond_ir, stage.IR), f"cond_ir must be an IR, got {type(cond_ir)}"
    assert isinstance(body_ir, stage.IR), f"body_ir must be an IR, got {type(body_ir)}"
    assert len(cond_ir.in_tree) == 1, "cond_ir must take exactly one positional argument"
    assert len(body_ir.in_tree) == 1, "body_ir must take exactly one positional argument"

    in_struct = tree.structure(body_ir.in_tree[0])
    out_struct = tree.structure(body_ir.out_tree)
    assert in_struct == out_struct, (
        f"body_ir must have identical input/output structure (f: State -> State).\n"
        f"in_struct:  {in_struct}\n"
        f"out_struct: {out_struct}"
    )
    return p_while_loop.bind(
        init_val,
        cond_ir=cond_ir,
        body_ir=body_ir,
        max_iters=max_iters,
    )


def impl_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
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
    cond_ir: stage.IR,
    body_ir: stage.IR,
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
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> Tree:
    del cond_ir, max_iters
    out_tree = tree.map(stage.aval_if_var, body_ir.out_tree)
    assert tree.structure(in_tree) == tree.structure(out_tree)

    def same_state(x, y):
        if isinstance(y, core.AVal) and not isinstance(x, core.AVal):
            # NOTE(asem): the key idea here is that in case inital state is a literal
            # and body returns AVal e.g.
            # >>> cond = af.trace(lambda x: False)("x")
            # >>> body = af.trace(lambda x: (x + "!"))("x")
            # >>> def program():
            # ...   return af.while_loop(cond, body, "hello", max_iters=10)
            # >>> ir = af.trace(program)()
            # >>> ir.call()  # "hello"
            # in here same_state normalizes hello -> StrAval
            x = core.avalof(x)
        return type(x) is type(y) and x == y

    assert all(tree.leaves(tree.map(same_state, in_tree, out_tree))), (
        "`while_loop` may run zero times: the initial state must match "
        "the body's output types and literal values"
    )
    return out_tree


def pushforward_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> TreePair:
    p_state, t_state = in_tree
    pf_body = ad.pushforward(body_ir)
    for _ in range(max_iters):
        if not cond_ir.call(p_state):
            break
        p_state, t_state = pf_body.call((p_state,), (t_state,))
    return p_state, t_state


async def apushforward_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> TreePair:
    p_state, t_state = in_tree
    pf_body = ad.pushforward(body_ir)
    for _ in range(max_iters):
        if not await cond_ir.acall(p_state):
            break
        p_state, t_state = await pf_body.acall((p_state,), (t_state,))
    return p_state, t_state


def pullback_fwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> TreePair:
    fwd, _ = ad.fwd_bwd(body_ir)
    out = in_tree
    residuals = []

    for _ in range(max_iters):
        if not cond_ir.call(out):
            break
        out, residual = fwd.call(out)
        residuals.append(residual)

    return out, residuals


async def apull_fwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> TreePair:
    fwd, _ = ad.fwd_bwd(body_ir)
    out = in_tree
    residuals = []

    for _ in range(max_iters):
        if not await cond_ir.acall(out):
            break
        out, residual = await fwd.acall(out)
        residuals.append(residual)

    return out, residuals


def batch_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> TreePair:
    spec, in_axes, init_val = in_tree
    b_sz = spec.size
    # NOTE(asem): in_tree is a SoA object, however we need to pass in only parts of the SoA
    # that are alive (:= still needs some work). so we need to convert from SoA to AoS
    # filter out dead items then convert back to SoA for the batched cond/body to work.
    # finally convert back to SoA for the output. the following code does exactly this with
    # some bookkeeping to handle divergence.

    # NOTE(asem): batched while loop with early exit. each item exits independently when
    # cond returns False, saving LLM calls on items that finish early.
    # example: Struct(text=batched, note=broadcast) with 3 items
    # >>> in_tree = Struct(text=["A","B","C"], note="v1")  # text batched, note broadcast
    # >>> in_axes = Struct(text=True, note=False)
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
    # >>> out_axes = Struct(text=True, note=True)

    # NOTE(asem): unbatch SoA -> AoS so each state can be tracked independently
    # and keep track of which items are not done. initially everything is alive
    states = [(x,) for x in spec.unstack(in_axes, init_val)]
    alive = [True] * b_sz

    # NOTE(asem): pre-batch cond and body IRs. True marks all leaves as batched.
    in_state_axes = tree.map(lambda _: True, body_ir.in_tree)
    in_cond_axes = in_state_axes
    in_body_axes = in_state_axes
    batched_cond = axis.batch(cond_ir, in_axes=in_cond_axes)
    batched_body = axis.batch(body_ir, in_axes=in_body_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        # NOTE(asem): check conditions only for alive items (transpose AoS -> SoA for call)
        alive_states = [states[i] for i in alive_idx]
        active_spec = axis.BatchSpec(tree.structure([0] * len(alive_states)))
        # NOTE(asem): move from AoS to SoA for alive states
        in_transposed_cond = active_spec.stack(in_cond_axes, alive_states)
        conds_result = batched_cond.call(*in_transposed_cond)
        # NOTE(asem): cond returns scalar bool, batched -> list. use unbatch for consistency.
        out_cond_axes = isinstance(conds_result, list)
        conds = active_spec.unstack(out_cond_axes, conds_result)
        # NOTE(asem): mark items as dead if cond returned False
        for idx, c in zip(alive_idx, conds):
            alive[idx] = c
        # NOTE(asem): run body ONLY on still-alive items
        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            body_spec = axis.BatchSpec(tree.structure([0] * len(still_alive_states)))
            in_transposed = body_spec.stack(in_body_axes, still_alive_states)
            out_transposed = batched_body.call(*in_transposed)
            out_axes = tree.map(stage.is_var, body_ir.out_tree)
            outputs = body_spec.unstack(out_axes, out_transposed)

            for batch_idx, value in zip(still_alive, outputs):
                states[batch_idx] = (value,)
    # NOTE(asem): transpose final states AoS -> SoA for batched output
    # only Var positions are batched; literal positions stay scalar
    out_axes = tree.map(stage.is_var, body_ir.out_tree)
    out_tree = spec.stack(out_axes, (state[0] for state in states))
    return out_tree, out_axes


async def abatch_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> TreePair:
    spec, in_axes, init_val = in_tree
    b_sz = spec.size

    # NOTE(asem): unbatch SoA -> AoS so each state can be tracked independently
    states = [(x,) for x in spec.unstack(in_axes, init_val)]
    alive = [True] * b_sz

    # NOTE(asem): pre-batch cond and body IRs
    in_state_axes = tree.map(lambda _: True, body_ir.in_tree)
    in_cond_axes = in_state_axes
    in_body_axes = in_state_axes
    batched_cond = axis.batch(cond_ir, in_axes=in_cond_axes)
    batched_body = axis.batch(body_ir, in_axes=in_body_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        alive_states = [states[i] for i in alive_idx]
        active_spec = axis.BatchSpec(tree.structure([0] * len(alive_states)))
        in_transposed_cond = active_spec.stack(in_cond_axes, alive_states)
        conds_result = await batched_cond.acall(*in_transposed_cond)
        out_cond_axes = isinstance(conds_result, list)
        conds = active_spec.unstack(out_cond_axes, conds_result)

        for idx, c in zip(alive_idx, conds):
            alive[idx] = c

        if still_alive := [i for i in alive_idx if alive[i]]:
            still_alive_states = [states[i] for i in still_alive]
            body_spec = axis.BatchSpec(tree.structure([0] * len(still_alive_states)))
            in_transposed = body_spec.stack(in_body_axes, still_alive_states)
            out_transposed = await batched_body.acall(*in_transposed)
            out_body_axes = tree.map(stage.is_var, body_ir.out_tree)
            outputs = body_spec.unstack(out_body_axes, out_transposed)

            for batch_idx, value in zip(still_alive, outputs):
                states[batch_idx] = (value,)

    out_axes = tree.map(stage.is_var, body_ir.out_tree)
    out_tree = spec.stack(out_axes, (state[0] for state in states))
    return out_tree, out_axes


def pullback_bwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> Tree:
    residuals, c = in_tree
    del cond_ir, max_iters
    _, bwd = ad.fwd_bwd(body_ir)

    for residual in reversed(residuals):
        (c,) = bwd.call(residual, c)

    return c


async def apull_bwd_while_loop(
    in_tree: Tree,
    /,
    *,
    cond_ir: stage.IR,
    body_ir: stage.IR,
    max_iters: int,
) -> Tree:
    residuals, c = in_tree
    del cond_ir, max_iters
    _, bwd = ad.fwd_bwd(body_ir)

    for residual in reversed(residuals):
        (c,) = await bwd.acall(residual, c)

    return c


def dce_while_loop(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    cond_ir = eqn.params["cond_ir"]
    body_ir = eqn.params["body_ir"]
    # NOTE(asem): every state leaf is loop-carried into later condition/body calls,
    # even if the caller only uses part of the final state.
    state_used = tree.map(lambda _: True, body_ir.out_tree)
    cond_ir = dead.dce(cond_ir)
    body_ir = dead.dce(body_ir, out_used=state_used)
    new_eqn = eqn.using(cond_ir=cond_ir, body_ir=body_ir)
    return dead.default_dce(new_eqn, out_used)


core.r_impl.set(p_while_loop, impl_while_loop)
core.r_aimpl.set(p_while_loop, aimpl_while_loop)
core.r_abstract.set(p_while_loop, abstract_while_loop)
core.r_batch.set(p_while_loop, batch_while_loop)
core.r_abatch.set(p_while_loop, abatch_while_loop)
core.r_push.set(p_while_loop, pushforward_while_loop)
core.r_apush.set(p_while_loop, apushforward_while_loop)
core.r_pull_fwd.set(p_while_loop, pullback_fwd_while_loop)
core.r_apull_fwd.set(p_while_loop, apull_fwd_while_loop)
core.r_pull_bwd.set(p_while_loop, pullback_bwd_while_loop)
core.r_apull_bwd.set(p_while_loop, apull_bwd_while_loop)
dead.register_dce(p_while_loop, dce_while_loop)


# ==================================================================================================
# FIXPOINT
# ==================================================================================================

p_fixpoint = core.Prim("fixpoint")


def cot_tree_accum(lhs: Tree, rhs: Tree, /) -> Tree:
    return tree.map(lambda l, r: ad.cot_accum([l, r]), lhs, rhs)


def fixpoint(
    step_ir: stage.IR,
    init_val: Tree,
    theta: Tree = (),
    *,
    max_iters: int,
    adj_iters: int = 1,
    equiv_ir: stage.IR | None = None,
) -> Tree:
    """Iterate ``step_ir`` until the state reaches a fixed point.

    ``step_ir`` must have shape ``(State, Theta) -> State``. The loop stops when
    the new state is equivalent to the previous state, or when ``max_iters``
    is reached. Equivalence defaults to structural equality of the state
    pytree; pass ``equiv_ir`` with shape ``(State, State) -> Bool`` to decide
    stability inside the program.

    Both AD directions approximate the implicit derivative at the returned
    state using up to ``adj_iters + 1`` terms and treat sensitivity to
    ``init_val`` as zero.
    """
    assert isinstance(step_ir, stage.IR), f"step_ir must be an IR, got {type(step_ir)}"
    assert len(step_ir.in_tree) == 2, "step_ir must take exactly two positional arguments"

    in_struct = tree.structure(step_ir.in_tree[0])
    out_struct = tree.structure(step_ir.out_tree)
    assert in_struct == out_struct, (
        f"step_ir must have identical state input/output structure (step: (State, Theta) -> State).\n"
        f"in_struct:  {in_struct}\n"
        f"out_struct: {out_struct}"
    )
    assert isinstance(max_iters, int) and max_iters >= 1, f"max_iters must be >= 1: {max_iters!r}"
    assert isinstance(adj_iters, int) and adj_iters >= 0, f"adj_iters must be >= 0: {adj_iters!r}"
    if equiv_ir is not None:
        assert isinstance(equiv_ir, stage.IR), f"equiv_ir must be an IR, got {type(equiv_ir)}"
        assert len(equiv_ir.in_tree) == 2, "equiv_ir must take two positional arguments"
        for side in equiv_ir.in_tree:
            assert tree.structure(side) == in_struct, (
                "equiv_ir inputs must both match the state structure"
            )
    return p_fixpoint.bind(
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
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> Tree:
    del adj_iters
    state, theta = in_tree

    for _ in range(max_iters):
        new_state = step_ir.call(state, theta)
        stable = (
            tree.equal(state, new_state) if equiv_ir is None else equiv_ir.call(state, new_state)
        )
        if stable:
            return new_state
        state = new_state
    return state


async def aimpl_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> Tree:
    del adj_iters
    state, theta = in_tree

    for _ in range(max_iters):
        new_state = await step_ir.acall(state, theta)
        stable = (
            tree.equal(state, new_state)
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
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> Tree:
    del in_tree, max_iters, adj_iters, equiv_ir
    return tree.map(stage.aval_if_var, step_ir.out_tree)


def pushforward_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> TreePair:
    p, (_, t_theta) = in_tree
    _, theta = p
    out = p_fixpoint.bind(
        p, step_ir=step_ir, max_iters=max_iters, adj_iters=adj_iters, equiv_ir=equiv_ir
    )
    t = core.s_tangent.zeros_like(out)
    if all(isinstance(x, core.Zero) for x in tree.leaves(t_theta)):
        return out, t

    pf_step_ir = ad.pushforward(step_ir)
    for _ in range(adj_iters + 1):
        _, t_next = pf_step_ir.call((out, theta), (t, t_theta))
        if tree.equal(t_next, t):
            return out, t_next
        t = t_next
    return out, t


async def apushforward_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> TreePair:
    p, (_, t_theta) = in_tree
    _, theta = p
    out = await p_fixpoint.abind(
        p, step_ir=step_ir, max_iters=max_iters, adj_iters=adj_iters, equiv_ir=equiv_ir
    )
    t = core.s_tangent.zeros_like(out)
    if all(isinstance(x, core.Zero) for x in tree.leaves(t_theta)):
        return out, t

    pf_step_ir = ad.pushforward(step_ir)
    for _ in range(adj_iters + 1):
        _, t_next = await pf_step_ir.acall((out, theta), (t, t_theta))
        if tree.equal(t_next, t):
            return out, t_next
        t = t_next
    return out, t


def pullback_fwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> TreePair:
    del adj_iters
    state, theta = in_tree
    fwd, _ = ad.fwd_bwd(step_ir)

    for _ in range(max_iters):
        new_state, residual = fwd.call(state, theta)
        stable = (
            tree.equal(state, new_state) if equiv_ir is None else equiv_ir.call(state, new_state)
        )
        if stable:
            return new_state, (new_state, theta, residual)
        state = new_state
    _, residual = fwd.call(state, theta)
    return state, (state, theta, residual)


async def apull_fwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> TreePair:
    del adj_iters
    state, theta = in_tree
    fwd, _ = ad.fwd_bwd(step_ir)

    for _ in range(max_iters):
        new_state, residual = await fwd.acall(state, theta)
        stable = (
            tree.equal(state, new_state)
            if equiv_ir is None
            else await equiv_ir.acall(state, new_state)
        )
        if stable:
            return new_state, (new_state, theta, residual)
        state = new_state
    _, residual = await fwd.acall(state, theta)
    return state, (state, theta, residual)


def pullback_bwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> Tree:
    del max_iters, equiv_ir
    residuals, g = in_tree
    x_star, theta, residual = residuals
    dx0 = core.s_cotangent.zeros_like(x_star)

    if all(isinstance(x, core.Zero) for x in tree.leaves(g)):
        return dx0, core.s_cotangent.zeros_like(theta)

    _, bwd = ad.fwd_bwd(step_ir)
    u = g

    for _ in range(adj_iters):
        x_bar, theta_bar = bwd.call(residual, u)
        next_u = cot_tree_accum(g, x_bar)
        if tree.equal(next_u, u):
            dtheta = theta_bar
            break
        u = next_u
    else:
        _, dtheta = bwd.call(residual, u)

    return dx0, dtheta


async def apull_bwd_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> Tree:
    del max_iters, equiv_ir
    residuals, g = in_tree
    x_star, theta, residual = residuals
    dx0 = core.s_cotangent.zeros_like(x_star)

    if all(isinstance(x, core.Zero) for x in tree.leaves(g)):
        return dx0, core.s_cotangent.zeros_like(theta)

    _, bwd = ad.fwd_bwd(step_ir)
    u = g

    for _ in range(adj_iters):
        x_bar, theta_bar = await bwd.acall(residual, u)
        next_u = cot_tree_accum(g, x_bar)
        if tree.equal(next_u, u):
            dtheta = theta_bar
            break
        u = next_u
    else:
        _, dtheta = await bwd.acall(residual, u)

    return dx0, dtheta


def batch_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> TreePair:
    spec, in_axes, in_values = in_tree
    b_sz = spec.size
    init_val, theta = in_values
    init_axes, theta_axes = in_axes
    states = spec.unstack(init_axes, init_val)
    thetas = spec.unstack(theta_axes, theta)
    alive = [True] * b_sz

    in_state_axes = tree.map(lambda _: True, step_ir.in_tree[0])
    in_theta_axes = tree.map(lambda _: True, step_ir.in_tree[1])
    in_axes = (in_state_axes, in_theta_axes)
    batched_step = axis.batch(step_ir, in_axes=in_axes)
    if equiv_ir is not None:
        equiv_axes = (in_state_axes, in_state_axes)
        batched_equiv = axis.batch(equiv_ir, in_axes=equiv_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        in_alive = [(states[i], thetas[i]) for i in alive_idx]
        active_spec = axis.BatchSpec(tree.structure([0] * len(in_alive)))
        in_transposed = active_spec.stack(in_axes, in_alive)
        out_transposed = batched_step.call(*in_transposed)
        out_axes = tree.map(stage.is_var, step_ir.out_tree)
        new_states = active_spec.unstack(out_axes, out_transposed)
        if equiv_ir is None:
            flags = [tree.equal(states[b], ns) for b, ns in zip(alive_idx, new_states)]
        else:
            pairs = [(states[b], ns) for b, ns in zip(alive_idx, new_states)]
            in_equiv = active_spec.stack(equiv_axes, pairs)
            flags = batched_equiv.call(*in_equiv)

        for flag, batch_idx, new_state in zip(flags, alive_idx, new_states):
            if flag:
                alive[batch_idx] = False
            states[batch_idx] = new_state

    out_axes = tree.map(stage.is_var, step_ir.out_tree)
    out_tree = spec.stack(out_axes, states)
    return out_tree, out_axes


async def abatch_fixpoint(
    in_tree: Tree,
    /,
    *,
    step_ir: stage.IR,
    max_iters: int,
    adj_iters: int,
    equiv_ir: stage.IR | None,
) -> TreePair:
    spec, in_axes, in_values = in_tree
    b_sz = spec.size
    init_val, theta = in_values
    init_axes, theta_axes = in_axes
    states = spec.unstack(init_axes, init_val)
    thetas = spec.unstack(theta_axes, theta)
    alive = [True] * b_sz

    in_state_axes = tree.map(lambda _: True, step_ir.in_tree[0])
    in_theta_axes = tree.map(lambda _: True, step_ir.in_tree[1])
    in_axes = (in_state_axes, in_theta_axes)
    batched_step = axis.batch(step_ir, in_axes=in_axes)
    if equiv_ir is not None:
        equiv_axes = (in_state_axes, in_state_axes)
        batched_equiv = axis.batch(equiv_ir, in_axes=equiv_axes)

    for _ in range(max_iters):
        if not (alive_idx := [i for i in range(b_sz) if alive[i]]):
            break

        in_alive = [(states[i], thetas[i]) for i in alive_idx]
        active_spec = axis.BatchSpec(tree.structure([0] * len(in_alive)))
        in_transposed = active_spec.stack(in_axes, in_alive)
        out_transposed = await batched_step.acall(*in_transposed)
        out_axes = tree.map(stage.is_var, step_ir.out_tree)
        new_states = active_spec.unstack(out_axes, out_transposed)
        if equiv_ir is None:
            flags = [tree.equal(states[b], ns) for b, ns in zip(alive_idx, new_states)]
        else:
            pairs = [(states[b], ns) for b, ns in zip(alive_idx, new_states)]
            in_equiv = active_spec.stack(equiv_axes, pairs)
            flags = await batched_equiv.acall(*in_equiv)

        for flag, batch_idx, new_state in zip(flags, alive_idx, new_states):
            if flag:
                alive[batch_idx] = False
            states[batch_idx] = new_state

    out_axes = tree.map(stage.is_var, step_ir.out_tree)
    out_tree = spec.stack(out_axes, states)
    return out_tree, out_axes


def dce_fixpoint(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    step_ir = eqn.params["step_ir"]
    equiv_ir = eqn.params["equiv_ir"]
    # NOTE(asem): every state leaf is loop-carried into the next step, even if
    # the caller only uses part of the final state.
    state_used = tree.map(lambda _: True, step_ir.out_tree)
    equiv_ir = None if equiv_ir is None else dead.dce(equiv_ir)
    step_ir = dead.dce(step_ir, out_used=state_used)
    new_eqn = eqn.using(step_ir=step_ir, equiv_ir=equiv_ir)
    return dead.default_dce(new_eqn, out_used)


core.r_impl.set(p_fixpoint, impl_fixpoint)
core.r_aimpl.set(p_fixpoint, aimpl_fixpoint)
core.r_abstract.set(p_fixpoint, abstract_fixpoint)
core.r_push.set(p_fixpoint, pushforward_fixpoint)
core.r_apush.set(p_fixpoint, apushforward_fixpoint)
core.r_batch.set(p_fixpoint, batch_fixpoint)
core.r_abatch.set(p_fixpoint, abatch_fixpoint)
core.r_pull_fwd.set(p_fixpoint, pullback_fwd_fixpoint)
core.r_apull_fwd.set(p_fixpoint, apull_fwd_fixpoint)
core.r_pull_bwd.set(p_fixpoint, pullback_bwd_fixpoint)
core.r_apull_bwd.set(p_fixpoint, apull_bwd_fixpoint)
dead.register_dce(p_fixpoint, dce_fixpoint)
