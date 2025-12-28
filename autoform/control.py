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
# ITERATE UNTIL
# ==================================================================================================

iterate_until_p = Primitive("iterate_until", tag="control")


type State = str
type Status = str
type Iterations = int
type Residuals = tuple[State, Status, Iterations]


def iterate_until(
    body: IR,
    init: Tree,
    goal: IR,
    max_iters: int = 10,
) -> Residuals:
    """Goal-directed iteration: iterate body until goal is satisfied.

    Args:
        body: IR for one refinement step, f: X -> X
        init: Initial state x₀
        goal: IR that returns truthy when goal is reached, g: X -> bool
        max_iters: Maximum iterations (safety bound)

    Returns:
        (final_state, status, n_iters) where:
        - final_state: The state when iteration stopped
        - status: "goal" if goal reached, "max" if bounded
        - n_iters: Number of iterations executed

    Example:
        >>> import autoform as af
        >>> def refine(draft):
        ...     msgs = [{"role": "user", "content": af.format("Improve: {}", draft)}]
        ...     return af.lm_call(msgs, model="gpt-4o-mini")
        >>> class Verdict(af.Struct):
        ...     is_good: bool
        >>> def verify(draft):
        ...     msgs = [{"role": "user", "content": af.format("Is good? {}", draft)}]
        ...     verdict = af.struct_lm_call(msgs, model="gpt-4o-mini", struct=Verdict)
        ...     return verdict.is_good  # Direct attribute access returns bool
        >>> body = af.build_ir(refine)("draft")
        >>> goal = af.build_ir(verify)("draft")
        >>> result, status, n = af.iterate_until(body, "my text", goal, max_iters=5) # doctest: +SKIP
    """
    assert isinstance(body, IR), f"body must be an IR, got {type(body)}"
    assert isinstance(goal, IR), f"goal must be an IR, got {type(goal)}"

    in_struct = treelib.structure(body.in_irtree)
    out_struct = treelib.structure(body.out_irtree)
    assert in_struct == out_struct, (
        f"body must have identical input/output structure (f: X -> X).\n"
        f"in_struct:  {in_struct}\n"
        f"out_struct: {out_struct}"
    )
    return iterate_until_p.bind(init, body=body, goal=goal, max_iters=max_iters)


@ft.partial(impl_rules.def_rule, iterate_until_p)
def impl_iterate_until(
    in_tree: Tree,
    *,
    body: IR,
    goal: IR,
    max_iters: int,
) -> Residuals:
    state: State = in_tree
    for t in range(max_iters):
        goal_result = call(goal)(state)
        assert isinstance(goal_result, bool), "goal must return bool"
        if goal_result:
            return (state, "goal", t)
        state = call(body)(state)

    return (state, "max", max_iters)


@ft.partial(eval_rules.def_rule, iterate_until_p)
def eval_iterate_until(
    in_tree: Tree,
    *,
    body: IR,
    goal: IR,
    max_iters: int,
) -> Residuals:
    del goal, max_iters
    out_state = treelib.map(lambda _: Var(), body.out_irtree, is_leaf=is_irvar)
    status = Var()
    n_iters = Var()
    return (out_state, status, n_iters)


@ft.partial(batch_rules.def_rule, iterate_until_p)
def batch_iterate_until(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    body: IR,
    goal: IR,
    max_iters: int,
) -> tuple[Residuals, Tree[bool]]:
    # NOTE(asem): the batch rule here is a bit more complicated than others
    # because it involves breaking the loop when a goal is reached. it would be inefficient
    # to keep running a maxiters loop for each example regardless of whether it reached the goal
    # or not. so we keep track of which examples are still alive and only run the loop for them
    # until all examples reach the goal or maxiters is reached

    # NOTE(asem): unbatch basically extracts an example from
    # a pytree according to the batch index. one thing to note
    # if the some part of the pytree mask is not batched, it will be broadcasted
    unbatch_state = ft.partial(unbatch_at, in_tree, in_batched)
    states = [unbatch_state(b) for b in range(batch_size)]

    # NOTE(asem): alive is a list of booleans that indicates which examples are still alive
    # statuses is a list of strings that indicates the status of each example
    # iterations is a list of integers that indicates the number of iterations for each example
    alive = [True] * batch_size

    # NOTE(asem): maybe expose this as an option
    statuses = ["running"] * batch_size
    iterations = [0] * batch_size

    batched_body = batch(body, in_axes=list)
    batched_goal = batch(goal, in_axes=list)

    for t in range(max_iters):
        alive_idx = [i for i in range(batch_size) if alive[i]]

        # NOTE(asem): all examples are done running
        if not alive_idx:
            break

        # NOTE(asem): batched goal check on alive examples
        alive_states = [states[i] for i in alive_idx]
        goals = call(batched_goal)(alive_states)

        # Handle case where goal returns literal (not batched)
        if isinstance(goals, bool):
            goals = [goals] * len(alive_idx)

        for i, g in zip(alive_idx, goals, strict=True):
            assert isinstance(g, bool), "goal must return bool"
            if g:
                alive[i] = False
                statuses[i] = "goal"
                iterations[i] = t

        # NOTE(asem): batched body application on still-alive examples
        still_alive = [i for i in alive_idx if alive[i]]
        if still_alive:
            still_alive_states = [states[i] for i in still_alive]
            new_states = call(batched_body)(still_alive_states)
            for i, s in zip(still_alive, new_states, strict=True):
                states[i] = s
                iterations[i] = t + 1

    # NOTE(asem): mark remaining as max
    for i in range(batch_size):
        if statuses[i] == "running":
            statuses[i] = "max"
            iterations[i] = max_iters

    # NOTE(asem): return batched outputs
    out_batched = (True, True, True)  # (states, statuses, iterations) all batched
    return (states, statuses, iterations), out_batched


@ft.partial(pull_fwd_rules.def_rule, iterate_until_p)
def pullback_fwd_iterate_until(
    in_tree: Tree,
    *,
    body: IR,
    goal: IR,
    max_iters: int,
) -> tuple[Tree, Tree]:
    state = in_tree
    trajectory = [state]

    for t in range(max_iters):
        reached = call(goal)(state)
        if reached:
            residuals = (trajectory, "goal", t, body, goal)
            return (state, "goal", t), residuals

        state = call(body)(state)
        trajectory.append(state)

    residuals = (trajectory, "max", max_iters, body, goal)
    return (state, "max", max_iters), residuals


@ft.partial(pull_bwd_rules.def_rule, iterate_until_p)
def pullback_bwd_iterate_until(
    residuals: Tree,
    out_cotangent: Tree,
    *,
    body: IR,
    goal: IR,
    max_iters: int,
) -> Tree:
    trajectory, status, n_iters, _, _ = residuals
    out_state_cotangent, _, _ = out_cotangent

    cotangent = out_state_cotangent
    pb_body = pullback(body)

    for t in reversed(range(n_iters)):
        state_t = trajectory[t]
        # NOTE(asem): backprop through body moving from  out_cot -> in_cot
        _, cotangent = call(pb_body)((state_t, cotangent))

    return cotangent
