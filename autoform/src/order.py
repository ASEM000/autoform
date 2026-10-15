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

"""Execution ordering primitives."""

from __future__ import annotations

import asyncio
import functools as ft
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.stage as stage
import autoform.src.utils as utils

zip = utils.strict_zip

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type IRList = list[stage.IR]
type FanoutPair = tuple[list[Tree], list[Tree]]
type FanoutResidual = tuple[list[Tree], IRList]
type FanoutFwdResult = tuple[list[Tree], FanoutResidual]
type BatchFanoutInput = tuple[int, list[bool], list[Tree]]
type BatchFanoutOutput = tuple[list[Tree], list[Tree[bool]]]
type DependsType[T] = tuple[T, tuple[Tree, ...]]
type DependsPair = tuple[DependsType[Tree], DependsType[Tree]]
type DependsFwdResult = tuple[Tree, DependsType[Tree]]
type DependsBwdInput = tuple[DependsType[Tree], Tree]
type BatchDependsInput = tuple[int, tuple[bool, tuple[bool, ...]], DependsType[Tree]]

# ==================================================================================================
# FANOUT
# ==================================================================================================


serial_fanout_flag: ContextVar[bool] = ContextVar("serial_fanout_flag", default=False)


@contextmanager
def serial_fanout() -> Generator[None, None, None]:
    """Run async fanout sequentially inside the context."""

    token = serial_fanout_flag.set(True)
    try:
        yield
    finally:
        serial_fanout_flag.reset(token)


p_fanout = core.Prim("fanout")


def impl_fanout(in_tree: list[Tree], /, *, irs: IRList) -> list[Tree]:
    assert len(in_tree) == len(irs)
    return [ir.call(*inp) for ir, inp in zip(irs, in_tree)]


async def aimpl_fanout(in_tree: list[Tree], /, *, irs: IRList) -> list[Tree]:
    assert len(in_tree) == len(irs)
    if len(irs) == 1:
        [ir], [inp] = irs, in_tree
        return [await ir.acall(*inp)]
    if serial_fanout_flag.get():
        return [await ir.acall(*inp) for ir, inp in zip(irs, in_tree)]
    return await asyncio.gather(*[ir.acall(*inp) for ir, inp in zip(irs, in_tree)])


def abstract_fanout(in_tree: list[Tree], /, *, irs: IRList) -> list[Tree]:
    return [utils.tree.map(stage.aval_if_var, ir.out_tree) for ir in irs]


def push_fanout(in_tree: FanoutPair, /, *, irs: IRList) -> FanoutPair:
    p, t = in_tree
    pf_irs = [ad.pushforward(ir) for ir in irs]
    pf_inputs = [(p, t) for p, t in zip(p, t)]
    results = p_fanout.bind(pf_inputs, irs=pf_irs)
    p_outs, t_outs = zip(*results)
    return list(p_outs), list(t_outs)


async def apush_fanout(in_tree: FanoutPair, /, *, irs: IRList) -> FanoutPair:
    p, t = in_tree
    pf_irs = [ad.pushforward(ir) for ir in irs]
    pf_inputs = [(p, t) for p, t in zip(p, t)]
    results = await p_fanout.abind(pf_inputs, irs=pf_irs)
    p_outs, t_outs = zip(*results)
    return list(p_outs), list(t_outs)


def pull_fwd_fanout(in_tree: list[Tree], /, *, irs: IRList) -> FanoutFwdResult:
    results = p_fanout.bind(in_tree, irs=irs)
    residuals = (in_tree, irs)
    return results, residuals


async def apull_fwd_fanout(in_tree: list[Tree], /, *, irs: IRList) -> FanoutFwdResult:
    results = await p_fanout.abind(in_tree, irs=irs)
    residuals = (in_tree, irs)
    return results, residuals


def pull_bwd_fanout(in_tree: Tree, /, *, irs: IRList) -> list[Tree]:
    residuals, out_c = in_tree
    inputs, _ = residuals
    pb_irs = [ad.pullback(ir) for ir in irs]
    pb_inputs = [(inp, cot) for inp, cot in zip(inputs, out_c)]
    results = p_fanout.bind(pb_inputs, irs=pb_irs)
    return [cot for _, cot in results]


async def apull_bwd_fanout(in_tree: Tree, /, *, irs: IRList) -> list[Tree]:
    residuals, out_c = in_tree
    inputs, _ = residuals
    pb_irs = [ad.pullback(ir) for ir in irs]
    pb_inputs = [(inp, cot) for inp, cot in zip(inputs, out_c)]
    results = await p_fanout.abind(pb_inputs, irs=pb_irs)
    return [cot for _, cot in results]


def batch_fanout(in_tree: BatchFanoutInput, /, *, irs: IRList) -> BatchFanoutOutput:
    _, in_batched, inputs = in_tree

    results: list[Tree] = []
    out_batched: list[Tree[bool]] = []

    for ir, inp, inp_batched in zip(irs, inputs, in_batched):
        if utils.batch_spec(inp, inp_batched) is None:
            results.append(ir.call(*inp))
            out_batched.append(utils.tree.map(lambda _: False, ir.out_tree))
        else:
            batched_ir = axis.batch(ir, in_axes=inp_batched)
            results.append(batched_ir.call(*inp))
            out_batched.append(utils.tree.map(lambda _: True, ir.out_tree))

    return results, out_batched


async def abatch_fanout(in_tree: BatchFanoutInput, /, *, irs: IRList) -> BatchFanoutOutput:
    _, in_batched, inputs = in_tree

    results: list[Tree] = []
    out_batched: list[Tree[bool]] = []

    for ir, inp, inp_batched in zip(irs, inputs, in_batched):
        if utils.batch_spec(inp, inp_batched) is None:
            results.append(await ir.acall(*inp))
            out_batched.append(utils.tree.map(lambda _: False, ir.out_tree))
        else:
            batched_ir = axis.batch(ir, in_axes=inp_batched)
            results.append(await batched_ir.acall(*inp))
            out_batched.append(utils.tree.map(lambda _: True, ir.out_tree))

    return results, out_batched


def dce_fanout(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    irs = eqn.params["irs"]

    # NOTE(asem): pruning each inner IR can change the outputs returned by fanout.
    # for example:
    # inner IR 1 => a(x) = concat(x, "!")
    # inner IR 2 => b(x) = concat(x, "?")
    # after DCE with [True, False] mask
    # fanout returns [a(x), None], but its out_tree still declares [a, b].
    # the fix is to enforce the mask on each IR's out_tree and the parent out_tree.
    def func(ir: stage.IR, used: dead.UsedTree):
        ir = stage.IR(list(ir.eqns), ir.in_tree, utils.mask(ir.out_tree, used))
        return dead.dce(ir)

    new_eqn = eqn.using(irs=utils.tree.map(func, irs, out_used))
    new_eqn.out_tree = utils.mask(eqn.out_tree, out_used)
    return dead.default_dce(new_eqn, out_used)


core.r_impl.set(p_fanout, impl_fanout)
core.r_aimpl.set(p_fanout, aimpl_fanout)
core.r_abstract.set(p_fanout, abstract_fanout)
core.r_batch.set(p_fanout, batch_fanout)
core.r_abatch.set(p_fanout, abatch_fanout)
core.r_push.set(p_fanout, push_fanout)
core.r_apush.set(p_fanout, apush_fanout)
core.r_pull_fwd.set(p_fanout, pull_fwd_fanout)
core.r_apull_fwd.set(p_fanout, apull_fwd_fanout)
core.r_pull_bwd.set(p_fanout, pull_bwd_fanout)
core.r_apull_bwd.set(p_fanout, apull_bwd_fanout)
dead.dce_rules[p_fanout] = dce_fanout


# ==================================================================================================
# SCHED
# ==================================================================================================


@ft.partial(utils.lru_cache, maxsize=256)
def sched[*A, R](
    ir: stage.IR[*A, R],
    /,
    *,
    cond: Callable[[stage.Eqn], bool] | None = None,
) -> stage.IR[*A, R]:
    """Schedule independent operations for parallel execution.

    Args:
        ir: The IR to schedule.
        cond: Predicate that takes an IR Equation and returns True if the
              equation should be parallelized. If None, all operations are
              candidates for parallelization.

    Returns:
        A new IR with independent operations grouped together for parallel execution.

    Example:
        >>> import autoform as af
        >>> import asyncio
        >>>
        >>> def parallel_calls(x):
        ...     a = af.lm.fill({"prompt": "Q1: " + x, "output": af.lm.Str()}, model="model-name")["output"]
        ...     b = af.lm.fill({"prompt": "Q2: " + x, "output": af.lm.Str()}, model="model-name")["output"]
        ...     return a + b
        >>>
        >>> ir = af.trace(parallel_calls)("input")
        >>> scheduled = af.sched(ir)
        >>>
        >>> # sync execution (sequential)
        >>> result = scheduled.call("hello") # doctest: +SKIP
        >>>
        >>> # async execution (concurrent via asyncio.gather)
        >>> result = asyncio.run(scheduled.acall("hello")) # doctest: +SKIP
    """
    levels: list[list[stage.Eqn]] = stage.toposort_levels(ir)
    out_eqns: list[stage.Eqn] = []
    cond = (lambda _: True) if cond is None else cond

    def recurse(leaf):
        return sched(leaf, cond=cond) if isinstance(leaf, stage.IR) else leaf

    def make_fanout(eqns: list[stage.Eqn]) -> stage.Eqn:
        # NOTE(asem): the created IR input must have non-repeated vars to avoid multiple
        # cotangent contribution, for example taking the pullback of
        # >>> func = lambda x: (x * x, x + 1)
        # simply copying the vars of multiply eqn then (e.g. [v0, v0]) incorrectly returns
        # (2 * x, 2 * x) that will get summed upstream.
        in_trees = [(tuple(dict.fromkeys(stage.var_leaves(eqn.in_tree))),) for eqn in eqns]
        irs = [stage.IR([eqn], inputs, eqn.out_tree) for eqn, inputs in zip(eqns, in_trees)]
        out_trees = [eqn.out_tree for eqn in eqns]
        return stage.Eqn(p_fanout, in_trees, out_trees, dict(irs=irs))

    for level in levels:
        eqns = [eqn.using(**utils.tree.map(recurse, eqn.params)) for eqn in level]
        seq_eqns = [eqn for eqn in eqns if not cond(eqn)]
        par_eqns = [eqn for eqn in eqns if cond(eqn)]
        out_eqns.extend([make_fanout(par_eqns)] if len(par_eqns) > 1 else par_eqns)
        out_eqns.extend(seq_eqns)

    return stage.IR(out_eqns, ir.in_tree, ir.out_tree)


# ==================================================================================================
# DEPENDS
# ==================================================================================================

p_depends = core.Prim("depends")


def depends[T](value: T, /, *deps) -> T:
    """Annotate that `value` depends on the evaluation of `deps`.

    This primitive inserts an ordering barrier without changing the forward
    value. The equations that produce `value` and `deps` may still run in the
    same scheduling level; the barrier's output is available only after
    `value` and all `deps` have been evaluated.

    Args:
        value: The main value to return.
        *deps: Values that `value` depends on.
    Returns:
        The original `value`, through a barrier that also depends on `deps`.
    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     a = "First: " + x
        ...     b = "Second: " + x
        ...     return af.depends(b, a)  # return b after a has also run
    """
    return p_depends.bind((value, deps))


def impl_depends[T](in_tree: DependsType[T], /) -> T:
    value, _ = in_tree
    return value


def abstract_depends(in_tree: DependsType[Tree], /) -> Tree:
    value, _ = in_tree
    return value


def push_depends(in_tree: DependsPair, /) -> TreePair:
    (p_value, p_deps), (t_value, t_deps) = in_tree
    out_p = p_depends.bind((p_value, p_deps))
    out_t = p_depends.bind((t_value, t_deps))
    return out_p, out_t


def pull_fwd_depends(in_tree: DependsType[Tree], /) -> DependsFwdResult:
    value, deps = in_tree
    return p_depends.bind((value, deps)), in_tree


def pull_bwd_depends(in_tree: DependsBwdInput, /) -> DependsType[Tree]:
    def make_c(x):
        if isinstance(x, core.Zero):
            return x
        return core.Zero(core.s_cotangent.map(core.avalof(x)))

    (_, deps), out_c = in_tree
    return out_c, utils.tree.map(make_c, deps)


def batch_depends(in_tree: BatchDependsInput, /) -> tuple[Tree, Tree[bool]]:
    _, (value_batched, _), (value, deps) = in_tree
    return p_depends.bind((value, deps)), value_batched


core.r_impl.set(p_depends, impl_depends)
core.r_aimpl.set(p_depends, utils.asyncify(impl_depends))
core.r_abstract.set(p_depends, abstract_depends)
core.r_batch.set(p_depends, batch_depends)
core.r_abatch.set(p_depends, utils.asyncify(batch_depends))
core.r_push.set(p_depends, push_depends)
core.r_apush.set(p_depends, utils.asyncify(push_depends))
core.r_pull_fwd.set(p_depends, pull_fwd_depends)
core.r_apull_fwd.set(p_depends, utils.asyncify(pull_fwd_depends))
core.r_pull_bwd.set(p_depends, pull_bwd_depends)
core.r_apull_bwd.set(p_depends, utils.asyncify(pull_bwd_depends))
