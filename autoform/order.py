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

import autoform.analysis as analysis
import autoform.core as core
import autoform.dead as dead
import autoform.utils as utils

__all__ = ["depends", "sched", "serial_fanout"]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type IRList = list[core.IR]
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


fanout_p = core.Prim("fanout")


def impl_fanout(in_tree: list[Tree], /, *, irs: IRList) -> list[Tree]:
    assert len(in_tree) == len(irs)
    return [ir.call(*inp) for ir, inp in zip(irs, in_tree, strict=True)]


async def aimpl_fanout(in_tree: list[Tree], /, *, irs: IRList) -> list[Tree]:
    assert len(in_tree) == len(irs)
    if len(irs) == 1:
        [ir], [inp] = irs, in_tree
        return [await ir.acall(*inp)]
    if serial_fanout_flag.get():
        return [await ir.acall(*inp) for ir, inp in zip(irs, in_tree, strict=True)]
    return await asyncio.gather(*[ir.acall(*inp) for ir, inp in zip(irs, in_tree, strict=True)])


def abstract_fanout(in_tree: list[Tree], /, *, irs: IRList) -> list[Tree]:
    return [utils.tree.map(core.aval_if_var, ir.out_tree) for ir in irs]


def push_fanout(in_tree: FanoutPair, /, *, irs: IRList) -> FanoutPair:
    import autoform.ad as ad

    primals, tangents = in_tree
    pf_irs = [ad.pushforward(ir) for ir in irs]
    pf_inputs = [(p, t) for p, t in zip(primals, tangents, strict=True)]
    results = fanout_p.bind(pf_inputs, irs=pf_irs)
    p_outs, t_outs = zip(*results)
    return list(p_outs), list(t_outs)


async def apush_fanout(in_tree: FanoutPair, /, *, irs: IRList) -> FanoutPair:
    import autoform.ad as ad

    primals, tangents = in_tree
    pf_irs = [ad.pushforward(ir) for ir in irs]
    pf_inputs = [(p, t) for p, t in zip(primals, tangents, strict=True)]
    results = await fanout_p.abind(pf_inputs, irs=pf_irs)
    p_outs, t_outs = zip(*results)
    return list(p_outs), list(t_outs)


def pull_fwd_fanout(in_tree: list[Tree], /, *, irs: IRList) -> FanoutFwdResult:
    results = fanout_p.bind(in_tree, irs=irs)
    residuals = (in_tree, irs)
    return results, residuals


async def apull_fwd_fanout(in_tree: list[Tree], /, *, irs: IRList) -> FanoutFwdResult:
    results = await fanout_p.abind(in_tree, irs=irs)
    residuals = (in_tree, irs)
    return results, residuals


def pull_bwd_fanout(in_tree: Tree, /, *, irs: IRList) -> list[Tree]:
    import autoform.ad as ad

    residuals, out_cotangent = in_tree
    inputs, _ = residuals
    pb_irs = [ad.pullback(ir) for ir in irs]
    pb_inputs = [(inp, cot) for inp, cot in zip(inputs, out_cotangent, strict=True)]
    results = fanout_p.bind(pb_inputs, irs=pb_irs)
    return [cot for _, cot in results]


async def apull_bwd_fanout(in_tree: Tree, /, *, irs: IRList) -> list[Tree]:
    import autoform.ad as ad

    residuals, out_cotangent = in_tree
    inputs, _ = residuals
    pb_irs = [ad.pullback(ir) for ir in irs]
    pb_inputs = [(inp, cot) for inp, cot in zip(inputs, out_cotangent, strict=True)]
    results = await fanout_p.abind(pb_inputs, irs=pb_irs)
    return [cot for _, cot in results]


def batch_fanout(in_tree: BatchFanoutInput, /, *, irs: IRList) -> BatchFanoutOutput:
    import autoform.axis as axis

    _, in_batched, inputs = in_tree

    results: list[Tree] = []
    out_batched: list[Tree[bool]] = []

    for ir, inp, inp_batched in zip(irs, inputs, in_batched, strict=True):
        if utils.batch_spec(inp, inp_batched) is None:
            results.append(ir.call(*inp))
            out_batched.append(utils.tree.map(lambda _: False, ir.out_tree))
        else:
            batched_ir = axis.batch(ir, in_axes=inp_batched)
            results.append(batched_ir.call(*inp))
            out_batched.append(utils.tree.map(lambda _: True, ir.out_tree))

    return results, out_batched


async def abatch_fanout(in_tree: BatchFanoutInput, /, *, irs: IRList) -> BatchFanoutOutput:
    import autoform.axis as axis

    _, in_batched, inputs = in_tree

    results: list[Tree] = []
    out_batched: list[Tree[bool]] = []

    for ir, inp, inp_batched in zip(irs, inputs, in_batched, strict=True):
        if utils.batch_spec(inp, inp_batched) is None:
            results.append(await ir.acall(*inp))
            out_batched.append(utils.tree.map(lambda _: False, ir.out_tree))
        else:
            batched_ir = axis.batch(ir, in_axes=inp_batched)
            results.append(await batched_ir.acall(*inp))
            out_batched.append(utils.tree.map(lambda _: True, ir.out_tree))

    return results, out_batched


core.impl_rules.set(fanout_p, impl_fanout)
core.impl_rules.aset(fanout_p, aimpl_fanout)
core.abstract_rules.set(fanout_p, abstract_fanout)
core.push_rules.set(fanout_p, push_fanout)
core.push_rules.aset(fanout_p, apush_fanout)
core.pull_fwd_rules.set(fanout_p, pull_fwd_fanout)
core.pull_fwd_rules.aset(fanout_p, apull_fwd_fanout)
core.pull_bwd_rules.set(fanout_p, pull_bwd_fanout)
core.pull_bwd_rules.aset(fanout_p, apull_bwd_fanout)
core.batch_rules.set(fanout_p, batch_fanout)
core.batch_rules.aset(fanout_p, abatch_fanout)


def dce_fanout(eqn: core.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    irs = eqn.params["irs"]
    new_irs = [dead.dce(ir, out_used=ou) for ir, ou in zip(irs, out_used, strict=True)]
    new_eqn = eqn.using(irs=new_irs)
    return dead.default_dce(new_eqn, out_used)


dead.dce_rules[fanout_p] = dce_fanout

# ==================================================================================================
# SCHED
# ==================================================================================================


@ft.partial(utils.lru_cache, maxsize=256)
def sched[*A, R](
    ir: core.IR[*A, R], /, *, cond: Callable[[core.Eqn], bool] | None = None
) -> core.IR[*A, R]:
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
        ...     msg1 = [dict(role="user", content=af.format("Q1: {}", x))]
        ...     msg2 = [dict(role="user", content=af.format("Q2: {}", x))]
        ...     a = af.lm_call(msg1, model="gpt-5.5")
        ...     b = af.lm_call(msg2, model="gpt-5.5")
        ...     return af.concat(a, b)
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
    levels: list[list[core.Eqn]] = analysis.toposort_levels(ir)
    out_eqns: list[core.Eqn] = []
    cond = (lambda _: True) if cond is None else cond

    def recurse(leaf):
        return sched(leaf, cond=cond) if isinstance(leaf, core.IR) else leaf

    def make_fanout(eqns: list[core.Eqn]) -> core.Eqn:
        irs = [core.IR([eqn], (eqn.in_tree,), eqn.out_tree) for eqn in eqns]
        in_tree = [(eqn.in_tree,) for eqn in eqns]
        out_tree = [eqn.out_tree for eqn in eqns]
        return core.Eqn(fanout_p, in_tree, out_tree, dict(irs=irs))

    for level in levels:
        eqns = [eqn.using(**utils.tree.map(recurse, eqn.params)) for eqn in level]
        seq_eqns = [eqn for eqn in eqns if not cond(eqn)]
        par_eqns = [eqn for eqn in eqns if cond(eqn)]
        out_eqns.extend([make_fanout(par_eqns)] if len(par_eqns) > 1 else par_eqns)
        out_eqns.extend(seq_eqns)

    return core.IR(out_eqns, ir.in_tree, ir.out_tree)


# ==================================================================================================
# DEPENDS
# ==================================================================================================

depends_p = core.Prim("depends")


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
        ...     a = af.format("First: {}", x)
        ...     b = af.format("Second: {}", x)
        ...     return af.depends(b, a)  # return b after a has also run
    """
    return depends_p.bind((value, deps))


def impl_depends[T](in_tree: DependsType[T], /) -> T:
    value, _ = in_tree
    return value


def abstract_depends(in_tree: DependsType[Tree], /) -> Tree:
    value, _ = in_tree
    return value


def push_depends(in_tree: DependsPair, /) -> TreePair:
    (primal_value, primal_deps), (tangent_value, tangent_deps) = in_tree
    p_out = depends_p.bind((primal_value, primal_deps))
    t_out = depends_p.bind((tangent_value, tangent_deps))
    return p_out, t_out


def pull_fwd_depends(in_tree: DependsType[Tree], /) -> DependsFwdResult:
    value, deps = in_tree
    return depends_p.bind((value, deps)), in_tree


def pull_bwd_depends(in_tree: DependsBwdInput, /) -> DependsType[Tree]:
    import autoform.ad as ad

    (_, deps), out_cotangent = in_tree
    return out_cotangent, utils.tree.map(
        lambda d: d if ad.is_zero(d) else ad.cotangent_zeroof(d), deps
    )


def batch_depends(in_tree: BatchDependsInput, /) -> core.BatchRuleResult:
    _, (value_batched, _), (value, deps) = in_tree
    return depends_p.bind((value, deps)), value_batched


core.impl_rules.set(depends_p, impl_depends)
core.impl_rules.aset(depends_p, utils.asyncify(impl_depends))
core.abstract_rules.set(depends_p, abstract_depends)
core.push_rules.set(depends_p, push_depends)
core.push_rules.aset(depends_p, utils.asyncify(push_depends))
core.pull_fwd_rules.set(depends_p, pull_fwd_depends)
core.pull_fwd_rules.aset(depends_p, utils.asyncify(pull_fwd_depends))
core.pull_bwd_rules.set(depends_p, pull_bwd_depends)
core.pull_bwd_rules.aset(depends_p, utils.asyncify(pull_bwd_depends))
core.batch_rules.set(depends_p, batch_depends)
core.batch_rules.aset(depends_p, utils.asyncify(batch_depends))
