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

"""Path-weight primitives."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Hashable

import autoform.core as core
import autoform.dead as dead
import autoform.memo as memo
import autoform.order as order
import autoform.utils as utils

__all__ = ["factor", "weighted"]

factor_p = core.Prim("factor")
dead.non_dce_primitives.add(factor_p)
memo.non_memoizable_primitives.add(factor_p)

number_type = (int, float, core.IntAVal, core.FloatAVal)


def factor(weight: float, /, *, name: Hashable | None = None) -> None:
    """Multiply the current path weight by ``weight``.

    ``factor`` is neutral during ordinary execution. When an IR is transformed
    with ``weighted``, each reached factor contributes to the returned path
    weight.
    """
    hash(name)
    factor_p.bind(weight, name=name)
    return None


def impl_factor(weight: float, /, *, name: Hashable | None = None) -> None:
    del name
    assert type(weight) in (int, float), f"Expected numeric factor weight: {weight!r}"
    assert math.isfinite(weight) and weight >= 0, (
        f"Expected finite non-negative factor weight: {weight!r}"
    )
    return ()


def abstract_factor(weight, /, *, name: Hashable | None = None) -> None:
    del name
    assert type(weight) in number_type, f"Expected number: {weight!r}"
    return ()


def pushforward_factor(in_tree, /, *, name: Hashable | None = None):
    weight, _ = in_tree
    factor_p.bind(weight, name=name)
    return (), ()


def pullback_fwd_factor(weight, /, *, name: Hashable | None = None):
    factor_p.bind(weight, name=name)
    return (), weight


def pullback_bwd_factor(in_tree, /, *, name: Hashable | None = None):
    import autoform.ad as ad

    del name
    weight, _ = in_tree
    return ad.cotangent_zeroof(weight)


def batch_factor(in_tree, /, *, name: Hashable | None = None):
    batch_size, in_batched, weight = in_tree

    if utils.batch_spec(weight, in_batched) is None:
        factor_p.bind(weight, name=name)
        return (), ()

    for b in range(batch_size):
        factor_p.bind(utils.batch_index(weight, in_batched, b), name=name)
    return (), ()


async def abatch_factor(in_tree, /, *, name: Hashable | None = None):
    batch_size, in_batched, weight = in_tree

    if utils.batch_spec(weight, in_batched) is None:
        await factor_p.abind(weight, name=name)
        return (), ()

    await asyncio.gather(*[
        factor_p.abind(utils.batch_index(weight, in_batched, b), name=name)
        for b in range(batch_size)
    ])
    return (), ()


core.impl_rules.set(factor_p, impl_factor)
core.impl_rules.aset(factor_p, utils.asyncify(impl_factor))
core.abstract_rules.set(factor_p, abstract_factor)
core.push_rules.set(factor_p, pushforward_factor)
core.push_rules.aset(factor_p, utils.asyncify(pushforward_factor))
core.pull_fwd_rules.set(factor_p, pullback_fwd_factor)
core.pull_fwd_rules.aset(factor_p, utils.asyncify(pullback_fwd_factor))
core.pull_bwd_rules.set(factor_p, pullback_bwd_factor)
core.pull_bwd_rules.aset(factor_p, utils.asyncify(pullback_bwd_factor))
core.batch_rules.set(factor_p, batch_factor)
core.batch_rules.aset(factor_p, abatch_factor)


weighted_call_p = core.Prim("weighted_call")


class PathWeightInterpreter(core.Interpreter):
    __slots__ = ["log_weight", "parent"]

    def __init__(self):
        self.parent = core.active_interpreter.get()
        self.log_weight = 0.0

    def factor(self, weight: float, /) -> None:
        self.log_weight += -math.inf if weight == 0 else math.log(weight)

    def interpret(self, prim: core.Prim, in_tree, /, **params):
        if prim is factor_p:
            impl_factor(in_tree, **params)
            self.factor(in_tree)
            return ()
        return self.parent.interpret(prim, in_tree, **params)

    async def ainterpret(self, prim: core.Prim, in_tree, /, **params):
        if prim is factor_p:
            impl_factor(in_tree, **params)
            self.factor(in_tree)
            return ()
        return await self.parent.ainterpret(prim, in_tree, **params)


def weighted(ir: core.IR, /) -> core.IR:
    """Transform an IR to return ``(output, path_weight)`` for one path."""
    assert isinstance(ir, core.IR), f"Expected IR, got {type(ir)}"

    def make_out(atom):
        if core.is_var(atom):
            return core.Var.fresh(aval=core.aval_if_var(atom), source=atom)
        return atom

    in_tree = ir.in_tree
    out_tree = (
        utils.tree.map(make_out, ir.out_tree),
        core.Var.fresh(aval=core.FloatAVal()),
    )
    eqn = core.Eqn(weighted_call_p, in_tree, out_tree, dict(ir=ir))
    return core.IR([eqn], in_tree, out_tree)


def impl_weighted_call(in_tree, /, *, ir: core.IR):
    interpreter = PathWeightInterpreter()
    with core.using_interpreter(interpreter):
        output = ir.call(*in_tree)
    weight = 0.0 if interpreter.log_weight == -math.inf else math.exp(interpreter.log_weight)
    return output, weight


async def aimpl_weighted_call(in_tree, /, *, ir: core.IR):
    interpreter = PathWeightInterpreter()
    with core.using_interpreter(interpreter):
        output = await ir.acall(*in_tree)
    weight = 0.0 if interpreter.log_weight == -math.inf else math.exp(interpreter.log_weight)
    return output, weight


def abstract_weighted_call(in_tree, /, *, ir: core.IR):
    del in_tree
    return utils.tree.map(core.aval_if_var, ir.out_tree), core.FloatAVal()


def unsupported_weighted_call_transform(transform: str) -> None:
    raise NotImplementedError(
        f"`{transform}(af.weighted(ir))` is not supported. Apply `af.weighted` after "
        f"`{transform}` if that is the intended path-weight semantics."
    )


def pushforward_weighted_call(in_tree, /, *, ir: core.IR):
    del in_tree, ir
    unsupported_weighted_call_transform("pushforward")


def pullback_fwd_weighted_call(in_tree, /, *, ir: core.IR):
    del in_tree, ir
    unsupported_weighted_call_transform("pullback")


def pullback_bwd_weighted_call(in_tree, /, *, ir: core.IR):
    del in_tree, ir
    unsupported_weighted_call_transform("pullback")


def batch_weighted_call(in_tree, /, *, ir: core.IR):
    batch_size, in_batched, in_values = in_tree

    if utils.batch_spec(in_values, in_batched) is None:
        return weighted_call_p.bind(in_values, ir=ir), False

    weighted_ir = weighted(ir)
    inputs = [utils.batch_index(in_values, in_batched, b) for b in range(batch_size)]
    out_bi = order.fanout_p.bind(inputs, irs=[weighted_ir] * batch_size)
    out_batched = utils.tree.map(lambda _: True, out_bi[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_weighted_call(in_tree, /, *, ir: core.IR):
    batch_size, in_batched, in_values = in_tree

    if utils.batch_spec(in_values, in_batched) is None:
        return await weighted_call_p.abind(in_values, ir=ir), False

    weighted_ir = weighted(ir)
    inputs = [utils.batch_index(in_values, in_batched, b) for b in range(batch_size)]
    out_bi = await order.fanout_p.abind(inputs, irs=[weighted_ir] * batch_size)
    out_batched = utils.tree.map(lambda _: True, out_bi[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


def dce_weighted_call(eqn: core.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    output_used, _ = out_used
    new_eqn = eqn.using(ir=dead.dce(eqn.params["ir"], out_used=output_used))
    return dead.default_dce(new_eqn, out_used)


core.impl_rules.set(weighted_call_p, impl_weighted_call)
core.impl_rules.aset(weighted_call_p, aimpl_weighted_call)
core.abstract_rules.set(weighted_call_p, abstract_weighted_call)
core.push_rules.set(weighted_call_p, pushforward_weighted_call)
core.push_rules.aset(weighted_call_p, utils.asyncify(pushforward_weighted_call))
core.pull_fwd_rules.set(weighted_call_p, pullback_fwd_weighted_call)
core.pull_fwd_rules.aset(weighted_call_p, utils.asyncify(pullback_fwd_weighted_call))
core.pull_bwd_rules.set(weighted_call_p, pullback_bwd_weighted_call)
core.pull_bwd_rules.aset(weighted_call_p, utils.asyncify(pullback_bwd_weighted_call))
core.batch_rules.set(weighted_call_p, batch_weighted_call)
core.batch_rules.aset(weighted_call_p, abatch_weighted_call)
dead.dce_rules[weighted_call_p] = dce_weighted_call
