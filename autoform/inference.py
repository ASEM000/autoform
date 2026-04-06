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

"""Inference primitives and transforms."""

from __future__ import annotations

import asyncio
import functools as ft
from collections.abc import Callable
from operator import add
from typing import cast

from autoform.core import (
    IR,
    AVal,
    Interpreter,
    IREqn,
    IRVar,
    Prim,
    TransformationTag,
    TypedAVal,
    abstract_rules,
    active_interpreter,
    batch_rules,
    impl_rules,
    ir_aval,
    is_irvar,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.dce import non_dce_primitives
from autoform.memoize import non_memoizable_primitives
from autoform.utils import (
    Tree,
    asyncify,
    batch_index,
    batch_spec,
    batch_transpose,
    lru_cache,
    treelib,
)

__all__ = ["factor", "weight"]


class InferenceTag(TransformationTag): ...


factor_p = Prim("factor", tag={InferenceTag})
weight_call_p = Prim("weight_call", tag={InferenceTag})
non_dce_primitives.add(factor_p)
non_memoizable_primitives.add(factor_p)


def accumulate(total: Tree | None, score: Tree) -> Tree:
    return score if total is None else treelib.map(add, total, score)


def unsupported_factor_transform(name: str):
    def rule(*_, **__):
        msg = f"`factor` has no default {name} rule. "
        msg += "Treat it as a scoring boundary or provide a custom transform rule later."
        raise NotImplementedError(msg)

    return rule


# ==================================================================================================
# FACTOR
# ==================================================================================================


def factor(*args, judge: Callable[..., float]) -> float:
    """Score the current execution point with a user-defined judge.

    ``judge`` is an opaque host callback that receives the positional operands
    and should return a log-weight contribution. ``factor`` itself is pure: it only
    returns the local score. Whole-program aggregation is provided by
    :func:`weight`.
    """

    assert callable(judge), f"`judge` must be callable, got {type(judge)}"
    return factor_p.bind(args, judge=judge)


def impl_factor(in_tree: Tree, /, *, judge: Callable[..., float]) -> float:
    score = judge(*in_tree)
    assert isinstance(score, (int, float)), f"`judge` must return a number, got {type(score)}"
    return float(score)


def abstract_factor(in_tree: Tree, /, *, judge: Callable[..., float]) -> AVal:
    del in_tree, judge
    return TypedAVal(float)


def batch_factor(in_tree: Tree, /, *, judge: Callable[..., float]) -> tuple[Tree, bool]:
    batch_size, in_batched, in_values = in_tree

    if (spec := batch_spec(in_values, in_batched)) is None:
        return factor_p.bind(in_values, judge=judge), False

    unbatch = ft.partial(batch_index, in_values, in_batched)
    results = [factor_p.bind(unbatch(b), judge=judge) for b in range(batch_size)]
    return spec.unflatten(results), True


impl_rules.set(factor_p, impl_factor)
impl_rules.aset(factor_p, asyncify(impl_factor))
abstract_rules.set(factor_p, abstract_factor)
push_rules.set(factor_p, unsupported_factor_transform("pushforward"))
push_rules.aset(factor_p, asyncify(unsupported_factor_transform("pushforward")))
pull_fwd_rules.set(factor_p, unsupported_factor_transform("pullback"))
pull_fwd_rules.aset(factor_p, asyncify(unsupported_factor_transform("pullback")))
pull_bwd_rules.set(factor_p, unsupported_factor_transform("pullback"))
pull_bwd_rules.aset(factor_p, asyncify(unsupported_factor_transform("pullback")))
batch_rules.set(factor_p, batch_factor)
batch_rules.aset(factor_p, asyncify(batch_factor))

# ==================================================================================================
# WEIGHT CALL
# ==================================================================================================


class WeightInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()
        self.total: Tree | None = None

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        out_tree = self.parent.interpret(prim, in_tree, **params)
        if prim is factor_p:
            self.total = accumulate(self.total, out_tree)
        return out_tree

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        out_tree = await self.parent.ainterpret(prim, in_tree, **params)
        if prim is factor_p:
            self.total = accumulate(self.total, out_tree)
        return out_tree


@ft.partial(lru_cache, maxsize=256)
def weight(ir: IR, /) -> IR:
    """Transform an IR to also return the summed executed factor scores.

    Creates a new IR that returns both the original output and the sum of all
    executed ``factor`` values along the path.

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: ``inputs -> (output, total_log_weight)``

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     y = af.concat(x, "!")
        ...     af.factor(x, judge=lambda s: float(len(s)))
        ...     af.factor(y, judge=lambda s: float(len(s)))
        ...     return y
        >>> ir = af.trace(program)("x")
        >>> weight_ir = af.weight(ir)
        >>> weight_ir.call("ab")
        ('ab!', 5.0)
    """

    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make(atom):
        return IRVar.fresh(aval=ir_aval(atom), source=atom) if is_irvar(atom) else atom

    in_ir_tree = treelib.map(make, ir.in_ir_tree)
    out_ir_tree = (treelib.map(make, ir.out_ir_tree), IRVar.fresh(aval=TypedAVal(float)))
    ir_eqn = IREqn(weight_call_p, in_ir_tree, out_ir_tree, dict(ir=ir))
    return IR([ir_eqn], in_ir_tree, out_ir_tree)


def impl_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, float]:
    with using_interpreter(WeightInterpreter()) as interpreter:
        out_tree = ir.call(*cast(tuple, in_tree))
    total = 0.0 if interpreter.total is None else interpreter.total
    return out_tree, total


async def aimpl_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, float]:
    with using_interpreter(WeightInterpreter()) as interpreter:
        out_tree = await ir.acall(*cast(tuple, in_tree))
    total = 0.0 if interpreter.total is None else interpreter.total
    return out_tree, total


def abstract_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, AVal]:
    del in_tree
    return treelib.map(ir_aval, ir.out_ir_tree), TypedAVal(float)


def unsupported_weight_transform(name: str):
    def rule(*_, **__):
        msg = f"`weight` has no default {name} rule yet. "
        msg += "Use it with ordinary execution or batch(weight(ir)) for now."
        raise NotImplementedError(msg)

    return rule


def batch_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree
    weight_ir = weight(ir)

    if batch_spec(in_values, in_batched) is None:
        result = weight_ir.call(*in_values)
        out_batched = treelib.map(lambda _: False, weight_ir.out_ir_tree)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    out_bi = [weight_ir.call(*unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, weight_ir.out_ir_tree)
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree
    weight_ir = weight(ir)

    if batch_spec(in_values, in_batched) is None:
        result = await weight_ir.acall(*in_values)
        out_batched = treelib.map(lambda _: False, weight_ir.out_ir_tree)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    out_bi = await asyncio.gather(*[weight_ir.acall(*unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, weight_ir.out_ir_tree)
    out_ib = batch_transpose(batch_size, out_batched, list(out_bi))
    return out_ib, out_batched


impl_rules.set(weight_call_p, impl_weight_call)
impl_rules.aset(weight_call_p, aimpl_weight_call)
abstract_rules.set(weight_call_p, abstract_weight_call)
push_rules.set(weight_call_p, unsupported_weight_transform("pushforward"))
push_rules.aset(weight_call_p, asyncify(unsupported_weight_transform("pushforward")))
pull_fwd_rules.set(weight_call_p, unsupported_weight_transform("pullback"))
pull_fwd_rules.aset(weight_call_p, asyncify(unsupported_weight_transform("pullback")))
pull_bwd_rules.set(weight_call_p, unsupported_weight_transform("pullback"))
pull_bwd_rules.aset(weight_call_p, asyncify(unsupported_weight_transform("pullback")))
batch_rules.set(weight_call_p, batch_weight_call)
batch_rules.aset(weight_call_p, abatch_weight_call)
