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
from operator import add, setitem
from typing import Any, cast

from autoform.core import (
    IR,
    AVal,
    Intercept,
    Interpreter,
    IREqn,
    IRLit,
    IRVal,
    IRVar,
    Prim,
    TransformationTag,
    TypedAVal,
    abstract_rules,
    active_interpreter,
    batch_rules,
    impl_rules,
    is_irlit,
    is_irvar,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_intercept,
    using_interpreter,
)
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


class FactorIntercept(Intercept): ...


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
    # NOTE(asem): mark factor with an intercept even though execution does not
    # read active_intercept here. this keeps factor semantically live for passes
    # like DCE and prevents it from being treated as an ordinary pure primitive.
    with using_intercept(FactorIntercept()):
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

    def make(atom: IRVal):
        return IRVar.fresh(type=atom.type, source=atom) if is_irvar(atom) else atom

    in_ir_tree = treelib.map(make, ir.in_ir_tree)
    out_ir_tree = (treelib.map(make, ir.out_ir_tree), IRVar.fresh(type=float))
    ir_eqn = IREqn(weight_call_p, None, in_ir_tree, out_ir_tree, dict(ir=ir))
    return IR([ir_eqn], in_ir_tree, out_ir_tree)


def impl_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, float]:
    env: dict[IRVar, Any] = {}

    def read(ir_val: IRVal) -> Any:
        return env[ir_val] if is_irvar(ir_val) else cast(IRLit, ir_val).value

    def check_input(ir_val: IRVal, value: Any):
        if is_irlit(ir_val):
            msg = f"Static input mismatch: expected {ir_val.value!r}, got {value!r}"
            assert ir_val.value == value, msg

    def write(ir_val: IRVal, value: Any):
        is_irvar(ir_val) and setitem(env, ir_val, value)

    treelib.map(check_input, ir.in_ir_tree, in_tree)
    treelib.map(write, ir.in_ir_tree, in_tree)
    interpreter = WeightInterpreter()
    with using_interpreter(interpreter):
        for ir_eqn in ir.ir_eqns:
            in_values = treelib.map(read, ir_eqn.in_ir_tree)
            out_values = ir_eqn.bind(in_values, **ir_eqn.params)
            treelib.map(write, ir_eqn.out_ir_tree, out_values)
    out_tree = treelib.map(read, ir.out_ir_tree)
    total = 0.0 if interpreter.total is None else interpreter.total
    return out_tree, total


async def aimpl_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, float]:
    env: dict[IRVar, Any] = {}

    def read(ir_val: IRVal) -> Any:
        return env[ir_val] if is_irvar(ir_val) else cast(IRLit, ir_val).value

    def check_input(ir_val: IRVal, value: Any):
        if is_irlit(ir_val):
            msg = f"Static input mismatch: expected {ir_val.value!r}, got {value!r}"
            assert ir_val.value == value, msg

    def write(ir_val: IRVal, value: Any):
        is_irvar(ir_val) and setitem(env, ir_val, value)

    treelib.map(check_input, ir.in_ir_tree, in_tree)
    treelib.map(write, ir.in_ir_tree, in_tree)
    interpreter = WeightInterpreter()
    with using_interpreter(interpreter):
        for ir_eqn in ir.ir_eqns:
            in_values = treelib.map(read, ir_eqn.in_ir_tree)
            out_values = await ir_eqn.abind(in_values, **ir_eqn.params)
            treelib.map(write, ir_eqn.out_ir_tree, out_values)
    out_tree = treelib.map(read, ir.out_ir_tree)
    total = 0.0 if interpreter.total is None else interpreter.total
    return out_tree, total


def abstract_weight_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, AVal]:
    del in_tree
    return treelib.map(lambda x: x.aval, ir.out_ir_tree), TypedAVal(float)


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
