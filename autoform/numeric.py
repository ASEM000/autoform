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

"""Numeric primitives."""

from __future__ import annotations

import functools as ft

import autoform.abstract as abstract
import autoform.core as core
import autoform.tracer as tracer
import autoform.utils as utils

__all__ = [
    "IntAVal",
    "FloatAVal",
    "BoolAVal",
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# TYPES
# ==================================================================================================


class IntAVal(abstract.AVal):
    """Abstract value for ``int`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1)
        >>> (x,) = ir.in_tree
        >>> x.aval
        IntAVal()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


abstract.aval_types[int] = lambda _: IntAVal()
tracer.trace_types.add(int)
abstract.primal_s.set(IntAVal, lambda aval: aval)


class FloatAVal(abstract.AVal):
    """Abstract value for ``float`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1.0)
        >>> (x,) = ir.in_tree
        >>> x.aval
        FloatAVal()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def zero(self) -> float:
        return 0.0

    def accumulate(self, cotangents: list[float], /) -> float:
        return sum(cotangents)


abstract.aval_types[float] = lambda _: FloatAVal()
tracer.trace_types.add(float)
abstract.primal_s.set(FloatAVal, lambda aval: aval)
abstract.tangent_s.set(FloatAVal, lambda aval: aval)
abstract.cotangent_s.set(FloatAVal, lambda aval: aval)


class BoolAVal(abstract.AVal):
    """Abstract value for ``bool`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(True)
        >>> (x,) = ir.in_tree
        >>> x.aval
        BoolAVal()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


abstract.aval_types[bool] = lambda _: BoolAVal()
tracer.trace_types.add(bool)
abstract.primal_s.set(BoolAVal, lambda aval: aval)
abstract.tangent_s.set(BoolAVal, lambda aval: aval)
abstract.cotangent_s.set(BoolAVal, lambda aval: aval)


def batch_unary(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, in_value = in_tree
    if (spec := utils.batch_spec(in_value, in_batched)) is None:
        return prim.bind(in_value), False
    result = [prim.bind(utils.batch_index(in_value, True, b)) for b in range(batch_size)]
    return spec.unflatten(result), True


def batch_binary(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    batch_size, in_batched, in_values = in_tree
    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return prim.bind(in_values, **params), False
    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    result = [prim.bind(unbatch(b), **params) for b in range(batch_size)]
    return spec.unflatten(result), True


# ==================================================================================================
# NEG
# ==================================================================================================

neg_p = core.Prim("neg")


def neg(a, /):
    if type(a) is int:
        a = float(a)
    return neg_p.bind(a)


def impl_neg(in_tree: Tree, /):
    assert type(in_tree) is float, f"Expected float: {in_tree!r}"
    return -in_tree


def abstract_neg(in_tree: Tree, /) -> FloatAVal:
    assert type(in_tree) in (float, FloatAVal), f"Expected float: {in_tree!r}"
    return FloatAVal()


def pushforward_neg(in_tree: Tree, /) -> TreePair:
    primal, tangent = in_tree
    return neg(primal), neg(abstract.materialize_zeros(tangent))


def pullback_fwd_neg(in_tree: Tree, /) -> TreePair:
    return neg(in_tree), None


def pullback_bwd_neg(in_tree: Tree, /) -> Tree:
    _, out_cotangent = in_tree
    return neg(out_cotangent)


def batch_neg(in_tree: Tree, /) -> TreePair:
    return batch_unary(neg_p, in_tree)


core.impl_rules[neg_p] = impl_neg
core.aimpl_rules[neg_p] = utils.asyncify(impl_neg)
core.abstract_rules[neg_p] = abstract_neg
core.push_rules[neg_p] = pushforward_neg
core.apush_rules[neg_p] = utils.asyncify(pushforward_neg)
core.pull_fwd_rules[neg_p] = pullback_fwd_neg
core.apull_fwd_rules[neg_p] = utils.asyncify(pullback_fwd_neg)
core.pull_bwd_rules[neg_p] = pullback_bwd_neg
core.apull_bwd_rules[neg_p] = utils.asyncify(pullback_bwd_neg)
core.batch_rules[neg_p] = batch_neg
core.abatch_rules[neg_p] = utils.asyncify(batch_neg)


# ==================================================================================================
# ADD
# ==================================================================================================

add_p = core.Prim("add")


def add(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return add_p.bind((a, b))


def impl_add(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a + b


def abstract_add(in_tree: Tree, /) -> FloatAVal:
    assert all(type(x) in (float, FloatAVal) for x in in_tree), f"Expected floats: {in_tree!r}"
    return FloatAVal()


def pushforward_add(in_tree: Tree, /) -> TreePair:
    primals, tangents = in_tree
    tangents = abstract.materialize_zeros(tangents)
    return add_p.bind(primals), add_p.bind(tangents)


def pullback_fwd_add(in_tree: Tree, /) -> TreePair:
    return add_p.bind(in_tree), None


def pullback_bwd_add(in_tree: Tree, /) -> Tree:
    _, out_cotangent = in_tree
    return out_cotangent, out_cotangent


def batch_add(in_tree: Tree, /) -> TreePair:
    return batch_binary(add_p, in_tree)


core.impl_rules[add_p] = impl_add
core.aimpl_rules[add_p] = utils.asyncify(impl_add)
core.abstract_rules[add_p] = abstract_add
core.push_rules[add_p] = pushforward_add
core.apush_rules[add_p] = utils.asyncify(pushforward_add)
core.pull_fwd_rules[add_p] = pullback_fwd_add
core.apull_fwd_rules[add_p] = utils.asyncify(pullback_fwd_add)
core.pull_bwd_rules[add_p] = pullback_bwd_add
core.apull_bwd_rules[add_p] = utils.asyncify(pullback_bwd_add)
core.batch_rules[add_p] = batch_add
core.abatch_rules[add_p] = utils.asyncify(batch_add)


# ==================================================================================================
# SUB
# ==================================================================================================

sub_p = core.Prim("sub")


def sub(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return sub_p.bind((a, b))


def impl_sub(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a - b


def abstract_sub(in_tree: Tree, /) -> FloatAVal:
    assert all(type(x) in (float, FloatAVal) for x in in_tree), f"Expected floats: {in_tree!r}"
    return FloatAVal()


def pushforward_sub(in_tree: Tree, /) -> TreePair:
    primals, tangents = in_tree
    tangents = abstract.materialize_zeros(tangents)
    return sub_p.bind(primals), sub_p.bind(tangents)


def pullback_fwd_sub(in_tree: Tree, /) -> TreePair:
    return sub_p.bind(in_tree), None


def pullback_bwd_sub(in_tree: Tree, /) -> Tree:
    _, out_cotangent = in_tree
    return out_cotangent, neg(out_cotangent)


def batch_sub(in_tree: Tree, /) -> TreePair:
    return batch_binary(sub_p, in_tree)


core.impl_rules[sub_p] = impl_sub
core.aimpl_rules[sub_p] = utils.asyncify(impl_sub)
core.abstract_rules[sub_p] = abstract_sub
core.push_rules[sub_p] = pushforward_sub
core.apush_rules[sub_p] = utils.asyncify(pushforward_sub)
core.pull_fwd_rules[sub_p] = pullback_fwd_sub
core.apull_fwd_rules[sub_p] = utils.asyncify(pullback_fwd_sub)
core.pull_bwd_rules[sub_p] = pullback_bwd_sub
core.apull_bwd_rules[sub_p] = utils.asyncify(pullback_bwd_sub)
core.batch_rules[sub_p] = batch_sub
core.abatch_rules[sub_p] = utils.asyncify(batch_sub)


# ==================================================================================================
# MUL
# ==================================================================================================

mul_p = core.Prim("mul")


def mul(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return mul_p.bind((a, b))


def impl_mul(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a * b


def abstract_mul(in_tree: Tree, /) -> FloatAVal:
    assert all(type(x) in (float, FloatAVal) for x in in_tree), f"Expected floats: {in_tree!r}"
    return FloatAVal()


def pushforward_mul(in_tree: Tree, /) -> TreePair:
    primals, tangents = in_tree
    a, b = primals
    da, db = abstract.materialize_zeros(tangents)
    return mul(a, b), add(mul(da, b), mul(a, db))


def pullback_fwd_mul(in_tree: Tree, /) -> TreePair:
    return mul_p.bind(in_tree), in_tree


def pullback_bwd_mul(in_tree: Tree, /) -> Tree:
    (a, b), out_cotangent = in_tree
    return mul(out_cotangent, b), mul(out_cotangent, a)


def batch_mul(in_tree: Tree, /) -> TreePair:
    return batch_binary(mul_p, in_tree)


core.impl_rules[mul_p] = impl_mul
core.aimpl_rules[mul_p] = utils.asyncify(impl_mul)
core.abstract_rules[mul_p] = abstract_mul
core.push_rules[mul_p] = pushforward_mul
core.apush_rules[mul_p] = utils.asyncify(pushforward_mul)
core.pull_fwd_rules[mul_p] = pullback_fwd_mul
core.apull_fwd_rules[mul_p] = utils.asyncify(pullback_fwd_mul)
core.pull_bwd_rules[mul_p] = pullback_bwd_mul
core.apull_bwd_rules[mul_p] = utils.asyncify(pullback_bwd_mul)
core.batch_rules[mul_p] = batch_mul
core.abatch_rules[mul_p] = utils.asyncify(batch_mul)


# ==================================================================================================
# DIV
# ==================================================================================================

div_p = core.Prim("div")


def div(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return div_p.bind((a, b))


def impl_div(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a / b


def abstract_div(in_tree: Tree, /) -> FloatAVal:
    assert all(type(x) in (float, FloatAVal) for x in in_tree), f"Expected floats: {in_tree!r}"
    return FloatAVal()


def pushforward_div(in_tree: Tree, /) -> TreePair:
    primals, tangents = in_tree
    a, b = primals
    da, db = abstract.materialize_zeros(tangents)
    return div(a, b), div(sub(mul(da, b), mul(a, db)), mul(b, b))


def pullback_fwd_div(in_tree: Tree, /) -> TreePair:
    return div_p.bind(in_tree), in_tree


def pullback_bwd_div(in_tree: Tree, /) -> Tree:
    (a, b), out_cotangent = in_tree
    a_cotangent = div(out_cotangent, b)
    b_cotangent = neg(div(mul(out_cotangent, a), mul(b, b)))
    return a_cotangent, b_cotangent


def batch_div(in_tree: Tree, /) -> TreePair:
    return batch_binary(div_p, in_tree)


core.impl_rules[div_p] = impl_div
core.aimpl_rules[div_p] = utils.asyncify(impl_div)
core.abstract_rules[div_p] = abstract_div
core.push_rules[div_p] = pushforward_div
core.apush_rules[div_p] = utils.asyncify(pushforward_div)
core.pull_fwd_rules[div_p] = pullback_fwd_div
core.apull_fwd_rules[div_p] = utils.asyncify(pullback_fwd_div)
core.pull_bwd_rules[div_p] = pullback_bwd_div
core.apull_bwd_rules[div_p] = utils.asyncify(pullback_bwd_div)
core.batch_rules[div_p] = batch_div
core.abatch_rules[div_p] = utils.asyncify(batch_div)


# ==================================================================================================
# COMPARISONS
# ==================================================================================================


def abstract_compare(in_tree: Tree, /) -> BoolAVal:
    assert all(type(x) in (float, FloatAVal) for x in in_tree)
    return BoolAVal()


def pushforward_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    primals, _ = in_tree
    return prim.bind(primals), abstract.Zero(abstract.tangent_s.map(BoolAVal()))


def pullback_fwd_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    return prim.bind(in_tree), in_tree


def pullback_bwd_compare(in_tree: Tree, /) -> Tree:
    def make_c(x):
        if isinstance(x, abstract.Zero):
            return x
        return abstract.Zero(abstract.cotangent_s.map(abstract.avalof(x)))

    primals, _ = in_tree
    return utils.tree.map(make_c, primals)


def batch_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    return batch_binary(prim, in_tree)


# ==================================================================================================
# EQ
# ==================================================================================================

eq_p = core.Prim("eq")


def eq(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return eq_p.bind((a, b))


def impl_eq(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a == b


pushforward_eq = ft.partial(pushforward_compare, eq_p)
pullback_fwd_eq = ft.partial(pullback_fwd_compare, eq_p)
batch_eq = ft.partial(batch_compare, eq_p)
core.impl_rules[eq_p] = impl_eq
core.aimpl_rules[eq_p] = utils.asyncify(impl_eq)
core.abstract_rules[eq_p] = abstract_compare
core.push_rules[eq_p] = pushforward_eq
core.apush_rules[eq_p] = utils.asyncify(pushforward_eq)
core.pull_fwd_rules[eq_p] = pullback_fwd_eq
core.apull_fwd_rules[eq_p] = utils.asyncify(pullback_fwd_eq)
core.pull_bwd_rules[eq_p] = pullback_bwd_compare
core.apull_bwd_rules[eq_p] = utils.asyncify(pullback_bwd_compare)
core.batch_rules[eq_p] = batch_eq
core.abatch_rules[eq_p] = utils.asyncify(batch_eq)


# ==================================================================================================
# NE
# ==================================================================================================

ne_p = core.Prim("ne")


def ne(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return ne_p.bind((a, b))


def impl_ne(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a != b


pushforward_ne = ft.partial(pushforward_compare, ne_p)
pullback_fwd_ne = ft.partial(pullback_fwd_compare, ne_p)
batch_ne = ft.partial(batch_compare, ne_p)
core.impl_rules[ne_p] = impl_ne
core.aimpl_rules[ne_p] = utils.asyncify(impl_ne)
core.abstract_rules[ne_p] = abstract_compare
core.push_rules[ne_p] = pushforward_ne
core.apush_rules[ne_p] = utils.asyncify(pushforward_ne)
core.pull_fwd_rules[ne_p] = pullback_fwd_ne
core.apull_fwd_rules[ne_p] = utils.asyncify(pullback_fwd_ne)
core.pull_bwd_rules[ne_p] = pullback_bwd_compare
core.apull_bwd_rules[ne_p] = utils.asyncify(pullback_bwd_compare)
core.batch_rules[ne_p] = batch_ne
core.abatch_rules[ne_p] = utils.asyncify(batch_ne)


# ==================================================================================================
# LT
# ==================================================================================================

lt_p = core.Prim("lt")


def lt(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return lt_p.bind((a, b))


def impl_lt(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a < b


pushforward_lt = ft.partial(pushforward_compare, lt_p)
pullback_fwd_lt = ft.partial(pullback_fwd_compare, lt_p)
batch_lt = ft.partial(batch_compare, lt_p)
core.impl_rules[lt_p] = impl_lt
core.aimpl_rules[lt_p] = utils.asyncify(impl_lt)
core.abstract_rules[lt_p] = abstract_compare
core.push_rules[lt_p] = pushforward_lt
core.apush_rules[lt_p] = utils.asyncify(pushforward_lt)
core.pull_fwd_rules[lt_p] = pullback_fwd_lt
core.apull_fwd_rules[lt_p] = utils.asyncify(pullback_fwd_lt)
core.pull_bwd_rules[lt_p] = pullback_bwd_compare
core.apull_bwd_rules[lt_p] = utils.asyncify(pullback_bwd_compare)
core.batch_rules[lt_p] = batch_lt
core.abatch_rules[lt_p] = utils.asyncify(batch_lt)


# ==================================================================================================
# LE
# ==================================================================================================

le_p = core.Prim("le")


def le(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return le_p.bind((a, b))


def impl_le(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a <= b


pushforward_le = ft.partial(pushforward_compare, le_p)
pullback_fwd_le = ft.partial(pullback_fwd_compare, le_p)
batch_le = ft.partial(batch_compare, le_p)
core.impl_rules[le_p] = impl_le
core.aimpl_rules[le_p] = utils.asyncify(impl_le)
core.abstract_rules[le_p] = abstract_compare
core.push_rules[le_p] = pushforward_le
core.apush_rules[le_p] = utils.asyncify(pushforward_le)
core.pull_fwd_rules[le_p] = pullback_fwd_le
core.apull_fwd_rules[le_p] = utils.asyncify(pullback_fwd_le)
core.pull_bwd_rules[le_p] = pullback_bwd_compare
core.apull_bwd_rules[le_p] = utils.asyncify(pullback_bwd_compare)
core.batch_rules[le_p] = batch_le
core.abatch_rules[le_p] = utils.asyncify(batch_le)


# ==================================================================================================
# GT
# ==================================================================================================

gt_p = core.Prim("gt")


def gt(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return gt_p.bind((a, b))


def impl_gt(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a > b


pushforward_gt = ft.partial(pushforward_compare, gt_p)
pullback_fwd_gt = ft.partial(pullback_fwd_compare, gt_p)
batch_gt = ft.partial(batch_compare, gt_p)
core.impl_rules[gt_p] = impl_gt
core.aimpl_rules[gt_p] = utils.asyncify(impl_gt)
core.abstract_rules[gt_p] = abstract_compare
core.push_rules[gt_p] = pushforward_gt
core.apush_rules[gt_p] = utils.asyncify(pushforward_gt)
core.pull_fwd_rules[gt_p] = pullback_fwd_gt
core.apull_fwd_rules[gt_p] = utils.asyncify(pullback_fwd_gt)
core.pull_bwd_rules[gt_p] = pullback_bwd_compare
core.apull_bwd_rules[gt_p] = utils.asyncify(pullback_bwd_compare)
core.batch_rules[gt_p] = batch_gt
core.abatch_rules[gt_p] = utils.asyncify(batch_gt)


# ==================================================================================================
# GE
# ==================================================================================================

ge_p = core.Prim("ge")


def ge(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return ge_p.bind((a, b))


def impl_ge(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a >= b


pushforward_ge = ft.partial(pushforward_compare, ge_p)
pullback_fwd_ge = ft.partial(pullback_fwd_compare, ge_p)
batch_ge = ft.partial(batch_compare, ge_p)
core.impl_rules[ge_p] = impl_ge
core.aimpl_rules[ge_p] = utils.asyncify(impl_ge)
core.abstract_rules[ge_p] = abstract_compare
core.push_rules[ge_p] = pushforward_ge
core.apush_rules[ge_p] = utils.asyncify(pushforward_ge)
core.pull_fwd_rules[ge_p] = pullback_fwd_ge
core.apull_fwd_rules[ge_p] = utils.asyncify(pullback_fwd_ge)
core.pull_bwd_rules[ge_p] = pullback_bwd_compare
core.apull_bwd_rules[ge_p] = utils.asyncify(pullback_bwd_compare)
core.batch_rules[ge_p] = batch_ge
core.abatch_rules[ge_p] = utils.asyncify(batch_ge)


tracer.dunder_rules[tracer.Dunder.NEG, FloatAVal] = neg
tracer.dunder_rules[tracer.Dunder.ADD, FloatAVal] = add
tracer.dunder_rules[tracer.Dunder.SUB, FloatAVal] = sub
tracer.dunder_rules[tracer.Dunder.MUL, FloatAVal] = mul
tracer.dunder_rules[tracer.Dunder.DIV, FloatAVal] = div
tracer.dunder_rules[tracer.Dunder.EQ, FloatAVal] = eq
tracer.dunder_rules[tracer.Dunder.NE, FloatAVal] = ne
tracer.dunder_rules[tracer.Dunder.LT, FloatAVal] = lt
tracer.dunder_rules[tracer.Dunder.LE, FloatAVal] = le
tracer.dunder_rules[tracer.Dunder.GT, FloatAVal] = gt
tracer.dunder_rules[tracer.Dunder.GE, FloatAVal] = ge
