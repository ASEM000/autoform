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

import autoform.src.core as core
import autoform.src.stage as stage
import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# TYPES
# ==================================================================================================


class AValInt(core.AVal):
    """Abstract value for ``int`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1)
        >>> (x,) = ir.in_tree
        >>> x.aval
        AValInt()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


core.aval_types[int] = lambda _: AValInt()
core.aval_types[AValInt] = lambda aval: aval
stage.trace_types.add(int)
core.s_primal.set(AValInt, lambda aval: aval)


class AValFloat(core.AVal):
    """Abstract value for ``float`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1.0)
        >>> (x,) = ir.in_tree
        >>> x.aval
        AValFloat()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class AlgebraFloat(core.Algebra):
    __slots__ = []

    def zero(self) -> float:
        return 0.0

    def accum(self, values: list[float], /) -> float:
        return sum(values)


core.algebra_rules[AValFloat] = lambda aval: AlgebraFloat(aval)
core.aval_types[float] = lambda _: AValFloat()
core.aval_types[AValFloat] = lambda aval: aval
stage.trace_types.add(float)
core.s_primal.set(AValFloat, lambda aval: aval)
core.s_tangent.set(AValFloat, lambda aval: aval)
core.s_cotangent.set(AValFloat, lambda aval: aval)


class AValBool(core.AVal):
    """Abstract value for ``bool`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(True)
        >>> (x,) = ir.in_tree
        >>> x.aval
        AValBool()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


core.aval_types[bool] = lambda _: AValBool()
core.aval_types[AValBool] = lambda aval: aval
stage.trace_types.add(bool)
core.s_primal.set(AValBool, lambda aval: aval)
core.s_tangent.set(AValBool, lambda aval: aval)
core.s_cotangent.set(AValBool, lambda aval: aval)


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

p_neg = core.Prim("neg")


def neg(a, /):
    if type(a) is int:
        a = float(a)
    return p_neg.bind(a)


def impl_neg(in_tree: Tree, /):
    assert type(in_tree) is float, f"Expected float: {in_tree!r}"
    return -in_tree


def abstract_neg(in_tree: Tree, /) -> AValFloat:
    assert type(in_tree) in (float, AValFloat), f"Expected float: {in_tree!r}"
    return AValFloat()


def pushforward_neg(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    t = core.algebraof(t.aval).zero() if isinstance(t, core.Zero) else t
    return neg(p), neg(t)


def pullback_fwd_neg(in_tree: Tree, /) -> TreePair:
    return neg(in_tree), None


def pullback_bwd_neg(in_tree: Tree, /) -> Tree:
    _, out_c = in_tree
    return neg(out_c)


def batch_neg(in_tree: Tree, /) -> TreePair:
    return batch_unary(p_neg, in_tree)


core.impl_rules.set(p_neg, impl_neg)
core.aimpl_rules.set(p_neg, utils.asyncify(impl_neg))
core.abstract_rules.set(p_neg, abstract_neg)
core.batch_rules.set(p_neg, batch_neg)
core.abatch_rules.set(p_neg, utils.asyncify(batch_neg))
core.push_rules.set(p_neg, pushforward_neg)
core.apush_rules.set(p_neg, utils.asyncify(pushforward_neg))
core.pull_fwd_rules.set(p_neg, pullback_fwd_neg)
core.apull_fwd_rules.set(p_neg, utils.asyncify(pullback_fwd_neg))
core.pull_bwd_rules.set(p_neg, pullback_bwd_neg)
core.apull_bwd_rules.set(p_neg, utils.asyncify(pullback_bwd_neg))


# ==================================================================================================
# ADD
# ==================================================================================================

p_add = core.Prim("add")


def add(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_add.bind((a, b))


def impl_add(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a + b


def abstract_add(in_tree: Tree, /) -> AValFloat:
    assert all(type(x) in (float, AValFloat) for x in in_tree), f"Expected floats: {in_tree!r}"
    return AValFloat()


def pushforward_add(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    zero = lambda x: core.algebraof(x.aval).zero() if isinstance(x, core.Zero) else x
    t = utils.tree.map(zero, t)
    return p_add.bind(p), p_add.bind(t)


def pullback_fwd_add(in_tree: Tree, /) -> TreePair:
    return p_add.bind(in_tree), None


def pullback_bwd_add(in_tree: Tree, /) -> Tree:
    _, out_c = in_tree
    return out_c, out_c


def batch_add(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_add, in_tree)


core.impl_rules.set(p_add, impl_add)
core.aimpl_rules.set(p_add, utils.asyncify(impl_add))
core.abstract_rules.set(p_add, abstract_add)
core.batch_rules.set(p_add, batch_add)
core.abatch_rules.set(p_add, utils.asyncify(batch_add))
core.push_rules.set(p_add, pushforward_add)
core.apush_rules.set(p_add, utils.asyncify(pushforward_add))
core.pull_fwd_rules.set(p_add, pullback_fwd_add)
core.apull_fwd_rules.set(p_add, utils.asyncify(pullback_fwd_add))
core.pull_bwd_rules.set(p_add, pullback_bwd_add)
core.apull_bwd_rules.set(p_add, utils.asyncify(pullback_bwd_add))


# ==================================================================================================
# SUB
# ==================================================================================================

p_sub = core.Prim("sub")


def sub(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_sub.bind((a, b))


def impl_sub(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a - b


def abstract_sub(in_tree: Tree, /) -> AValFloat:
    assert all(type(x) in (float, AValFloat) for x in in_tree), f"Expected floats: {in_tree!r}"
    return AValFloat()


def pushforward_sub(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    zero = lambda x: core.algebraof(x.aval).zero() if isinstance(x, core.Zero) else x
    t = utils.tree.map(zero, t)
    return p_sub.bind(p), p_sub.bind(t)


def pullback_fwd_sub(in_tree: Tree, /) -> TreePair:
    return p_sub.bind(in_tree), None


def pullback_bwd_sub(in_tree: Tree, /) -> Tree:
    _, out_c = in_tree
    return out_c, neg(out_c)


def batch_sub(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_sub, in_tree)


core.impl_rules.set(p_sub, impl_sub)
core.aimpl_rules.set(p_sub, utils.asyncify(impl_sub))
core.abstract_rules.set(p_sub, abstract_sub)
core.batch_rules.set(p_sub, batch_sub)
core.abatch_rules.set(p_sub, utils.asyncify(batch_sub))
core.push_rules.set(p_sub, pushforward_sub)
core.apush_rules.set(p_sub, utils.asyncify(pushforward_sub))
core.pull_fwd_rules.set(p_sub, pullback_fwd_sub)
core.apull_fwd_rules.set(p_sub, utils.asyncify(pullback_fwd_sub))
core.pull_bwd_rules.set(p_sub, pullback_bwd_sub)
core.apull_bwd_rules.set(p_sub, utils.asyncify(pullback_bwd_sub))


# ==================================================================================================
# MUL
# ==================================================================================================

p_mul = core.Prim("mul")


def mul(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_mul.bind((a, b))


def impl_mul(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a * b


def abstract_mul(in_tree: Tree, /) -> AValFloat:
    assert all(type(x) in (float, AValFloat) for x in in_tree), f"Expected floats: {in_tree!r}"
    return AValFloat()


def pushforward_mul(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    a, b = p
    zero = lambda x: core.algebraof(x.aval).zero() if isinstance(x, core.Zero) else x
    da, db = utils.tree.map(zero, t)
    return mul(a, b), add(mul(da, b), mul(a, db))


def pullback_fwd_mul(in_tree: Tree, /) -> TreePair:
    return p_mul.bind(in_tree), in_tree


def pullback_bwd_mul(in_tree: Tree, /) -> Tree:
    (a, b), out_c = in_tree
    return mul(out_c, b), mul(out_c, a)


def batch_mul(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_mul, in_tree)


core.impl_rules.set(p_mul, impl_mul)
core.aimpl_rules.set(p_mul, utils.asyncify(impl_mul))
core.abstract_rules.set(p_mul, abstract_mul)
core.batch_rules.set(p_mul, batch_mul)
core.abatch_rules.set(p_mul, utils.asyncify(batch_mul))
core.push_rules.set(p_mul, pushforward_mul)
core.apush_rules.set(p_mul, utils.asyncify(pushforward_mul))
core.pull_fwd_rules.set(p_mul, pullback_fwd_mul)
core.apull_fwd_rules.set(p_mul, utils.asyncify(pullback_fwd_mul))
core.pull_bwd_rules.set(p_mul, pullback_bwd_mul)
core.apull_bwd_rules.set(p_mul, utils.asyncify(pullback_bwd_mul))


# ==================================================================================================
# DIV
# ==================================================================================================

p_div = core.Prim("div")


def div(a, b, /):
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_div.bind((a, b))


def impl_div(in_tree: Tree, /):
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a / b


def abstract_div(in_tree: Tree, /) -> AValFloat:
    assert all(type(x) in (float, AValFloat) for x in in_tree), f"Expected floats: {in_tree!r}"
    return AValFloat()


def pushforward_div(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    a, b = p
    zero = lambda x: core.algebraof(x.aval).zero() if isinstance(x, core.Zero) else x
    da, db = utils.tree.map(zero, t)
    return div(a, b), div(sub(mul(da, b), mul(a, db)), mul(b, b))


def pullback_fwd_div(in_tree: Tree, /) -> TreePair:
    return p_div.bind(in_tree), in_tree


def pullback_bwd_div(in_tree: Tree, /) -> Tree:
    (a, b), out_c = in_tree
    c_a = div(out_c, b)
    c_b = neg(div(mul(out_c, a), mul(b, b)))
    return c_a, c_b


def batch_div(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_div, in_tree)


core.impl_rules.set(p_div, impl_div)
core.aimpl_rules.set(p_div, utils.asyncify(impl_div))
core.abstract_rules.set(p_div, abstract_div)
core.batch_rules.set(p_div, batch_div)
core.abatch_rules.set(p_div, utils.asyncify(batch_div))
core.push_rules.set(p_div, pushforward_div)
core.apush_rules.set(p_div, utils.asyncify(pushforward_div))
core.pull_fwd_rules.set(p_div, pullback_fwd_div)
core.apull_fwd_rules.set(p_div, utils.asyncify(pullback_fwd_div))
core.pull_bwd_rules.set(p_div, pullback_bwd_div)
core.apull_bwd_rules.set(p_div, utils.asyncify(pullback_bwd_div))


# ==================================================================================================
# COMPARISONS
# ==================================================================================================


def abstract_compare(in_tree: Tree, /) -> AValBool:
    assert all(type(x) in (float, AValFloat) for x in in_tree)
    return AValBool()


def pushforward_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    p, _ = in_tree
    return prim.bind(p), core.Zero(core.s_tangent.map(AValBool()))


def pullback_fwd_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    return prim.bind(in_tree), in_tree


def pullback_bwd_compare(in_tree: Tree, /) -> Tree:
    def make_c(x):
        if isinstance(x, core.Zero):
            return x
        return core.Zero(core.s_cotangent.map(core.avalof(x)))

    p, _ = in_tree
    return utils.tree.map(make_c, p)


def batch_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    return batch_binary(prim, in_tree)


# ==================================================================================================
# EQ
# ==================================================================================================

p_eq = core.Prim("eq")


def eq(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_eq.bind((a, b))


def impl_eq(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a == b


pushforward_eq = ft.partial(pushforward_compare, p_eq)
pullback_fwd_eq = ft.partial(pullback_fwd_compare, p_eq)
batch_eq = ft.partial(batch_compare, p_eq)


core.impl_rules.set(p_eq, impl_eq)
core.aimpl_rules.set(p_eq, utils.asyncify(impl_eq))
core.abstract_rules.set(p_eq, abstract_compare)
core.batch_rules.set(p_eq, batch_eq)
core.abatch_rules.set(p_eq, utils.asyncify(batch_eq))
core.push_rules.set(p_eq, pushforward_eq)
core.apush_rules.set(p_eq, utils.asyncify(pushforward_eq))
core.pull_fwd_rules.set(p_eq, pullback_fwd_eq)
core.apull_fwd_rules.set(p_eq, utils.asyncify(pullback_fwd_eq))
core.pull_bwd_rules.set(p_eq, pullback_bwd_compare)
core.apull_bwd_rules.set(p_eq, utils.asyncify(pullback_bwd_compare))


# ==================================================================================================
# NE
# ==================================================================================================

p_ne = core.Prim("ne")


def ne(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_ne.bind((a, b))


def impl_ne(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a != b


pushforward_ne = ft.partial(pushforward_compare, p_ne)
pullback_fwd_ne = ft.partial(pullback_fwd_compare, p_ne)
batch_ne = ft.partial(batch_compare, p_ne)


core.impl_rules.set(p_ne, impl_ne)
core.aimpl_rules.set(p_ne, utils.asyncify(impl_ne))
core.abstract_rules.set(p_ne, abstract_compare)
core.batch_rules.set(p_ne, batch_ne)
core.abatch_rules.set(p_ne, utils.asyncify(batch_ne))
core.push_rules.set(p_ne, pushforward_ne)
core.apush_rules.set(p_ne, utils.asyncify(pushforward_ne))
core.pull_fwd_rules.set(p_ne, pullback_fwd_ne)
core.apull_fwd_rules.set(p_ne, utils.asyncify(pullback_fwd_ne))
core.pull_bwd_rules.set(p_ne, pullback_bwd_compare)
core.apull_bwd_rules.set(p_ne, utils.asyncify(pullback_bwd_compare))


# ==================================================================================================
# LT
# ==================================================================================================

p_lt = core.Prim("lt")


def lt(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_lt.bind((a, b))


def impl_lt(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a < b


pushforward_lt = ft.partial(pushforward_compare, p_lt)
pullback_fwd_lt = ft.partial(pullback_fwd_compare, p_lt)
batch_lt = ft.partial(batch_compare, p_lt)


core.impl_rules.set(p_lt, impl_lt)
core.aimpl_rules.set(p_lt, utils.asyncify(impl_lt))
core.abstract_rules.set(p_lt, abstract_compare)
core.batch_rules.set(p_lt, batch_lt)
core.abatch_rules.set(p_lt, utils.asyncify(batch_lt))
core.push_rules.set(p_lt, pushforward_lt)
core.apush_rules.set(p_lt, utils.asyncify(pushforward_lt))
core.pull_fwd_rules.set(p_lt, pullback_fwd_lt)
core.apull_fwd_rules.set(p_lt, utils.asyncify(pullback_fwd_lt))
core.pull_bwd_rules.set(p_lt, pullback_bwd_compare)
core.apull_bwd_rules.set(p_lt, utils.asyncify(pullback_bwd_compare))


# ==================================================================================================
# LE
# ==================================================================================================

p_le = core.Prim("le")


def le(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_le.bind((a, b))


def impl_le(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a <= b


pushforward_le = ft.partial(pushforward_compare, p_le)
pullback_fwd_le = ft.partial(pullback_fwd_compare, p_le)
batch_le = ft.partial(batch_compare, p_le)


core.impl_rules.set(p_le, impl_le)
core.aimpl_rules.set(p_le, utils.asyncify(impl_le))
core.abstract_rules.set(p_le, abstract_compare)
core.batch_rules.set(p_le, batch_le)
core.abatch_rules.set(p_le, utils.asyncify(batch_le))
core.push_rules.set(p_le, pushforward_le)
core.apush_rules.set(p_le, utils.asyncify(pushforward_le))
core.pull_fwd_rules.set(p_le, pullback_fwd_le)
core.apull_fwd_rules.set(p_le, utils.asyncify(pullback_fwd_le))
core.pull_bwd_rules.set(p_le, pullback_bwd_compare)
core.apull_bwd_rules.set(p_le, utils.asyncify(pullback_bwd_compare))


# ==================================================================================================
# GT
# ==================================================================================================

p_gt = core.Prim("gt")


def gt(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_gt.bind((a, b))


def impl_gt(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a > b


pushforward_gt = ft.partial(pushforward_compare, p_gt)
pullback_fwd_gt = ft.partial(pullback_fwd_compare, p_gt)
batch_gt = ft.partial(batch_compare, p_gt)


core.impl_rules.set(p_gt, impl_gt)
core.aimpl_rules.set(p_gt, utils.asyncify(impl_gt))
core.abstract_rules.set(p_gt, abstract_compare)
core.batch_rules.set(p_gt, batch_gt)
core.abatch_rules.set(p_gt, utils.asyncify(batch_gt))
core.push_rules.set(p_gt, pushforward_gt)
core.apush_rules.set(p_gt, utils.asyncify(pushforward_gt))
core.pull_fwd_rules.set(p_gt, pullback_fwd_gt)
core.apull_fwd_rules.set(p_gt, utils.asyncify(pullback_fwd_gt))
core.pull_bwd_rules.set(p_gt, pullback_bwd_compare)
core.apull_bwd_rules.set(p_gt, utils.asyncify(pullback_bwd_compare))


# ==================================================================================================
# GE
# ==================================================================================================

p_ge = core.Prim("ge")


def ge(a, b, /) -> bool:
    if type(a) is int:
        a = float(a)
    if type(b) is int:
        b = float(b)
    return p_ge.bind((a, b))


def impl_ge(in_tree: Tree, /) -> bool:
    a, b = in_tree
    assert type(a) is type(b) is float, f"Expected floats: {in_tree!r}"
    return a >= b


pushforward_ge = ft.partial(pushforward_compare, p_ge)
pullback_fwd_ge = ft.partial(pullback_fwd_compare, p_ge)
batch_ge = ft.partial(batch_compare, p_ge)


core.impl_rules.set(p_ge, impl_ge)
core.aimpl_rules.set(p_ge, utils.asyncify(impl_ge))
core.abstract_rules.set(p_ge, abstract_compare)
core.batch_rules.set(p_ge, batch_ge)
core.abatch_rules.set(p_ge, utils.asyncify(batch_ge))
core.push_rules.set(p_ge, pushforward_ge)
core.apush_rules.set(p_ge, utils.asyncify(pushforward_ge))
core.pull_fwd_rules.set(p_ge, pullback_fwd_ge)
core.apull_fwd_rules.set(p_ge, utils.asyncify(pullback_fwd_ge))
core.pull_bwd_rules.set(p_ge, pullback_bwd_compare)
core.apull_bwd_rules.set(p_ge, utils.asyncify(pullback_bwd_compare))


stage.dunder_rules[stage.Dunder.NEG, AValFloat] = neg
stage.dunder_rules[stage.Dunder.ADD, AValFloat] = add
stage.dunder_rules[stage.Dunder.SUB, AValFloat] = sub
stage.dunder_rules[stage.Dunder.MUL, AValFloat] = mul
stage.dunder_rules[stage.Dunder.DIV, AValFloat] = div
stage.dunder_rules[stage.Dunder.EQ, AValFloat] = eq
stage.dunder_rules[stage.Dunder.NE, AValFloat] = ne
stage.dunder_rules[stage.Dunder.LT, AValFloat] = lt
stage.dunder_rules[stage.Dunder.LE, AValFloat] = le
stage.dunder_rules[stage.Dunder.GT, AValFloat] = gt
stage.dunder_rules[stage.Dunder.GE, AValFloat] = ge
