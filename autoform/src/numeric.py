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

import autoform.src.ad as ad
import autoform.src.core as core
import autoform.src.pp as pp
import autoform.src.stage as stage
import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# TYPES
# ==================================================================================================


class AInt(core.AVal):
    """Abstract value for ``int`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1)
        >>> (x,) = ir.in_tree
        >>> x.aval
        AInt()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def join(self, other, /):
        match other:
            case bool():
                raise TypeError(f"Cannot join {self!r} and {other!r}")
            case int() | AInt() | core.Zero(aval=AInt()):
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")


stage.register_trace_type(int, lambda value: AInt())
stage.register_trace_type(AInt, lambda aval: aval)
core.s_primal.set(AInt, lambda aval: aval)


class AFloat(core.AVal):
    """Abstract value for ``float`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(1.0)
        >>> (x,) = ir.in_tree
        >>> x.aval
        AFloat()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def zero(self) -> float:
        return 0.0

    def add(self, left, right, /):
        return left + right

    def join(self, other, /):
        match other:
            case float() | AFloat() | core.Zero(aval=AFloat()):
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")


stage.register_trace_type(float, lambda value: AFloat())
stage.register_trace_type(AFloat, lambda aval: aval)
core.s_primal.set(AFloat, lambda aval: aval)
core.s_tangent.set(AFloat, lambda aval: aval)
core.s_cotangent.set(AFloat, lambda aval: aval)


class ABool(core.AVal):
    """Abstract value for ``bool`` leaves.

    Example:
        >>> import autoform as af
        >>> ir = af.trace(lambda x: x)(True)
        >>> (x,) = ir.in_tree
        >>> x.aval
        ABool()
    """

    __slots__ = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def join(self, other, /):
        match other:
            case bool() | ABool() | core.Zero(aval=ABool()):
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")


@ft.partial(pp.register, AInt)
@ft.partial(pp.register, AFloat)
@ft.partial(pp.register, ABool)
def pretty_scalar(aval):
    return pp.style(type(aval).__name__, pp.Code.BLUE), "()"


stage.register_trace_type(bool, lambda value: ABool())
stage.register_trace_type(ABool, lambda aval: aval)
core.s_primal.set(ABool, lambda aval: aval)
core.s_tangent.set(ABool, lambda aval: aval)
core.s_cotangent.set(ABool, lambda aval: aval)


def abstract_float(values: Tree, /) -> AFloat:
    return AFloat()


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


@utils.dispatch
def abstract_neg(value: Tree, /) -> AFloat:
    assert False, f"Expected float: {value!r}"


abstract_neg.register(float, abstract_float)
abstract_neg.register(AFloat, abstract_float)


def pushforward_neg(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    return neg(p), neg(t)


def pullback_fwd_neg(in_tree: Tree, /) -> TreePair:
    return neg(in_tree), None


def pullback_bwd_neg(in_tree: Tree, /) -> Tree:
    _, out_c = in_tree
    return neg(out_c)


def batch_neg(in_tree: Tree, /) -> TreePair:
    return batch_unary(p_neg, in_tree)


core.r_impl.set(p_neg, impl_neg)
core.r_aimpl.set(p_neg, utils.asyncify(impl_neg))
core.r_abstract.set(p_neg, abstract_neg)
core.r_batch.set(p_neg, batch_neg)
core.r_abatch.set(p_neg, utils.asyncify(batch_neg))
core.r_push.set(p_neg, ad.concrete_zeros(pushforward_neg))
core.r_apush.set(p_neg, utils.asyncify(ad.concrete_zeros(pushforward_neg)))
core.r_pull_fwd.set(p_neg, pullback_fwd_neg)
core.r_apull_fwd.set(p_neg, utils.asyncify(pullback_fwd_neg))
core.r_pull_bwd.set(p_neg, pullback_bwd_neg)
core.r_apull_bwd.set(p_neg, utils.asyncify(pullback_bwd_neg))


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


@utils.dispatch
def abstract_add(values: Tree, /) -> AFloat:
    assert False, f"Expected floats: {values!r}"


abstract_add.register((float, float), abstract_float)
abstract_add.register((float, AFloat), abstract_float)
abstract_add.register((AFloat, float), abstract_float)
abstract_add.register((AFloat, AFloat), abstract_float)


def pushforward_add(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    return p_add.bind(p), p_add.bind(t)


def pullback_fwd_add(in_tree: Tree, /) -> TreePair:
    return p_add.bind(in_tree), None


def pullback_bwd_add(in_tree: Tree, /) -> Tree:
    _, out_c = in_tree
    return out_c, out_c


def batch_add(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_add, in_tree)


core.r_impl.set(p_add, impl_add)
core.r_aimpl.set(p_add, utils.asyncify(impl_add))
core.r_abstract.set(p_add, abstract_add)
core.r_batch.set(p_add, batch_add)
core.r_abatch.set(p_add, utils.asyncify(batch_add))
core.r_push.set(p_add, ad.concrete_zeros(pushforward_add))
core.r_apush.set(p_add, utils.asyncify(ad.concrete_zeros(pushforward_add)))
core.r_pull_fwd.set(p_add, pullback_fwd_add)
core.r_apull_fwd.set(p_add, utils.asyncify(pullback_fwd_add))
core.r_pull_bwd.set(p_add, pullback_bwd_add)
core.r_apull_bwd.set(p_add, utils.asyncify(pullback_bwd_add))


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


@utils.dispatch
def abstract_sub(values: Tree, /) -> AFloat:
    assert False, f"Expected floats: {values!r}"


abstract_sub.register((float, float), abstract_float)
abstract_sub.register((float, AFloat), abstract_float)
abstract_sub.register((AFloat, float), abstract_float)
abstract_sub.register((AFloat, AFloat), abstract_float)


def pushforward_sub(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    return p_sub.bind(p), p_sub.bind(t)


def pullback_fwd_sub(in_tree: Tree, /) -> TreePair:
    return p_sub.bind(in_tree), None


def pullback_bwd_sub(in_tree: Tree, /) -> Tree:
    _, out_c = in_tree
    return out_c, neg(out_c)


def batch_sub(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_sub, in_tree)


core.r_impl.set(p_sub, impl_sub)
core.r_aimpl.set(p_sub, utils.asyncify(impl_sub))
core.r_abstract.set(p_sub, abstract_sub)
core.r_batch.set(p_sub, batch_sub)
core.r_abatch.set(p_sub, utils.asyncify(batch_sub))
core.r_push.set(p_sub, ad.concrete_zeros(pushforward_sub))
core.r_apush.set(p_sub, utils.asyncify(ad.concrete_zeros(pushforward_sub)))
core.r_pull_fwd.set(p_sub, pullback_fwd_sub)
core.r_apull_fwd.set(p_sub, utils.asyncify(pullback_fwd_sub))
core.r_pull_bwd.set(p_sub, pullback_bwd_sub)
core.r_apull_bwd.set(p_sub, utils.asyncify(pullback_bwd_sub))


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


@utils.dispatch
def abstract_mul(values: Tree, /) -> AFloat:
    assert False, f"Expected floats: {values!r}"


abstract_mul.register((float, float), abstract_float)
abstract_mul.register((float, AFloat), abstract_float)
abstract_mul.register((AFloat, float), abstract_float)
abstract_mul.register((AFloat, AFloat), abstract_float)


def pushforward_mul(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    a, b = p
    da, db = t
    return mul(a, b), add(mul(da, b), mul(a, db))


def pullback_fwd_mul(in_tree: Tree, /) -> TreePair:
    return p_mul.bind(in_tree), in_tree


def pullback_bwd_mul(in_tree: Tree, /) -> Tree:
    (a, b), out_c = in_tree
    return mul(out_c, b), mul(out_c, a)


def batch_mul(in_tree: Tree, /) -> TreePair:
    return batch_binary(p_mul, in_tree)


core.r_impl.set(p_mul, impl_mul)
core.r_aimpl.set(p_mul, utils.asyncify(impl_mul))
core.r_abstract.set(p_mul, abstract_mul)
core.r_batch.set(p_mul, batch_mul)
core.r_abatch.set(p_mul, utils.asyncify(batch_mul))
core.r_push.set(p_mul, ad.concrete_zeros(pushforward_mul))
core.r_apush.set(p_mul, utils.asyncify(ad.concrete_zeros(pushforward_mul)))
core.r_pull_fwd.set(p_mul, pullback_fwd_mul)
core.r_apull_fwd.set(p_mul, utils.asyncify(pullback_fwd_mul))
core.r_pull_bwd.set(p_mul, pullback_bwd_mul)
core.r_apull_bwd.set(p_mul, utils.asyncify(pullback_bwd_mul))


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


@utils.dispatch
def abstract_div(values: Tree, /) -> AFloat:
    assert False, f"Expected floats: {values!r}"


abstract_div.register((float, float), abstract_float)
abstract_div.register((float, AFloat), abstract_float)
abstract_div.register((AFloat, float), abstract_float)
abstract_div.register((AFloat, AFloat), abstract_float)


def pushforward_div(in_tree: Tree, /) -> TreePair:
    p, t = in_tree
    a, b = p
    da, db = t
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


core.r_impl.set(p_div, impl_div)
core.r_aimpl.set(p_div, utils.asyncify(impl_div))
core.r_abstract.set(p_div, abstract_div)
core.r_batch.set(p_div, batch_div)
core.r_abatch.set(p_div, utils.asyncify(batch_div))
core.r_push.set(p_div, ad.concrete_zeros(pushforward_div))
core.r_apush.set(p_div, utils.asyncify(ad.concrete_zeros(pushforward_div)))
core.r_pull_fwd.set(p_div, pullback_fwd_div)
core.r_apull_fwd.set(p_div, utils.asyncify(pullback_fwd_div))
core.r_pull_bwd.set(p_div, pullback_bwd_div)
core.r_apull_bwd.set(p_div, utils.asyncify(pullback_bwd_div))


# ==================================================================================================
# COMPARISONS
# ==================================================================================================


def abstract_bool(values: Tree, /) -> ABool:
    return ABool()


def pushforward_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    p, _ = in_tree
    out_p = prim.bind(p)
    return out_p, core.s_tangent.zeros_like(out_p)


def pullback_fwd_compare(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    return prim.bind(in_tree), in_tree


def pullback_bwd_compare(in_tree: Tree, /) -> Tree:
    p, _ = in_tree
    return core.s_cotangent.zeros_like(p)


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


@utils.dispatch
def abstract_eq(values: Tree, /) -> ABool:
    assert False, f"Expected floats: {values!r}"


abstract_eq.register((float, float), abstract_bool)
abstract_eq.register((float, AFloat), abstract_bool)
abstract_eq.register((AFloat, float), abstract_bool)
abstract_eq.register((AFloat, AFloat), abstract_bool)


pushforward_eq = ft.partial(pushforward_compare, p_eq)
pullback_fwd_eq = ft.partial(pullback_fwd_compare, p_eq)
batch_eq = ft.partial(batch_compare, p_eq)


core.r_impl.set(p_eq, impl_eq)
core.r_aimpl.set(p_eq, utils.asyncify(impl_eq))
core.r_abstract.set(p_eq, abstract_eq)
core.r_batch.set(p_eq, batch_eq)
core.r_abatch.set(p_eq, utils.asyncify(batch_eq))
core.r_push.set(p_eq, pushforward_eq)
core.r_apush.set(p_eq, utils.asyncify(pushforward_eq))
core.r_pull_fwd.set(p_eq, pullback_fwd_eq)
core.r_apull_fwd.set(p_eq, utils.asyncify(pullback_fwd_eq))
core.r_pull_bwd.set(p_eq, pullback_bwd_compare)
core.r_apull_bwd.set(p_eq, utils.asyncify(pullback_bwd_compare))


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


@utils.dispatch
def abstract_ne(values: Tree, /) -> ABool:
    assert False, f"Expected floats: {values!r}"


abstract_ne.register((float, float), abstract_bool)
abstract_ne.register((float, AFloat), abstract_bool)
abstract_ne.register((AFloat, float), abstract_bool)
abstract_ne.register((AFloat, AFloat), abstract_bool)


pushforward_ne = ft.partial(pushforward_compare, p_ne)
pullback_fwd_ne = ft.partial(pullback_fwd_compare, p_ne)
batch_ne = ft.partial(batch_compare, p_ne)


core.r_impl.set(p_ne, impl_ne)
core.r_aimpl.set(p_ne, utils.asyncify(impl_ne))
core.r_abstract.set(p_ne, abstract_ne)
core.r_batch.set(p_ne, batch_ne)
core.r_abatch.set(p_ne, utils.asyncify(batch_ne))
core.r_push.set(p_ne, pushforward_ne)
core.r_apush.set(p_ne, utils.asyncify(pushforward_ne))
core.r_pull_fwd.set(p_ne, pullback_fwd_ne)
core.r_apull_fwd.set(p_ne, utils.asyncify(pullback_fwd_ne))
core.r_pull_bwd.set(p_ne, pullback_bwd_compare)
core.r_apull_bwd.set(p_ne, utils.asyncify(pullback_bwd_compare))


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


@utils.dispatch
def abstract_lt(values: Tree, /) -> ABool:
    assert False, f"Expected floats: {values!r}"


abstract_lt.register((float, float), abstract_bool)
abstract_lt.register((float, AFloat), abstract_bool)
abstract_lt.register((AFloat, float), abstract_bool)
abstract_lt.register((AFloat, AFloat), abstract_bool)


pushforward_lt = ft.partial(pushforward_compare, p_lt)
pullback_fwd_lt = ft.partial(pullback_fwd_compare, p_lt)
batch_lt = ft.partial(batch_compare, p_lt)


core.r_impl.set(p_lt, impl_lt)
core.r_aimpl.set(p_lt, utils.asyncify(impl_lt))
core.r_abstract.set(p_lt, abstract_lt)
core.r_batch.set(p_lt, batch_lt)
core.r_abatch.set(p_lt, utils.asyncify(batch_lt))
core.r_push.set(p_lt, pushforward_lt)
core.r_apush.set(p_lt, utils.asyncify(pushforward_lt))
core.r_pull_fwd.set(p_lt, pullback_fwd_lt)
core.r_apull_fwd.set(p_lt, utils.asyncify(pullback_fwd_lt))
core.r_pull_bwd.set(p_lt, pullback_bwd_compare)
core.r_apull_bwd.set(p_lt, utils.asyncify(pullback_bwd_compare))


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


@utils.dispatch
def abstract_le(values: Tree, /) -> ABool:
    assert False, f"Expected floats: {values!r}"


abstract_le.register((float, float), abstract_bool)
abstract_le.register((float, AFloat), abstract_bool)
abstract_le.register((AFloat, float), abstract_bool)
abstract_le.register((AFloat, AFloat), abstract_bool)


pushforward_le = ft.partial(pushforward_compare, p_le)
pullback_fwd_le = ft.partial(pullback_fwd_compare, p_le)
batch_le = ft.partial(batch_compare, p_le)


core.r_impl.set(p_le, impl_le)
core.r_aimpl.set(p_le, utils.asyncify(impl_le))
core.r_abstract.set(p_le, abstract_le)
core.r_batch.set(p_le, batch_le)
core.r_abatch.set(p_le, utils.asyncify(batch_le))
core.r_push.set(p_le, pushforward_le)
core.r_apush.set(p_le, utils.asyncify(pushforward_le))
core.r_pull_fwd.set(p_le, pullback_fwd_le)
core.r_apull_fwd.set(p_le, utils.asyncify(pullback_fwd_le))
core.r_pull_bwd.set(p_le, pullback_bwd_compare)
core.r_apull_bwd.set(p_le, utils.asyncify(pullback_bwd_compare))


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


@utils.dispatch
def abstract_gt(values: Tree, /) -> ABool:
    assert False, f"Expected floats: {values!r}"


abstract_gt.register((float, float), abstract_bool)
abstract_gt.register((float, AFloat), abstract_bool)
abstract_gt.register((AFloat, float), abstract_bool)
abstract_gt.register((AFloat, AFloat), abstract_bool)


pushforward_gt = ft.partial(pushforward_compare, p_gt)
pullback_fwd_gt = ft.partial(pullback_fwd_compare, p_gt)
batch_gt = ft.partial(batch_compare, p_gt)


core.r_impl.set(p_gt, impl_gt)
core.r_aimpl.set(p_gt, utils.asyncify(impl_gt))
core.r_abstract.set(p_gt, abstract_gt)
core.r_batch.set(p_gt, batch_gt)
core.r_abatch.set(p_gt, utils.asyncify(batch_gt))
core.r_push.set(p_gt, pushforward_gt)
core.r_apush.set(p_gt, utils.asyncify(pushforward_gt))
core.r_pull_fwd.set(p_gt, pullback_fwd_gt)
core.r_apull_fwd.set(p_gt, utils.asyncify(pullback_fwd_gt))
core.r_pull_bwd.set(p_gt, pullback_bwd_compare)
core.r_apull_bwd.set(p_gt, utils.asyncify(pullback_bwd_compare))


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


@utils.dispatch
def abstract_ge(values: Tree, /) -> ABool:
    assert False, f"Expected floats: {values!r}"


abstract_ge.register((float, float), abstract_bool)
abstract_ge.register((float, AFloat), abstract_bool)
abstract_ge.register((AFloat, float), abstract_bool)
abstract_ge.register((AFloat, AFloat), abstract_bool)


pushforward_ge = ft.partial(pushforward_compare, p_ge)
pullback_fwd_ge = ft.partial(pullback_fwd_compare, p_ge)
batch_ge = ft.partial(batch_compare, p_ge)


core.r_impl.set(p_ge, impl_ge)
core.r_aimpl.set(p_ge, utils.asyncify(impl_ge))
core.r_abstract.set(p_ge, abstract_ge)
core.r_batch.set(p_ge, batch_ge)
core.r_abatch.set(p_ge, utils.asyncify(batch_ge))
core.r_push.set(p_ge, pushforward_ge)
core.r_apush.set(p_ge, utils.asyncify(pushforward_ge))
core.r_pull_fwd.set(p_ge, pullback_fwd_ge)
core.r_apull_fwd.set(p_ge, utils.asyncify(pullback_fwd_ge))
core.r_pull_bwd.set(p_ge, pullback_bwd_compare)
core.r_apull_bwd.set(p_ge, utils.asyncify(pullback_bwd_compare))


stage.register_dunder(stage.Dunder.NEG, AFloat, neg)
stage.register_dunder(stage.Dunder.ADD, AFloat, add)
stage.register_dunder(stage.Dunder.SUB, AFloat, sub)
stage.register_dunder(stage.Dunder.MUL, AFloat, mul)
stage.register_dunder(stage.Dunder.DIV, AFloat, div)
stage.register_dunder(stage.Dunder.EQ, AFloat, eq)
stage.register_dunder(stage.Dunder.NE, AFloat, ne)
stage.register_dunder(stage.Dunder.LT, AFloat, lt)
stage.register_dunder(stage.Dunder.LE, AFloat, le)
stage.register_dunder(stage.Dunder.GT, AFloat, gt)
stage.register_dunder(stage.Dunder.GE, AFloat, ge)
