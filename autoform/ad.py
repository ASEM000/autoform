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

"""Automatic differentiation over text"""

from __future__ import annotations

import asyncio
import functools as ft
from collections import defaultdict
from collections.abc import Callable
from typing import Any

__all__ = [
    "Zero",
    "is_zero",
    "zero_registry",
    "materialize",
    "accumulate_cotangents",
    "cotangent_accumulators",
    "pushforward",
    "pullback",
]

from autoform.core import (
    IR,
    BoxedInterpreter,
    IREqn,
    IRVar,
    Prim,
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
from autoform.dce import dce, dce_rules, default_dce
from autoform.utils import Tree, batch_index, batch_spec, batch_transpose, lru_cache, treelib

# ==================================================================================================
# ZERO
# ==================================================================================================


class Zero:
    __slots__ = ["type"]

    def __init__(self, type: type, /):
        self.type = type

    def __repr__(self):
        return f"Zero({self.type.__name__})"

    def __eq__(self, other):
        return isinstance(other, Zero) and self.type == other.type

    def __hash__(self):
        return hash(("Zero", self.type))


def is_zero(x) -> bool:
    return isinstance(x, Zero)


def zero_aval(aval, /) -> Zero:
    assert isinstance(aval, TypedAVal), f"Expected TypedAVal, got {aval!r}"
    return Zero(aval.type)


zero_registry: dict[type, Any] = {}
zero_registry[str] = ""
zero = lambda v: v if is_zero(v) else Zero(type(v))


def materialize(x: Tree, /) -> Tree:
    """Replace each Zero leaf in a pytree with its concrete zero value.

    Raises:
        TypeError: If a ``Zero`` has a type with no registered concrete
            zero (e.g. ``Zero(bool)``). This indicates an invalid gradient
            path through a non-differentiable type.
    """

    def map_func(x):
        if not is_zero(x):
            return x
        if x.type not in zero_registry:
            raise TypeError(f"Cannot materialize Zero({x.type.__name__})")
        return zero_registry[x.type]

    return treelib.map(map_func, x, is_leaf=is_zero)


def all_zero(x: Tree, /) -> bool:
    return all(is_zero(leaf) for leaf in treelib.leaves(x, is_leaf=is_zero))


# ==================================================================================================
# PUSHFORWARD
# ==================================================================================================

pushforward_call_p = Prim("pushforward_call")


class PushforwardBox:
    __slots__ = ["owner", "primal", "tangent"]

    def __init__(self, owner, primal, tangent):
        self.owner = owner
        self.primal = primal
        self.tangent = tangent


class PushforwardInterpreter(BoxedInterpreter[PushforwardBox]):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        p, t = value
        return treelib.map(lambda p, t: PushforwardBox(self, p, t), p, t)

    def unbox(self, values: Tree, /) -> tuple[Tree, Tree]:
        # NOTE(asem): pushforward is structural, so this is not fixing a current
        # perturbation-confusion bug. Ownership only keeps values from other
        # interpreter instances opaque to this one.

        def primal(v):
            return v.primal if isinstance(v, PushforwardBox) and v.owner is self else v

        def tangent(v):
            return v.tangent if isinstance(v, PushforwardBox) and v.owner is self else zero(v)

        return treelib.map(primal, values), treelib.map(tangent, values)

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        p_in, t_in = self.unbox(in_tree)
        with using_interpreter(self.parent):
            p_out, t_out = push_rules.get(prim)((p_in, t_in), **params)
        return self.box((p_out, t_out))

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        p_in, t_in = self.unbox(in_tree)
        with using_interpreter(self.parent):
            p_out, t_out = await push_rules.aget(prim)((p_in, t_in), **params)
        return self.box((p_out, t_out))


@ft.partial(lru_cache, maxsize=256)
def pushforward(ir: IR, /) -> IR:
    """Transform an IR to compute primals and tangents (forward-mode AD).

    Creates a new IR that propagates tangent (perturbation) alongside
    primal values.

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(p_in, t_in) -> (p_out, t_out)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.trace(program)("a", "b")
        >>> pf_ir = af.pushforward(ir)
        >>> p_out, t_out = pf_ir.call(("Hello", " World"), ("dx", "dy"))
        >>> p_out
        'Hello World'
        >>> t_out
        'dxdy'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        return IRVar.fresh(aval=ir_aval(atom), source=atom) if is_irvar(atom) else atom

    def make_t(atom):
        return IRVar.fresh(aval=ir_aval(atom), source=atom) if is_irvar(atom) else Zero(type(atom))

    p_in_ir = treelib.map(make_p, ir.in_ir_tree)
    t_in_ir = treelib.map(make_t, ir.in_ir_tree)
    in_ir_tree = (p_in_ir, t_in_ir)
    p_out_ir = treelib.map(make_p, ir.out_ir_tree)
    t_out_ir = treelib.map(make_t, ir.out_ir_tree)
    out_ir_tree = (p_out_ir, t_out_ir)
    ir_eqn = IREqn(pushforward_call_p, in_ir_tree, out_ir_tree, dict(ir=ir))
    return IR([ir_eqn], in_ir_tree, out_ir_tree)


def impl_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    with using_interpreter(PushforwardInterpreter(parent=active_interpreter.get())) as pusher:

        def custom_bind(ir_eqn: IREqn, boxed_in: Tree, /) -> Tree:
            p_in, t_in = pusher.unbox(boxed_in)
            if not all_zero(t_in):
                return ir_eqn.bind(boxed_in, **ir_eqn.params)
            with using_interpreter(pusher.parent):
                p_out = ir_eqn.bind(p_in, **ir_eqn.params)
            return pusher.box((p_out, treelib.map(zero, p_out)))

        ir_eqn, boxed_in = next(gen := ir.walk(*pusher.box(in_tree)))
        while ir_eqn:
            ir_eqn, boxed_in = gen.send(custom_bind(ir_eqn, boxed_in))
        return pusher.unbox(boxed_in)


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    with using_interpreter(PushforwardInterpreter(parent=active_interpreter.get())) as pusher:

        async def custom_abind(ir_eqn: IREqn, boxed_in: Tree, /) -> Tree:
            p_in, t_in = pusher.unbox(boxed_in)
            if not all_zero(t_in):
                return await ir_eqn.abind(boxed_in, **ir_eqn.params)
            with using_interpreter(pusher.parent):
                p_out = await ir_eqn.abind(p_in, **ir_eqn.params)
            return pusher.box((p_out, treelib.map(zero, p_out)))

        ir_eqn, boxed_in = next(gen := ir.walk(*pusher.box(in_tree)))
        while ir_eqn:
            ir_eqn, boxed_in = gen.send(await custom_abind(ir_eqn, boxed_in))
        return pusher.unbox(boxed_in)


def abstract_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    out = treelib.map(ir_aval, ir.out_ir_tree)
    return out, out


def pushforward_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    p, t = in_tree
    (p_in, t_in), (t_p_in, t_t_in) = p, t
    pf_ir = pushforward(ir)
    p_out = pf_ir.call(p_in, t_in)
    t_out = pf_ir.call(t_p_in, t_t_in)
    return p_out, t_out


async def apushforward_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    p, t = in_tree
    (p_in, t_in), (t_p_in, t_t_in) = p, t
    pf_ir = pushforward(ir)
    p_out = await pf_ir.acall(p_in, t_in)
    t_out = await pf_ir.acall(t_p_in, t_t_in)
    return p_out, t_out


def pullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = pf_ir.call(p_in, t_in)
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


async def apullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = await pf_ir.acall(p_in, t_in)
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


def pullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, c_out = in_tree
    p_in, _ = residuals
    c_p_out, c_t_out = c_out
    pb_ir = pullback(ir)
    _, c_p_in = pb_ir.call(p_in, c_p_out)
    _, c_t_in = pb_ir.call(p_in, c_t_out)
    return (c_p_in, c_t_in)


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, c_out = in_tree
    p_in, _ = residuals
    c_p_out, c_t_out = c_out
    pb_ir = pullback(ir)
    _, c_p_in = await pb_ir.acall(p_in, c_p_out)
    _, c_t_in = await pb_ir.acall(p_in, c_t_out)
    return (c_p_in, c_t_in)


def batch_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if batch_spec(in_values, in_batched) is None:
        pf_ir = pushforward(ir)
        result = pf_ir.call(*in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = [pf_ir.call(unbatch_p(b), unbatch_t(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, pf_ir.out_ir_tree)
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    bs, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if batch_spec(in_values, in_batched) is None:
        pf_ir = pushforward(ir)
        result = await pf_ir.acall(*in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = await asyncio.gather(*[pf_ir.acall(unbatch_p(b), unbatch_t(b)) for b in range(bs)])
    out_batched = treelib.map(lambda _: True, pf_ir.out_ir_tree)
    out_ib = batch_transpose(bs, out_batched, list(out_bi))
    return out_ib, out_batched


impl_rules.set(pushforward_call_p, impl_pushforward_call)
impl_rules.aset(pushforward_call_p, aimpl_pushforward_call)
abstract_rules.set(pushforward_call_p, abstract_pushforward_call)
push_rules.set(pushforward_call_p, pushforward_pushforward_call)
push_rules.aset(pushforward_call_p, apushforward_pushforward_call)
pull_fwd_rules.set(pushforward_call_p, pullback_fwd_pushforward_call)
pull_fwd_rules.aset(pushforward_call_p, apullback_fwd_pushforward_call)
pull_bwd_rules.set(pushforward_call_p, pullback_bwd_pushforward_call)
pull_bwd_rules.aset(pushforward_call_p, apullback_bwd_pushforward_call)
batch_rules.set(pushforward_call_p, batch_pushforward_call)
batch_rules.aset(pushforward_call_p, abatch_pushforward_call)


def dce_pushforward_call(ir_eqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    p_used, t_used = out_used
    original_out_used = treelib.map(lambda p, t: p or t, p_used, t_used)
    new_eqn = ir_eqn.using(ir=dce(ir_eqn.params["ir"], out_used=original_out_used))
    return default_dce(new_eqn, out_used)


dce_rules[pushforward_call_p] = dce_pushforward_call


# ==================================================================================================
# PULLBACK
# ==================================================================================================

pullback_call_p = Prim("pullback_call")


cotangent_accumulators: dict[type, Callable[[list], Any]] = {}
cotangent_accumulators[str] = lambda cs: "".join(cs)


def accumulate_cotangents(cotangents: list[Any]) -> Any:
    non_zero = [c for c in cotangents if not is_zero(c)]
    if not non_zero:
        return cotangents[0]  # all zeros — return first (preserves type)
    if len(non_zero) == 1:
        return non_zero[0]
    first, *_ = non_zero
    for typ in cotangent_accumulators:
        if isinstance(first, typ):
            return cotangent_accumulators[typ](non_zero)
    return sum(non_zero[1:], non_zero[0])


class PullbackFwdBox:
    __slots__ = ["owner", "primal"]

    def __init__(self, owner, primal):
        self.owner = owner
        self.primal = primal


class PullbackFwdInterpreter(BoxedInterpreter[PullbackFwdBox]):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        return treelib.map(lambda p: PullbackFwdBox(self, p), value)

    def unbox(self, values: Tree, /) -> Tree:
        def primal(v):
            return v.primal if isinstance(v, PullbackFwdBox) and v.owner is self else v

        return treelib.map(primal, values)

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        p_in = self.unbox(in_tree)
        with using_interpreter(self.parent):
            p_out, residuals = pull_fwd_rules.get(prim)(p_in, **params)
        return self.box(p_out), residuals

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        p_in = self.unbox(in_tree)
        with using_interpreter(self.parent):
            p_out, residuals = await pull_fwd_rules.aget(prim)(p_in, **params)
        return self.box(p_out), residuals


class PullbackBwdBox:
    __slots__ = ["owner", "cotangent"]

    def __init__(self, owner, cotangent):
        self.owner = owner
        self.cotangent = cotangent


def transpose_walk(ir: IR, c_out: Tree, /):
    c_env: defaultdict[IRVar, list[Any]] = defaultdict(list)

    def write_c(atom, value: Any):
        is_irvar(atom) and c_env[atom].append(value)

    def read_c(atom) -> Any:
        if not is_irvar(atom):
            return Zero(type(atom))
        if not (cs := c_env[atom]):
            return zero_aval(atom.aval)
        return accumulate_cotangents(cs)

    treelib.map(write_c, ir.out_ir_tree, c_out)
    for ir_eqn in reversed(ir.ir_eqns):
        c_out = treelib.map(read_c, ir_eqn.out_ir_tree)
        c_in = yield ir_eqn, c_out
        treelib.map(write_c, ir_eqn.in_ir_tree, c_in)
    yield None, treelib.map(read_c, ir.in_ir_tree)


class PullbackBwdInterpreter(BoxedInterpreter[PullbackBwdBox]):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        return treelib.map(lambda c: PullbackBwdBox(self, c), value)

    def unbox(self, values: Tree, /) -> Tree:
        def cotangent(v):
            return v.cotangent if isinstance(v, PullbackBwdBox) and v.owner is self else v

        return treelib.map(cotangent, values)

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        residuals, c_out = in_tree
        c_out = self.unbox(c_out)
        with using_interpreter(self.parent):
            c_in = pull_bwd_rules.get(prim)((residuals, c_out), **params)
        return self.box(c_in)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        residuals, c_out = in_tree
        c_out = self.unbox(c_out)
        with using_interpreter(self.parent):
            c_in = await pull_bwd_rules.aget(prim)((residuals, c_out), **params)
        return self.box(c_in)


@ft.partial(lru_cache, maxsize=256)
def pullback(ir: IR, /) -> IR:
    """Transform an IR to compute outputs and input cotangents (reverse-mode AD).

    Creates a new IR that computes gradients by backpropagating cotangent
    (adjoint).

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(inputs, output_cotangents) -> (outputs, input_cotangents)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.trace(program)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> outputs, cotangents = pb_ir.call(("Hello", " World"), "feedback")
        >>> outputs
        'Hello World'
        >>> cotangents  # Gradient flows back to both inputs
        ('feedback', 'feedback')
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        return IRVar.fresh(aval=ir_aval(atom), source=atom) if is_irvar(atom) else atom

    def make_c(atom):
        return IRVar.fresh(aval=ir_aval(atom), source=atom) if is_irvar(atom) else Zero(type(atom))

    p_in_ir = treelib.map(make_p, ir.in_ir_tree)
    c_out_ir = treelib.map(make_c, ir.out_ir_tree)
    in_ir_tree = (p_in_ir, c_out_ir)
    p_out_ir = treelib.map(make_p, ir.out_ir_tree)
    c_in_ir = treelib.map(make_c, ir.in_ir_tree)
    out_ir_tree = (p_out_ir, c_in_ir)
    ir_eqn = IREqn(pullback_call_p, in_ir_tree, out_ir_tree, dict(ir=ir))
    return IR([ir_eqn], in_ir_tree, out_ir_tree)


def impl_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree

    res: dict[IREqn, Tree] = {}
    parent = active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    with using_interpreter(fwd):

        def custom_bind(ir_eqn: IREqn, boxed_in: Tree, /) -> Tree:
            boxed_out, residuals = ir_eqn.bind(boxed_in, **ir_eqn.params)
            res[ir_eqn] = residuals
            return boxed_out

        ir_eqn, boxed_in = next(gen := ir.walk(*fwd.box(p_in)))
        while ir_eqn:
            ir_eqn, boxed_in = gen.send(custom_bind(ir_eqn, boxed_in))

    with using_interpreter(bwd):

        def custom_bind(ir_eqn: IREqn, c_out: Tree, /) -> Tree:
            residuals = res[ir_eqn]
            boxed_c_out = bwd.box(c_out)
            boxed_c_in = ir_eqn.bind((residuals, boxed_c_out), **ir_eqn.params)
            return bwd.unbox(boxed_c_in)

        ir_eqn, c_out = next(gen := transpose_walk(ir, c_out))
        while ir_eqn:
            ir_eqn, c_out = gen.send(custom_bind(ir_eqn, c_out))

    return fwd.unbox(boxed_in), c_out


async def aimpl_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree

    res: dict[IREqn, Tree] = {}
    parent = active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    with using_interpreter(fwd):

        async def custom_abind(ir_eqn: IREqn, boxed_in: Tree, /) -> Tree:
            boxed_out, residuals = await ir_eqn.abind(boxed_in, **ir_eqn.params)
            res[ir_eqn] = residuals
            return boxed_out

        ir_eqn, boxed_in = next(gen := ir.walk(*fwd.box(p_in)))
        while ir_eqn:
            ir_eqn, boxed_in = gen.send(await custom_abind(ir_eqn, boxed_in))

    with using_interpreter(bwd):

        async def custom_abind(ir_eqn: IREqn, c_out: Tree, /) -> Tree:
            residuals = res[ir_eqn]
            boxed_c_out = bwd.box(c_out)
            boxed_c_in = await ir_eqn.abind((residuals, boxed_c_out), **ir_eqn.params)
            return bwd.unbox(boxed_c_in)

        ir_eqn, c_out = next(gen := transpose_walk(ir, c_out))
        while ir_eqn:
            ir_eqn, c_out = gen.send(await custom_abind(ir_eqn, c_out))

    return fwd.unbox(boxed_in), c_out


def abstract_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    p_out = treelib.map(ir_aval, ir.out_ir_tree)
    c_in = treelib.map(ir_aval, ir.in_ir_tree)
    return p_out, c_in


def pushforward_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    p, t = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = p, t
    pb_ir = pullback(ir)
    p_out, c_in = pb_ir.call(p_in, c_out)
    t_p_out, t_c_in = pb_ir.call(t_p_in, t_c_out)
    return (p_out, c_in), (t_p_out, t_c_in)


async def apushforward_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    p, t = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = p, t
    pb_ir = pullback(ir)
    p_out, c_in = await pb_ir.acall(p_in, c_out)
    t_p_out, t_c_in = await pb_ir.acall(t_p_in, t_c_out)
    return (p_out, c_in), (t_p_out, t_c_in)


def pullback_fwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    p_out, c_in = pb_ir.call(p_in, c_out)
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


async def apullback_fwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    p_out, c_in = await pb_ir.acall(p_in, c_out)
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


def pullback_bwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, c = in_tree
    p_in, _, _, _ = residuals
    c_p_out, c_c_in = c
    pb_ir = pullback(ir)
    _, c_p_in = pb_ir.call(p_in, c_p_out)
    pf_ir = pushforward(ir)
    _, c_c_out = pf_ir.call(p_in, c_c_in)
    return (c_p_in, c_c_out)


async def apullback_bwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, c = in_tree
    p_in, _, _, _ = residuals
    c_p_out, c_c_in = c
    pb_ir = pullback(ir)
    _, c_p_in = await pb_ir.acall(p_in, c_p_out)
    pf_ir = pushforward(ir)
    _, c_c_out = await pf_ir.acall(p_in, c_c_in)
    return (c_p_in, c_c_out)


def batch_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    size, in_batched, in_values = in_tree
    (p_cols, c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = pb_ir.call(*in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(batch_index, c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = [pb_ir.call(unbatch_p(b), unbatch_c(b)) for b in range(size)]
    out_batched = treelib.map(lambda _: True, pb_ir.out_ir_tree)
    out_ib = batch_transpose(size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    size, in_batched, in_values = in_tree
    (p_cols, c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = await pb_ir.acall(*in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(batch_index, c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = await asyncio.gather(*[pb_ir.acall(unbatch_p(b), unbatch_c(b)) for b in range(size)])
    out_batched = treelib.map(lambda _: True, pb_ir.out_ir_tree)
    out_ib = batch_transpose(size, out_batched, list(out_bi))
    return out_ib, out_batched


impl_rules.set(pullback_call_p, impl_pullback_call)
impl_rules.aset(pullback_call_p, aimpl_pullback_call)
abstract_rules.set(pullback_call_p, abstract_pullback_call)
push_rules.set(pullback_call_p, pushforward_pullback_call)
push_rules.aset(pullback_call_p, apushforward_pullback_call)
pull_fwd_rules.set(pullback_call_p, pullback_fwd_pullback_call)
pull_fwd_rules.aset(pullback_call_p, apullback_fwd_pullback_call)
pull_bwd_rules.set(pullback_call_p, pullback_bwd_pullback_call)
pull_bwd_rules.aset(pullback_call_p, apullback_bwd_pullback_call)
batch_rules.set(pullback_call_p, batch_pullback_call)
batch_rules.aset(pullback_call_p, abatch_pullback_call)


def dce_pullback_call(ir_eqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    out, in_cot = out_used
    used = treelib.any(in_cot)
    inner_ir = ir_eqn.params["ir"] if used else dce(ir_eqn.params["ir"], out_used=out)
    new_eqn = ir_eqn.using(ir=inner_ir)
    return default_dce(new_eqn, out_used)


dce_rules[pullback_call_p] = dce_pullback_call
