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
from operator import setitem
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
    is_irvar,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.dce import dce, dce_rules, default_dce
from autoform.utils import Tree, batch_index, batch_spec, batch_transpose, lru_cache, treelib


class ADTag(TransformationTag): ...


# ==================================================================================================
# ZERO
# ==================================================================================================


class Zero:
    __slots__ = "type"

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


# ==================================================================================================
# PUSHFORWARD
# ==================================================================================================

pushforward_call_p = Prim("pushforward_call", tag={ADTag})


class PushforwardInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return push_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return await push_rules.aget(prim)(in_tree, **params)


@ft.partial(lru_cache, maxsize=256)
def pushforward(ir: IR, /) -> IR:
    """Transform an IR to compute primals and tangents (forward-mode AD).

    Creates a new IR that propagates tangent (perturbation) alongside
    primal values.

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(primals, tangents) -> (out_primalputs, out_tangentputs)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.trace(program)("a", "b")
        >>> pf_ir = af.pushforward(ir)
        >>> primals, tangents = pf_ir.call(("Hello", " World"), ("dx", "dy"))
        >>> primals
        'Hello World'
        >>> tangents
        'dxdy'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom: IRVal):
        return IRVar.fresh(aval=atom.aval, source=atom) if is_irvar(atom) else atom

    def make_t(atom: IRVal):
        return (
            IRVar.fresh(aval=atom.aval, source=atom)
            if is_irvar(atom)
            else IRLit(Zero(type(atom.value)))
        )

    p_in_ir_tree = treelib.map(make_p, ir.in_ir_tree)
    t_in_ir_tree = treelib.map(make_t, ir.in_ir_tree)
    in_ir_tree = (p_in_ir_tree, t_in_ir_tree)
    out_p_ir_tree = treelib.map(make_p, ir.out_ir_tree)
    out_t_ir_tree = treelib.map(make_t, ir.out_ir_tree)
    out_ir_tree = (out_p_ir_tree, out_t_ir_tree)
    ir_eqn = IREqn(pushforward_call_p, None, in_ir_tree, out_ir_tree, dict(ir=ir))
    return IR([ir_eqn], in_ir_tree, out_ir_tree)


def impl_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, t_in_tree) = in_tree

    p_env: dict[IRVar, Any] = {}
    t_env: dict[IRVar, Any] = {}

    def write_p(atom: IRVal, value: Any):
        is_irvar(atom) and setitem(p_env, atom, value)

    def write_t(atom: IRVal, value: Any):
        is_irvar(atom) and setitem(t_env, atom, value)

    def read_p(atom: IRVal) -> Any:
        return p_env[atom] if is_irvar(atom) else atom.value

    def read_t(atom: IRVal) -> Any:
        return t_env[atom] if is_irvar(atom) else Zero(type(atom.value))

    treelib.map(write_p, ir.in_ir_tree, p_in_tree)
    treelib.map(write_t, ir.in_ir_tree, t_in_tree)

    with using_interpreter(PushforwardInterpreter()):
        for ir_eqn in ir.ir_eqns:
            p_in_ir_eqn = treelib.map(read_p, ir_eqn.in_ir_tree)
            t_in_ir_eqn = treelib.map(read_t, ir_eqn.in_ir_tree)
            in_tree = (p_in_ir_eqn, t_in_ir_eqn)
            out_p_ir_eqn, out_t_ir_eqn = ir_eqn.bind(in_tree, **ir_eqn.params)
            treelib.map(write_p, ir_eqn.out_ir_tree, out_p_ir_eqn)
            treelib.map(write_t, ir_eqn.out_ir_tree, out_t_ir_eqn)

    out_p_tree = treelib.map(read_p, ir.out_ir_tree)
    out_t_tree = treelib.map(read_t, ir.out_ir_tree)
    return out_p_tree, out_t_tree


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, t_in_tree) = in_tree

    p_env: dict[IRVar, Any] = {}
    t_env: dict[IRVar, Any] = {}

    def write_p(atom: IRVal, value: Any):
        is_irvar(atom) and setitem(p_env, atom, value)

    def write_t(atom: IRVal, value: Any):
        is_irvar(atom) and setitem(t_env, atom, value)

    def read_p(atom: IRVal) -> Any:
        return p_env[atom] if is_irvar(atom) else atom.value

    def read_t(atom: IRVal) -> Any:
        return t_env[atom] if is_irvar(atom) else Zero(type(atom.value))

    treelib.map(write_p, ir.in_ir_tree, p_in_tree)
    treelib.map(write_t, ir.in_ir_tree, t_in_tree)

    with using_interpreter(PushforwardInterpreter()):
        for ir_eqn in ir.ir_eqns:
            p_in_ir_eqn = treelib.map(read_p, ir_eqn.in_ir_tree)
            t_in_ir_eqn = treelib.map(read_t, ir_eqn.in_ir_tree)
            in_tree = (p_in_ir_eqn, t_in_ir_eqn)
            out_p_ir_eqn, out_t_ir_eqn = await ir_eqn.abind(in_tree, **ir_eqn.params)
            treelib.map(write_p, ir_eqn.out_ir_tree, out_p_ir_eqn)
            treelib.map(write_t, ir_eqn.out_ir_tree, out_t_ir_eqn)

    out_p_tree = treelib.map(read_p, ir.out_ir_tree)
    out_t_tree = treelib.map(read_t, ir.out_ir_tree)
    return out_p_tree, out_t_tree


def abstract_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    out = treelib.map(lambda x: x.aval, ir.out_ir_tree)
    return out, out


def pushforward_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, t_in), (p_in_t, t_in_t) = primals, tangents
    pf_ir = pushforward(ir)
    p_out = pf_ir.call(p_in, t_in)
    t_out = pf_ir.call(p_in_t, t_in_t)
    return p_out, t_out


async def apushforward_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, t_in), (p_in_t, t_in_t) = primals, tangents
    pf_ir = pushforward(ir)
    p_out = await pf_ir.acall(p_in, t_in)
    t_out = await pf_ir.acall(p_in_t, t_in_t)
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
    residuals, out_cotangent = in_tree
    in_p, _ = residuals
    out_c_p, out_c_t = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = pb_ir.call(in_p, out_c_p)
    _, in_c_t = pb_ir.call(in_p, out_c_t)
    return (in_c_p, in_c_t)


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    in_p, _ = residuals
    out_c_p, out_c_t = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = await pb_ir.acall(in_p, out_c_p)
    _, in_c_t = await pb_ir.acall(in_p, out_c_t)
    return (in_c_p, in_c_t)


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
    primals_used, tangents_used = out_used
    original_out_used = treelib.map(lambda p, t: p or t, primals_used, tangents_used)
    new_eqn = ir_eqn.using(ir=dce(ir_eqn.params["ir"], out_used=original_out_used))
    return default_dce(new_eqn, out_used)


dce_rules[pushforward_call_p] = dce_pushforward_call


# ==================================================================================================
# PULLBACK
# ==================================================================================================

pullback_call_p = Prim("pullback_call", tag={ADTag})


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


class PullbackFwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return pull_fwd_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return await pull_fwd_rules.aget(prim)(in_tree, **params)


class PullbackBwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return pull_bwd_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return await pull_bwd_rules.aget(prim)(in_tree, **params)


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
        return IRVar.fresh(aval=atom.aval, source=atom) if is_irvar(atom) else atom

    def make_c(atom):
        return (
            IRVar.fresh(aval=atom.aval, source=atom)
            if is_irvar(atom)
            else IRLit(Zero(type(atom.value)))
        )

    in_p = treelib.map(make_p, ir.in_ir_tree)
    out_c = treelib.map(make_c, ir.out_ir_tree)
    in_ir_tree = (in_p, out_c)
    out_p = treelib.map(make_p, ir.out_ir_tree)
    in_c = treelib.map(make_c, ir.in_ir_tree)
    out_ir_tree = (out_p, in_c)
    ir_eqn = IREqn(pullback_call_p, None, in_ir_tree, out_ir_tree, dict(ir=ir))
    return IR([ir_eqn], in_ir_tree, out_ir_tree)


def impl_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, out_c_tree) = in_tree

    p_env: dict[IRVar, Any] = {}
    res_env: dict[int, Tree] = {}
    c_env: defaultdict[IRVar, list[Any]] = defaultdict(list)

    def write_p(atom: IRVal, value: Any):
        is_irvar(atom) and setitem(p_env, atom, value)

    def read_p(atom: IRVal) -> Any:
        return p_env[atom] if is_irvar(atom) else atom.value

    def write_c(atom: IRVal, value: Any):
        is_irvar(atom) and c_env[atom].append(value)

    def read_c(atom: IRVal) -> Any:
        if not is_irvar(atom):
            return Zero(type(atom.value))
        if not (cs := c_env[atom]):
            return zero_aval(atom.aval)
        return accumulate_cotangents(cs)

    treelib.map(write_p, ir.in_ir_tree, p_in_tree)

    with using_interpreter(PullbackFwdInterpreter()):
        for i, ir_eqn in enumerate(ir.ir_eqns):
            p_in_ir_eqn = treelib.map(read_p, ir_eqn.in_ir_tree)
            out_p_ir_eqn, residuals = ir_eqn.bind(p_in_ir_eqn, **ir_eqn.params)
            res_env[i] = residuals
            treelib.map(write_p, ir_eqn.out_ir_tree, out_p_ir_eqn)

    treelib.map(write_c, ir.out_ir_tree, out_c_tree)

    with using_interpreter(PullbackBwdInterpreter()):
        for i, ir_eqn in enumerate(reversed(ir.ir_eqns)):
            idx = len(ir.ir_eqns) - 1 - i
            residuals = res_env[idx]
            out_c_ir_eqn = treelib.map(read_c, ir_eqn.out_ir_tree)
            c_in_ir_eqn = ir_eqn.bind((residuals, out_c_ir_eqn), **ir_eqn.params)
            treelib.map(write_c, ir_eqn.in_ir_tree, c_in_ir_eqn)

    out_p_tree = treelib.map(read_p, ir.out_ir_tree)
    in_c_tree = treelib.map(read_c, ir.in_ir_tree)
    return out_p_tree, in_c_tree


async def aimpl_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, out_c_tree) = in_tree

    p_env: dict[IRVar, Any] = {}
    res_env: dict[int, Tree] = {}
    c_env: defaultdict[IRVar, list[Any]] = defaultdict(list)

    def write_p(atom: IRVal, value: Any):
        is_irvar(atom) and setitem(p_env, atom, value)

    def read_p(atom: IRVal) -> Any:
        return p_env[atom] if is_irvar(atom) else atom.value

    def write_c(atom: IRVal, value: Any):
        is_irvar(atom) and c_env[atom].append(value)

    def read_c(atom: IRVal) -> Any:
        if not is_irvar(atom):
            return Zero(type(atom.value))
        if not (cs := c_env[atom]):
            return zero_aval(atom.aval)
        return accumulate_cotangents(cs)

    treelib.map(write_p, ir.in_ir_tree, p_in_tree)

    with using_interpreter(PullbackFwdInterpreter()):
        for i, ir_eqn in enumerate(ir.ir_eqns):
            p_in_ir_eqn = treelib.map(read_p, ir_eqn.in_ir_tree)
            out_p_ir_eqn, residuals = await ir_eqn.abind(p_in_ir_eqn, **ir_eqn.params)
            res_env[i] = residuals
            treelib.map(write_p, ir_eqn.out_ir_tree, out_p_ir_eqn)

    treelib.map(write_c, ir.out_ir_tree, out_c_tree)

    with using_interpreter(PullbackBwdInterpreter()):
        for i, ir_eqn in enumerate(reversed(ir.ir_eqns)):
            idx = len(ir.ir_eqns) - 1 - i
            residuals = res_env[idx]
            out_c_ir_eqn = treelib.map(read_c, ir_eqn.out_ir_tree)
            c_in_ir_eqn = await ir_eqn.abind((residuals, out_c_ir_eqn), **ir_eqn.params)
            treelib.map(write_c, ir_eqn.in_ir_tree, c_in_ir_eqn)

    out_p_tree = treelib.map(read_p, ir.out_ir_tree)
    in_c_tree = treelib.map(read_c, ir.in_ir_tree)
    return out_p_tree, in_c_tree


def abstract_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    out_p = treelib.map(lambda x: x.aval, ir.out_ir_tree)
    in_c = treelib.map(lambda x: x.aval, ir.in_ir_tree)
    return out_p, in_c


def pushforward_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = primals, tangents
    pb_ir = pullback(ir)
    out_p, in_c = pb_ir.call(p_in, c_out)
    t_out_p, t_in_c = pb_ir.call(t_p_in, t_c_out)
    return (out_p, in_c), (t_out_p, t_in_c)


async def apushforward_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = primals, tangents
    pb_ir = pullback(ir)
    out_p, in_c = await pb_ir.acall(p_in, c_out)
    t_out_p, t_in_c = await pb_ir.acall(t_p_in, t_c_out)
    return (out_p, in_c), (t_out_p, t_in_c)


def pullback_fwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = pb_ir.call(p_in, c_out)
    residuals = (p_in, c_out, out_p, in_c)
    return (out_p, in_c), residuals


async def apullback_fwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = await pb_ir.acall(p_in, c_out)
    residuals = (p_in, c_out, out_p, in_c)
    return (out_p, in_c), residuals


def pullback_bwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    p_in, c_out, _, _ = residuals
    out_c_p, in_c_c = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = pb_ir.call(p_in, out_c_p)
    pf_ir = pushforward(ir)
    _, in_c_cout = pf_ir.call(p_in, in_c_c)
    return (in_c_p, in_c_cout)


async def apullback_bwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    p_in, c_out, _, _ = residuals
    out_c_p, in_c_c = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = await pb_ir.acall(p_in, out_c_p)
    pf_ir = pushforward(ir)
    _, in_c_cout = await pf_ir.acall(p_in, in_c_c)
    return (in_c_p, in_c_cout)


def batch_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    size, in_batched, in_values = in_tree
    (p_cols, out_c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = pb_ir.call(*in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(batch_index, out_c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = [pb_ir.call(unbatch_p(b), unbatch_c(b)) for b in range(size)]
    out_batched = treelib.map(lambda _: True, pb_ir.out_ir_tree)
    out_ib = batch_transpose(size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    size, in_batched, in_values = in_tree
    (p_cols, out_c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = await pb_ir.acall(*in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(batch_index, out_c_cols, c_batched)
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
