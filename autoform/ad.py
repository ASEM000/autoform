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

"""Automatic differentiation"""

from __future__ import annotations

import functools as ft
from collections import defaultdict
from typing import Any

import autoform.core as core
import autoform.dead as dead
import autoform.order as order
import autoform.stage as stage
import autoform.utils as utils

type Tree[T] = utils.Tree[T]


__all__ = ["cot_accum", "pushforward", "pullback"]

type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# PUSHFORWARD
# ==================================================================================================

pushforward_call_p = core.Prim("pushforward_call")


class PushforwardBox:
    __slots__ = ["owner", "primal", "tangent"]

    def __init__(self, owner, primal, tangent):
        self.owner = owner
        self.primal = primal
        self.tangent = tangent


core.aval_types[PushforwardBox] = lambda value: core.avalof(value.primal)


class PushforwardInterpreter(core.Interpreter):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        p, t = value
        return utils.tree.map(lambda p, t: PushforwardBox(self, p, t), p, t)

    def unbox(self, values: Tree, /) -> TreePair:
        # NOTE(asem): pushforward is structural, so this is not fixing a current
        # perturbation-confusion bug. Ownership only keeps values from other
        # interpreter instances opaque to this one.

        def primal(v):
            return v.primal if isinstance(v, PushforwardBox) and v.owner is self else v

        def tangent(v):
            if isinstance(v, PushforwardBox) and v.owner is self:
                return v.tangent
            return core.Zero(core.tangent_s.map(core.avalof(v)))

        return utils.tree.map(primal, values), utils.tree.map(tangent, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        p_in, t_in = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            p_out, t_out = core.push_rules.get(prim)((p_in, t_in), **params)
        return self.box((p_out, t_out))

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        p_in, t_in = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            p_out, t_out = await core.apush_rules.get(prim)((p_in, t_in), **params)
        return self.box((p_out, t_out))


@ft.partial(utils.lru_cache, maxsize=256)
def pushforward(ir: stage.IR, /) -> stage.IR:
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
        ...     return x + y
        >>> ir = af.trace(program)("a", "b")
        >>> pf_ir = af.pushforward(ir)
        >>> p_out, t_out = pf_ir.call(("Hello", " World"), ("dx", "dy"))
        >>> p_out
        'Hello World'
        >>> t_out
        'dxdy'
    """
    assert isinstance(ir, stage.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=stage.aval_if_var(atom), source=atom)
        return atom

    def make_t(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.tangent_s.map(atom.aval), source=atom)
        return core.Zero(core.tangent_s.map(core.avalof(atom)))

    p_in_ir = utils.tree.map(make_p, ir.in_tree)
    t_in_ir = utils.tree.map(make_t, ir.in_tree)
    in_tree = (p_in_ir, t_in_ir)
    p_out_ir = utils.tree.map(make_p, ir.out_tree)
    t_out_ir = utils.tree.map(make_t, ir.out_tree)
    out_tree = (p_out_ir, t_out_ir)
    eqn = stage.Eqn(pushforward_call_p, in_tree, out_tree, dict(ir=ir))
    return stage.IR([eqn], in_tree, out_tree)


def zero_tangent(x):
    return core.Zero(core.tangent_s.map(core.avalof(x)))


class PFEnv:
    __slots__ = ["primals", "tangents"]

    def __init__(self):
        self.primals = {}
        self.tangents = {}

    def read_p(self, atom, /):
        if not stage.is_var(atom):
            return atom
        value = self.primals[atom]
        stage.no_stage_typecheck(value, atom.aval)
        return value

    def read_t(self, atom, /):
        if not stage.is_var(atom):
            return zero_tangent(atom)
        value = self.tangents[atom]
        stage.no_stage_typecheck(value, core.tangent_s.map(atom.aval))
        return value

    def write(self, atom, primal, tangent, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(primal, atom.aval)
            stage.no_stage_typecheck(tangent, core.tangent_s.map(atom.aval))
            self.primals[atom] = primal
            self.tangents[atom] = tangent


def impl_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    env = PFEnv()

    def fwd_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        p_in, t_in = in_tree
        if all(isinstance(x, core.Zero) for x in utils.tree.leaves(t_in)):
            with core.using_interpreter(parent):
                p_out = eqn.bind(p_in, **eqn.params)
            return p_out, utils.tree.map(zero_tangent, p_out)
        with core.using_interpreter(pusher):
            boxed_out = eqn.bind(pusher.box((p_in, t_in)), **eqn.params)
        return pusher.unbox(boxed_out)

    utils.tree.map(env.write, ir.in_tree, *in_tree)
    for eqn in ir.eqns:
        p_in = utils.tree.map(env.read_p, eqn.in_tree)
        t_in = utils.tree.map(env.read_t, eqn.in_tree)
        p_out, t_out = fwd_bind(eqn, (p_in, t_in))
        utils.tree.map(env.write, eqn.out_tree, p_out, t_out)
    return utils.tree.map(env.read_p, ir.out_tree), utils.tree.map(env.read_t, ir.out_tree)


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    env = PFEnv()

    async def fwd_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        p_in, t_in = in_tree
        if all(isinstance(x, core.Zero) for x in utils.tree.leaves(t_in)):
            with core.using_interpreter(parent):
                p_out = await eqn.abind(p_in, **eqn.params)
            return p_out, utils.tree.map(zero_tangent, p_out)
        with core.using_interpreter(pusher):
            boxed_out = await eqn.abind(pusher.box((p_in, t_in)), **eqn.params)
        return pusher.unbox(boxed_out)

    utils.tree.map(env.write, ir.in_tree, *in_tree)
    for eqn in ir.eqns:
        p_in = utils.tree.map(env.read_p, eqn.in_tree)
        t_in = utils.tree.map(env.read_t, eqn.in_tree)
        p_out, t_out = await fwd_bind(eqn, (p_in, t_in))
        utils.tree.map(env.write, eqn.out_tree, p_out, t_out)
    return utils.tree.map(env.read_p, ir.out_tree), utils.tree.map(env.read_t, ir.out_tree)


def abstract_pushforward_call(_: Tree, /, *, ir: stage.IR) -> TreePair:
    def tangent_aval(atom):
        if stage.is_var(atom):
            return core.tangent_s.map(atom.aval)
        return core.Zero(core.tangent_s.map(core.avalof(atom)))

    p_out = utils.tree.map(stage.aval_if_var, ir.out_tree)
    t_out = utils.tree.map(tangent_aval, ir.out_tree)
    return p_out, t_out


def pushforward_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    with core.using_interpreter(pusher):
        return pusher.unbox(impl_pushforward_call(pusher.box(in_tree), ir=ir))


async def apushforward_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    with core.using_interpreter(pusher):
        return pusher.unbox(await aimpl_pushforward_call(pusher.box(in_tree), ir=ir))


def pullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = pf_ir.call(p_in, t_in)
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


async def apullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = await pf_ir.acall(p_in, t_in)
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


def pullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, c_out = in_tree
    p_in, t_in = residuals
    c_p_out, c_t_out = c_out
    primals, tangents = (p_in, c_t_out), (t_in, c_p_out)
    (_, c_t_in), (_, c_p_in) = pushforward_pullback_call((primals, tangents), ir=ir)
    return c_p_in, c_t_in


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, c_out = in_tree
    p_in, t_in = residuals
    c_p_out, c_t_out = c_out
    primals, tangents = (p_in, c_t_out), (t_in, c_p_out)
    (_, c_t_in), (_, c_p_in) = await apushforward_pullback_call((primals, tangents), ir=ir)
    return c_p_in, c_t_in


def batch_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    batch_size, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        pf_ir = pushforward(ir)
        result = pf_ir.call(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(utils.batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = [pf_ir.call(unbatch_p(b), unbatch_t(b)) for b in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, pf_ir.out_tree)
    out_ib = utils.batch_transpose(batch_size, out_batched, spec.unflatten(out_bi))
    return out_ib, out_batched


async def abatch_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    bs, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        pf_ir = pushforward(ir)
        result = await pf_ir.acall(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(utils.batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)

    inputs = [(unbatch_p(b), unbatch_t(b)) for b in range(bs)]
    out_bi = await order.fanout_p.abind(inputs, irs=[pf_ir] * bs)
    out_batched = utils.tree.map(lambda _: True, pf_ir.out_tree)
    out_ib = utils.batch_transpose(bs, out_batched, spec.unflatten(out_bi))
    return out_ib, out_batched


def dce_pushforward_call(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    p_used, t_used = out_used
    original_out_used = utils.tree.map(lambda p, t: p or t, p_used, t_used)
    new_eqn = eqn.using(ir=dead.dce(eqn.params["ir"], out_used=original_out_used))
    return dead.default_dce(new_eqn, out_used)


core.impl_rules.set(pushforward_call_p, impl_pushforward_call)
core.aimpl_rules.set(pushforward_call_p, aimpl_pushforward_call)
core.abstract_rules.set(pushforward_call_p, abstract_pushforward_call)
core.batch_rules.set(pushforward_call_p, batch_pushforward_call)
core.abatch_rules.set(pushforward_call_p, abatch_pushforward_call)
core.push_rules.set(pushforward_call_p, pushforward_pushforward_call)
core.apush_rules.set(pushforward_call_p, apushforward_pushforward_call)
core.pull_fwd_rules.set(pushforward_call_p, pullback_fwd_pushforward_call)
core.apull_fwd_rules.set(pushforward_call_p, apullback_fwd_pushforward_call)
core.pull_bwd_rules.set(pushforward_call_p, pullback_bwd_pushforward_call)
core.apull_bwd_rules.set(pushforward_call_p, apullback_bwd_pushforward_call)
dead.dce_rules[pushforward_call_p] = dce_pushforward_call


# ==================================================================================================
# PULLBACK
# ==================================================================================================

pullback_call_p = core.Prim("pullback_call")
cot_accum_p = core.Prim("cot_accum")


def cot_accum(cots: list[Any | core.Zero]) -> Any:
    assert cots
    non_zero = [c for c in cots if not isinstance(c, core.Zero)]
    if not non_zero:
        # NOTE(asem): all output paths into the same input are zero.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.trace(f)("...")
        # >>> z = af.core.Zero(af.core.primal_s.map(af.core.avalof("")))
        # >>> af.pullback(ir).call(("a",), (z, z))
        # (('a', 'a'), (Zero(StrAVal()),))
        first_zero, *rest_zero = cots
        assert all(core.avalof(c) == core.avalof(first_zero) for c in rest_zero)
        return first_zero
    if len(non_zero) == 1:
        # NOTE(asem): exactly one output path into the same input is live.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.trace(f)("...")
        # >>> z = af.core.Zero(af.core.primal_s.map(af.core.avalof("")))
        # >>> af.pullback(ir).call(("a",), ("df", z))
        # (('a', 'a'), ('df',))
        return non_zero[0]
    first, *_ = non_zero
    if not utils.tree.is_leaf(first):
        # NOTE(asem): non-leaf cotangents accumulate matching leaves.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.batch(af.trace(f)("..."))
        # >>> af.pullback(ir).call((["a", "b"],), (["G0", "G1"], ["H0", "H1"]))
        # ((['a', 'b'], ['a', 'b']), (['G0H0', 'G1H1'],))
        return utils.tree.map(lambda *cs: cot_accum(list(cs)), *non_zero)
    # NOTE(asem): leaf cotangents use their aval's accumulation method.
    # >>> def f(x):
    # ...     return x + x
    # >>> ir = af.trace(f)("...")
    # >>> af.pullback(ir).call(("a",), "df")
    # ('aa', ('dfdf',))
    return cot_accum_p.bind(non_zero)


def impl_cot_accum(cots: list[Any], /) -> Any:
    aval = core.avalof(cots[0])
    return aval.accum(cots)


def abstract_cot_accum(cots: list[Any], /) -> core.AVal:
    first = cots[0]
    return first if isinstance(first, core.AVal) else core.avalof(first)


def pushforward_cot_accum(in_tree: TreePair, /) -> TreePair:
    p_cots, t_cots = in_tree
    return cot_accum(p_cots), cot_accum(t_cots)


def pullback_fwd_cot_accum(cots: list[Any], /) -> TreePair:
    return cot_accum(cots), len(cots)


def pullback_bwd_cot_accum(in_tree: TreePair, /) -> list[Any]:
    num_cots, c_out = in_tree
    return [c_out] * num_cots


def batch_cot_accum(in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, cots = in_tree
    if (spec := utils.batch_spec(cots, in_batched)) is None:
        return cot_accum(cots), False
    unbatch = ft.partial(utils.batch_index, cots, in_batched)
    out_bi = [cot_accum(unbatch(i)) for i in range(batch_size)]
    return spec.unflatten(out_bi), True


core.impl_rules.set(cot_accum_p, impl_cot_accum)
core.aimpl_rules.set(cot_accum_p, utils.asyncify(impl_cot_accum))
core.abstract_rules.set(cot_accum_p, abstract_cot_accum)
core.batch_rules.set(cot_accum_p, batch_cot_accum)
core.abatch_rules.set(cot_accum_p, utils.asyncify(batch_cot_accum))
core.push_rules.set(cot_accum_p, pushforward_cot_accum)
core.apush_rules.set(cot_accum_p, utils.asyncify(pushforward_cot_accum))
core.pull_fwd_rules.set(cot_accum_p, pullback_fwd_cot_accum)
core.apull_fwd_rules.set(cot_accum_p, utils.asyncify(pullback_fwd_cot_accum))
core.pull_bwd_rules.set(cot_accum_p, pullback_bwd_cot_accum)
core.apull_bwd_rules.set(cot_accum_p, utils.asyncify(pullback_bwd_cot_accum))


class PullbackFwdBox:
    __slots__ = ["owner", "primal"]

    def __init__(self, owner, primal):
        self.owner = owner
        self.primal = primal


core.aval_types[PullbackFwdBox] = lambda value: core.avalof(value.primal)


class PullbackFwdInterpreter(core.Interpreter):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        return utils.tree.map(lambda p: PullbackFwdBox(self, p), value)

    def unbox(self, values: Tree, /) -> Tree:
        def primal(v):
            return v.primal if isinstance(v, PullbackFwdBox) and v.owner is self else v

        return utils.tree.map(primal, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        p_in = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            p_out, residuals = core.pull_fwd_rules.get(prim)(p_in, **params)
        return self.box(p_out), residuals

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        p_in = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            p_out, residuals = await core.apull_fwd_rules.get(prim)(p_in, **params)
        return self.box(p_out), residuals


class PullbackBwdBox:
    __slots__ = ["owner", "cotangent"]

    def __init__(self, owner, cotangent):
        self.owner = owner
        self.cotangent = cotangent


core.aval_types[PullbackBwdBox] = lambda value: core.avalof(value.cotangent)


class PullbackBwdInterpreter(core.Interpreter):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        return utils.tree.map(lambda c: PullbackBwdBox(self, c), value)

    def unbox(self, values: Tree, /) -> Tree:
        def cotangent(v):
            return v.cotangent if isinstance(v, PullbackBwdBox) and v.owner is self else v

        return utils.tree.map(cotangent, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        residuals, c_out = in_tree
        c_out = self.unbox(c_out)
        with core.using_interpreter(self.parent):
            c_in = core.pull_bwd_rules.get(prim)((residuals, c_out), **params)
        return self.box(c_in)

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        residuals, c_out = in_tree
        c_out = self.unbox(c_out)
        with core.using_interpreter(self.parent):
            c_in = await core.apull_bwd_rules.get(prim)((residuals, c_out), **params)
        return self.box(c_in)


@ft.partial(utils.lru_cache, maxsize=256)
def pullback(ir: stage.IR, /) -> stage.IR:
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
        ...     return x + y
        >>> ir = af.trace(program)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> outputs, cotangents = pb_ir.call(("Hello", " World"), "feedback")
        >>> outputs
        'Hello World'
        >>> cotangents  # Gradient flows back to both inputs
        ('feedback', 'feedback')
    """
    assert isinstance(ir, stage.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=stage.aval_if_var(atom), source=atom)
        return atom

    def make_c(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.cotangent_s.map(atom.aval), source=atom)
        return core.Zero(core.cotangent_s.map(core.avalof(atom)))

    p_in_ir = utils.tree.map(make_p, ir.in_tree)
    c_out_ir = utils.tree.map(make_c, ir.out_tree)
    in_tree = (p_in_ir, c_out_ir)
    p_out_ir = utils.tree.map(make_p, ir.out_tree)
    c_in_ir = utils.tree.map(make_c, ir.in_tree)
    out_tree = (p_out_ir, c_in_ir)
    eqn = stage.Eqn(pullback_call_p, in_tree, out_tree, dict(ir=ir))
    return stage.IR([eqn], in_tree, out_tree)


class PBEnv:
    __slots__ = ["primals", "cotangents"]

    def __init__(self):
        self.primals = {}
        self.cotangents = defaultdict(list)

    def read_p(self, atom, /):
        if not stage.is_var(atom):
            return atom
        value = self.primals[atom]
        stage.no_stage_typecheck(value, atom.aval)
        return value

    def write_p(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, atom.aval)
            self.primals[atom] = value

    def read_c(self, atom, /):
        aval = core.cotangent_s.map(core.avalof(atom))
        if not stage.is_var(atom) or not (values := self.cotangents[atom]):
            value = core.Zero(aval)
        else:
            for value in values:
                stage.no_stage_typecheck(value, aval)
            value = cot_accum(values)
        stage.no_stage_typecheck(value, aval)
        return value

    def write_c(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.cotangent_s.map(atom.aval))
            self.cotangents[atom].append(value)


def impl_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (p_in, c_out) = in_tree

    res: dict[stage.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    def fwd_bind(eqn: stage.Eqn, p_in: Tree, /) -> Tree:
        with core.using_interpreter(fwd):
            boxed_out, residuals = eqn.bind(fwd.box(p_in), **eqn.params)
        res[eqn] = residuals
        return fwd.unbox(boxed_out)

    def bwd_bind(eqn: stage.Eqn, c_out: Tree, /) -> Tree:
        residuals = res[eqn]
        with core.using_interpreter(bwd):
            boxed_c_in = eqn.bind((residuals, bwd.box(c_out)), **eqn.params)
        return bwd.unbox(boxed_c_in)

    env = PBEnv()

    utils.tree.map(env.write_p, ir.in_tree, p_in)
    for eqn in ir.eqns:
        p_in = utils.tree.map(env.read_p, eqn.in_tree)
        p_out = fwd_bind(eqn, p_in)
        utils.tree.map(env.write_p, eqn.out_tree, p_out)
    p_out = utils.tree.map(env.read_p, ir.out_tree)

    utils.tree.map(env.write_c, ir.out_tree, c_out)
    for eqn in reversed(ir.eqns):
        c_out = utils.tree.map(env.read_c, eqn.out_tree)
        c_in = bwd_bind(eqn, c_out)
        utils.tree.map(env.write_c, eqn.in_tree, c_in)
    return p_out, utils.tree.map(env.read_c, ir.in_tree)


async def aimpl_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (p_in, c_out) = in_tree

    res: dict[stage.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    async def fwd_bind(eqn: stage.Eqn, p_in: Tree, /) -> Tree:
        with core.using_interpreter(fwd):
            boxed_out, residuals = await eqn.abind(fwd.box(p_in), **eqn.params)
        res[eqn] = residuals
        return fwd.unbox(boxed_out)

    async def bwd_bind(eqn: stage.Eqn, c_out: Tree, /) -> Tree:
        residuals = res[eqn]
        with core.using_interpreter(bwd):
            boxed_c_in = await eqn.abind((residuals, bwd.box(c_out)), **eqn.params)
        return bwd.unbox(boxed_c_in)

    env = PBEnv()

    utils.tree.map(env.write_p, ir.in_tree, p_in)
    for eqn in ir.eqns:
        p_in = utils.tree.map(env.read_p, eqn.in_tree)
        p_out = await fwd_bind(eqn, p_in)
        utils.tree.map(env.write_p, eqn.out_tree, p_out)
    p_out = utils.tree.map(env.read_p, ir.out_tree)

    utils.tree.map(env.write_c, ir.out_tree, c_out)
    for eqn in reversed(ir.eqns):
        c_out = utils.tree.map(env.read_c, eqn.out_tree)
        c_in = await bwd_bind(eqn, c_out)
        utils.tree.map(env.write_c, eqn.in_tree, c_in)
    return p_out, utils.tree.map(env.read_c, ir.in_tree)


def abstract_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    def cotangent_aval(atom):
        if stage.is_var(atom):
            return core.cotangent_s.map(atom.aval)
        return core.Zero(core.cotangent_s.map(core.avalof(atom)))

    p_out = utils.tree.map(stage.aval_if_var, ir.out_tree)
    c_in = utils.tree.map(cotangent_aval, ir.in_tree)
    return p_out, c_in


def pushforward_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    with core.using_interpreter(pusher):
        return pusher.unbox(impl_pullback_call(pusher.box(in_tree), ir=ir))


async def apushforward_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    with core.using_interpreter(pusher):
        return pusher.unbox(await aimpl_pullback_call(pusher.box(in_tree), ir=ir))


def pullback_fwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    p_out, c_in = pb_ir.call(p_in, c_out)
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


async def apullback_fwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    p_out, c_in = await pb_ir.acall(p_in, c_out)
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


def pullback_bwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, c = in_tree
    p_in, c_out, _, _ = residuals
    c_p_out, c_c_in = c
    primals, tangents = (p_in, c_out), (c_c_in, c_p_out)
    (_, _), (c_c_out, c_p_in) = pushforward_pullback_call((primals, tangents), ir=ir)
    return c_p_in, c_c_out


async def apullback_bwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, c = in_tree
    p_in, c_out, _, _ = residuals
    c_p_out, c_c_in = c
    primals, tangents = (p_in, c_out), (c_c_in, c_p_out)
    (_, _), (c_c_out, c_p_in) = await apushforward_pullback_call((primals, tangents), ir=ir)
    return c_p_in, c_c_out


def batch_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    size, in_batched, in_values = in_tree
    (p_cols, c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        pb_ir = pullback(ir)
        result = pb_ir.call(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(utils.batch_index, c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = [pb_ir.call(unbatch_p(b), unbatch_c(b)) for b in range(size)]
    out_batched = utils.tree.map(lambda _: True, pb_ir.out_tree)
    out_ib = utils.batch_transpose(size, out_batched, spec.unflatten(out_bi))
    return out_ib, out_batched


async def abatch_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    size, in_batched, in_values = in_tree
    (p_cols, c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        pb_ir = pullback(ir)
        result = await pb_ir.acall(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(utils.batch_index, c_cols, c_batched)
    pb_ir = pullback(ir)

    inputs = [(unbatch_p(b), unbatch_c(b)) for b in range(size)]
    out_bi = await order.fanout_p.abind(inputs, irs=[pb_ir] * size)
    out_batched = utils.tree.map(lambda _: True, pb_ir.out_tree)
    out_ib = utils.batch_transpose(size, out_batched, spec.unflatten(out_bi))
    return out_ib, out_batched


def dce_pullback_call(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    _, in_cot = out_used
    used = utils.tree.any(in_cot)
    # NOTE(asem): when input cotangents are used, avoid threading the output mask
    # to DCE on the IR, to avoid pruning paths still needed for cotangents even if
    # the output is not needed. for example
    # >>> def program(x):
    # ...    a = x + "!"
    # ...    b = x + "?"
    # ...    return a, b
    # >>> pb = af.pullback(af.trace(program)("x"))
    # >>> (a, b), (dx,) = pb.call(("x",), ("g", "h"))
    # even if b is not used, dx has contribution from b cotangent (h),
    # even when dx is unused, the wrapper still runs backward and consumes (g, h).
    # keep both primal outputs so their structure matches the incoming cotangents.
    inner_ir = eqn.params["ir"] if used else dead.dce(eqn.params["ir"])
    new_eqn = eqn.using(ir=inner_ir)
    return dead.default_dce(new_eqn, out_used)


core.impl_rules.set(pullback_call_p, impl_pullback_call)
core.aimpl_rules.set(pullback_call_p, aimpl_pullback_call)
core.abstract_rules.set(pullback_call_p, abstract_pullback_call)
core.batch_rules.set(pullback_call_p, batch_pullback_call)
core.abatch_rules.set(pullback_call_p, abatch_pullback_call)
core.push_rules.set(pullback_call_p, pushforward_pullback_call)
core.apush_rules.set(pullback_call_p, apushforward_pullback_call)
core.pull_fwd_rules.set(pullback_call_p, pullback_fwd_pullback_call)
core.apull_fwd_rules.set(pullback_call_p, apullback_fwd_pullback_call)
core.pull_bwd_rules.set(pullback_call_p, pullback_bwd_pullback_call)
core.apull_bwd_rules.set(pullback_call_p, apullback_bwd_pullback_call)
dead.dce_rules[pullback_call_p] = dce_pullback_call
