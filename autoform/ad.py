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

        def p(v):
            return v.primal if isinstance(v, PushforwardBox) and v.owner is self else v

        def t(v):
            if isinstance(v, PushforwardBox) and v.owner is self:
                return v.tangent
            return core.Zero(core.tangent_s.map(core.avalof(v)))

        return utils.tree.map(p, values), utils.tree.map(t, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p, in_t = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            out_p, out_t = core.push_rules.get(prim)((in_p, in_t), **params)
        return self.box((out_p, out_t))

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p, in_t = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            out_p, out_t = await core.apush_rules.get(prim)((in_p, in_t), **params)
        return self.box((out_p, out_t))


@ft.partial(utils.lru_cache, maxsize=256)
def pushforward(ir: stage.IR, /) -> stage.IR:
    """Transform an IR to compute primals and tangents (forward-mode AD).

    Creates a new IR that propagates tangent (perturbation) alongside
    primal values.

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(in_p, in_t) -> (out_p, out_t)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return x + y
        >>> ir = af.trace(program)("a", "b")
        >>> pf_ir = af.pushforward(ir)
        >>> out_p, out_t = pf_ir.call(("Hello", " World"), ("dx", "dy"))
        >>> out_p
        'Hello World'
        >>> out_t
        'dxdy'
    """
    assert isinstance(ir, stage.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.primal_s.map(atom.aval), source=atom)
        return atom

    def make_t(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.tangent_s.map(atom.aval), source=atom)
        return core.Zero(core.tangent_s.map(core.avalof(atom)))

    in_p_ir = utils.tree.map(make_p, ir.in_tree)
    in_t_ir = utils.tree.map(make_t, ir.in_tree)
    in_tree = (in_p_ir, in_t_ir)
    out_p_ir = utils.tree.map(make_p, ir.out_tree)
    out_t_ir = utils.tree.map(make_t, ir.out_tree)
    out_tree = (out_p_ir, out_t_ir)
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
        stage.no_stage_typecheck(value, core.primal_s.map(atom.aval))
        return value

    def read_t(self, atom, /):
        if not stage.is_var(atom):
            return zero_tangent(atom)
        value = self.tangents[atom]
        stage.no_stage_typecheck(value, core.tangent_s.map(atom.aval))
        return value

    def write_p(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.primal_s.map(atom.aval))
            self.primals[atom] = value

    def write_t(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.tangent_s.map(atom.aval))
            self.tangents[atom] = value


def impl_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    env = PFEnv()

    def fwd_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        in_p, in_t = in_tree
        if all(isinstance(x, core.Zero) for x in utils.tree.leaves(in_t)) and all(
            core.primal_s.map(atom.aval) == atom.aval
            for atom in utils.tree.leaves((eqn.in_tree, eqn.out_tree))
            if stage.is_var(atom)
        ):
            with core.using_interpreter(parent):
                out_p = eqn.bind(in_p, **eqn.params)
            return out_p, utils.tree.map(zero_tangent, out_p)
        with core.using_interpreter(pusher):
            out_boxed = eqn.bind(pusher.box((in_p, in_t)), **eqn.params)
        return pusher.unbox(out_boxed)

    in_p, in_t = in_tree
    utils.tree.map(env.write_p, ir.in_tree, in_p)
    utils.tree.map(env.write_t, ir.in_tree, in_t)
    for eqn in ir.eqns:
        in_p = utils.tree.map(env.read_p, eqn.in_tree)
        in_t = utils.tree.map(env.read_t, eqn.in_tree)
        out_p, out_t = fwd_bind(eqn, (in_p, in_t))
        utils.tree.map(env.write_p, eqn.out_tree, out_p)
        utils.tree.map(env.write_t, eqn.out_tree, out_t)
    return utils.tree.map(env.read_p, ir.out_tree), utils.tree.map(env.read_t, ir.out_tree)


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interpreter.get()
    pusher = PushforwardInterpreter(parent=parent)
    env = PFEnv()

    async def fwd_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        in_p, in_t = in_tree
        if all(isinstance(x, core.Zero) for x in utils.tree.leaves(in_t)) and all(
            core.primal_s.map(atom.aval) == atom.aval
            for atom in utils.tree.leaves((eqn.in_tree, eqn.out_tree))
            if stage.is_var(atom)
        ):
            with core.using_interpreter(parent):
                out_p = await eqn.abind(in_p, **eqn.params)
            return out_p, utils.tree.map(zero_tangent, out_p)
        with core.using_interpreter(pusher):
            out_boxed = await eqn.abind(pusher.box((in_p, in_t)), **eqn.params)
        return pusher.unbox(out_boxed)

    in_p, in_t = in_tree
    utils.tree.map(env.write_p, ir.in_tree, in_p)
    utils.tree.map(env.write_t, ir.in_tree, in_t)
    for eqn in ir.eqns:
        in_p = utils.tree.map(env.read_p, eqn.in_tree)
        in_t = utils.tree.map(env.read_t, eqn.in_tree)
        out_p, out_t = await fwd_bind(eqn, (in_p, in_t))
        utils.tree.map(env.write_p, eqn.out_tree, out_p)
        utils.tree.map(env.write_t, eqn.out_tree, out_t)
    return utils.tree.map(env.read_p, ir.out_tree), utils.tree.map(env.read_t, ir.out_tree)


def abstract_pushforward_call(_: Tree, /, *, ir: stage.IR) -> TreePair:
    def t_aval(atom):
        if stage.is_var(atom):
            return core.tangent_s.map(atom.aval)
        return core.Zero(core.tangent_s.map(core.avalof(atom)))

    def p_aval(atom):
        if stage.is_var(atom):
            return core.primal_s.map(atom.aval)
        return atom

    out_p = utils.tree.map(p_aval, ir.out_tree)
    out_t = utils.tree.map(t_aval, ir.out_tree)
    return out_p, out_t


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
    (in_p, in_t) = in_tree
    pf_ir = pushforward(ir)
    out_p, out_t = pf_ir.call(in_p, in_t)
    residuals = (in_p, in_t)
    return (out_p, out_t), residuals


async def apullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (in_p, in_t) = in_tree
    pf_ir = pushforward(ir)
    out_p, out_t = await pf_ir.acall(in_p, in_t)
    residuals = (in_p, in_t)
    return (out_p, out_t), residuals


def pullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, out_c = in_tree
    in_p, in_t = residuals
    out_c_p, out_c_t = out_c
    p, t = (in_p, out_c_t), (in_t, out_c_p)
    (_, in_c_t), (_, in_c_p) = pushforward_pullback_call((p, t), ir=ir)
    return in_c_p, in_c_t


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, out_c = in_tree
    in_p, in_t = residuals
    out_c_p, out_c_t = out_c
    p, t = (in_p, out_c_t), (in_t, out_c_p)
    (_, in_c_t), (_, in_c_p) = await apushforward_pullback_call((p, t), ir=ir)
    return in_c_p, in_c_t


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
    out_original_used = utils.tree.map(lambda p, t: p or t, p_used, t_used)
    new_eqn = eqn.using(ir=dead.dce(eqn.params["ir"], out_used=out_original_used))
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
    num_cots, out_c = in_tree
    return [out_c] * num_cots


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
        def p(v):
            return v.primal if isinstance(v, PullbackFwdBox) and v.owner is self else v

        return utils.tree.map(p, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            out_p, residuals = core.pull_fwd_rules.get(prim)(in_p, **params)
        return self.box(out_p), residuals

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            out_p, residuals = await core.apull_fwd_rules.get(prim)(in_p, **params)
        return self.box(out_p), residuals


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
        def c(v):
            return v.cotangent if isinstance(v, PullbackBwdBox) and v.owner is self else v

        return utils.tree.map(c, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        residuals, out_c = in_tree
        out_c = self.unbox(out_c)
        with core.using_interpreter(self.parent):
            in_c = core.pull_bwd_rules.get(prim)((residuals, out_c), **params)
        return self.box(in_c)

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        residuals, out_c = in_tree
        out_c = self.unbox(out_c)
        with core.using_interpreter(self.parent):
            in_c = await core.apull_bwd_rules.get(prim)((residuals, out_c), **params)
        return self.box(in_c)


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
        >>> outputs, c = pb_ir.call(("Hello", " World"), "feedback")
        >>> outputs
        'Hello World'
        >>> c  # Gradient flows back to both inputs
        ('feedback', 'feedback')
    """
    assert isinstance(ir, stage.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.primal_s.map(atom.aval), source=atom)
        return atom

    def make_c(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.cotangent_s.map(atom.aval), source=atom)
        return core.Zero(core.cotangent_s.map(core.avalof(atom)))

    in_p_ir = utils.tree.map(make_p, ir.in_tree)
    out_c_ir = utils.tree.map(make_c, ir.out_tree)
    in_tree = (in_p_ir, out_c_ir)
    out_p_ir = utils.tree.map(make_p, ir.out_tree)
    in_c_ir = utils.tree.map(make_c, ir.in_tree)
    out_tree = (out_p_ir, in_c_ir)
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
        stage.no_stage_typecheck(value, core.primal_s.map(atom.aval))
        return value

    def write_p(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.primal_s.map(atom.aval))
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
    (in_p, out_c) = in_tree

    res: dict[stage.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    def fwd_bind(eqn: stage.Eqn, in_p: Tree, /) -> Tree:
        with core.using_interpreter(fwd):
            out_boxed, residuals = eqn.bind(fwd.box(in_p), **eqn.params)
        res[eqn] = residuals
        return fwd.unbox(out_boxed)

    def bwd_bind(eqn: stage.Eqn, out_c: Tree, /) -> Tree:
        residuals = res[eqn]
        with core.using_interpreter(bwd):
            in_boxed_c = eqn.bind((residuals, bwd.box(out_c)), **eqn.params)
        return bwd.unbox(in_boxed_c)

    env = PBEnv()

    utils.tree.map(env.write_p, ir.in_tree, in_p)
    for eqn in ir.eqns:
        in_p = utils.tree.map(env.read_p, eqn.in_tree)
        out_p = fwd_bind(eqn, in_p)
        utils.tree.map(env.write_p, eqn.out_tree, out_p)
    out_p = utils.tree.map(env.read_p, ir.out_tree)

    utils.tree.map(env.write_c, ir.out_tree, out_c)
    for eqn in reversed(ir.eqns):
        out_c = utils.tree.map(env.read_c, eqn.out_tree)
        in_c = bwd_bind(eqn, out_c)
        utils.tree.map(env.write_c, eqn.in_tree, in_c)
    return out_p, utils.tree.map(env.read_c, ir.in_tree)


async def aimpl_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (in_p, out_c) = in_tree

    res: dict[stage.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    async def fwd_bind(eqn: stage.Eqn, in_p: Tree, /) -> Tree:
        with core.using_interpreter(fwd):
            out_boxed, residuals = await eqn.abind(fwd.box(in_p), **eqn.params)
        res[eqn] = residuals
        return fwd.unbox(out_boxed)

    async def bwd_bind(eqn: stage.Eqn, out_c: Tree, /) -> Tree:
        residuals = res[eqn]
        with core.using_interpreter(bwd):
            in_boxed_c = await eqn.abind((residuals, bwd.box(out_c)), **eqn.params)
        return bwd.unbox(in_boxed_c)

    env = PBEnv()

    utils.tree.map(env.write_p, ir.in_tree, in_p)
    for eqn in ir.eqns:
        in_p = utils.tree.map(env.read_p, eqn.in_tree)
        out_p = await fwd_bind(eqn, in_p)
        utils.tree.map(env.write_p, eqn.out_tree, out_p)
    out_p = utils.tree.map(env.read_p, ir.out_tree)

    utils.tree.map(env.write_c, ir.out_tree, out_c)
    for eqn in reversed(ir.eqns):
        out_c = utils.tree.map(env.read_c, eqn.out_tree)
        in_c = await bwd_bind(eqn, out_c)
        utils.tree.map(env.write_c, eqn.in_tree, in_c)
    return out_p, utils.tree.map(env.read_c, ir.in_tree)


def abstract_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    def c_aval(atom):
        if stage.is_var(atom):
            return core.cotangent_s.map(atom.aval)
        return core.Zero(core.cotangent_s.map(core.avalof(atom)))

    def p_aval(atom):
        if stage.is_var(atom):
            return core.primal_s.map(atom.aval)
        return atom

    out_p = utils.tree.map(p_aval, ir.out_tree)
    in_c = utils.tree.map(c_aval, ir.in_tree)
    return out_p, in_c


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
    (in_p, out_c) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = pb_ir.call(in_p, out_c)
    residuals = (in_p, out_c, out_p, in_c)
    return (out_p, in_c), residuals


async def apullback_fwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    (in_p, out_c) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = await pb_ir.acall(in_p, out_c)
    residuals = (in_p, out_c, out_p, in_c)
    return (out_p, in_c), residuals


def pullback_bwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, c = in_tree
    in_p, out_c, _, _ = residuals
    out_c_p, in_c_c = c
    p, t = (in_p, out_c), (in_c_c, out_c_p)
    (_, _), (out_c_c, in_c_p) = pushforward_pullback_call((p, t), ir=ir)
    return in_c_p, out_c_c


async def apullback_bwd_pullback_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, c = in_tree
    in_p, out_c, _, _ = residuals
    out_c_p, in_c_c = c
    p, t = (in_p, out_c), (in_c_c, out_c_p)
    (_, _), (out_c_c, in_c_p) = await apushforward_pullback_call((p, t), ir=ir)
    return in_c_p, out_c_c


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
