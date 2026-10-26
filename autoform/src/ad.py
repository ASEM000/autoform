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

import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.order as order
import autoform.src.stage as stage
import autoform.src.tree as tree
import autoform.src.untyped as untyped
import autoform.src.utils as utils

type Tree[T] = Any


type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# PUSHFORWARD
# ==================================================================================================

p_pushforward_call = core.Prim("pushforward_call")


def concrete_zeros[R](rule: R, /) -> R:
    @ft.wraps(rule)
    def wrapper(in_tree, /, **params):
        first, second = in_tree
        return rule((first, core.instantiate(second)), **params)

    return wrapper


class PushforwardBox:
    __slots__ = ["owner", "primal", "tangent"]

    def __init__(self, owner, primal, tangent):
        self.owner = owner
        self.primal = primal
        self.tangent = tangent


core.aval_types[PushforwardBox] = lambda value: core.avalof(value.primal)


def expand_zeros(tangent: Tree, primal: Tree, /) -> Tree:
    def expand(t, p):
        if isinstance(t, core.Zero) and not tree.is_leaf(p):
            return core.s_tangent.zeros_like(p)
        return t

    leaves, spec = tree.flatten(tangent, is_leaf=lambda x: isinstance(x, core.Zero))
    return spec.unflatten(map(expand, leaves, spec.flatten_up_to(primal)))


class PushInterp(core.Interp):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        p, t = value
        t = expand_zeros(t, p)

        def box(p, t):
            if isinstance(t, core.Zero) and t == core.s_tangent.zeros_like(p):
                return p
            return PushforwardBox(self, p, t)

        return tree.map(box, p, t)

    def unbox(self, values: Tree, /) -> TreePair:
        # NOTE(asem): pushforward is structural, so this is not fixing a current
        # perturbation-confusion bug. Ownership only keeps values from other
        # interpreter instances opaque to this one.

        def primal(v):
            return v.primal if isinstance(v, PushforwardBox) and v.owner is self else v

        def tangent(v):
            if isinstance(v, PushforwardBox) and v.owner is self:
                return v.tangent
            return core.s_tangent.zeros_like(v)

        return tree.map(primal, values), tree.map(tangent, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p, in_t = self.unbox(in_tree)
        with core.using_interp(self.parent):
            out_p, out_t = core.r_push.get(prim)((in_p, in_t), **params)
        return self.box((out_p, out_t))

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p, in_t = self.unbox(in_tree)
        with core.using_interp(self.parent):
            out_p, out_t = await core.r_apush.get(prim)((in_p, in_t), **params)
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
            return stage.Var.fresh(aval=core.s_primal.map(atom.aval), source=atom)
        return atom

    def make_t(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.s_tangent.map(atom.aval), source=atom)
        return core.s_tangent.zeros_like(atom)

    in_p_ir = tree.map(make_p, ir.in_tree)
    in_t_ir = tree.map(make_t, ir.in_tree)
    in_tree = (in_p_ir, in_t_ir)
    out_p_ir = tree.map(make_p, ir.out_tree)
    out_t_ir = tree.map(make_t, ir.out_tree)
    out_tree = (out_p_ir, out_t_ir)
    eqn = stage.Eqn(p_pushforward_call, in_tree, out_tree, dict(ir=ir))
    return stage.IR([eqn], in_tree, out_tree)


class PFEnv:
    __slots__ = ["primals", "tangents"]

    def __init__(self):
        self.primals = {}
        self.tangents = {}

    def read_p(self, atom, /):
        if not stage.is_var(atom):
            return atom
        value = self.primals[atom]
        stage.no_stage_typecheck(value, core.s_primal.map(atom.aval))
        return value

    def read_t(self, atom, /):
        if not stage.is_var(atom):
            return core.s_tangent.zeros_like(atom)
        value = self.tangents[atom]
        stage.no_stage_typecheck(value, core.s_tangent.map(atom.aval))
        return value

    def write_p(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.s_primal.map(atom.aval))
            self.primals[atom] = value

    def write_t(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.s_tangent.map(atom.aval))
            self.tangents[atom] = value


def impl_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interp.get()
    pusher = PushInterp(parent=parent)
    env = PFEnv()

    def fwd_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        in_p, in_t = in_tree
        if all(isinstance(x, core.Zero) for x in tree.leaves(in_t)) and all(
            core.s_primal.map(atom.aval) == atom.aval
            for atom in tree.leaves((eqn.in_tree, eqn.out_tree))
            if stage.is_var(atom)
        ):
            with core.using_interp(parent):
                out_p = eqn.bind(in_p, **eqn.params)
            return out_p, core.s_tangent.zeros_like(out_p)
        with core.using_interp(pusher):
            out_boxed = eqn.bind(pusher.box((in_p, in_t)), **eqn.params)
        return pusher.unbox(out_boxed)

    in_p, in_t = in_tree
    tree.map(env.write_p, ir.in_tree, in_p)
    tree.map(env.write_t, ir.in_tree, in_t)
    for eqn in ir.eqns:
        in_p = tree.map(env.read_p, eqn.in_tree)
        in_t = tree.map(env.read_t, eqn.in_tree)
        out_p, out_t = fwd_bind(eqn, (in_p, in_t))
        tree.map(env.write_p, eqn.out_tree, out_p)
        tree.map(env.write_t, eqn.out_tree, out_t)
    return tree.map(env.read_p, ir.out_tree), tree.map(env.read_t, ir.out_tree)


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interp.get()
    pusher = PushInterp(parent=parent)
    env = PFEnv()

    async def fwd_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        in_p, in_t = in_tree
        if all(isinstance(x, core.Zero) for x in tree.leaves(in_t)) and all(
            core.s_primal.map(atom.aval) == atom.aval
            for atom in tree.leaves((eqn.in_tree, eqn.out_tree))
            if stage.is_var(atom)
        ):
            with core.using_interp(parent):
                out_p = await eqn.abind(in_p, **eqn.params)
            return out_p, core.s_tangent.zeros_like(out_p)
        with core.using_interp(pusher):
            out_boxed = await eqn.abind(pusher.box((in_p, in_t)), **eqn.params)
        return pusher.unbox(out_boxed)

    in_p, in_t = in_tree
    tree.map(env.write_p, ir.in_tree, in_p)
    tree.map(env.write_t, ir.in_tree, in_t)
    for eqn in ir.eqns:
        in_p = tree.map(env.read_p, eqn.in_tree)
        in_t = tree.map(env.read_t, eqn.in_tree)
        out_p, out_t = await fwd_bind(eqn, (in_p, in_t))
        tree.map(env.write_p, eqn.out_tree, out_p)
        tree.map(env.write_t, eqn.out_tree, out_t)
    return tree.map(env.read_p, ir.out_tree), tree.map(env.read_t, ir.out_tree)


def abstract_pushforward_call(_: Tree, /, *, ir: stage.IR) -> TreePair:
    def t_aval(atom):
        if stage.is_var(atom):
            return core.s_tangent.map(atom.aval)
        return core.s_tangent.zeros_like(atom)

    def p_aval(atom):
        if stage.is_var(atom):
            return core.s_primal.map(atom.aval)
        return atom

    out_p = tree.map(p_aval, ir.out_tree)
    out_t = tree.map(t_aval, ir.out_tree)
    return out_p, out_t


def pushforward_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interp.get()
    pusher = PushInterp(parent=parent)
    with core.using_interp(pusher):
        return pusher.unbox(impl_pushforward_call(pusher.box(in_tree), ir=ir))


async def apushforward_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    parent = core.active_interp.get()
    pusher = PushInterp(parent=parent)
    with core.using_interp(pusher):
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
    (_, in_c_t), (_, in_c_p) = pushforward_pullback((p, t), ir=ir)
    return in_c_p, in_c_t


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> Tree:
    residuals, out_c = in_tree
    in_p, in_t = residuals
    out_c_p, out_c_t = out_c
    p, t = (in_p, out_c_t), (in_t, out_c_p)
    (_, in_c_t), (_, in_c_p) = await apushforward_pullback((p, t), ir=ir)
    return in_c_p, in_c_t


def batch_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    spec, in_axes, in_values = in_tree
    pf_ir = pushforward(ir)
    items = tree.unstack(spec, in_axes, in_values)
    results = (pf_ir.call(*x) for x in items)
    out_axes = tree.map(lambda _: True, pf_ir.out_tree)
    return tree.stack(spec, out_axes, results), out_axes


async def abatch_pushforward_call(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    spec, in_axes, in_values = in_tree
    pf_ir = pushforward(ir)
    inputs = tree.unstack(spec, in_axes, in_values)
    results = await order.p_fanout.abind(inputs, irs=[pf_ir] * len(inputs))
    out_axes = tree.map(lambda _: True, pf_ir.out_tree)
    return tree.stack(spec, out_axes, results), out_axes


def dce_pushforward_call(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    p_used, t_used = out_used
    out_original_used = tree.map(lambda p, t: p or t, p_used, t_used)
    new_eqn = eqn.using(ir=dead.dce(eqn.params["ir"], out_used=out_original_used))
    return dead.default_dce(new_eqn, out_used)


core.r_impl.set(p_pushforward_call, impl_pushforward_call)
core.r_aimpl.set(p_pushforward_call, aimpl_pushforward_call)
core.r_abstract.set(p_pushforward_call, abstract_pushforward_call)
core.r_batch.set(p_pushforward_call, batch_pushforward_call)
core.r_abatch.set(p_pushforward_call, abatch_pushforward_call)
core.r_push.set(p_pushforward_call, pushforward_pushforward_call)
core.r_apush.set(p_pushforward_call, apushforward_pushforward_call)
core.r_pull_fwd.set(p_pushforward_call, pullback_fwd_pushforward_call)
core.r_apull_fwd.set(p_pushforward_call, apullback_fwd_pushforward_call)
core.r_pull_bwd.set(p_pushforward_call, pullback_bwd_pushforward_call)
core.r_apull_bwd.set(p_pushforward_call, apullback_bwd_pushforward_call)
dead.register_dce(p_pushforward_call, dce_pushforward_call)


# ==================================================================================================
# PULLBACK
# ==================================================================================================

p_pullback_fwd_call = core.Prim("pullback_fwd_call")
p_pullback_bwd_call = core.Prim("pullback_bwd_call")
p_cot_accum = core.Prim("cot_accum")


def cot_accum(cots: list[Any | core.Zero]) -> Any:
    assert cots
    non_zero = [c for c in cots if not isinstance(c, core.Zero)]
    if not non_zero:
        # NOTE(asem): all output paths into the same input are zero.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.trace(f)("...")
        # >>> z = af.extend.Zero(af.extend.s_primal.map(af.extend.avalof("")))
        # >>> af.pullback(ir).call(("a",), (z, z))
        # (('a', 'a'), (Zero(AStr()),))
        first_zero, *rest_zero = cots
        assert all(core.avalof(c) == core.avalof(first_zero) for c in rest_zero)
        return first_zero
    if len(non_zero) == 1:
        # NOTE(asem): exactly one output path into the same input is live.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.trace(f)("...")
        # >>> z = af.extend.Zero(af.extend.s_primal.map(af.extend.avalof("")))
        # >>> af.pullback(ir).call(("a",), ("df", z))
        # (('a', 'a'), ('df',))
        return non_zero[0]
    first, *_ = non_zero
    if not tree.is_leaf(first):
        # NOTE(asem): non-leaf cotangents accumulate matching leaves.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.batch(af.trace(f)("..."))
        # >>> af.pullback(ir).call((["a", "b"],), (["G0", "G1"], ["H0", "H1"]))
        # ((['a', 'b'], ['a', 'b']), (['G0H0', 'G1H1'],))
        return tree.map(lambda *cs: cot_accum(list(cs)), *non_zero)
    # NOTE(asem): leaf cotangents use their aval's addition.
    # >>> def f(x):
    # ...     return x + x
    # >>> ir = af.trace(f)("...")
    # >>> af.pullback(ir).call(("a",), "df")
    # ('aa', ('dfdf',))
    return p_cot_accum.bind(non_zero, n=len(non_zero))


def impl_cot_accum(cots: list[Any], /, *, n: int) -> Any:
    del n
    aval = core.avalof(cots[0])
    return ft.reduce(aval.add, cots, aval.zero())


def abstract_cot_accum(cots: list[Any], /, *, n: int) -> core.AVal:
    del n
    first = cots[0]
    return first if isinstance(first, core.AVal) else core.avalof(first)


def pushforward_cot_accum(in_tree: TreePair, /, *, n: int) -> TreePair:
    del n
    p_cots, t_cots = in_tree
    return cot_accum(p_cots), cot_accum(t_cots)


def pullback_fwd_cot_accum(cots: list[Any], /, *, n: int) -> TreePair:
    del n
    return cot_accum(cots), None


def pullback_bwd_cot_accum(in_tree: TreePair, /, *, n: int) -> list[Any]:
    _, out_c = in_tree
    return [out_c] * n


def batch_cot_accum(in_tree: Tree, /, *, n: int) -> TreePair:
    del n
    spec, in_axes, in_values = in_tree
    items = tree.unstack(spec, in_axes, in_values)
    results = (cot_accum(x) for x in items)
    return tree.stack(spec, True, results), True


core.r_impl.set(p_cot_accum, impl_cot_accum)
core.r_aimpl.set(p_cot_accum, utils.asyncify(impl_cot_accum))
core.r_abstract.set(p_cot_accum, abstract_cot_accum)
core.r_batch.set(p_cot_accum, batch_cot_accum)
core.r_abatch.set(p_cot_accum, utils.asyncify(batch_cot_accum))
core.r_push.set(p_cot_accum, pushforward_cot_accum)
core.r_apush.set(p_cot_accum, utils.asyncify(pushforward_cot_accum))
core.r_pull_fwd.set(p_cot_accum, pullback_fwd_cot_accum)
core.r_apull_fwd.set(p_cot_accum, utils.asyncify(pullback_fwd_cot_accum))
core.r_pull_bwd.set(p_cot_accum, pullback_bwd_cot_accum)
core.r_apull_bwd.set(p_cot_accum, utils.asyncify(pullback_bwd_cot_accum))


class PullbackFwdBox:
    __slots__ = ["owner", "primal"]

    def __init__(self, owner, primal):
        self.owner = owner
        self.primal = primal


core.aval_types[PullbackFwdBox] = lambda value: core.avalof(value.primal)


class PullFwdInterp(core.Interp):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        return tree.map(lambda p: PullbackFwdBox(self, p), value)

    def unbox(self, values: Tree, /) -> Tree:
        def p(v):
            return v.primal if isinstance(v, PullbackFwdBox) and v.owner is self else v

        return tree.map(p, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p = self.unbox(in_tree)
        with core.using_interp(self.parent):
            out_p, residuals = core.r_pull_fwd.get(prim)(in_p, **params)
        return self.box(out_p), residuals

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_p = self.unbox(in_tree)
        with core.using_interp(self.parent):
            out_p, residuals = await core.r_apull_fwd.get(prim)(in_p, **params)
        return self.box(out_p), residuals


class PullbackBwdBox:
    __slots__ = ["owner", "cotangent"]

    def __init__(self, owner, cotangent):
        self.owner = owner
        self.cotangent = cotangent


core.aval_types[PullbackBwdBox] = lambda value: core.avalof(value.cotangent)


class PullBwdInterp(core.Interp):
    __slots__ = ["parent"]

    def __init__(self, *, parent):
        self.parent = parent

    def box(self, value, /) -> Tree:
        return tree.map(lambda c: PullbackBwdBox(self, c), value)

    def unbox(self, values: Tree, /) -> Tree:
        def c(v):
            return v.cotangent if isinstance(v, PullbackBwdBox) and v.owner is self else v

        return tree.map(c, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        residuals, out_c = in_tree
        out_c = self.unbox(out_c)
        with core.using_interp(self.parent):
            in_c = core.r_pull_bwd.get(prim)((residuals, out_c), **params)
        return self.box(in_c)

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        residuals, out_c = in_tree
        out_c = self.unbox(out_c)
        with core.using_interp(self.parent):
            in_c = await core.r_apull_bwd.get(prim)((residuals, out_c), **params)
        return self.box(in_c)


@ft.partial(utils.lru_cache, maxsize=256)
def fwd_bwd(ir: stage.IR, /) -> tuple[stage.IR, stage.IR]:
    """Split reverse-mode AD into forward and backward IRs.

    Example:
        >>> import autoform as af
        >>> fwd, bwd = af.fwd_bwd(af.trace(lambda x: x * x)(2.0))
        >>> out, residual = fwd.call(3.0)
        >>> out, bwd.call(residual, 1.0)
        (9.0, (6.0,))
    """
    assert isinstance(ir, stage.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.s_primal.map(atom.aval), source=atom)
        return atom

    def make_c(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=core.s_cotangent.map(atom.aval), source=atom)
        return core.s_cotangent.zeros_like(atom)

    in_p = tree.map(make_p, ir.in_tree)
    out_p = tree.map(make_p, ir.out_tree)
    out_c = tree.map(make_c, ir.out_tree)
    in_c = tree.map(make_c, ir.in_tree)
    residual = stage.Var.fresh(aval=untyped.AUntyped())
    fwd_eqn = stage.Eqn(p_pullback_fwd_call, in_p, (out_p, residual), dict(ir=ir))
    bwd_eqn = stage.Eqn(p_pullback_bwd_call, (residual, out_c), in_c, dict(ir=ir))
    fwd = stage.IR([fwd_eqn], in_p, (out_p, residual))
    bwd = stage.IR([bwd_eqn], (residual, out_c), in_c)
    return fwd, bwd


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
    fwd, bwd = fwd_bwd(ir)
    (out_p, _), (_, out_c) = fwd.out_tree, bwd.in_tree
    return stage.IR([*fwd.eqns, *bwd.eqns], (fwd.in_tree, out_c), (out_p, bwd.out_tree))


class PBEnv:
    __slots__ = ["primals", "cotangents"]

    def __init__(self):
        self.primals = {}
        self.cotangents = defaultdict(list)

    def read_p(self, atom, /):
        if not stage.is_var(atom):
            return atom
        value = self.primals[atom]
        stage.no_stage_typecheck(value, core.s_primal.map(atom.aval))
        return value

    def write_p(self, atom, value, /):
        if stage.is_var(atom):
            stage.no_stage_typecheck(value, core.s_primal.map(atom.aval))
            self.primals[atom] = value

    def read_c(self, atom, /):
        aval = core.s_cotangent.map(core.avalof(atom))
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
            stage.no_stage_typecheck(value, core.s_cotangent.map(atom.aval))
            self.cotangents[atom].append(value)

    def is_zero_c(self, value, /) -> bool:
        return all(
            isinstance(c, core.Zero)
            for atom in tree.leaves(value)
            if stage.is_var(atom)
            for c in self.cotangents[atom]
        )


def impl_pullback_fwd_call(
    in_p: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    inputs, res = in_p, []
    fwd = PullFwdInterp(parent=core.active_interp.get())
    env = PBEnv()
    tree.map(env.write_p, ir.in_tree, in_p)
    for eqn in ir.eqns:
        in_p = tree.map(env.read_p, eqn.in_tree)
        with core.using_interp(fwd):
            out_boxed, residuals = eqn.bind(fwd.box(in_p), **eqn.params)
        res.append(residuals)
        tree.map(env.write_p, eqn.out_tree, fwd.unbox(out_boxed))
    out_p = tree.map(env.read_p, ir.out_tree)
    return out_p, untyped.Untyped((inputs, tuple(res)))


async def aimpl_pullback_fwd_call(
    in_p: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    inputs, res = in_p, []
    fwd = PullFwdInterp(parent=core.active_interp.get())
    env = PBEnv()
    tree.map(env.write_p, ir.in_tree, in_p)
    for eqn in ir.eqns:
        in_p = tree.map(env.read_p, eqn.in_tree)
        with core.using_interp(fwd):
            out_boxed, residuals = await eqn.abind(fwd.box(in_p), **eqn.params)
        res.append(residuals)
        tree.map(env.write_p, eqn.out_tree, fwd.unbox(out_boxed))
    out_p = tree.map(env.read_p, ir.out_tree)
    return out_p, untyped.Untyped((inputs, tuple(res)))


def impl_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    residual, out_c = in_tree
    _, res = residual.value
    bwd = PullBwdInterp(parent=core.active_interp.get())
    env = PBEnv()
    tree.map(env.write_c, ir.out_tree, out_c)
    for eqn, residuals in reversed(list(zip(ir.eqns, res, strict=True))):
        if env.is_zero_c(eqn.out_tree):
            continue
        out_c = tree.map(env.read_c, eqn.out_tree)
        with core.using_interp(bwd):
            in_boxed_c = eqn.bind((residuals, bwd.box(out_c)), **eqn.params)
        tree.map(env.write_c, eqn.in_tree, bwd.unbox(in_boxed_c))
    return tree.map(env.read_c, ir.in_tree)


async def aimpl_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    residual, out_c = in_tree
    _, res = residual.value
    bwd = PullBwdInterp(parent=core.active_interp.get())
    env = PBEnv()
    tree.map(env.write_c, ir.out_tree, out_c)
    for eqn, residuals in reversed(list(zip(ir.eqns, res, strict=True))):
        if env.is_zero_c(eqn.out_tree):
            continue
        out_c = tree.map(env.read_c, eqn.out_tree)
        with core.using_interp(bwd):
            in_boxed_c = await eqn.abind((residuals, bwd.box(out_c)), **eqn.params)
        tree.map(env.write_c, eqn.in_tree, bwd.unbox(in_boxed_c))
    return tree.map(env.read_c, ir.in_tree)


def impl_pullback(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    in_p, out_c = in_tree
    out_p, residual = impl_pullback_fwd_call(in_p, ir=ir)
    return out_p, impl_pullback_bwd_call((residual, out_c), ir=ir)


async def aimpl_pullback(in_tree: Tree, /, *, ir: stage.IR) -> TreePair:
    in_p, out_c = in_tree
    out_p, residual = await aimpl_pullback_fwd_call(in_p, ir=ir)
    return out_p, await aimpl_pullback_bwd_call((residual, out_c), ir=ir)


def pushforward_pullback(in_tree: TreePair, /, *, ir: stage.IR) -> TreePair:
    pusher = PushInterp(parent=core.active_interp.get())
    with core.using_interp(pusher):
        return pusher.unbox(impl_pullback(pusher.box(in_tree), ir=ir))


async def apushforward_pullback(in_tree: TreePair, /, *, ir: stage.IR) -> TreePair:
    pusher = PushInterp(parent=core.active_interp.get())
    with core.using_interp(pusher):
        return pusher.unbox(await aimpl_pullback(pusher.box(in_tree), ir=ir))


def abstract_pullback_fwd_call(
    _: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    def p_aval(atom):
        if stage.is_var(atom):
            return core.s_primal.map(atom.aval)
        return atom

    return tree.map(p_aval, ir.out_tree), untyped.AUntyped()


def abstract_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    def c_aval(atom):
        if stage.is_var(atom):
            return core.s_cotangent.map(atom.aval)
        return core.s_cotangent.zeros_like(atom)

    return tree.map(c_aval, ir.in_tree)


def pushforward_pullback_fwd_call(
    in_tree: TreePair,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    pusher = PushInterp(parent=core.active_interp.get())
    with core.using_interp(pusher):
        out = impl_pullback_fwd_call(pusher.box(in_tree), ir=ir)
    return pusher.unbox(out)


async def apushforward_pullback_fwd_call(
    in_tree: TreePair,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    pusher = PushInterp(parent=core.active_interp.get())
    with core.using_interp(pusher):
        out = await aimpl_pullback_fwd_call(pusher.box(in_tree), ir=ir)
    return pusher.unbox(out)


def pushforward_pullback_bwd_call(
    in_tree: TreePair,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    pusher = PushInterp(parent=core.active_interp.get())
    with core.using_interp(pusher):
        out = impl_pullback_bwd_call(pusher.box(in_tree), ir=ir)
    return pusher.unbox(out)


async def apushforward_pullback_bwd_call(
    in_tree: TreePair,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    pusher = PushInterp(parent=core.active_interp.get())
    with core.using_interp(pusher):
        out = await aimpl_pullback_bwd_call(pusher.box(in_tree), ir=ir)
    return pusher.unbox(out)


# NOTE(asem): a residual's cotangent is the cotangent of the inputs it saved, boxed so it
# flows as one ``AUntyped()`` leaf like the residual itself.
def unbox_cotangent(residual_c: Tree, /) -> Tree:
    return residual_c if isinstance(residual_c, core.Zero) else residual_c.value


def pullback_fwd_pullback_fwd_call(
    in_p: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    out_p, residual = impl_pullback_fwd_call(in_p, ir=ir)
    return (out_p, residual), residual


async def apullback_fwd_pullback_fwd_call(
    in_p: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    out_p, residual = await aimpl_pullback_fwd_call(in_p, ir=ir)
    return (out_p, residual), residual


def pullback_bwd_pullback_fwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    residual, (out_c, residual_c) = in_tree
    in_c = impl_pullback_bwd_call((residual, out_c), ir=ir)
    return cot_accum([in_c, unbox_cotangent(residual_c)])


async def apullback_bwd_pullback_fwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    residual, (out_c, residual_c) = in_tree
    in_c = await aimpl_pullback_bwd_call((residual, out_c), ir=ir)
    return cot_accum([in_c, unbox_cotangent(residual_c)])


def pullback_fwd_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    return impl_pullback_bwd_call(in_tree, ir=ir), in_tree


async def apullback_fwd_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    return await aimpl_pullback_bwd_call(in_tree, ir=ir), in_tree


def pullback_bwd_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    (residual, out_c), in_c_c = in_tree
    in_p, _ = residual.value
    out_c_t = core.s_tangent.zeros_like(out_c)
    (_, _), (out_c_c, in_c_p) = pushforward_pullback(((in_p, out_c), (in_c_c, out_c_t)), ir=ir)
    return untyped.Untyped(in_c_p), out_c_c


async def apullback_bwd_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> Tree:
    (residual, out_c), in_c_c = in_tree
    in_p, _ = residual.value
    out_c_t = core.s_tangent.zeros_like(out_c)
    (_, _), (out_c_c, in_c_p) = await apushforward_pullback(
        ((in_p, out_c), (in_c_c, out_c_t)),
        ir=ir,
    )
    return untyped.Untyped(in_c_p), out_c_c


def batch_pullback_half(ir: stage.IR, in_tree: Tree, /) -> TreePair:
    spec, in_axes, in_values = in_tree
    items = tree.unstack(spec, in_axes, in_values)
    results = (ir.call(*x) for x in items)
    out_axes = tree.map(lambda _: True, ir.out_tree)
    return tree.stack(spec, out_axes, results), out_axes


async def abatch_pullback_half(ir: stage.IR, in_tree: Tree, /) -> TreePair:
    spec, in_axes, in_values = in_tree
    inputs = tree.unstack(spec, in_axes, in_values)
    results = await order.p_fanout.abind(inputs, irs=[ir] * len(inputs))
    out_axes = tree.map(lambda _: True, ir.out_tree)
    return tree.stack(spec, out_axes, results), out_axes


def batch_pullback_fwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    fwd, _ = fwd_bwd(ir)
    return batch_pullback_half(fwd, in_tree)


async def abatch_pullback_fwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    fwd, _ = fwd_bwd(ir)
    return await abatch_pullback_half(fwd, in_tree)


def batch_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    _, bwd = fwd_bwd(ir)
    return batch_pullback_half(bwd, in_tree)


async def abatch_pullback_bwd_call(
    in_tree: Tree,
    /,
    *,
    ir: stage.IR,
) -> TreePair:
    _, bwd = fwd_bwd(ir)
    return await abatch_pullback_half(bwd, in_tree)


def dce_pullback_fwd_call(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    out_p_used, residual_used = out_used
    inner_ir = eqn.params["ir"]
    if not residual_used:
        inner_ir = dead.dce(inner_ir, out_used=out_p_used)
    return dead.default_dce(eqn.using(ir=inner_ir), out_used)


core.r_impl.set(p_pullback_fwd_call, impl_pullback_fwd_call)
core.r_aimpl.set(p_pullback_fwd_call, aimpl_pullback_fwd_call)
core.r_abstract.set(p_pullback_fwd_call, abstract_pullback_fwd_call)
core.r_batch.set(p_pullback_fwd_call, batch_pullback_fwd_call)
core.r_abatch.set(p_pullback_fwd_call, abatch_pullback_fwd_call)
core.r_push.set(p_pullback_fwd_call, pushforward_pullback_fwd_call)
core.r_apush.set(p_pullback_fwd_call, apushforward_pullback_fwd_call)
core.r_pull_fwd.set(p_pullback_fwd_call, pullback_fwd_pullback_fwd_call)
core.r_apull_fwd.set(p_pullback_fwd_call, apullback_fwd_pullback_fwd_call)
core.r_pull_bwd.set(p_pullback_fwd_call, pullback_bwd_pullback_fwd_call)
core.r_apull_bwd.set(p_pullback_fwd_call, apullback_bwd_pullback_fwd_call)


core.r_impl.set(p_pullback_bwd_call, impl_pullback_bwd_call)
core.r_aimpl.set(p_pullback_bwd_call, aimpl_pullback_bwd_call)
core.r_abstract.set(p_pullback_bwd_call, abstract_pullback_bwd_call)
core.r_batch.set(p_pullback_bwd_call, batch_pullback_bwd_call)
core.r_abatch.set(p_pullback_bwd_call, abatch_pullback_bwd_call)
core.r_push.set(p_pullback_bwd_call, pushforward_pullback_bwd_call)
core.r_apush.set(p_pullback_bwd_call, apushforward_pullback_bwd_call)
core.r_pull_fwd.set(p_pullback_bwd_call, pullback_fwd_pullback_bwd_call)
core.r_apull_fwd.set(p_pullback_bwd_call, apullback_fwd_pullback_bwd_call)
core.r_pull_bwd.set(p_pullback_bwd_call, pullback_bwd_pullback_bwd_call)
core.r_apull_bwd.set(p_pullback_bwd_call, apullback_bwd_pullback_bwd_call)
dead.register_dce(p_pullback_fwd_call, dce_pullback_fwd_call)
