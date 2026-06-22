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
from collections.abc import Callable
from typing import Any, TypeGuard

__all__ = [
    "Zero",
    "is_zero",
    "zero_rules",
    "zeroof",
    "materialize",
    "cot_acc",
    "cot_acc_rules",
    "pushforward",
    "pullback",
]

import autoform.core as core
import autoform.dce as dce
import autoform.scheduling as scheduling
import autoform.utils as utils

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]

# ==================================================================================================
# ZERO
# ==================================================================================================


class Zero[T: core.AVal]:
    """Symbolic zero cotangent for an abstract value.

    ``Zero`` keeps reverse-mode and pushforward rules from materializing a
    concrete zero until one is actually needed. Use :func:`materialize` to
    replace symbolic zeros with concrete values.

    Example:
        >>> import autoform.extend as afe
        >>> z = afe.Zero(afe.StrAVal())
        >>> afe.materialize(z)
        ''
    """

    __slots__ = ["aval"]

    def __init__(self, aval: T, /):
        assert isinstance(aval, core.AVal), f"Expected AVal, got {aval!r}"
        self.aval = aval

    def __repr__(self):
        return f"Zero({self.aval!r})"

    def __eq__(self, other):
        return isinstance(other, Zero) and self.aval == other.aval

    def __hash__(self):
        return hash((type(self), self.aval))


def is_zero(x, /) -> TypeGuard[Zero]:
    """Return whether the input is a symbolic zero cotangent.

    This is intended for rule implementations that need to preserve symbolic
    zeros instead of treating them as ordinary values.
    """
    return isinstance(x, Zero)


zero_rules: dict[type[core.AVal], Callable[[core.AVal], Any]] = {}
zero_rules[core.StrAVal] = lambda _: ""
core.aval_rules[Zero] = lambda z: z.aval


def zeroof(v, /) -> Zero:
    """Return a symbolic zero with the same aval as ``v``.

    If ``v`` is already a symbolic zero, it is returned unchanged.

    Args:
        v: Concrete value, IR value, or symbolic zero.

    Returns:
        A ``Zero`` carrying ``avalof(v)``.
    """
    return v if is_zero(v) else Zero(core.avalof(v))


def materialize(x: Tree, /) -> Tree:
    """Replace each Zero leaf in a pytree with its concrete zero value.

    ``materialize`` is useful inside transform rules before calling primitives
    that expect real runtime values instead of symbolic zeros.

    Args:
        x: Pytree that may contain ``Zero`` leaves.

    Returns:
        A pytree with the same structure as ``x`` where each symbolic zero has
        been replaced by its registered concrete zero value.

    Raises:
        TypeError: If a ``Zero`` has a type with no registered concrete
            zero (e.g. ``Zero(BoolAVal())``). This indicates an invalid gradient
            path through a non-differentiable type.
    """

    def map_func(x):
        if not is_zero(x):
            return x
        if (rule := zero_rules.get(type(x.aval))) is None:
            raise TypeError(f"Cannot materialize {x!r}")
        return rule(x.aval)

    return utils.tree.map(map_func, x, is_leaf=is_zero)


def all_zero(x: Tree, /) -> bool:
    return all(is_zero(leaf) for leaf in utils.tree.leaves(x, is_leaf=is_zero))


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


class PushforwardInterpreter(core.BoxedInterpreter[PushforwardBox]):
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
            return v.tangent if isinstance(v, PushforwardBox) and v.owner is self else zeroof(v)

        return utils.tree.map(primal, values), utils.tree.map(tangent, values)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        p_in, t_in = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            p_out, t_out = core.push_rules.get(prim)((p_in, t_in), **params)
        return self.box((p_out, t_out))

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        p_in, t_in = self.unbox(in_tree)
        with core.using_interpreter(self.parent):
            p_out, t_out = await core.push_rules.aget(prim)((p_in, t_in), **params)
        return self.box((p_out, t_out))


@ft.partial(utils.lru_cache, maxsize=256)
def pushforward(ir: core.IR, /) -> core.IR:
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
    assert isinstance(ir, core.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if core.is_irvar(atom):
            return core.Var.fresh(aval=core.ir_aval(atom), source=atom)
        return atom

    def make_t(atom):
        if core.is_irvar(atom):
            return core.Var.fresh(aval=core.ir_aval(atom), source=atom)
        return zeroof(atom)

    p_in_ir = utils.tree.map(make_p, ir.in_tree)
    t_in_ir = utils.tree.map(make_t, ir.in_tree)
    in_tree = (p_in_ir, t_in_ir)
    p_out_ir = utils.tree.map(make_p, ir.out_tree)
    t_out_ir = utils.tree.map(make_t, ir.out_tree)
    out_tree = (p_out_ir, t_out_ir)
    eqn = core.Eqn(pushforward_call_p, in_tree, out_tree, dict(ir=ir))
    return core.IR([eqn], in_tree, out_tree)


def impl_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    pusher = PushforwardInterpreter(parent=core.active_interpreter.get())
    with core.using_interpreter(pusher):

        def custom_bind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            p_in, t_in = pusher.unbox(boxed_in)
            if not all_zero(t_in):
                return eqn.bind(boxed_in, **eqn.params)
            with core.using_interpreter(pusher.parent):
                p_out = eqn.bind(p_in, **eqn.params)
            return pusher.box((p_out, utils.tree.map(zeroof, p_out)))

        eqn, boxed_in = next(gen := ir.walk(*pusher.box(in_tree)))
        while eqn:
            eqn, boxed_in = gen.send(custom_bind(eqn, boxed_in))
        return pusher.unbox(boxed_in)


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    pusher = PushforwardInterpreter(parent=core.active_interpreter.get())
    with core.using_interpreter(pusher):

        async def custom_abind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            p_in, t_in = pusher.unbox(boxed_in)
            if not all_zero(t_in):
                return await eqn.abind(boxed_in, **eqn.params)
            with core.using_interpreter(pusher.parent):
                p_out = await eqn.abind(p_in, **eqn.params)
            return pusher.box((p_out, utils.tree.map(zeroof, p_out)))

        eqn, boxed_in = next(gen := ir.walk(*pusher.box(in_tree)))
        while eqn:
            eqn, boxed_in = gen.send(await custom_abind(eqn, boxed_in))
        return pusher.unbox(boxed_in)


def abstract_pushforward_call(_: Tree, /, *, ir: core.IR) -> TreePair:
    out = utils.tree.map(core.ir_aval, ir.out_tree)
    return out, out


def pushforward_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    p, t = in_tree
    (p_in, t_in), (t_p_in, t_t_in) = p, t
    pf_ir = pushforward(ir)
    p_out = pf_ir.call(p_in, t_in)
    t_out = pf_ir.call(t_p_in, t_t_in)
    return p_out, t_out


async def apushforward_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    p, t = in_tree
    (p_in, t_in), (t_p_in, t_t_in) = p, t
    pf_ir = pushforward(ir)
    p_out = await pf_ir.acall(p_in, t_in)
    t_out = await pf_ir.acall(t_p_in, t_t_in)
    return p_out, t_out


def pullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = pf_ir.call(p_in, t_in)
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


async def apullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = await pf_ir.acall(p_in, t_in)
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


def pullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> Tree:
    residuals, c_out = in_tree
    p_in, _ = residuals
    c_p_out, c_t_out = c_out
    pb_ir = pullback(ir)
    _, c_p_in = pb_ir.call(p_in, c_p_out)
    _, c_t_in = pb_ir.call(p_in, c_t_out)
    return (c_p_in, c_t_in)


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> Tree:
    residuals, c_out = in_tree
    p_in, _ = residuals
    c_p_out, c_t_out = c_out
    pb_ir = pullback(ir)
    _, c_p_in = await pb_ir.acall(p_in, c_p_out)
    _, c_t_in = await pb_ir.acall(p_in, c_t_out)
    return (c_p_in, c_t_in)


def batch_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    batch_size, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if utils.batch_spec(in_values, in_batched) is None:
        pf_ir = pushforward(ir)
        result = pf_ir.call(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(utils.batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = [pf_ir.call(unbatch_p(b), unbatch_t(b)) for b in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, pf_ir.out_tree)
    out_ib = utils.batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pushforward_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    bs, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if utils.batch_spec(in_values, in_batched) is None:
        pf_ir = pushforward(ir)
        result = await pf_ir.acall(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(utils.batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)

    inputs = [(unbatch_p(b), unbatch_t(b)) for b in range(bs)]
    out_bi = await scheduling.gather_p.abind(inputs, irs=[pf_ir] * bs)
    out_batched = utils.tree.map(lambda _: True, pf_ir.out_tree)
    out_ib = utils.batch_transpose(bs, out_batched, list(out_bi))
    return out_ib, out_batched


core.impl_rules.set(pushforward_call_p, impl_pushforward_call)
core.impl_rules.aset(pushforward_call_p, aimpl_pushforward_call)
core.abstract_rules.set(pushforward_call_p, abstract_pushforward_call)
core.push_rules.set(pushforward_call_p, pushforward_pushforward_call)
core.push_rules.aset(pushforward_call_p, apushforward_pushforward_call)
core.pull_fwd_rules.set(pushforward_call_p, pullback_fwd_pushforward_call)
core.pull_fwd_rules.aset(pushforward_call_p, apullback_fwd_pushforward_call)
core.pull_bwd_rules.set(pushforward_call_p, pullback_bwd_pushforward_call)
core.pull_bwd_rules.aset(pushforward_call_p, apullback_bwd_pushforward_call)
core.batch_rules.set(pushforward_call_p, batch_pushforward_call)
core.batch_rules.aset(pushforward_call_p, abatch_pushforward_call)


def dce_pushforward_call(eqn: core.Eqn, out_used: dce.UsedTree, /) -> dce.DCEResult:
    p_used, t_used = out_used
    original_out_used = utils.tree.map(lambda p, t: p or t, p_used, t_used)
    new_eqn = eqn.using(ir=dce.dce(eqn.params["ir"], out_used=original_out_used))
    return dce.default_dce(new_eqn, out_used)


dce.dce_rules[pushforward_call_p] = dce_pushforward_call


# ==================================================================================================
# PULLBACK
# ==================================================================================================

pullback_call_p = core.Prim("pullback_call")


cot_acc_rules: dict[type[core.AVal], Callable[[list[Any], core.AVal], Any]] = {}
cot_acc_rules[core.StrAVal] = lambda cs, _: "".join(cs)
cot_acc_rules[core.IntAVal] = lambda cs, _: sum(cs)
cot_acc_rules[core.FloatAVal] = lambda cs, _: sum(cs)


def cot_acc(cots: list[Any | Zero]) -> Any:
    assert cots
    non_zero = [c for c in cots if not is_zero(c)]
    if not non_zero:
        # NOTE(asem): all output paths into the same input are zero.
        # >>> def f(x):
        # ...     return (x, x)
        # >>> ir = af.trace(f)("...")
        # >>> z = af.ad.Zero(af.core.StrAVal())
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
        # >>> z = af.ad.Zero(af.core.StrAVal())
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
        return utils.tree.map(lambda *cs: cot_acc(list(cs)), *non_zero)
    # NOTE(asem): leaf cotangents use the accumulator registered for their aval.
    # >>> def f(x):
    # ...     return x + x
    # >>> ir = af.trace(f)("...")
    # >>> af.pullback(ir).call(("a",), "df")
    # ('aa', ('dfdf',))
    aval = core.avalof(first)
    if (rule := cot_acc_rules.get(type(aval))) is None:
        raise TypeError(f"No cotangent accumulator registered for {aval!r}")
    return rule(non_zero, aval)


class PullbackFwdBox:
    __slots__ = ["owner", "primal"]

    def __init__(self, owner, primal):
        self.owner = owner
        self.primal = primal


class PullbackFwdInterpreter(core.BoxedInterpreter[PullbackFwdBox]):
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
            p_out, residuals = await core.pull_fwd_rules.aget(prim)(p_in, **params)
        return self.box(p_out), residuals


class PullbackBwdBox:
    __slots__ = ["owner", "cotangent"]

    def __init__(self, owner, cotangent):
        self.owner = owner
        self.cotangent = cotangent


def transpose_walk(ir: core.IR, c_out: Tree, /):
    # NOTE(asem): walk the IR in reverse accumulating cotangents in an environment.
    # used for pullback backward pass.
    c_env: defaultdict[core.Var, list[Any]] = defaultdict(list)

    def write_c(atom, value: Any):
        # NOTE(asem): cotangent contributions are collected by appending them to the
        # same Var entry in `c_env`. accumulation happens when that Var is read.
        # for example:
        # >>> def f(x): return x + x
        # >>> ir = af.trace(f)("...")
        # >>> out, (dx,) = af.pullback(ir).call(("...",), "df")
        # the trace contains `concat(x, x)`. during transpose, the concat pullback
        # returns one cotangent for each concat input. since both inputs are the same
        # Var `x`, `c_env[x]` receives two cotangents: ["df", "df"].
        # `read_c(x)` then calls `cot_acc`, which combines them using
        # the registered `StrAVal` accumulator.
        core.is_irvar(atom) and c_env[atom].append(value)

    def read_c(atom) -> Any:
        if not core.is_irvar(atom):
            return zeroof(atom)
        if not (cs := c_env[atom]):
            return zeroof(atom)
        return cot_acc(cs)

    utils.tree.map(write_c, ir.out_tree, c_out)
    for eqn in reversed(ir.eqns):
        c_out = utils.tree.map(read_c, eqn.out_tree)
        c_in = yield eqn, c_out
        utils.tree.map(write_c, eqn.in_tree, c_in)
    yield None, utils.tree.map(read_c, ir.in_tree)


class PullbackBwdInterpreter(core.BoxedInterpreter[PullbackBwdBox]):
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
            c_in = await core.pull_bwd_rules.aget(prim)((residuals, c_out), **params)
        return self.box(c_in)


@ft.partial(utils.lru_cache, maxsize=256)
def pullback(ir: core.IR, /) -> core.IR:
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
    assert isinstance(ir, core.IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        if core.is_irvar(atom):
            return core.Var.fresh(aval=core.ir_aval(atom), source=atom)
        return atom

    def make_c(atom):
        if core.is_irvar(atom):
            return core.Var.fresh(aval=core.ir_aval(atom), source=atom)
        return zeroof(atom)

    p_in_ir = utils.tree.map(make_p, ir.in_tree)
    c_out_ir = utils.tree.map(make_c, ir.out_tree)
    in_tree = (p_in_ir, c_out_ir)
    p_out_ir = utils.tree.map(make_p, ir.out_tree)
    c_in_ir = utils.tree.map(make_c, ir.in_tree)
    out_tree = (p_out_ir, c_in_ir)
    eqn = core.Eqn(pullback_call_p, in_tree, out_tree, dict(ir=ir))
    return core.IR([eqn], in_tree, out_tree)


def impl_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    (p_in, c_out) = in_tree

    res: dict[core.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    with core.using_interpreter(fwd):

        def custom_bind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            boxed_out, residuals = eqn.bind(boxed_in, **eqn.params)
            res[eqn] = residuals
            return boxed_out

        eqn, boxed_in = next(gen := ir.walk(*fwd.box(p_in)))
        while eqn:
            eqn, boxed_in = gen.send(custom_bind(eqn, boxed_in))

    with core.using_interpreter(bwd):

        def custom_bind(eqn: core.Eqn, c_out: Tree, /) -> Tree:
            residuals = res[eqn]
            boxed_c_out = bwd.box(c_out)
            boxed_c_in = eqn.bind((residuals, boxed_c_out), **eqn.params)
            return bwd.unbox(boxed_c_in)

        eqn, c_out = next(gen := transpose_walk(ir, c_out))
        while eqn:
            eqn, c_out = gen.send(custom_bind(eqn, c_out))

    return fwd.unbox(boxed_in), c_out


async def aimpl_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    (p_in, c_out) = in_tree

    res: dict[core.Eqn, Tree] = {}
    parent = core.active_interpreter.get()
    fwd = PullbackFwdInterpreter(parent=parent)
    bwd = PullbackBwdInterpreter(parent=parent)

    with core.using_interpreter(fwd):

        async def custom_abind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            boxed_out, residuals = await eqn.abind(boxed_in, **eqn.params)
            res[eqn] = residuals
            return boxed_out

        eqn, boxed_in = next(gen := ir.walk(*fwd.box(p_in)))
        while eqn:
            eqn, boxed_in = gen.send(await custom_abind(eqn, boxed_in))

    with core.using_interpreter(bwd):

        async def custom_abind(eqn: core.Eqn, c_out: Tree, /) -> Tree:
            residuals = res[eqn]
            boxed_c_out = bwd.box(c_out)
            boxed_c_in = await eqn.abind((residuals, boxed_c_out), **eqn.params)
            return bwd.unbox(boxed_c_in)

        eqn, c_out = next(gen := transpose_walk(ir, c_out))
        while eqn:
            eqn, c_out = gen.send(await custom_abind(eqn, c_out))

    return fwd.unbox(boxed_in), c_out


def abstract_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    p_out = utils.tree.map(core.ir_aval, ir.out_tree)
    c_in = utils.tree.map(core.ir_aval, ir.in_tree)
    return p_out, c_in


def pushforward_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    p, t = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = p, t
    pb_ir = pullback(ir)
    p_out, c_in = pb_ir.call(p_in, c_out)
    t_p_out, t_c_in = pb_ir.call(t_p_in, t_c_out)
    return (p_out, c_in), (t_p_out, t_c_in)


async def apushforward_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    p, t = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = p, t
    pb_ir = pullback(ir)
    p_out, c_in = await pb_ir.acall(p_in, c_out)
    t_p_out, t_c_in = await pb_ir.acall(t_p_in, t_c_out)
    return (p_out, c_in), (t_p_out, t_c_in)


def pullback_fwd_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    p_out, c_in = pb_ir.call(p_in, c_out)
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


async def apullback_fwd_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    p_out, c_in = await pb_ir.acall(p_in, c_out)
    residuals = (p_in, c_out, p_out, c_in)
    return (p_out, c_in), residuals


def pullback_bwd_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> Tree:
    residuals, c = in_tree
    p_in, _, _, _ = residuals
    c_p_out, c_c_in = c
    pb_ir = pullback(ir)
    _, c_p_in = pb_ir.call(p_in, c_p_out)
    pf_ir = pushforward(ir)
    _, c_c_out = pf_ir.call(p_in, c_c_in)
    return (c_p_in, c_c_out)


async def apullback_bwd_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> Tree:
    residuals, c = in_tree
    p_in, _, _, _ = residuals
    c_p_out, c_c_in = c
    pb_ir = pullback(ir)
    _, c_p_in = await pb_ir.acall(p_in, c_p_out)
    pf_ir = pushforward(ir)
    _, c_c_out = await pf_ir.acall(p_in, c_c_in)
    return (c_p_in, c_c_out)


def batch_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    size, in_batched, in_values = in_tree
    (p_cols, c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if utils.batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = pb_ir.call(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(utils.batch_index, c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = [pb_ir.call(unbatch_p(b), unbatch_c(b)) for b in range(size)]
    out_batched = utils.tree.map(lambda _: True, pb_ir.out_tree)
    out_ib = utils.batch_transpose(size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pullback_call(in_tree: Tree, /, *, ir: core.IR) -> TreePair:
    size, in_batched, in_values = in_tree
    (p_cols, c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if utils.batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = await pb_ir.acall(*in_values)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(utils.batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(utils.batch_index, c_cols, c_batched)
    pb_ir = pullback(ir)

    inputs = [(unbatch_p(b), unbatch_c(b)) for b in range(size)]
    out_bi = await scheduling.gather_p.abind(inputs, irs=[pb_ir] * size)
    out_batched = utils.tree.map(lambda _: True, pb_ir.out_tree)
    out_ib = utils.batch_transpose(size, out_batched, list(out_bi))
    return out_ib, out_batched


core.impl_rules.set(pullback_call_p, impl_pullback_call)
core.impl_rules.aset(pullback_call_p, aimpl_pullback_call)
core.abstract_rules.set(pullback_call_p, abstract_pullback_call)
core.push_rules.set(pullback_call_p, pushforward_pullback_call)
core.push_rules.aset(pullback_call_p, apushforward_pullback_call)
core.pull_fwd_rules.set(pullback_call_p, pullback_fwd_pullback_call)
core.pull_fwd_rules.aset(pullback_call_p, apullback_fwd_pullback_call)
core.pull_bwd_rules.set(pullback_call_p, pullback_bwd_pullback_call)
core.pull_bwd_rules.aset(pullback_call_p, apullback_bwd_pullback_call)
core.batch_rules.set(pullback_call_p, batch_pullback_call)
core.batch_rules.aset(pullback_call_p, abatch_pullback_call)


def dce_pullback_call(eqn: core.Eqn, out_used: dce.UsedTree, /) -> dce.DCEResult:
    out, in_cot = out_used
    used = utils.tree.any(in_cot)
    inner_ir = eqn.params["ir"] if used else dce.dce(eqn.params["ir"], out_used=out)
    new_eqn = eqn.using(ir=inner_ir)
    return dce.default_dce(new_eqn, out_used)


dce.dce_rules[pullback_call_p] = dce_pullback_call
