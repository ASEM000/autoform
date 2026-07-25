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

import autoform.core as core
import autoform.utils as utils

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


class PushforwardBox:
    __slots__ = ["owner", "primal", "tangent"]

    def __init__(self, owner, primal, tangent):
        self.owner = owner
        self.primal = primal
        self.tangent = tangent


core.aval_rules[PushforwardBox] = lambda box: core.avalof(box.primal)


def pair(owner, primals: Tree, tangents: Tree, /) -> Tree:
    return utils.tree.map(lambda p, t: PushforwardBox(owner, p, t), primals, tangents)


def unpair(owner, values: Tree, /) -> TreePair:
    # NOTE(asem): ownership keeps boxes from other transform instances opaque.
    def primal(value):
        if isinstance(value, PushforwardBox) and value.owner is owner:
            return value.primal
        return value

    def tangent(value):
        if isinstance(value, PushforwardBox) and value.owner is owner:
            return value.tangent
        return zeroof(value)

    return utils.tree.map(primal, values), utils.tree.map(tangent, values)


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
        if core.is_var(atom):
            return core.Var.fresh(aval=core.aval_if_var(atom), source=atom)
        return atom

    def make_t(atom):
        if core.is_var(atom):
            return core.Var.fresh(aval=core.aval_if_var(atom), source=atom)
        return zeroof(atom)

    p_in_ir = utils.tree.map(make_p, ir.in_tree)
    t_in_ir = utils.tree.map(make_t, ir.in_tree)
    in_tree = (p_in_ir, t_in_ir)

    with core.using_interpreter(core.TraceInterpreter()) as tracer:
        boxed_p_in, boxed_t_in = tracer.box(in_tree)

        def custom_bind(eqn: core.Eqn, paired_in: Tree, /) -> Tree:
            p_in, t_in = unpair(tracer, paired_in)
            with core.tag(*eqn.tags):
                if all_zero(t_in):
                    p_out = eqn.bind(p_in, **eqn.params)
                    t_out = utils.tree.map(zeroof, p_out)
                else:
                    p_out, t_out = core.push_rules.get(eqn.prim)((p_in, t_in), **eqn.params)
            return pair(tracer, p_out, t_out)

        eqn, paired_out = next(gen := ir.walk(*pair(tracer, boxed_p_in, boxed_t_in)))
        while eqn:
            eqn, paired_out = gen.send(custom_bind(eqn, paired_out))

    p_out, t_out = unpair(tracer, paired_out)
    return core.IR(tracer.eqns, in_tree, tracer.unbox((p_out, t_out)))


# ==================================================================================================
# PULLBACK
# ==================================================================================================

acc_p = core.Prim("cotangent_accumulate")


cot_acc_rules: dict[type[core.AVal], Callable[[list[Any], core.AVal], Any]] = {}
cot_acc_rules[core.StrAVal] = lambda cs, _: "".join(cs)
cot_acc_rules[core.IntAVal] = lambda cs, _: sum(cs)
cot_acc_rules[core.FloatAVal] = lambda cs, _: sum(cs)


def cot_acc(cots: list[Any | Zero], /, *, aval: core.AVal | None = None) -> Any:
    assert cots
    assert aval is None or isinstance(aval, core.AVal), f"Expected AVal, got {aval!r}"
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
        assert aval is None or core.avalof(first_zero) == aval
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
    accumulator_aval = core.avalof(first) if aval is None else aval
    traced = next((cot for cot in non_zero if isinstance(cot, core.TraceBox)), None)
    if traced is not None:
        with core.using_interpreter(traced.owner):
            return acc_p.bind(non_zero, aval=accumulator_aval)
    if (rule := cot_acc_rules.get(type(accumulator_aval))) is None:
        raise TypeError(f"No cotangent accumulator registered for {accumulator_aval!r}")
    return rule(non_zero, accumulator_aval)


def impl_acc(cots: list[Any | Zero], /, *, aval: core.AVal) -> Any:
    return cot_acc(cots, aval=aval)


def abstract_acc(cots: list[Any], /, *, aval: core.AVal) -> core.AVal:
    assert isinstance(aval, core.AVal), f"Expected AVal, got {aval!r}"
    for cot in cots:
        cot_aval = cot if isinstance(cot, core.AVal) else core.avalof(cot)
        assert cot_aval == aval, f"Expected cotangent aval {aval!r}, got {cot_aval!r}"
    return aval


core.impl_rules.set(acc_p, impl_acc)
core.impl_rules.aset(acc_p, utils.asyncify(impl_acc))
core.abstract_rules.set(acc_p, abstract_acc)


def push_acc(in_tree: TreePair, /, *, aval: core.AVal) -> TreePair:
    primals, tangents = in_tree
    return acc_p.bind(primals, aval=aval), acc_p.bind(tangents, aval=aval)


def pull_fwd_acc(cots: list[Any], /, *, aval: core.AVal) -> TreePair:
    return acc_p.bind(cots, aval=aval), len(cots)


def pull_bwd_acc(in_tree: Tree, /, *, aval: core.AVal) -> list[Any]:
    del aval
    count, cotangent = in_tree
    return [cotangent] * count


def batch_acc(in_tree: Tree, /, *, aval: core.AVal) -> TreePair:
    batch_size, in_batched, cots = in_tree
    if (spec := utils.batch_spec(cots, in_batched)) is None:
        return acc_p.bind(cots, aval=aval), False
    unbatch = ft.partial(utils.batch_index, cots, in_batched)
    results = [acc_p.bind(unbatch(b), aval=aval) for b in range(batch_size)]
    return spec.unflatten(results), True


core.push_rules.set(acc_p, push_acc)
core.push_rules.aset(acc_p, utils.asyncify(push_acc))
core.pull_fwd_rules.set(acc_p, pull_fwd_acc)
core.pull_fwd_rules.aset(acc_p, utils.asyncify(pull_fwd_acc))
core.pull_bwd_rules.set(acc_p, pull_bwd_acc)
core.pull_bwd_rules.aset(acc_p, utils.asyncify(pull_bwd_acc))
core.batch_rules.set(acc_p, batch_acc)
core.batch_rules.aset(acc_p, utils.asyncify(batch_acc))


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
        core.is_var(atom) and c_env[atom].append(value)

    def read_c(atom) -> Any:
        if not core.is_var(atom):
            return zeroof(atom)
        if not (cs := c_env[atom]):
            return zeroof(atom)
        return cot_acc(cs, aval=atom.aval)

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
        if core.is_var(atom):
            return core.Var.fresh(aval=core.aval_if_var(atom), source=atom)
        return atom

    def make_c(atom):
        if core.is_var(atom):
            return core.Var.fresh(aval=core.aval_if_var(atom), source=atom)
        return zeroof(atom)

    p_in_ir = utils.tree.map(make_p, ir.in_tree)
    c_out_ir = utils.tree.map(make_c, ir.out_tree)
    in_tree = (p_in_ir, c_out_ir)
    res: dict[core.Eqn, Tree] = {}

    with core.using_interpreter(core.TraceInterpreter()) as tracer:
        boxed_p_in, boxed_c_out = tracer.box(in_tree)

        def custom_fwd_bind(eqn: core.Eqn, p_in: Tree, /) -> Tree:
            with core.tag(*eqn.tags):
                p_out, residuals = core.pull_fwd_rules.get(eqn.prim)(p_in, **eqn.params)
            res[eqn] = residuals
            return p_out

        eqn, p_out = next(gen := ir.walk(*boxed_p_in))
        while eqn:
            eqn, p_out = gen.send(custom_fwd_bind(eqn, p_out))

        def custom_bwd_bind(eqn: core.Eqn, c_out: Tree, /) -> Tree:
            residuals = res[eqn]
            with core.tag(*eqn.tags):
                return core.pull_bwd_rules.get(eqn.prim)(
                    (residuals, c_out),
                    **eqn.params,
                )

        eqn, c_in = next(gen := transpose_walk(ir, boxed_c_out))
        while eqn:
            eqn, c_in = gen.send(custom_bwd_bind(eqn, c_in))

    return core.IR(tracer.eqns, in_tree, tracer.unbox((p_out, c_in)))
