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

"""Axis-based batch transformation."""

from __future__ import annotations

import functools as ft
from typing import Any

import autoform.src.ad as ad
import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.order as order
import autoform.src.pp as pp
import autoform.src.stage as stage
import autoform.src.tree as tree
import autoform.src.utils as utils

type Tree[T] = Any


zip = utils.strict_zip

type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# BATCH
# ==================================================================================================


def common_spec(value: Tree, in_axes: Tree[bool], /) -> tree.PyTreeSpec | None:
    shape = tree.structure(in_axes)
    values = shape.flatten_up_to(value)
    flags = tree.leaves(in_axes)
    specs = [
        tree.structure(x, is_leaf=lambda node: node is not x)
        for x, use in zip(values, flags)
        if use
    ]
    if not specs:
        return None
    spec, *rest = specs
    assert all(spec == other for other in rest), "Mismatched container types among batched leaves."
    return spec


class ABatch(core.AVal):
    # NOTE(asem): no aval_type rule can infer ABatch from a concrete container
    # as containers are later introduced at the call site. unlike jax the atomic unit is not
    # the array object but any thing really.
    def __init__(self, base: core.AVal):
        # TODO(asem): maybe exapand with useful metadata here

        assert isinstance(base, core.AVal), f"Expected AVal, got {base!r}"
        self.base = base

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.base == other.base

    def __hash__(self) -> int:
        return hash((type(self), self.base))

    def join(self, other, /):
        match other:
            case ABatch(base=base) | core.Zero(aval=ABatch(base=base)):
                return type(self)(self.base.join(base))
            case _ if not tree.is_leaf(other):
                base = self.base
                for x in tree.leaves(other, is_leaf=lambda x: x is not other):
                    base = base.join(x)
                return type(self)(base)
        raise TypeError(f"Cannot join {self!r} and {other!r}")


@ft.partial(pp.register, ABatch)
def pretty_batch(aval):
    name = pp.style(type(aval).__name__, pp.Theme.TYPE)
    return name, pp.bracket("(", [pp.pretty(aval.base)], ")")


stage.register_trace_type(ABatch, lambda aval: aval)
core.s_primal.set(ABatch, lambda aval: ABatch(core.s_primal.map(aval.base)))
core.s_tangent.set(ABatch, lambda aval: ABatch(core.s_tangent.map(aval.base)))
core.s_cotangent.set(ABatch, lambda aval: ABatch(core.s_cotangent.map(aval.base)))


def is_axis_spec(v) -> bool:
    return isinstance(v, bool)


def assert_trees(out_axes: Tree, out_ir: Tree, prim_name: str) -> Tree:
    expected_axes = tree.map(lambda _: False, out_ir)
    is_bool_leaf = lambda v: isinstance(v, bool)
    out_spec = tree.structure(out_axes, is_leaf=is_bool_leaf)
    expected_spec = tree.structure(expected_axes, is_leaf=is_bool_leaf)
    if out_spec != expected_spec:
        raise ValueError(
            f"Primitive '{prim_name}' batch_rule returned out_axes with structure {out_spec}, "
            f"but expected structure {expected_spec} to match output. "
            f"out_axes must match the structure of the output exactly."
        )
    return out_axes


def broadcast_batch_out(spec, out_v: Tree, out_axes: Tree[bool], /) -> Tree:
    batch_size = spec.num_leaves
    out_spec = tree.structure(out_axes, is_leaf=is_axis_spec)
    out_flat = out_spec.flatten_up_to(out_v)
    flat_out_axes = tree.leaves(out_axes, is_leaf=is_axis_spec)

    def broadcast_leaf(v, b):
        return v if b else spec.unflatten([v] * batch_size)

    return out_spec.unflatten(map(broadcast_leaf, out_flat, flat_out_axes))


def unbatch_zeros(in_v: Tree, in_axes: Tree[bool], /) -> TreePair:
    def is_batched_zero(b, v):
        return b and isinstance(v, core.Zero) and isinstance(core.avalof(v), ABatch)

    is_zero = tree.map(is_batched_zero, in_axes, in_v)
    out_v = tree.map(lambda z, v: core.Zero(v.aval.base) if z else v, is_zero, in_v)
    out_axes = tree.map(lambda z, b: b and not z, is_zero, in_axes)
    return out_v, out_axes


p_batch_call = core.Prim("batch_call")


def batch(ir: stage.IR, /, *, in_axes: Tree[bool] = True) -> stage.IR:
    """Transform an IR to process batched inputs.

    Creates a batched version of the IR that processes multiple inputs
    simultaneously. Use `in_axes` to specify which inputs are batched
    (True) vs broadcast (False).

    Args:
        ir: The IR to transform.
        in_axes: Axis specification tree matching input structure.
            - True: This input is batched (a collection of values).
            - False: This input is broadcast (same value for all batch items).

    Returns:
        A new IR that takes batched inputs and returns batched outputs.

    Example:
        >>> import autoform as af
        >>> def greet(greeting, name):
        ...     return greeting + name
        >>> ir = af.trace(greet)("Hi", "World")
        >>> # Batch over names, broadcast greeting
        >>> batched = af.batch(ir, in_axes=(False, True))
        >>> batched.call("Hello, ", ["x0", "x1", "x2"])
        ['Hello, x0', 'Hello, x1', 'Hello, x2']
    """
    assert isinstance(ir, stage.IR), f"Expected IR, got {type(ir)}"
    expanded_in_axes = tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    has_batched = any(tree.leaves(expanded_in_axes, is_leaf=is_axis_spec))

    def maybe_batched(aval, is_batched: bool):
        return ABatch(aval) if is_batched else aval

    def make_in(atom, is_batched: bool):
        if not stage.is_var(atom):
            return atom
        return stage.Var.fresh(aval=maybe_batched(atom.aval, is_batched), source=atom)

    def make_out(atom):
        if stage.is_var(atom):
            return stage.Var.fresh(aval=maybe_batched(atom.aval, has_batched), source=atom)
        if has_batched:
            return stage.Var.fresh(aval=maybe_batched(core.avalof(atom), True))
        return atom

    in_v_ir = tree.map(make_in, ir.in_tree, expanded_in_axes)
    out_v_ir = tree.map(make_out, ir.out_tree)
    eqn = stage.Eqn(p_batch_call, in_v_ir, out_v_ir, dict(ir=ir, in_axes=in_axes))
    return stage.IR([eqn], in_v_ir, out_v_ir)


class BatchBox:
    __slots__ = ["owner", "value", "batched"]

    def __init__(self, owner, value, batched):
        self.owner = owner
        self.value = value
        self.batched = batched


def avalof_batch_box(box: BatchBox, /) -> core.AVal:
    if not box.batched:
        return core.avalof(box.value)
    if tree.is_leaf(box.value):
        raise TypeError("Expected a batch container to infer BatchBox aval")
    if not (items := tree.leaves(box.value, is_leaf=lambda x: x is not box.value)):
        raise TypeError("Cannot infer BatchBox aval from an empty batch")
    item0, *rest = items
    aval = core.avalof(item0)
    if any(core.avalof(item) != aval for item in rest):
        raise TypeError("Cannot infer BatchBox aval from items with different avals")
    return aval


core.aval_types[BatchBox] = avalof_batch_box


class BatchInterp(core.Interp):
    __slots__ = ["parent", "spec"]

    def __init__(self, *, spec: tree.PyTreeSpec, parent):
        self.parent = parent
        self.spec = spec

    def box(self, value, /) -> Tree:
        v, b = value
        # NOTE(asem): ``b`` is a prefix spec, not necessarily the same structure
        # as ``v``. For example, v=["a", "b"] and b=True means the whole list is
        # one batched leaf, not two leaves.
        spec = tree.structure(b, is_leaf=is_axis_spec)
        v = spec.flatten_up_to(v)
        b = tree.leaves(b, is_leaf=is_axis_spec)
        return spec.unflatten(BatchBox(self, v, b) for v, b in zip(v, b))

    def unbox(self, v: Tree, /) -> TreePair:
        def value(v):
            return v.value if isinstance(v, BatchBox) and v.owner is self else v

        def batched(v):
            return v.batched if isinstance(v, BatchBox) and v.owner is self else False

        return tree.map(value, v), tree.map(batched, v)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        in_v, in_axes = self.unbox(in_tree)
        with core.using_interp(self.parent):
            if not any(tree.leaves(in_axes)):
                return prim.bind(in_v, **params)
            out_v, out_axes = core.r_batch.get(prim)((self.spec, in_axes, in_v), **params)
        return self.box((out_v, out_axes))

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        # NOTE(asem): async batch rules must be explicitly seted - no fallback to sync.
        in_v, in_axes = self.unbox(in_tree)
        with core.using_interp(self.parent):
            if not any(tree.leaves(in_axes)):
                return await prim.abind(in_v, **params)
            out_v, out_axes = await core.r_abatch.get(prim)((self.spec, in_axes, in_v), **params)
        return self.box((out_v, out_axes))


class BatchEnv:
    __slots__ = ["values", "batched"]

    def __init__(self):
        self.values = {}
        self.batched = {}

    def read_v(self, atom, /):
        if not stage.is_var(atom):
            return atom
        value = self.values[atom]
        aval = ABatch(atom.aval) if self.batched[atom] else atom.aval
        stage.no_stage_typecheck(value, aval)
        return value

    def read_b(self, atom, /):
        return self.batched[atom] if stage.is_var(atom) else False

    def write_v(self, atom, value, /):
        if stage.is_var(atom):
            aval = ABatch(atom.aval) if self.batched[atom] else atom.aval
            stage.no_stage_typecheck(value, aval)
            self.values[atom] = value

    def write_b(self, atom, is_batched, /):
        if stage.is_var(atom):
            self.batched[atom] = is_batched


def impl_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    # NOTE(asem): ``in_axes`` only marks which leaves are batched.
    # the actual batch container comes from runtime data.
    # >>> in_tree = ReviewState(code=["a", "b"], has_bugs=[True, False])
    # >>> in_axes = True
    # >>> in_axes = ReviewState(code=True, has_bugs=True)
    # >>> batch_size = 2
    in_v = in_tree
    in_axes = tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)

    if not any(tree.leaves(in_axes)):
        return ir.call(*in_v)

    in_v, in_axes = unbatch_zeros(in_v, in_axes)
    spec = common_spec(in_v, in_axes)
    if spec is None:
        raise TypeError("Cannot infer batch layout from symbolic zeros alone")

    batch_size = spec.num_leaves
    # NOTE(asem): this case can be something like
    # >>> def program(v):
    # ...     return "constant string"
    # >>> ir = af.trace(program)("input")
    # >>> batched = af.batch(ir, in_axes=True)
    # >>> batched.call([])
    assert batch_size, "batch size must be > 0"

    batcher = BatchInterp(spec=spec, parent=core.active_interp.get())

    env = BatchEnv()

    def batch_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        in_v, in_axes = in_tree
        if not any(tree.leaves(in_axes)):
            return eqn.bind(in_v, **eqn.params), tree.map(lambda _: False, eqn.out_tree)
        with core.using_interp(batcher):
            out_boxed = eqn.bind(batcher.box(in_tree), **eqn.params)
        out_v, out_axes = batcher.unbox(out_boxed)
        return out_v, assert_trees(out_axes, eqn.out_tree, eqn.prim.name)

    tree.map(env.write_b, ir.in_tree, in_axes)
    tree.map(env.write_v, ir.in_tree, in_v)
    for eqn in ir.eqns:
        in_v = tree.map(env.read_v, eqn.in_tree)
        in_axes = tree.map(env.read_b, eqn.in_tree)
        out_v, out_axes = batch_bind(eqn, (in_v, in_axes))
        tree.map(env.write_b, eqn.out_tree, out_axes)
        tree.map(env.write_v, eqn.out_tree, out_v)
    out_v = tree.map(env.read_v, ir.out_tree)
    out_axes = tree.map(env.read_b, ir.out_tree)
    return broadcast_batch_out(spec, out_v, out_axes)


async def aimpl_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    in_v = in_tree
    in_axes = tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)

    if not any(tree.leaves(in_axes)):
        return await ir.acall(*in_v)

    in_v, in_axes = unbatch_zeros(in_v, in_axes)
    spec = common_spec(in_v, in_axes)
    if spec is None:
        raise TypeError("Cannot infer batch layout from symbolic zeros alone")

    batch_size = spec.num_leaves
    assert batch_size, "batch size must be > 0"

    batcher = BatchInterp(spec=spec, parent=core.active_interp.get())

    env = BatchEnv()

    async def batch_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        in_v, in_axes = in_tree
        if not any(tree.leaves(in_axes)):
            out_v = await eqn.abind(in_v, **eqn.params)
            return out_v, tree.map(lambda _: False, eqn.out_tree)
        with core.using_interp(batcher):
            out_boxed = await eqn.abind(batcher.box(in_tree), **eqn.params)
        out_v, out_axes = batcher.unbox(out_boxed)
        return out_v, assert_trees(out_axes, eqn.out_tree, eqn.prim.name)

    tree.map(env.write_b, ir.in_tree, in_axes)
    tree.map(env.write_v, ir.in_tree, in_v)
    for eqn in ir.eqns:
        in_v = tree.map(env.read_v, eqn.in_tree)
        in_axes = tree.map(env.read_b, eqn.in_tree)
        out_v, out_axes = await batch_bind(eqn, (in_v, in_axes))
        tree.map(env.write_b, eqn.out_tree, out_axes)
        tree.map(env.write_v, eqn.out_tree, out_v)
    out_v = tree.map(env.read_v, ir.out_tree)
    out_axes = tree.map(env.read_b, ir.out_tree)
    return broadcast_batch_out(spec, out_v, out_axes)


def abstract_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    del in_tree
    in_axes = tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    has_batched = any(tree.leaves(in_axes, is_leaf=is_axis_spec))

    def maybe_batched(aval, is_batched: bool):
        return ABatch(aval) if is_batched else aval

    def out_aval(atom):
        if stage.is_var(atom):
            return maybe_batched(atom.aval, has_batched)
        if has_batched:
            return maybe_batched(core.avalof(atom), True)
        return atom

    return tree.map(out_aval, ir.out_tree)


def pushforward_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    p, t = in_tree
    pf_ir = ad.pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return batch_pf_ir.call(p, t)


async def apushforward_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    p, t = in_tree
    pf_ir = ad.pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return await batch_pf_ir.acall(p, t)


def pullback_fwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    fwd, _ = ad.fwd_bwd(ir)
    return batch(fwd, in_axes=in_axes).call(*in_tree)


async def apullback_fwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    fwd, _ = ad.fwd_bwd(ir)
    return await batch(fwd, in_axes=in_axes).acall(*in_tree)


def pullback_bwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    residual, out_c = in_tree
    in_axes = tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    _, bwd = ad.fwd_bwd(ir)
    if not any(tree.leaves(in_axes)):
        return bwd.call(residual, out_c)
    spec = common_spec(residual, True)
    in_c = batch(bwd, in_axes=(True, True)).call(residual, out_c)

    def accum(batched, c):
        return c if batched else ad.cot_accum(spec.flatten_up_to(c))

    return tree.map(accum, in_axes, in_c)


async def apullback_bwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    residual, out_c = in_tree
    in_axes = tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    _, bwd = ad.fwd_bwd(ir)
    if not any(tree.leaves(in_axes)):
        return await bwd.acall(residual, out_c)
    spec = common_spec(residual, True)
    in_c = await batch(bwd, in_axes=(True, True)).acall(residual, out_c)

    def accum(batched, c):
        return c if batched else ad.cot_accum(spec.flatten_up_to(c))

    return tree.map(accum, in_axes, in_c)


def batch_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    spec, outer_in_axes, in_v = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    items = tree.unstack(spec, outer_in_axes, in_v)
    results = (batched_ir.call(*x) for x in items)
    out_axes = tree.map(lambda _: True, ir.out_tree)
    return tree.stack(spec, out_axes, results), out_axes


async def abatch_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    spec, outer_in_axes, in_v = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    inputs = tree.unstack(spec, outer_in_axes, in_v)
    results = await order.p_fanout.abind(inputs, irs=[batched_ir] * len(inputs))
    out_axes = tree.map(lambda _: True, ir.out_tree)
    return tree.stack(spec, out_axes, results), out_axes


def dce_batch_call(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    new_eqn = eqn.using(ir=dead.dce(eqn.params["ir"], out_used=out_used))
    return dead.default_dce(new_eqn, out_used)


core.r_impl.set(p_batch_call, impl_batch_call)
core.r_aimpl.set(p_batch_call, aimpl_batch_call)
core.r_abstract.set(p_batch_call, abstract_batch_call)
core.r_batch.set(p_batch_call, batch_batch_call)
core.r_abatch.set(p_batch_call, abatch_batch_call)
core.r_push.set(p_batch_call, pushforward_batch_call)
core.r_apush.set(p_batch_call, apushforward_batch_call)
core.r_pull_fwd.set(p_batch_call, pullback_fwd_batch_call)
core.r_apull_fwd.set(p_batch_call, apullback_fwd_batch_call)
core.r_pull_bwd.set(p_batch_call, pullback_bwd_batch_call)
core.r_apull_bwd.set(p_batch_call, apullback_bwd_batch_call)
dead.register_dce(p_batch_call, dce_batch_call)
