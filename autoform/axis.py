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

import autoform.ad as ad
import autoform.core as core
import autoform.dead as dead
import autoform.order as order
import autoform.stage as stage
import autoform.utils as utils

type Tree[T] = utils.Tree[T]


__all__ = ["batch"]

zip = utils.strict_zip

type TreePair = tuple[Tree, Tree]


# ==================================================================================================
# BATCH
# ==================================================================================================


class BatchAVal(core.AVal):
    # NOTE(asem): no aval_type rule can infer BatchAVal from a concrete container
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

    def check(self, value, /) -> None:
        if utils.tree.is_leaf(value):
            actual = core.avalof(value)
            if not isinstance(actual, type(self)):
                raise TypeError(f"Expected {self!r}, got {actual!r}")
            self.base.check(actual.base)
            return
        utils.tree.map(self.base.check, value, is_leaf=lambda x: x is not value)


core.aval_types[BatchAVal] = lambda aval: aval
core.primal_s.set(BatchAVal, lambda aval: BatchAVal(core.primal_s.map(aval.base)))
core.tangent_s.set(BatchAVal, lambda aval: BatchAVal(core.tangent_s.map(aval.base)))
core.cotangent_s.set(BatchAVal, lambda aval: BatchAVal(core.cotangent_s.map(aval.base)))


def is_axis_spec(v) -> bool:
    return isinstance(v, bool)


def assert_trees(b: Tree, out_ir: Tree, prim_name: str) -> Tree:
    expected_b = utils.tree.map(lambda _: False, out_ir)
    is_bool_leaf = lambda v: isinstance(v, bool)
    b_spec = utils.tree.structure(b, is_leaf=is_bool_leaf)
    expected_spec = utils.tree.structure(expected_b, is_leaf=is_bool_leaf)
    if b_spec != expected_spec:
        raise ValueError(
            f"Primitive '{prim_name}' batch_rule returned out_batched with structure {b_spec}, "
            f"but expected structure {expected_spec} to match output. "
            f"out_batched must match the structure of the output exactly."
        )
    return b


def broadcast_batch_out(spec, v_out: Tree, b_out: Tree[bool], /) -> Tree:
    batch_size = spec.num_children
    out_spec = utils.tree.structure(b_out, is_leaf=is_axis_spec)
    flat_out = out_spec.flatten_up_to(v_out)
    flat_b_out = utils.tree.leaves(b_out, is_leaf=is_axis_spec)

    def broadcast_leaf(v, b):
        return v if b else spec.unflatten([v] * batch_size)

    return out_spec.unflatten(map(broadcast_leaf, flat_out, flat_b_out))


def unbatch_zeros(v_in: Tree, b_in: Tree[bool], /) -> TreePair:
    def is_batched_zero(b, v):
        return b and isinstance(v, core.Zero) and isinstance(core.avalof(v), BatchAVal)

    is_zero = utils.tree.map(is_batched_zero, b_in, v_in)
    v_out = utils.tree.map(lambda z, v: core.Zero(v.aval.base) if z else v, is_zero, v_in)
    b_out = utils.tree.map(lambda z, b: b and not z, is_zero, b_in)
    return v_out, b_out


batch_call_p = core.Prim("batch_call")


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
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    has_batched = any(utils.tree.leaves(b_in, is_leaf=is_axis_spec))

    def maybe_batched(aval, is_batched: bool):
        return BatchAVal(aval) if is_batched else aval

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

    v_in_ir = utils.tree.map(make_in, ir.in_tree, b_in)
    v_out_ir = utils.tree.map(make_out, ir.out_tree)
    eqn = stage.Eqn(batch_call_p, v_in_ir, v_out_ir, dict(ir=ir, in_axes=in_axes))
    return stage.IR([eqn], v_in_ir, v_out_ir)


class BatchBox:
    __slots__ = ["owner", "value", "batched"]

    def __init__(self, owner, value, batched):
        self.owner = owner
        self.value = value
        self.batched = batched


def avalof_batch_box(box: BatchBox, /) -> core.AVal:
    if not box.batched:
        return core.avalof(box.value)
    if utils.tree.is_leaf(box.value):
        raise TypeError("Expected a batch container to infer BatchBox aval")
    if not (items := utils.tree.leaves(box.value, is_leaf=lambda x: x is not box.value)):
        raise TypeError("Cannot infer BatchBox aval from an empty batch")
    item0, *rest = items
    aval = core.avalof(item0)
    if any(core.avalof(item) != aval for item in rest):
        raise TypeError("Cannot infer BatchBox aval from items with different avals")
    return aval


core.aval_types[BatchBox] = avalof_batch_box


class BatchInterpreter(core.Interpreter):
    __slots__ = ["parent", "batch_size"]

    def __init__(self, *, batch_size: int, parent):
        self.parent = parent
        self.batch_size = batch_size

    def box(self, value, /) -> Tree:
        v, b = value
        # NOTE(asem): ``b`` is a prefix spec, not necessarily the same structure
        # as ``v``. For example, v=["a", "b"] and b=True means the whole list is
        # one batched leaf, not two leaves.
        spec = utils.tree.structure(b, is_leaf=is_axis_spec)
        v = spec.flatten_up_to(v)
        b = utils.tree.leaves(b, is_leaf=is_axis_spec)
        return spec.unflatten(BatchBox(self, v, b) for v, b in zip(v, b))

    def unbox(self, v: Tree, /) -> TreePair:
        def value(v):
            return v.value if isinstance(v, BatchBox) and v.owner is self else v

        def batched(v):
            return v.batched if isinstance(v, BatchBox) and v.owner is self else False

        return utils.tree.map(value, v), utils.tree.map(batched, v)

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        v_in, b_in = self.unbox(in_tree)
        b_sz = self.batch_size
        with core.using_interpreter(self.parent):
            v_out, b_out = core.batch_rules.get(prim)((b_sz, b_in, v_in), **params)
        return self.box((v_out, b_out))

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        # NOTE(asem): async batch rules must be explicitly seted - no fallback to sync.
        v_in, b_in = self.unbox(in_tree)
        b_sz = self.batch_size
        with core.using_interpreter(self.parent):
            v_out, b_out = await core.abatch_rules.get(prim)((b_sz, b_in, v_in), **params)
        return self.box((v_out, b_out))


class BatchEnv:
    __slots__ = ["values", "batched"]

    def __init__(self):
        self.values = {}
        self.batched = {}

    def read_v(self, atom, /):
        if not stage.is_var(atom):
            return atom
        value = self.values[atom]
        aval = BatchAVal(atom.aval) if self.batched[atom] else atom.aval
        stage.no_stage_typecheck(value, aval)
        return value

    def read_b(self, atom, /):
        return self.batched[atom] if stage.is_var(atom) else False

    def write(self, atom, value, is_batched, /):
        if stage.is_var(atom):
            aval = BatchAVal(atom.aval) if is_batched else atom.aval
            stage.no_stage_typecheck(value, aval)
            self.values[atom] = value
            self.batched[atom] = is_batched


def impl_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    # NOTE(asem): ``in_axes`` only marks which leaves are batched.
    # the actual batch container comes from runtime data.
    # >>> in_tree = ReviewState(code=["a", "b"], has_bugs=[True, False])
    # >>> in_axes = True
    # >>> b_in = ReviewState(code=True, has_bugs=True)
    # >>> batch_size = 2
    v_in = in_tree
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)

    if not any(utils.tree.leaves(b_in)):
        return ir.call(*v_in)

    v_in, b_in = unbatch_zeros(v_in, b_in)
    spec = utils.batch_spec(v_in, b_in)
    if spec is None:
        raise TypeError("Cannot infer batch layout from symbolic zeros alone")

    batch_size = spec.num_children
    # NOTE(asem): this case can be something like
    # >>> def program(v):
    # ...     return "constant string"
    # >>> ir = af.trace(program)("input")
    # >>> batched = af.batch(ir, in_axes=True)
    # >>> batched.call([])
    assert batch_size, "batch size must be > 0"

    batcher = BatchInterpreter(batch_size=batch_size, parent=core.active_interpreter.get())

    env = BatchEnv()

    def batch_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        with core.using_interpreter(batcher):
            boxed_out = eqn.bind(batcher.box(in_tree), **eqn.params)
        v_out, b_out = batcher.unbox(boxed_out)
        return v_out, assert_trees(b_out, eqn.out_tree, eqn.prim.name)

    utils.tree.map(env.write, ir.in_tree, v_in, b_in)
    for eqn in ir.eqns:
        v_in = utils.tree.map(env.read_v, eqn.in_tree)
        b_in = utils.tree.map(env.read_b, eqn.in_tree)
        v_out, b_out = batch_bind(eqn, (v_in, b_in))
        utils.tree.map(env.write, eqn.out_tree, v_out, b_out)
    v_out = utils.tree.map(env.read_v, ir.out_tree)
    b_out = utils.tree.map(env.read_b, ir.out_tree)
    return broadcast_batch_out(spec, v_out, b_out)


async def aimpl_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    v_in = in_tree
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)

    if not any(utils.tree.leaves(b_in)):
        return await ir.acall(*v_in)

    v_in, b_in = unbatch_zeros(v_in, b_in)
    spec = utils.batch_spec(v_in, b_in)
    if spec is None:
        raise TypeError("Cannot infer batch layout from symbolic zeros alone")

    batch_size = spec.num_children
    assert batch_size, "batch size must be > 0"

    batcher = BatchInterpreter(batch_size=batch_size, parent=core.active_interpreter.get())

    env = BatchEnv()

    async def batch_bind(eqn: stage.Eqn, in_tree: TreePair, /) -> TreePair:
        with core.using_interpreter(batcher):
            boxed_out = await eqn.abind(batcher.box(in_tree), **eqn.params)
        v_out, b_out = batcher.unbox(boxed_out)
        return v_out, assert_trees(b_out, eqn.out_tree, eqn.prim.name)

    utils.tree.map(env.write, ir.in_tree, v_in, b_in)
    for eqn in ir.eqns:
        v_in = utils.tree.map(env.read_v, eqn.in_tree)
        b_in = utils.tree.map(env.read_b, eqn.in_tree)
        v_out, b_out = await batch_bind(eqn, (v_in, b_in))
        utils.tree.map(env.write, eqn.out_tree, v_out, b_out)
    v_out = utils.tree.map(env.read_v, ir.out_tree)
    b_out = utils.tree.map(env.read_b, ir.out_tree)
    return broadcast_batch_out(spec, v_out, b_out)


def abstract_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    del in_tree
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    has_batched = any(utils.tree.leaves(b_in, is_leaf=is_axis_spec))

    def maybe_batched(aval, is_batched: bool):
        return BatchAVal(aval) if is_batched else aval

    def out_aval(atom):
        if stage.is_var(atom):
            return maybe_batched(atom.aval, has_batched)
        if has_batched:
            return maybe_batched(core.avalof(atom), True)
        return atom

    return utils.tree.map(out_aval, ir.out_tree)


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
    v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    v_out = batched_ir.call(*v_in)
    residuals = (v_in, in_axes)
    return v_out, residuals


async def apullback_fwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    v_out = await batched_ir.acall(*v_in)
    residuals = (v_in, in_axes)
    return v_out, residuals


def pullback_bwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    residuals, c_out = in_tree
    p, _ = residuals
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    pb_ir = ad.pullback(ir)
    if (spec := utils.batch_spec(p, b_in)) is None:
        return pb_ir.call(p, c_out)[1]
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, c_in = batch_pb_ir.call(p, c_out)

    def accum(batched, cotangents):
        return cotangents if batched else ad.cot_accum(spec.flatten_up_to(cotangents))

    return utils.tree.map(accum, b_in, c_in)


async def apullback_bwd_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> Tree:
    residuals, c_out = in_tree
    p, _ = residuals
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    pb_ir = ad.pullback(ir)
    if (spec := utils.batch_spec(p, b_in)) is None:
        return (await pb_ir.acall(p, c_out))[1]
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, c_in = await batch_pb_ir.acall(p, c_out)

    def accum(batched, cotangents):
        return cotangents if batched else ad.cot_accum(spec.flatten_up_to(cotangents))

    return utils.tree.map(accum, b_in, c_in)


def batch_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    batch_size, b_in, v_in = in_tree
    # NOTE(asem): nested batch rule. b_in tells us which positions are batched.
    # we use b_in's structure to flatten the data, index each batch item,
    # then unflatten back to the original container type.
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(utils.batch_index, v_in, b_in)
    v_bi = [batched_ir.call(*unbatch(b)) for b in range(batch_size)]
    b_out = utils.tree.map(lambda _: True, ir.out_tree)
    v_out = utils.batch_transpose(batch_size, b_out, v_bi)
    return v_out, b_out


async def abatch_batch_call(in_tree: Tree, /, *, ir: stage.IR, in_axes: Tree) -> TreePair:
    batch_size, b_in, v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(utils.batch_index, v_in, b_in)

    inputs = [unbatch(b) for b in range(batch_size)]
    v_bi = await order.fanout_p.abind(inputs, irs=[batched_ir] * batch_size)
    b_out = utils.tree.map(lambda _: True, ir.out_tree)
    v_out = utils.batch_transpose(batch_size, b_out, list(v_bi))
    return v_out, b_out


def dce_batch_call(eqn: stage.Eqn, out_used: dead.UsedTree, /) -> dead.DCEResult:
    new_eqn = eqn.using(ir=dead.dce(eqn.params["ir"], out_used=out_used))
    return dead.default_dce(new_eqn, out_used)


core.impl_rules.set(batch_call_p, impl_batch_call)
core.aimpl_rules.set(batch_call_p, aimpl_batch_call)
core.abstract_rules.set(batch_call_p, abstract_batch_call)
core.batch_rules.set(batch_call_p, batch_batch_call)
core.abatch_rules.set(batch_call_p, abatch_batch_call)
core.push_rules.set(batch_call_p, pushforward_batch_call)
core.apush_rules.set(batch_call_p, apushforward_batch_call)
core.pull_fwd_rules.set(batch_call_p, pullback_fwd_batch_call)
core.apull_fwd_rules.set(batch_call_p, apullback_fwd_batch_call)
core.pull_bwd_rules.set(batch_call_p, pullback_bwd_batch_call)
core.apull_bwd_rules.set(batch_call_p, apullback_bwd_batch_call)
dead.dce_rules[batch_call_p] = dce_batch_call
