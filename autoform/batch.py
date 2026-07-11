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

"""Batch transformation"""

from __future__ import annotations

import functools as ft

import autoform.ad as ad
import autoform.core as core
import autoform.dce as dce
import autoform.scheduling as scheduling
import autoform.utils as utils

__all__ = ["batch"]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]

# ==================================================================================================
# BATCH
# ==================================================================================================


class BatchAVal(core.AVal):
    # NOTE(asem): unlike atomic AVals(e.g StrAVal), no aval_type rule can be registered
    # as containers are later introduced at the call site. unlike jax the atomic unit is not
    # the array object but any thing really.
    def __init__(self, base: core.AVal):
        # TODO(asem): maybe exapand with useful metadata here

        assert core.is_aval(base), f"Expected AVal, got {base!r}"
        self.base = base

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, BatchAVal) and self.base == other.base

    def __hash__(self) -> int:
        return hash((type(self), self.base))


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


batch_call_p = core.Prim("batch_call")


def batch(ir: core.IR, /, *, in_axes: Tree[bool] = True) -> core.IR:
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
        ...     return af.concat(greeting, name)
        >>> ir = af.trace(greet)("Hi", "World")
        >>> # Batch over names, broadcast greeting
        >>> batched = af.batch(ir, in_axes=(False, True))
        >>> batched.call("Hello, ", ["x0", "x1", "x2"])
        ['Hello, x0', 'Hello, x1', 'Hello, x2']
    """
    assert isinstance(ir, core.IR), f"Expected IR, got {type(ir)}"
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    has_batched = any(utils.tree.leaves(b_in, is_leaf=is_axis_spec))

    def maybe_batched(aval, is_batched: bool):
        return BatchAVal(aval) if is_batched else aval

    def make_in(atom, is_batched: bool):
        if not core.is_var(atom):
            return atom
        return core.Var.fresh(aval=maybe_batched(atom.aval, is_batched), source=atom)

    def make_out(atom):
        if core.is_var(atom):
            return core.Var.fresh(aval=maybe_batched(atom.aval, has_batched), source=atom)
        if has_batched:
            return core.Var.fresh(aval=maybe_batched(core.avalof(atom), True))
        return atom

    v_in_ir = utils.tree.map(make_in, ir.in_tree, b_in)
    v_out_ir = utils.tree.map(make_out, ir.out_tree)
    eqn = core.Eqn(batch_call_p, v_in_ir, v_out_ir, dict(ir=ir, in_axes=in_axes))
    return core.IR([eqn], v_in_ir, v_out_ir)


class BatchBox:
    __slots__ = ["owner", "value", "batched"]

    def __init__(self, owner, value, batched):
        self.owner = owner
        self.value = value
        self.batched = batched


class BatchInterpreter(core.BoxedInterpreter[BatchBox]):
    __slots__ = ["parent", "batch_size"]

    def __init__(self, *, batch_size: int, parent):
        self.parent = parent
        self.batch_size = batch_size

    def box(self, batched, value, /) -> BatchBox:
        return BatchBox(self, value, batched)

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
        return utils.tree.map(self.box, b_out, v_out, is_leaf=is_axis_spec)

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params):
        # NOTE(asem): async batch rules must be explicitly seted - no fallback to sync.
        v_in, b_in = self.unbox(in_tree)
        b_sz = self.batch_size
        with core.using_interpreter(self.parent):
            v_out, b_out = await core.batch_rules.aget(prim)((b_sz, b_in, v_in), **params)
        return utils.tree.map(self.box, b_out, v_out, is_leaf=is_axis_spec)


def impl_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> Tree:
    # NOTE(asem): ``in_axes`` only marks which leaves are batched.
    # the actual batch container comes from runtime data.
    # >>> in_tree = ReviewState(code=["a", "b"], has_bugs=[True, False])
    # >>> in_axes = True
    # >>> b_in = ReviewState(code=True, has_bugs=True)
    # >>> batch_size = 2
    v_in = in_tree
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)

    if (spec := utils.batch_spec(v_in, b_in)) is None:
        return ir.call(*v_in)

    batch_size = spec.num_children
    # NOTE(asem): this case can be something like
    # >>> def program(v):
    # ...     return af.format("constant string")
    # >>> ir = af.trace(program)("input")
    # >>> batched = af.batch(ir, in_axes=True)
    # >>> batched.call([])
    assert batch_size, "batch size must be > 0"

    batcher = BatchInterpreter(batch_size=batch_size, parent=core.active_interpreter.get())
    with core.using_interpreter(batcher):

        def custom_bind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            boxed_out = eqn.bind(boxed_in, **eqn.params)
            v_out, b_out = batcher.unbox(boxed_out)
            b_out = assert_trees(b_out, eqn.out_tree, eqn.prim.name)
            return utils.tree.map(batcher.box, b_out, v_out, is_leaf=is_axis_spec)

        eqn, boxed_in = next(
            gen := ir.walk(*utils.tree.map(batcher.box, b_in, v_in, is_leaf=is_axis_spec))
        )
        while eqn:
            eqn, boxed_in = gen.send(custom_bind(eqn, boxed_in))

    v_out, b_out = batcher.unbox(boxed_in)
    return broadcast_batch_out(spec, v_out, b_out)


async def aimpl_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> Tree:
    v_in = in_tree
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)

    if (spec := utils.batch_spec(v_in, b_in)) is None:
        return await ir.acall(*v_in)

    batch_size = spec.num_children
    assert batch_size, "batch size must be > 0"

    batcher = BatchInterpreter(batch_size=batch_size, parent=core.active_interpreter.get())
    with core.using_interpreter(batcher):

        async def custom_abind(eqn: core.Eqn, boxed_in: Tree, /) -> Tree:
            boxed_out = await eqn.abind(boxed_in, **eqn.params)
            v_out, b_out = batcher.unbox(boxed_out)
            b_out = assert_trees(b_out, eqn.out_tree, eqn.prim.name)
            return utils.tree.map(batcher.box, b_out, v_out, is_leaf=is_axis_spec)

        eqn, boxed_in = next(
            gen := ir.walk(*utils.tree.map(batcher.box, b_in, v_in, is_leaf=is_axis_spec))
        )
        while eqn:
            eqn, boxed_in = gen.send(await custom_abind(eqn, boxed_in))

    v_out, b_out = batcher.unbox(boxed_in)
    return broadcast_batch_out(spec, v_out, b_out)


def abstract_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> Tree:
    del in_tree
    b_in = utils.tree.broadcast_prefix(in_axes, ir.in_tree, is_leaf=is_axis_spec)
    has_batched = any(utils.tree.leaves(b_in, is_leaf=is_axis_spec))

    def maybe_batched(aval, is_batched: bool):
        return BatchAVal(aval) if is_batched else aval

    def out_aval(atom):
        if core.is_var(atom):
            return maybe_batched(atom.aval, has_batched)
        if has_batched:
            return maybe_batched(core.avalof(atom), True)
        return atom

    return utils.tree.map(out_aval, ir.out_tree)


def pushforward_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> TreePair:
    p, t = in_tree
    pf_ir = ad.pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return batch_pf_ir.call(p, t)


async def apushforward_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> TreePair:
    p, t = in_tree
    pf_ir = ad.pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return await batch_pf_ir.acall(p, t)


def pullback_fwd_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> TreePair:
    v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    v_out = batched_ir.call(*v_in)
    residuals = (v_in, in_axes)
    return v_out, residuals


async def apullback_fwd_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> TreePair:
    v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    v_out = await batched_ir.acall(*v_in)
    residuals = (v_in, in_axes)
    return v_out, residuals


def pullback_bwd_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> Tree:
    residuals, c_out = in_tree
    p, _ = residuals
    pb_ir = ad.pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, c_in = batch_pb_ir.call(p, c_out)
    return c_in


async def apullback_bwd_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> Tree:
    residuals, c_out = in_tree
    p, _ = residuals
    pb_ir = ad.pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, c_in = await batch_pb_ir.acall(p, c_out)
    return c_in


def batch_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> TreePair:
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


async def abatch_batch_call(in_tree: Tree, /, *, ir: core.IR, in_axes: Tree) -> TreePair:
    batch_size, b_in, v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(utils.batch_index, v_in, b_in)

    inputs = [unbatch(b) for b in range(batch_size)]
    v_bi = await scheduling.gather_p.abind(inputs, irs=[batched_ir] * batch_size)
    b_out = utils.tree.map(lambda _: True, ir.out_tree)
    v_out = utils.batch_transpose(batch_size, b_out, list(v_bi))
    return v_out, b_out


core.impl_rules.set(batch_call_p, impl_batch_call)
core.impl_rules.aset(batch_call_p, aimpl_batch_call)
core.abstract_rules.set(batch_call_p, abstract_batch_call)
core.push_rules.set(batch_call_p, pushforward_batch_call)
core.push_rules.aset(batch_call_p, apushforward_batch_call)
core.pull_fwd_rules.set(batch_call_p, pullback_fwd_batch_call)
core.pull_fwd_rules.aset(batch_call_p, apullback_fwd_batch_call)
core.pull_bwd_rules.set(batch_call_p, pullback_bwd_batch_call)
core.pull_bwd_rules.aset(batch_call_p, apullback_bwd_batch_call)
core.batch_rules.set(batch_call_p, batch_batch_call)
core.batch_rules.aset(batch_call_p, abatch_batch_call)


def dce_batch_call(eqn: core.Eqn, out_used: dce.UsedTree, /) -> dce.DCEResult:
    new_eqn = eqn.using(ir=dce.dce(eqn.params["ir"], out_used=out_used))
    return dce.default_dce(new_eqn, out_used)


dce.dce_rules[batch_call_p] = dce_batch_call
