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

import asyncio
import functools as ft
from operator import setitem
from typing import Any

from autoform.ad import pullback, pushforward
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
    is_irvar,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.dce import dce, dce_rules, default_dce
from autoform.utils import Tree, batch_index, batch_spec, batch_transpose, treelib

# ==================================================================================================
# BATCH
# ==================================================================================================


def is_axis_spec(v) -> bool:
    return isinstance(v, bool)


def assert_trees(b: Tree, out_ir: Tree, prim_name: str) -> Tree:
    expected_b = treelib.map(lambda _: False, out_ir)
    is_bool_leaf = lambda v: isinstance(v, bool)
    b_spec = treelib.structure(b, is_leaf=is_bool_leaf)
    expected_spec = treelib.structure(expected_b, is_leaf=is_bool_leaf)
    if b_spec != expected_spec:
        raise ValueError(
            f"Primitive '{prim_name}' batch_rule returned out_batched with structure {b_spec}, "
            f"but expected structure {expected_spec} to match output. "
            f"out_batched must match the structure of the output exactly."
        )
    return b


def broadcast_batch_out(spec, v_out: Tree, b_out: Tree[bool], /) -> Tree:
    batch_size = spec.num_children
    out_spec = treelib.structure(b_out, is_leaf=is_axis_spec)
    flat_out = out_spec.flatten_up_to(v_out)
    flat_b_out = treelib.leaves(b_out, is_leaf=is_axis_spec)

    def broadcast_leaf(v, b):
        return v if b else spec.unflatten([v] * batch_size)

    return out_spec.unflatten(map(broadcast_leaf, flat_out, flat_b_out))


batch_call_p = Prim("batch_call")


def batch(ir: IR, /, *, in_axes: Tree[bool] = True) -> IR:
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
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"
    b_in = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)
    has_batched = any(treelib.leaves(b_in, is_leaf=is_axis_spec))

    def make_in(atom, is_batched: bool):
        if not is_irvar(atom):
            return atom
        del is_batched
        return IRVar.fresh(aval=atom.aval, source=atom)

    def make_out(atom):
        if is_irvar(atom):
            return IRVar.fresh(aval=atom.aval, source=atom)
        if has_batched:
            return IRVar.fresh(aval=TypedAVal(type(atom)))
        return atom

    v_in_ir = treelib.map(make_in, ir.in_ir_tree, b_in)
    v_out_ir = treelib.map(make_out, ir.out_ir_tree)
    eqn = IREqn(batch_call_p, v_in_ir, v_out_ir, dict(ir=ir, in_axes=in_axes))
    return IR([eqn], v_in_ir, v_out_ir)


class BatchBox:
    __slots__ = ["owner", "value", "batched"]

    def __init__(self, owner, value, batched):
        self.owner = owner
        self.value = value
        self.batched = batched


class BatchInterpreter(BoxedInterpreter[BatchBox]):
    __slots__ = ["parent", "batch_size"]

    def __init__(self, *, batch_size: int):
        self.parent = active_interpreter.get()
        self.batch_size = batch_size

    def box(self, value, /) -> Tree:
        v, b = value
        # NOTE(asem): ``b`` is a prefix spec, not necessarily the same structure
        # as ``v``. For example, v=["a", "b"] and b=True means the whole list is
        # one batched leaf, not two leaves.
        spec = treelib.structure(b, is_leaf=is_axis_spec)
        v = spec.flatten_up_to(v)
        b = treelib.leaves(b, is_leaf=is_axis_spec)
        return spec.unflatten(BatchBox(self, v, b) for v, b in zip(v, b, strict=True))

    def unbox(self, v: Tree, /) -> tuple[Tree, Tree]:
        def value(v):
            return v.value if isinstance(v, BatchBox) and v.owner is self else v

        def batched(v):
            return v.batched if isinstance(v, BatchBox) and v.owner is self else False

        return treelib.map(value, v), treelib.map(batched, v)

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        v_in, b_in = self.unbox(in_tree)
        with using_interpreter(self.parent):
            v_out, b_out = batch_rules.get(prim)((self.batch_size, b_in, v_in), **params)
        return self.box((v_out, b_out))

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        # NOTE(asem): async batch rules must be explicitly seted - no fallback to sync.
        v_in, b_in = self.unbox(in_tree)
        with using_interpreter(self.parent):
            v_out, b_out = await batch_rules.aget(prim)((self.batch_size, b_in, v_in), **params)
        return self.box((v_out, b_out))


def impl_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    # NOTE(asem): ``in_axes`` only marks which leaves are batched.
    # the actual batch container comes from runtime data.
    # >>> in_tree = ReviewState(code=["a", "b"], has_bugs=[True, False])
    # >>> in_axes = True
    # >>> b_in = ReviewState(code=True, has_bugs=True)
    # >>> batch_size = 2
    v_in = in_tree
    b_in = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)

    if (spec := batch_spec(v_in, b_in)) is None:
        return ir.call(*v_in)

    batch_size = spec.num_children
    # NOTE(asem): this case can be something like
    # >>> def program(v):
    # ...     return af.format("constant string")
    # >>> ir = af.trace(program)("input")
    # >>> batched = af.batch(ir, in_axes=True)
    # >>> batched.call([])
    assert batch_size, "batch size must be > 0"

    env: dict[IRVar, Any] = {}
    interpreter = BatchInterpreter(batch_size=batch_size)

    def write(atom, value: Any):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom) -> Any:
        return env[atom] if is_irvar(atom) else atom

    treelib.map(write, ir.in_ir_tree, interpreter.box((v_in, b_in)))

    with using_interpreter(interpreter):
        for ir_eqn in ir.ir_eqns:
            v_in_eqn = treelib.map(read, ir_eqn.in_ir_tree)
            v_out_eqn = ir_eqn.bind(v_in_eqn, **ir_eqn.params)
            v_out, b_out = interpreter.unbox(v_out_eqn)
            b_out = assert_trees(b_out, ir_eqn.out_ir_tree, ir_eqn.prim.name)
            treelib.map(write, ir_eqn.out_ir_tree, interpreter.box((v_out, b_out)))

    v_out, b_out = interpreter.unbox(treelib.map(read, ir.out_ir_tree))
    return broadcast_batch_out(spec, v_out, b_out)


async def aimpl_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    v_in = in_tree
    b_in = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)

    if (spec := batch_spec(v_in, b_in)) is None:
        return await ir.acall(*v_in)

    batch_size = spec.num_children
    assert batch_size, "batch size must be > 0"

    env: dict[IRVar, Any] = {}
    interpreter = BatchInterpreter(batch_size=batch_size)

    def write(atom, value: Any):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom) -> Any:
        return env[atom] if is_irvar(atom) else atom

    treelib.map(write, ir.in_ir_tree, interpreter.box((v_in, b_in)))

    with using_interpreter(interpreter):
        for ir_eqn in ir.ir_eqns:
            v_in_eqn = treelib.map(read, ir_eqn.in_ir_tree)
            v_out_eqn = await ir_eqn.abind(v_in_eqn, **ir_eqn.params)
            v_out, b_out = interpreter.unbox(v_out_eqn)
            b_out = assert_trees(b_out, ir_eqn.out_ir_tree, ir_eqn.prim.name)
            treelib.map(write, ir_eqn.out_ir_tree, interpreter.box((v_out, b_out)))

    v_out, b_out = interpreter.unbox(treelib.map(read, ir.out_ir_tree))
    return broadcast_batch_out(spec, v_out, b_out)


def abstract_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    del in_tree
    b_in = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)
    has_batched = any(treelib.leaves(b_in, is_leaf=is_axis_spec))

    def out_aval(atom):
        if is_irvar(atom):
            return atom.aval
        if has_batched:
            return TypedAVal(type(atom))
        return atom

    return treelib.map(out_aval, ir.out_ir_tree)


def pushforward_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    p, t = in_tree
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return batch_pf_ir.call(p, t)


async def apushforward_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    p, t = in_tree
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return await batch_pf_ir.acall(p, t)


def pullback_fwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    v_out = batched_ir.call(*v_in)
    residuals = (v_in, in_axes)
    return v_out, residuals


async def apullback_fwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    v_out = await batched_ir.acall(*v_in)
    residuals = (v_in, in_axes)
    return v_out, residuals


def pullback_bwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    residuals, c_out = in_tree
    p, _ = residuals
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, c_in = batch_pb_ir.call(p, c_out)
    return c_in


async def apullback_bwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    residuals, c_out = in_tree
    p, _ = residuals
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, c_in = await batch_pb_ir.acall(p, c_out)
    return c_in


def batch_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    batch_size, b_in, v_in = in_tree
    # NOTE(asem): nested batch rule. b_in tells us which positions are batched.
    # we use b_in's structure to flatten the data, index each batch item,
    # then unflatten back to the original container type.
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(batch_index, v_in, b_in)
    v_bi = [batched_ir.call(*unbatch(b)) for b in range(batch_size)]
    b_out = treelib.map(lambda _: True, ir.out_ir_tree)
    v_out = batch_transpose(batch_size, b_out, v_bi)
    return v_out, b_out


async def abatch_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    batch_size, b_in, v_in = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(batch_index, v_in, b_in)
    v_bi = await asyncio.gather(*[batched_ir.acall(*unbatch(b)) for b in range(batch_size)])
    b_out = treelib.map(lambda _: True, ir.out_ir_tree)
    v_out = batch_transpose(batch_size, b_out, list(v_bi))
    return v_out, b_out


impl_rules.set(batch_call_p, impl_batch_call)
impl_rules.aset(batch_call_p, aimpl_batch_call)
abstract_rules.set(batch_call_p, abstract_batch_call)
push_rules.set(batch_call_p, pushforward_batch_call)
push_rules.aset(batch_call_p, apushforward_batch_call)
pull_fwd_rules.set(batch_call_p, pullback_fwd_batch_call)
pull_fwd_rules.aset(batch_call_p, apullback_fwd_batch_call)
pull_bwd_rules.set(batch_call_p, pullback_bwd_batch_call)
pull_bwd_rules.aset(batch_call_p, apullback_bwd_batch_call)
batch_rules.set(batch_call_p, batch_batch_call)
batch_rules.aset(batch_call_p, abatch_batch_call)


def dce_batch_call(ir_eqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    new_eqn = ir_eqn.using(ir=dce(ir_eqn.params["ir"], out_used=out_used))
    return default_dce(new_eqn, out_used)


dce_rules[batch_call_p] = dce_batch_call
