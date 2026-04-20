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
from collections.abc import Hashable
from operator import setitem
from typing import Any

from autoform.ad import pullback, pushforward
from autoform.core import (
    IR,
    Interpreter,
    IREqn,
    IRVar,
    Prim,
    TypedAVal,
    abstract_rules,
    active_interpreter,
    add_axis,
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


def is_axis_spec(x) -> bool:
    return isinstance(x, bool)


def assert_trees(batch_tree: Tree, ir_tree: Tree, prim_name: str) -> Tree:
    expected_batch_tree = treelib.map(lambda _: False, ir_tree)
    is_bool_leaf = lambda x: isinstance(x, bool)
    batch_spec = treelib.structure(batch_tree, is_leaf=is_bool_leaf)
    expected_spec = treelib.structure(expected_batch_tree, is_leaf=is_bool_leaf)
    if batch_spec != expected_spec:
        raise ValueError(
            f"Primitive '{prim_name}' batch_rule returned out_batched with structure {batch_spec}, "
            f"but expected structure {expected_spec} to match output. "
            f"out_batched must match the structure of the output exactly."
        )
    return batch_tree


def broadcast_batch_out(spec, out_tree: Tree, out_batched_tree: Tree[bool], /) -> Tree:
    batch_size = spec.num_children
    out_spec = treelib.structure(out_batched_tree, is_leaf=is_axis_spec)
    flat_out = out_spec.flatten_up_to(out_tree)
    flat_out_b = treelib.leaves(out_batched_tree, is_leaf=is_axis_spec)

    def broadcast_leaf(v, b):
        return v if b else spec.unflatten([v] * batch_size)

    return out_spec.unflatten(map(broadcast_leaf, flat_out, flat_out_b))


batch_call_p = Prim("batch_call")


def batch(ir: IR, /, *, in_axes: Tree[bool] = True, axis_name: Hashable | None = None) -> IR:
    """Transform an IR to process batched inputs.

    Creates a batched version of the IR that processes multiple inputs
    simultaneously. Use `in_axes` to specify which inputs are batched
    (True) vs broadcast (False).

    Args:
        ir: The IR to transform.
        in_axes: Axis specification tree matching input structure.
            - True: This input is batched (a collection of values).
            - False: This input is broadcast (same value for all batch items).
        axis_name: Optional name for the mapped axis. Named-axis primitives
            such as ``axis_index``, ``axis_size``, and ``axis_gather`` can
            refer to this name while the batched IR runs.

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
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)
    has_batched_input = any(treelib.leaves(in_batched_tree, is_leaf=is_axis_spec))

    def make_in(atom, is_batched: bool):
        if not is_irvar(atom):
            return atom
        del is_batched
        return IRVar.fresh(aval=atom.aval, source=atom)

    def make_out(atom):
        if is_irvar(atom):
            return IRVar.fresh(aval=atom.aval, source=atom)
        if has_batched_input:
            return IRVar.fresh(aval=TypedAVal(type(atom)))
        return atom

    in_b_ir_tree = treelib.map(make_in, ir.in_ir_tree, in_batched_tree)
    out_b_ir_tree = treelib.map(make_out, ir.out_ir_tree)
    eqn = IREqn(
        batch_call_p,
        in_b_ir_tree,
        out_b_ir_tree,
        dict(ir=ir, in_axes=in_axes, axis_name=axis_name),
    )
    return IR([eqn], in_b_ir_tree, out_b_ir_tree)


class BatchInterpreter(Interpreter):
    __slots__ = ["parent", "batch_size"]

    def __init__(self, *, batch_size: int):
        self.parent = active_interpreter.get()
        self.batch_size = batch_size

    def interpret(self, prim: Prim, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return batch_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params):
        # NOTE(asem): async batch rules must be explicitly seted - no fallback to sync.
        with using_interpreter(self.parent):
            return await batch_rules.aget(prim)(in_tree, **params)


def impl_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None) -> Tree:
    # NOTE(asem): ``in_axes`` only marks which leaves are batched.
    # the actual batch container comes from runtime data.
    # >>> in_tree = ReviewState(code=["a", "b"], has_bugs=[True, False])
    # >>> in_axes = True
    # >>> in_batched_tree = ReviewState(code=True, has_bugs=True)
    # >>> batch_size = 2
    col_tree = in_tree
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)

    if (spec := batch_spec(col_tree, in_batched_tree)) is None:
        return ir.call(*col_tree)

    batch_size = spec.num_children
    # NOTE(asem): this case can be something like
    # >>> def program(x):
    # ...     return af.format("constant string")
    # >>> ir = af.trace(program)("input")
    # >>> batched = af.batch(ir, in_axes=True)
    # >>> batched.call([])
    assert batch_size, "batch size must be > 0"

    v_env: dict[IRVar, Any] = {}
    b_env: dict[IRVar, bool] = {}

    def write_v(atom, value: Any):
        is_irvar(atom) and setitem(v_env, atom, value)

    def write_b(atom, is_batched: bool):
        is_irvar(atom) and setitem(b_env, atom, is_batched)

    def read_v(atom) -> Any:
        return v_env[atom] if is_irvar(atom) else atom

    def read_b(atom) -> bool:
        return b_env[atom] if is_irvar(atom) else False

    treelib.map(write_v, ir.in_ir_tree, col_tree)
    treelib.map(write_b, ir.in_ir_tree, in_batched_tree)

    with (
        add_axis(axis_name, batch_size, spec),
        using_interpreter(BatchInterpreter(batch_size=batch_size)),
    ):
        for ir_eqn in ir.ir_eqns:
            in_vals = treelib.map(read_v, ir_eqn.in_ir_tree)
            in_batched = treelib.map(read_b, ir_eqn.in_ir_tree)
            in_tree = (batch_size, in_batched, in_vals)
            out_vals, out_batched = ir_eqn.bind(in_tree, **ir_eqn.params)
            treelib.map(write_v, ir_eqn.out_ir_tree, out_vals)
            out_batched = assert_trees(out_batched, ir_eqn.out_ir_tree, ir_eqn.prim.name)
            treelib.map(write_b, ir_eqn.out_ir_tree, out_batched)
    out_vals = treelib.map(read_v, ir.out_ir_tree)
    out_batched = treelib.map(read_b, ir.out_ir_tree)
    return broadcast_batch_out(spec, out_vals, out_batched)


async def aimpl_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> Tree:
    col_tree = in_tree
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)

    if (spec := batch_spec(col_tree, in_batched_tree)) is None:
        return await ir.acall(*col_tree)

    batch_size = spec.num_children
    assert batch_size, "batch size must be > 0"

    v_env: dict[IRVar, Any] = {}
    b_env: dict[IRVar, bool] = {}

    def write_v(atom, value: Any):
        is_irvar(atom) and setitem(v_env, atom, value)

    def write_b(atom, is_batched: bool):
        is_irvar(atom) and setitem(b_env, atom, is_batched)

    def read_v(atom) -> Any:
        return v_env[atom] if is_irvar(atom) else atom

    def read_b(atom) -> bool:
        return b_env[atom] if is_irvar(atom) else False

    treelib.map(write_v, ir.in_ir_tree, col_tree)
    treelib.map(write_b, ir.in_ir_tree, in_batched_tree)

    with (
        add_axis(axis_name, batch_size, spec),
        using_interpreter(BatchInterpreter(batch_size=batch_size)),
    ):
        for ir_eqn in ir.ir_eqns:
            in_vals = treelib.map(read_v, ir_eqn.in_ir_tree)
            in_batched = treelib.map(read_b, ir_eqn.in_ir_tree)
            in_tree = (batch_size, in_batched, in_vals)
            out_vals, out_batched = await ir_eqn.abind(in_tree, **ir_eqn.params)
            treelib.map(write_v, ir_eqn.out_ir_tree, out_vals)
            out_batched = assert_trees(out_batched, ir_eqn.out_ir_tree, ir_eqn.prim.name)
            treelib.map(write_b, ir_eqn.out_ir_tree, out_batched)
    out_vals = treelib.map(read_v, ir.out_ir_tree)
    out_batched = treelib.map(read_b, ir.out_ir_tree)
    return broadcast_batch_out(spec, out_vals, out_batched)


def abstract_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> Tree:
    del in_tree
    del axis_name
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_ir_tree, is_leaf=is_axis_spec)
    has_batched_input = any(treelib.leaves(in_batched_tree, is_leaf=is_axis_spec))

    def out_aval(atom):
        if is_irvar(atom):
            return atom.aval
        if has_batched_input:
            return TypedAVal(type(atom))
        return atom

    return treelib.map(out_aval, ir.out_ir_tree)


def pushforward_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_cols, t_cols = primals, tangents
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes), axis_name=axis_name)
    return batch_pf_ir.call(p_cols, t_cols)


async def apushforward_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_cols, t_cols = primals, tangents
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes), axis_name=axis_name)
    return await batch_pf_ir.acall(p_cols, t_cols)


def pullback_fwd_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> tuple[Tree, Tree]:
    col_tree = in_tree
    batched_ir = batch(ir, in_axes=in_axes, axis_name=axis_name)
    out_cols = batched_ir.call(*col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


async def apullback_fwd_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> tuple[Tree, Tree]:
    col_tree = in_tree
    batched_ir = batch(ir, in_axes=in_axes, axis_name=axis_name)
    out_cols = await batched_ir.acall(*col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


def pullback_bwd_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> Tree:
    residuals, out_cotangent = in_tree
    p_cols, _ = residuals
    out_c_cols = out_cotangent
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True), axis_name=axis_name)
    _, in_c_cols = batch_pb_ir.call(p_cols, out_c_cols)
    return in_c_cols


async def apullback_bwd_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> Tree:
    residuals, out_cotangent = in_tree
    p_cols, _ = residuals
    out_c_cols = out_cotangent
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True), axis_name=axis_name)
    _, in_c_cols = await batch_pb_ir.acall(p_cols, out_c_cols)
    return in_c_cols


def batch_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> tuple[Tree, Tree]:
    batch_size, in_batched, col_cols = in_tree
    # NOTE(asem): nested batch rule. in_batched tells us which positions are batched.
    # we use in_batched's structure to flatten the data, index each batch item,
    # then unflatten back to the original container type.
    batched_ir = batch(ir, in_axes=in_axes, axis_name=axis_name)
    unbatch = ft.partial(batch_index, col_cols, in_batched)
    out_bi = [batched_ir.call(*unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, ir.out_ir_tree)
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_batch_call(
    in_tree: Tree, /, *, ir: IR, in_axes: Tree, axis_name: Hashable | None
) -> tuple[Tree, Tree]:
    batch_size, in_batched, col_cols = in_tree
    batched_ir = batch(ir, in_axes=in_axes, axis_name=axis_name)
    unbatch = ft.partial(batch_index, col_cols, in_batched)
    out_bi = await asyncio.gather(*[batched_ir.acall(*unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, ir.out_ir_tree)
    out_ib = batch_transpose(batch_size, out_batched, list(out_bi))
    return out_ib, out_batched


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
