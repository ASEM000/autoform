"""Batch transformation"""

from __future__ import annotations

import asyncio
import functools as ft
from operator import setitem
from typing import Any, cast

from autoform.ad import pullback, pushforward
from autoform.core import (
    IR,
    Interpreter,
    IREqn,
    IRLit,
    IRVar,
    Primitive,
    TransformationTag,
    acall,
    active_effect,
    active_interpreter,
    batch_rules,
    call,
    eval_rules,
    impl_rules,
    iratom_to_evaltype,
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


class BatchTag(TransformationTag): ...


def is_axis_spec(x) -> bool:
    return isinstance(x, bool)


def assert_trees(batch_tree: Tree, irtree: Tree, prim_name: str) -> Tree:
    expected_batch_tree = treelib.map(lambda _: False, irtree)
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


batch_call_p = Primitive("batch_call", tag={BatchTag})


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
        >>> call(batched)(("Hello, ", ["Alice", "Bob", "Carol"]))
        ['Hello, Alice', 'Hello, Bob', 'Hello, Carol']
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_b(atom):
        return IRVar.fresh(type=atom.type, source=atom) if is_irvar(atom) else atom

    in_b_irtree = treelib.map(make_b, ir.in_irtree)
    out_b_irtree = treelib.map(make_b, ir.out_irtree)
    # NOTE(asem): effect on the wrapper IREqn is unused at execution time.
    # impl_batch_call never reads active_effect, and no EffectInterpreter handler
    # targets batch_call_p. inner IREqns carry their own effects and restore them
    # via ireqn.bind(). captured here only to preserve the IR structure contract.
    effect = active_effect.get()
    eqn = IREqn(batch_call_p, effect, in_b_irtree, out_b_irtree, dict(ir=ir, in_axes=in_axes))
    return IR([eqn], in_b_irtree, out_b_irtree)


class BatchInterpreter(Interpreter):
    def __init__(self, *, batch_size: int):
        self.parent = active_interpreter.get()
        self.batch_size = batch_size

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return batch_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params):
        # NOTE(asem): async batch rules must be explicitly seted - no fallback to sync.
        with using_interpreter(self.parent):
            return await batch_rules.aget(prim)(in_tree, **params)


def impl_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    # NOTE(asem): in_axes is Tree[bool] specifying which positions are batched.
    # container type is inferred from actual data, not specified in in_axes.
    #
    # >>> in_tree = ReviewState(code=["a","b"], has_bugs=[T,F])  # actual data
    # >>> in_axes = True  # scalar -> batch everything
    # >>> in_batched_tree = ReviewState(code=True, has_bugs=True)  # broadcast to match IR
    # >>> batch_size = 2  # inferred from len(in_tree.code)
    col_tree = in_tree
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_irtree, is_leaf=is_axis_spec)

    if (spec := batch_spec(col_tree, in_batched_tree)) is None:
        return call(ir)(col_tree)

    batch_size = spec.num_children
    # NOTE(asem): this case can be something like
    # >>> def program(x):
    # ...     return af.format("constant string")
    # >>> ir = af.trace(program)("input")
    # >>> batched = af.batch(ir, in_axes=True)
    # >>> call(batched)([])
    assert batch_size, "batch size must be > 0"

    v_env: dict[IRVar, Any] = {}
    b_env: dict[IRVar, bool] = {}

    def write_v(atom, value: Any):
        is_irvar(atom) and setitem(v_env, atom, value)

    def write_b(atom, is_batched: bool):
        is_irvar(atom) and setitem(b_env, atom, is_batched)

    def read_v(atom) -> Any:
        return v_env[atom] if is_irvar(atom) else cast(IRLit, atom).value

    def read_b(atom) -> bool:
        return b_env[atom] if is_irvar(atom) else False

    treelib.map(write_v, ir.in_irtree, col_tree)
    treelib.map(write_b, ir.in_irtree, in_batched_tree)

    with using_interpreter(BatchInterpreter(batch_size=batch_size)):
        for ireqn in ir.ireqns:
            in_vals = treelib.map(read_v, ireqn.in_irtree)
            in_batched = treelib.map(read_b, ireqn.in_irtree)
            in_tree = (batch_size, in_batched, in_vals)
            out_vals, out_batched = ireqn.bind(in_tree, **ireqn.params)
            treelib.map(write_v, ireqn.out_irtree, out_vals)
            out_batched = assert_trees(out_batched, ireqn.out_irtree, ireqn.prim.name)
            treelib.map(write_b, ireqn.out_irtree, out_batched)

    return treelib.map(read_v, ir.out_irtree)


async def aimpl_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    col_tree = in_tree
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_irtree, is_leaf=is_axis_spec)

    if (spec := batch_spec(col_tree, in_batched_tree)) is None:
        return await acall(ir)(col_tree)

    batch_size = spec.num_children
    assert batch_size, "batch size must be > 0"

    v_env: dict[IRVar, Any] = {}
    b_env: dict[IRVar, bool] = {}

    def write_v(atom, value: Any):
        is_irvar(atom) and setitem(v_env, atom, value)

    def write_b(atom, is_batched: bool):
        is_irvar(atom) and setitem(b_env, atom, is_batched)

    def read_v(atom) -> Any:
        return v_env[atom] if is_irvar(atom) else cast(IRLit, atom).value

    def read_b(atom) -> bool:
        return b_env[atom] if is_irvar(atom) else False

    treelib.map(write_v, ir.in_irtree, col_tree)
    treelib.map(write_b, ir.in_irtree, in_batched_tree)

    with using_interpreter(BatchInterpreter(batch_size=batch_size)):
        for ireqn in ir.ireqns:
            in_vals = treelib.map(read_v, ireqn.in_irtree)
            in_batched = treelib.map(read_b, ireqn.in_irtree)
            in_tree = (batch_size, in_batched, in_vals)
            out_vals, out_batched = await ireqn.abind(in_tree, **ireqn.params)
            treelib.map(write_v, ireqn.out_irtree, out_vals)
            out_batched = assert_trees(out_batched, ireqn.out_irtree, ireqn.prim.name)
            treelib.map(write_b, ireqn.out_irtree, out_batched)
    return treelib.map(read_v, ir.out_irtree)


def eval_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    return treelib.map(iratom_to_evaltype, ir.out_irtree)


def pushforward_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_cols, t_cols = primals, tangents
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return call(batch_pf_ir)((p_cols, t_cols))


async def apushforward_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_cols, t_cols = primals, tangents
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return await acall(batch_pf_ir)((p_cols, t_cols))


def pullback_fwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    col_tree = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    out_cols = call(batched_ir)(col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


async def apullback_fwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    col_tree = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    out_cols = await acall(batched_ir)(col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


def pullback_bwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    residuals, out_cotangent = in_tree
    p_cols, _ = residuals
    out_c_cols = out_cotangent
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, in_c_cols = call(batch_pb_ir)((p_cols, out_c_cols))
    return in_c_cols


async def apullback_bwd_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> Tree:
    residuals, out_cotangent = in_tree
    p_cols, _ = residuals
    out_c_cols = out_cotangent
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, in_c_cols = await acall(batch_pb_ir)((p_cols, out_c_cols))
    return in_c_cols


def batch_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    batch_size, in_batched, col_cols = in_tree
    # NOTE(asem): nested batch rule. in_batched tells us which positions are batched.
    # we use in_batched's structure to flatten the data, index each batch item,
    # then unflatten back to the original container type.
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(batch_index, col_cols, in_batched)
    out_bi = [call(batched_ir)(unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, ir.out_irtree)
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_batch_call(in_tree: Tree, /, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    batch_size, in_batched, col_cols = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    unbatch = ft.partial(batch_index, col_cols, in_batched)
    out_bi = await asyncio.gather(*[acall(batched_ir)(unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, ir.out_irtree)
    out_ib = batch_transpose(batch_size, out_batched, list(out_bi))
    return out_ib, out_batched


impl_rules.set(batch_call_p, impl_batch_call)
impl_rules.aset(batch_call_p, aimpl_batch_call)
eval_rules.set(batch_call_p, eval_batch_call)
push_rules.set(batch_call_p, pushforward_batch_call)
push_rules.aset(batch_call_p, apushforward_batch_call)
pull_fwd_rules.set(batch_call_p, pullback_fwd_batch_call)
pull_fwd_rules.aset(batch_call_p, apullback_fwd_batch_call)
pull_bwd_rules.set(batch_call_p, pullback_bwd_batch_call)
pull_bwd_rules.aset(batch_call_p, apullback_bwd_batch_call)
batch_rules.set(batch_call_p, batch_batch_call)
batch_rules.aset(batch_call_p, abatch_batch_call)


def dce_batch_call(ireqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    new_eqn = ireqn.using(ir=dce(ireqn.params["ir"], out_used=out_used))
    return default_dce(new_eqn, out_used)


dce_rules[batch_call_p] = dce_batch_call
