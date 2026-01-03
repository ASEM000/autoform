"""Batch transformation"""

from __future__ import annotations

import asyncio
import functools as ft
import typing as tp
from operator import setitem

from autoform.ad import pullback, pushforward
from autoform.core import (
    IR,
    Interpreter,
    IREqn,
    IRLit,
    IRVar,
    Primitive,
    TransformationTag,
    Value,
    acall,
    async_rules,
    batch_rules,
    call,
    dce_rules,
    default_dce,
    eval_rules,
    get_interpreter,
    impl_rules,
    iratom_to_evaltype,
    is_irvar,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.optims import dce
from autoform.utils import Tree, lru_cache, transpose_batch, treelib

# ==================================================================================================
# BATCH
# ==================================================================================================


class IRBVar(IRVar): ...


class BatchTag(TransformationTag): ...


def is_axis_spec(x) -> bool:
    return isinstance(x, bool)


def infer_batch_size(tree: Tree, in_axes: Tree) -> int:
    # NOTE(asem): infer batch size by finding the first batched (True) position.
    # in_axes specifies ONLY which positions are batched, not the container type.
    # The container type is inferred from the actual data in `tree`.
    #
    # >>> tree = ReviewState(code=["a", "b", "c"], has_bugs=[T, F, T])
    # >>> in_axes = ReviewState(code=True, has_bugs=True)
    # >>> axes_spec = PyTreeSpec(ReviewState(*, *))  # structure with 2 leaves
    # >>> axes_leaves = [True, True]
    # >>> tree_leaves = [["a","b","c"], [T,F,T]]  # flattened to match spec
    # >>> batch_size = len(["a","b","c"]) = 3
    axes_spec = treelib.structure(in_axes, is_leaf=is_axis_spec)
    axes_leaves = treelib.leaves(in_axes, is_leaf=is_axis_spec)
    tree_leaves = axes_spec.flatten_up_to(tree)
    return next((len(v) for v, a in zip(tree_leaves, axes_leaves) if a), 0)


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


@ft.partial(lru_cache, maxsize=256)
def batch(ir: IR, in_axes: Tree[bool] = True) -> IR:
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
        >>> ir = af.build_ir(greet)("Hi", "World")
        >>> # Batch over names, broadcast greeting
        >>> batched = af.batch(ir, in_axes=(False, True))
        >>> call(batched)(("Hello, ", ["Alice", "Bob", "Carol"]))
        ['Hello, Alice', 'Hello, Bob', 'Hello, Carol']
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_b(atom):
        return IRBVar.fresh(source=atom) if is_irvar(atom) else atom

    in_b_irtree = treelib.map(make_b, ir.in_irtree)
    out_b_irtree = treelib.map(make_b, ir.out_irtree)
    eqn = IREqn(batch_call_p, in_b_irtree, out_b_irtree, dict(ir=ir, in_axes=in_axes))
    return IR([eqn], in_b_irtree, out_b_irtree)


class BatchInterpreter(Interpreter):
    def __init__(self, *, batch_size: int):
        self.parent = get_interpreter()
        self.batch_size = batch_size

    def interpret(self, prim: Primitive, in_tree: Tree, **params):
        with using_interpreter(self.parent):
            batch_size, in_batched, in_values = in_tree
            return batch_rules[prim](batch_size, in_batched, in_values, **params)


@ft.partial(impl_rules.def_rule, batch_call_p)
def impl_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    # NOTE(asem): in_axes is Tree[bool] specifying which positions are batched.
    # container type is inferred from actual data, not specified in in_axes.
    #
    # >>> in_tree = ReviewState(code=["a","b"], has_bugs=[T,F])  # actual data
    # >>> in_axes = True  # scalar -> batch everything
    # >>> in_batched_tree = ReviewState(code=True, has_bugs=True)  # broadcast to match IR
    # >>> batch_size = 2  # inferred from len(in_tree.code)
    col_tree = in_tree
    in_batched_tree = treelib.broadcast_prefix(in_axes, ir.in_irtree, is_leaf=is_axis_spec)
    batch_size = infer_batch_size(col_tree, in_batched_tree)

    v_env: dict[IRVar, Value | list[Value]] = {}
    b_env: dict[IRVar, bool] = {}

    def write_v(atom, value: Value | list[Value]):
        is_irvar(atom) and setitem(v_env, atom, value)

    def write_b(atom, is_batched: bool):
        is_irvar(atom) and setitem(b_env, atom, is_batched)

    def read_v(atom) -> Value | list[Value]:
        return v_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    def read_b(atom) -> bool:
        return b_env[atom] if is_irvar(atom) else False

    treelib.map(write_v, ir.in_irtree, col_tree)
    treelib.map(write_b, ir.in_irtree, in_batched_tree)

    with using_interpreter(BatchInterpreter(batch_size=batch_size)):
        for ireqn in ir.ireqns:
            in_vals = treelib.map(read_v, ireqn.in_irtree)
            in_batched = treelib.map(read_b, ireqn.in_irtree)
            in_tree = (batch_size, in_batched, in_vals)
            out_vals, out_batched = ireqn.prim.bind(in_tree, **ireqn.params)
            treelib.map(write_v, ireqn.out_irtree, out_vals)
            out_batched = assert_trees(out_batched, ireqn.out_irtree, ireqn.prim.name)
            treelib.map(write_b, ireqn.out_irtree, out_batched)

    return treelib.map(read_v, ir.out_irtree)


@ft.partial(eval_rules.def_rule, batch_call_p)
def eval_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    return treelib.map(iratom_to_evaltype, ir.out_irtree)


@ft.partial(push_rules.def_rule, batch_call_p)
def pushforward_batch_call(
    primals: Tree,
    tangents: Tree,
    *,
    ir: IR,
    in_axes: Tree,
) -> tuple[Tree, Tree]:
    p_cols, t_cols = primals, tangents
    pf_ir = pushforward(ir)
    batch_pf_ir = batch(pf_ir, in_axes=(in_axes, in_axes))
    return call(batch_pf_ir)((p_cols, t_cols))


@ft.partial(pull_fwd_rules.def_rule, batch_call_p)
def pullback_fwd_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    col_tree = in_tree
    batched_ir = batch(ir, in_axes=in_axes)
    out_cols = call(batched_ir)(col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


@ft.partial(pull_bwd_rules.def_rule, batch_call_p)
def pullback_bwd_batch_call(residuals: Tree, out_cotangent: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    p_cols, _ = residuals
    out_c_cols = out_cotangent
    pb_ir = pullback(ir)
    batch_pb_ir = batch(pb_ir, in_axes=(in_axes, True))
    _, in_c_cols = call(batch_pb_ir)((p_cols, out_c_cols))
    return in_c_cols


@ft.partial(batch_rules.def_rule, batch_call_p)
def batch_batch_call(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    ir: IR,
    in_axes: Tree,
) -> tuple[Tree, Tree]:
    # NOTE(asem): nested batch rule. in_batched tells us which positions are batched.
    # we use in_batched's structure to flatten the data, index each batch item,
    # then unflatten back to the original container type.
    col_cols = in_tree
    batched_ir = batch(ir, in_axes=in_axes)

    batched_spec = treelib.structure(in_batched, is_leaf=is_axis_spec)
    batched_leaves = treelib.leaves(in_batched, is_leaf=is_axis_spec)
    col_cols_flat = batched_spec.flatten_up_to(col_cols)

    def get(b):
        indexed = [v[b] if is_b else v for v, is_b in zip(col_cols_flat, batched_leaves)]
        return batched_spec.unflatten(indexed)

    out_bi = [call(batched_ir)(get(b)) for b in range(batch_size)]

    out_batched = treelib.map(lambda _: True, ir.out_irtree)
    out_ib = transpose_batch(batch_size, out_batched, out_bi)
    return out_ib, out_batched


@ft.partial(async_rules.def_rule, batch_call_p)
async def async_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    col_tree = in_tree
    axes_tree = treelib.broadcast_prefix(in_axes, ir.in_irtree, is_leaf=is_axis_spec)
    batch_size = infer_batch_size(col_tree, axes_tree)

    axes_spec = treelib.structure(axes_tree, is_leaf=is_axis_spec)
    axes_leaves = treelib.leaves(axes_tree, is_leaf=is_axis_spec)
    col_tree_flat = axes_spec.flatten_up_to(col_tree)

    async def run_item(b: int):
        indexed = [v[b] if a else v for v, a in zip(col_tree_flat, axes_leaves)]
        value = axes_spec.unflatten(indexed)
        return await acall(ir)(value)

    out_bi = await asyncio.gather(*[run_item(b) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, ir.out_irtree)
    return transpose_batch(batch_size, out_batched, out_bi)


@ft.partial(dce_rules.def_rule, batch_call_p)
def dce_batch_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    new_eqn = ireqn.using(ir=dce(ireqn.params["ir"]))
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn
