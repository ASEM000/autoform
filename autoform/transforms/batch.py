"""Batch transformation"""

from __future__ import annotations

import asyncio
import functools as ft
import typing as tp
from collections.abc import Callable
from operator import setitem

from autoform.core import Interpreter, get_interp, using_interp
from autoform.core import IR, IREqn, IRLit, IRVar, Value, Var, is_irvar
from autoform.core import (
    Primitive,
    async_rules,
    batch_rules,
    dce_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, lru_cache, treelib


class IRBVar(IRVar): ...


# ==================================================================================================
# BATCHING UTILITIES
# ==================================================================================================


def is_axis_spec(x) -> bool:
    return x is None or isinstance(x, type)


def make_is_batch_leaf(in_axes: Tree) -> Callable[[Tree], bool]:
    batch_types = tuple(treelib.leaves(in_axes, is_leaf=lambda x: isinstance(x, type)))
    return (lambda x: isinstance(x, batch_types)) if batch_types else (lambda _: False)


def infer_batch_size(tree: Tree, in_axes: Tree) -> int:
    is_batch_leaf = make_is_batch_leaf(in_axes)
    axes_leaves = treelib.leaves(in_axes, is_leaf=is_axis_spec)
    col_leaves = treelib.leaves(tree, is_leaf=is_batch_leaf)
    return next((len(v) for v, a in zip(col_leaves, axes_leaves) if a is not None), 0)


def broadcast_in_axes_prefix(in_axes: Tree, tree: Tree) -> Tree:
    is_batch_leaf = make_is_batch_leaf(in_axes)
    is_leaf = lambda x: is_axis_spec(x) or is_batch_leaf(x)
    return treelib.broadcast_prefix(in_axes, tree, is_leaf=is_leaf)


def in_axes_to_batch_tree(in_axes: Tree) -> Tree[bool]:
    return treelib.map(lambda ax: ax is not None, in_axes, is_leaf=is_axis_spec)


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


# ==================================================================================================
# BATCH CALL
# ==================================================================================================

batch_call_p = Primitive("batch_call", tag="transformation")


class BatchInterpreter(Interpreter):
    def __init__(self, *, batch_size: int):
        self.parent = get_interp()
        self.batch_size = batch_size

    def process(self, prim: Primitive, in_tree: Tree, **params):
        with using_interp(self.parent):
            batch_size, in_batched, in_values = in_tree
            return batch_rules[prim](batch_size, in_batched, in_values, **params)


@ft.partial(impl_rules.def_rule, batch_call_p)
def impl_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    col_tree = in_tree
    axes_tree = broadcast_in_axes_prefix(in_axes, col_tree)
    in_batched_tree: Tree[bool] = in_axes_to_batch_tree(axes_tree)
    in_batched_tree = treelib.broadcast_prefix(in_batched_tree, ir.in_irtree)
    batch_size = infer_batch_size(col_tree, in_axes)

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

    with using_interp(BatchInterpreter(batch_size=batch_size)):
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
    del ir, in_axes
    is_var = lambda x: isinstance(x, Var)
    return treelib.map(lambda _: Var(), in_tree, is_leaf=is_var)


@ft.partial(push_rules.def_rule, batch_call_p)
def pushforward_batch_call(
    primals: Tree,
    tangents: Tree,
    *,
    ir: IR,
    in_axes: Tree,
) -> tuple[Tree, Tree]:
    from autoform.evaluation import run_ir
    from autoform.transforms.ad import pushforward_ir

    p_cols, t_cols = primals, tangents
    pf_ir = pushforward_ir(ir)
    batch_pf_ir = batch_ir(pf_ir, in_axes=(in_axes, in_axes))
    return run_ir(batch_pf_ir, (p_cols, t_cols))


@ft.partial(pull_fwd_rules.def_rule, batch_call_p)
def pullback_fwd_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> tuple[Tree, Tree]:
    from autoform.evaluation import run_ir

    col_tree = in_tree
    batched_ir = batch_ir(ir, in_axes=in_axes)
    out_cols = run_ir(batched_ir, col_tree)
    residuals = (col_tree, in_axes)
    return out_cols, residuals


@ft.partial(pull_bwd_rules.def_rule, batch_call_p)
def pullback_bwd_batch_call(residuals: Tree, cotangent_out: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    from autoform.evaluation import run_ir
    from autoform.transforms.ad import pullback_ir

    p_cols, _ = residuals
    c_out_cols = cotangent_out
    pb_ir = pullback_ir(ir)
    batch_pb_ir = batch_ir(pb_ir, in_axes=(in_axes, list))
    _, c_in_cols = run_ir(batch_pb_ir, (p_cols, c_out_cols))
    return c_in_cols


@ft.partial(batch_rules.def_rule, batch_call_p)
def batch_batch_call(
    batch_size: int,
    in_batched: Tree,
    in_tree: Tree,
    *,
    ir: IR,
    in_axes: Tree,
) -> tuple[Tree, Tree]:
    from autoform.evaluation import run_ir

    col_cols = in_tree

    in_axes_tree = broadcast_in_axes_prefix(in_axes, col_cols)
    get_is_leaf = make_is_batch_leaf(in_axes_tree)
    batched_ir = batch_ir(ir, in_axes=in_axes)

    def get(b):
        get_at_b = lambda v, a: v if a is None else v[b]
        return treelib.map(get_at_b, col_cols, in_axes_tree, is_leaf=get_is_leaf)

    results = [run_ir(batched_ir, get(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, ir.out_irtree)
    out_spec = treelib.structure(ir.out_irtree)
    leaves_bi = [out_spec.flatten_up_to(r) for r in results]
    num_leaves = out_spec.num_leaves
    stacked = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(num_leaves)]
    out_cols = out_spec.unflatten(stacked)
    return out_cols, out_batched


@ft.partial(async_rules.def_rule, batch_call_p)
async def async_batch_call(in_tree: Tree, *, ir: IR, in_axes: Tree) -> Tree:
    from autoform.evaluation import arun_ir

    col_tree = in_tree

    axes_tree = broadcast_in_axes_prefix(in_axes, col_tree)
    run_is_leaf = make_is_batch_leaf(axes_tree)
    batch_size = infer_batch_size(col_tree, in_axes)

    async def run_item(b: int):
        def get(v, a):
            return v if a is None else v[b]

        value = treelib.map(get, col_tree, axes_tree, is_leaf=run_is_leaf)
        return await arun_ir(ir, value)

    results = await asyncio.gather(*[run_item(b) for b in range(batch_size)])
    out_spec = treelib.structure(ir.out_irtree)
    leaves_bi = [out_spec.flatten_up_to(r) for r in results]
    stacked = [[leaves_bi[b][i] for b in range(batch_size)] for i in range(out_spec.num_leaves)]
    return out_spec.unflatten(stacked)


@ft.partial(dce_rules.def_rule, batch_call_p)
def dce_batch_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    from autoform.transforms.optims import default_dce, dce_ir

    new_eqn = ireqn.using(ir=dce_ir(ireqn.params["ir"]))
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn


# ==================================================================================================
# BATCH IR TRANSFORMATION
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def batch_ir(ir: IR, in_axes: Tree[type | None] = list) -> IR:
    """Transform an IR to process batched inputs.

    Creates a batched version of the IR that processes multiple inputs
    simultaneously. Use `in_axes` to specify which inputs are batched (type
    like `list`) vs broadcast (None).

    Args:
        ir: The IR to transform.
        in_axes: Axis specification tree matching input structure.
            - Container type: This input is batched (e.g. list of values).
            - `None`: This input is broadcast (same value for all batch items).

    Returns:
        A new IR that takes batched inputs and returns batched outputs.

    Example:
        >>> import autoform as af
        >>> def greet(greeting, name):
        ...     return af.concat(greeting, name)
        >>> ir = af.build_ir(greet, "Hi", "World")
        >>> # Batch over names contained in list, broadcast greeting
        >>> batched = af.batch_ir(ir, in_axes=(None, list))
        >>> af.run_ir(batched, ("Hello, ", ["Alice", "Bob", "Carol"]))
        ['Hello, Alice', 'Hello, Bob', 'Hello, Carol']
    """

    def make_b(atom):
        return IRBVar.fresh(source=atom) if is_irvar(atom) else atom

    b_in_irtree = treelib.map(make_b, ir.in_irtree)
    b_out_irtree = treelib.map(make_b, ir.out_irtree)
    eqn = IREqn(batch_call_p, b_in_irtree, b_out_irtree, dict(ir=ir, in_axes=in_axes))
    return IR([eqn], b_in_irtree, b_out_irtree)
