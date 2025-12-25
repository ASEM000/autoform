"""Harvest primitives"""

from __future__ import annotations

import functools as ft
import typing as tp
from operator import setitem

from autoform.core import Interpreter, build_ir, get_interp, using_interp
from autoform.core import IR, EvalType, IRLit, IRVar, Value, Var, is_irvar, is_var
from autoform.core import (
    Primitive,
    batch_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, treelib, lru_cache

# ==================================================================================================
# SOW
# ==================================================================================================

sow_p = Primitive("sow", tag="core")


def sow(in_tree: Tree, /, *, tag: tp.Hashable, name: tp.Hashable) -> Tree:
    """Tag a value with a category and name for later collection.

    `sow` marks a value with a `tag` (category) and `name` (unique identifier)
    that can be collected by `reap_ir`. It acts as an identity operation in
    normal execution, but when run under a `ReapInterpreter`, the sown values
    are captured.

    Args:
        in_tree: The value to sow (returned unchanged).
        tag: Category for filtering (e.g., "debug", "cache", "metrics").
        name: Unique identifier within the tag namespace.

    Returns:
        The input value unchanged.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.sow(af.format("Q: {}", x), tag="debug", name="prompt")
        ...     response = af.concat(prompt, " A: 42")
        ...     return af.sow(response, tag="debug", name="response")
        >>> ir = af.build_ir(program, "test")
        >>> reap = af.reap_ir(ir, tag="debug")
        >>> result, reaped = af.run_ir(reap, "What is 6*7?")
        >>> result
        'Q: What is 6*7? A: 42'
        >>> reaped["prompt"]
        'Q: What is 6*7?'
        >>> reaped["response"]
        'Q: What is 6*7? A: 42'
    """
    assert hash(tag) is not None, "Tag must be hashable"
    assert hash(name) is not None, "Name must be hashable"
    return sow_p.bind(in_tree, tag=tag, name=name)


@ft.partial(impl_rules.def_rule, sow_p)
def impl_sow(in_tree: Tree, *, tag: tp.Hashable, name: tp.Hashable) -> Tree:
    del tag, name
    return in_tree


@ft.partial(eval_rules.def_rule, sow_p)
def eval_sow(in_tree: Tree[EvalType], *, tag: tp.Hashable, name: tp.Hashable) -> Tree[EvalType]:
    del tag, name
    return in_tree


@ft.partial(push_rules.def_rule, sow_p)
def pushforward_sow(
    primal: Tree,
    tangent: Tree,
    *,
    tag: tp.Hashable,
    name: tp.Hashable,
) -> tuple[Tree, Tree]:
    del tag, name
    return primal, tangent


@ft.partial(pull_fwd_rules.def_rule, sow_p)
def pullback_fwd_sow(in_tree: Tree, *, tag: tp.Hashable, name: tp.Hashable) -> tuple[Tree, Tree]:
    del tag, name
    return in_tree, in_tree


@ft.partial(pull_bwd_rules.def_rule, sow_p)
def pullback_bwd_sow(
    in_residuals: Tree,
    out_cotangent: Tree,
    *,
    tag: tp.Hashable,
    name: tp.Hashable,
) -> Tree:
    del in_residuals, tag, name
    return out_cotangent


@ft.partial(batch_rules.def_rule, sow_p)
def batch_sow(
    _: int,
    in_batched: Tree,
    x: Tree,
    *,
    tag: tp.Hashable,
    name: tp.Hashable,
) -> tuple[Tree, Tree]:
    del tag, name
    return x, in_batched


# ==================================================================================================
# REAP CALL PRIMITIVE
# ==================================================================================================

reap_call_p = Primitive("reap_call", tag="harvest")

type Reaped = dict[tp.Hashable, Tree]


@ft.partial(impl_rules.def_rule, reap_call_p)
def impl_reap_call(in_tree: Tree, *, ir: IR, tag: tp.Hashable) -> tuple[Tree, Reaped]:
    reaped: Reaped = {}
    env: dict[IRVar, Value] = {}

    def write(atom, value: Value):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom) -> Value:
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    treelib.map(write, ir.in_irtree, in_tree)

    for ireqn in ir.ireqns:
        in_ireqn = treelib.map(read, ireqn.in_irtree)
        out_ireqn = ireqn.prim.bind(in_ireqn, **ireqn.params)
        treelib.map(write, ireqn.out_irtree, out_ireqn)
        if ireqn.prim == sow_p and ireqn.params.get("tag") == tag:
            reaped[ireqn.params["name"]] = out_ireqn

    result = treelib.map(read, ir.out_irtree)
    return result, reaped


@ft.partial(eval_rules.def_rule, reap_call_p)
def eval_reap_call(in_tree: Tree, *, ir: IR, tag: tp.Hashable) -> tuple[Tree, Var]:
    del ir, tag
    out = treelib.map(lambda _: Var(), in_tree, is_leaf=is_var)
    return out, Var()


@ft.partial(push_rules.def_rule, reap_call_p)
def push_reap_call(
    primals: Tree,
    tangents: Tree,
    *,
    ir: IR,
    tag: tp.Hashable,
) -> tuple[tuple[Tree, Reaped], tuple[Tree, Reaped]]:
    from autoform.transforms.ad import pushforward_ir

    pf_ir = pushforward_ir(ir)
    p_result, p_reaped = reap_call_p.bind((primals, tangents), ir=pf_ir, tag=tag)
    primal_out, tangent_out = p_result
    return (primal_out, p_reaped), (tangent_out, {})


@ft.partial(pull_fwd_rules.def_rule, reap_call_p)
def pull_fwd_reap_call(
    in_tree: Tree,
    *,
    ir: IR,
    tag: tp.Hashable,
) -> tuple[tuple[Tree, Reaped], tuple[Tree, Reaped]]:
    result, reaped = impl_reap_call(in_tree, ir=ir, tag=tag)
    return (result, reaped), (in_tree, reaped)


@ft.partial(pull_bwd_rules.def_rule, reap_call_p)
def pull_bwd_reap_call(
    residuals: tuple[Tree, Reaped],
    cotangent: tuple[Tree, Reaped],
    *,
    ir: IR,
    tag: tp.Hashable,
) -> Tree:
    from autoform.evaluation import run_ir
    from autoform.transforms.ad import pullback_ir

    in_tree, _ = residuals
    ct_result, _ = cotangent
    pb_ir = pullback_ir(ir)
    _, ct_in = run_ir(pb_ir, (in_tree, ct_result))
    return ct_in


@ft.partial(batch_rules.def_rule, reap_call_p)
def batch_reap_call(
    batch_size: int,
    in_batched: Tree,
    in_values: Tree,
    *,
    ir: IR,
    tag: tp.Hashable,
) -> tuple[tuple[Tree, Reaped], tuple[Tree, bool]]:
    from autoform.transforms.batch import batch_ir

    batched_ir = batch_ir(ir)
    result, reaped = reap_call_p.bind(in_values, ir=batched_ir, tag=tag)
    return (result, reaped), (in_batched, False)


@ft.partial(lru_cache, maxsize=256)
def reap_ir(ir: IR, *, tag: tp.Hashable) -> IR:
    """Transform IR to return (result, reaped_dict).

    Args:
        ir: The intermediate representation to transform.
        tag: The tag to filter sown values by.

    Returns:
        A new IR that outputs (original_result, reaped_dict).

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.sow(af.format("Q: {}", x), tag="debug", name="prompt")
        ...     return af.concat(prompt, " A: 42")
        >>> ir = af.build_ir(program, "test")
        >>> reap = af.reap_ir(ir, tag="debug")
        >>> result, reaped = af.run_ir(reap, "What?")
        >>> result
        'Q: What? A: 42'
        >>> reaped
        {'prompt': 'Q: What?'}
    """
    assert isinstance(ir, IR), f"{type(ir)=} is not an IR instance."

    def func(in_tree):
        return reap_call_p.bind(in_tree, ir=ir, tag=tag)

    return build_ir(func, treelib.map(lambda _: "", ir.in_irtree))


# ==================================================================================================
# PLANT CALL PRIMITIVE
# ==================================================================================================

plant_call_p = Primitive("plant_call", tag="harvest")


class PlantInterpreter(Interpreter):
    def __init__(self, *, tag: tp.Hashable, plants: dict[tp.Hashable, Tree]):
        self.tag = tag
        self.plants = plants
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        if (
            prim == sow_p
            and params.get("tag") == self.tag
            and (name := params.get("name")) in self.plants
        ):
            return self.plants[name]
        with using_interp(self.parent):
            return self.parent.process(prim, in_tree, **params)


@ft.partial(impl_rules.def_rule, plant_call_p)
def impl_plant_call(
    in_tree: Tree,
    *,
    ir: IR,
    tag: tp.Hashable,
    plants: dict[tp.Hashable, Tree],
) -> Tree:
    env: dict[IRVar, Value] = {}

    def write(atom, value: Value):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom) -> Value:
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    treelib.map(write, ir.in_irtree, in_tree)

    for ireqn in ir.ireqns:
        in_ireqn = treelib.map(read, ireqn.in_irtree)
        if (
            ireqn.prim == sow_p
            and ireqn.params.get("tag") == tag
            and ireqn.params["name"] in plants
        ):
            out_ireqn = plants[ireqn.params["name"]]
        else:
            out_ireqn = ireqn.prim.bind(in_ireqn, **ireqn.params)
        treelib.map(write, ireqn.out_irtree, out_ireqn)

    return treelib.map(read, ir.out_irtree)


@ft.partial(eval_rules.def_rule, plant_call_p)
def eval_plant_call(
    in_tree: Tree, *, ir: IR, tag: tp.Hashable, plants: dict[tp.Hashable, Tree]
) -> Tree:
    del ir, tag, plants
    return treelib.map(lambda _: Var(), in_tree, is_leaf=is_var)


@ft.partial(push_rules.def_rule, plant_call_p)
def push_plant_call(
    primals: Tree,
    tangents: Tree,
    *,
    ir: IR,
    tag: tp.Hashable,
    plants: dict[tp.Hashable, Tree],
) -> tuple[Tree, Tree]:
    from autoform.transforms.ad import pushforward_ir

    pf_ir = pushforward_ir(ir)
    primal_tangent = plant_call_p.bind((primals, tangents), ir=pf_ir, tag=tag, plants=plants)
    primal_out, tangent_out = primal_tangent
    return primal_out, tangent_out


@ft.partial(pull_fwd_rules.def_rule, plant_call_p)
def pull_fwd_plant_call(
    in_tree: Tree, *, ir: IR, tag: tp.Hashable, plants: dict[tp.Hashable, Tree]
) -> tuple[Tree, Tree]:
    result = impl_plant_call(in_tree, ir=ir, tag=tag, plants=plants)
    return result, in_tree  # residuals = input


@ft.partial(pull_bwd_rules.def_rule, plant_call_p)
def pull_bwd_plant_call(
    residuals: Tree, cotangent: Tree, *, ir: IR, tag: tp.Hashable, plants: dict[tp.Hashable, Tree]
) -> Tree:
    from autoform.evaluation import run_ir
    from autoform.transforms.ad import pullback_ir

    in_tree = residuals
    pb_ir = pullback_ir(ir)
    _, ct_in = run_ir(pb_ir, (in_tree, cotangent))
    return ct_in


@ft.partial(batch_rules.def_rule, plant_call_p)
def batch_plant_call(
    batch_size: int,
    in_batched: Tree,
    in_values: Tree,
    *,
    ir: IR,
    tag: tp.Hashable,
    plants: dict[tp.Hashable, Tree],
) -> tuple[Tree, Tree]:
    from autoform.transforms.batch import batch_ir

    batched_ir = batch_ir(ir)
    result = plant_call_p.bind(in_values, ir=batched_ir, tag=tag, plants=plants)
    return result, in_batched


def plant_ir(ir: IR, plants: dict[tp.Hashable, Tree], *, tag: tp.Hashable) -> IR:
    """Transform IR to inject planted values at sow locations.

    Args:
        ir: The intermediate representation to transform.
        plants: Dictionary mapping sow names to values to inject.
        tag: The tag to filter sow locations by.

    Returns:
        A new IR with planted values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.sow(af.concat("Hello, ", x), tag="cache", name="greeting")
        >>> ir = af.build_ir(program, "test")
        >>> planted = af.plant_ir(ir, {"greeting": "CACHED"}, tag="cache")
        >>> af.run_ir(planted, "World")
        'CACHED'
    """
    assert isinstance(ir, IR), f"{type(ir)=} is not an IR instance."

    def func(in_tree):
        return plant_call_p.bind(in_tree, ir=ir, tag=tag, plants=plants)

    return build_ir(func, treelib.map(lambda _: "", ir.in_irtree))
