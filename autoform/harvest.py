"""Harvest primitives"""

from __future__ import annotations

import functools as ft
import typing as tp

from autoform.core import Interpreter, build_ir, get_interp, using_interp
from autoform.core import IR, EvalType, IRVar, Var, is_irvar, is_var
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
    p = sow(primal, tag=(tag, "primal"), name=name)
    t = sow(tangent, tag=(tag, "tangent"), name=name)
    return p, t


@ft.partial(pull_fwd_rules.def_rule, sow_p)
def pullback_fwd_sow(in_tree: Tree, *, tag: tp.Hashable, name: tp.Hashable) -> tuple[Tree, Tree]:
    out = sow(in_tree, tag=(tag, "primal"), name=name)
    return out, out


@ft.partial(pull_bwd_rules.def_rule, sow_p)
def pullback_bwd_sow(
    in_residuals: Tree,
    out_cotangent: Tree,
    *,
    tag: tp.Hashable,
    name: tp.Hashable,
) -> Tree:
    del in_residuals
    return sow(out_cotangent, tag=(tag, "cotangent"), name=name)


@ft.partial(batch_rules.def_rule, sow_p)
def batch_sow(
    _: int,
    in_batched: Tree,
    x: Tree,
    *,
    tag: tp.Hashable,
    name: tp.Hashable,
) -> tuple[Tree, Tree]:
    return sow(x, tag=(tag, "batch"), name=name), in_batched


# ==================================================================================================
# REAP
# ==================================================================================================

reap_call_p = Primitive("reap_call", tag="harvest")

type Reaped = dict[tp.Hashable, Tree]


class ReapInterpreter(Interpreter):
    def __init__(self, *, tag: tp.Hashable):
        self.tag = tag
        self.reaped: Reaped = {}
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        result = self.parent.process(prim, in_tree, **params)
        if prim == sow_p and params.get("tag") == self.tag:
            self.reaped[params["name"]] = result
        return result


@ft.partial(impl_rules.def_rule, reap_call_p)
def impl_reap_call(in_tree: Tree, *, ir: IR, tag: tp.Hashable) -> tuple[Tree, Reaped]:
    from autoform.evaluation import run_ir

    with using_interp(ReapInterpreter(tag=tag)) as reap:
        result = run_ir(ir, in_tree)
    return result, reap.reaped


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
    from autoform.ad import pushforward_ir

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
    from autoform.ad import pullback_ir

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
    from autoform.batch import batch_ir

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
# PLANT
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
    from autoform.evaluation import run_ir

    with using_interp(PlantInterpreter(tag=tag, plants=plants)):
        return run_ir(ir, in_tree)


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
    from autoform.ad import pushforward_ir

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
    from autoform.ad import pullback_ir

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
    from autoform.batch import batch_ir

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


# ==================================================================================================
# SPLIT AND MERGE IR
# ==================================================================================================


def split_ir(ir: IR, *, tag: tp.Hashable, name: tp.Hashable) -> tuple[IR, IR]:
    """Split IR at a sow point into (before, after).

    Finds the first `sow` equation matching the given tag and name, then splits
    the IR into two parts: equations before (inclusive of the sow) and after.

    Args:
        ir: The intermediate representation to split.
        tag: The tag to match on the sow.
        name: The name to match on the sow.

    Returns:
        A tuple of (ir_before, ir_after) where:
        - ir_before: IR from input to the sow point (sow output is the output)
        - ir_after: IR from the sow output to the original output

    Raises:
        ValueError: If no matching sow is found.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     a = af.concat("Step1: ", x)
        ...     mid = af.sow(a, tag="split", name="checkpoint")
        ...     b = af.concat(mid, " -> Step2")
        ...     return b
        >>> ir = af.build_ir(program, "test")
        >>> ir1, ir2 = af.split_ir(ir, tag="split", name="checkpoint")
        >>> af.run_ir(ir1, "input")
        'Step1: input'
        >>> af.run_ir(ir2, "Step1: input")
        'Step1: input -> Step2'
    """
    assert isinstance(ir, IR), f"{type(ir)=} is not an IR instance."

    split_idx = split_out_irtree = None

    for i, ireqn in enumerate(ir.ireqns):
        if (
            ireqn.prim == sow_p
            and ireqn.params.get("tag") == tag
            and ireqn.params.get("name") == name
        ):
            split_idx = i
            split_out_irtree = ireqn.out_irtree
            break

    if split_idx is None:
        return ir, IR([], (), ())

    ir1_eqns = ir.ireqns[: split_idx + 1]
    ir1 = IR(ireqns=ir1_eqns, in_irtree=ir.in_irtree, out_irtree=split_out_irtree)

    ir2_eqns = ir.ireqns[split_idx + 1 :]
    ir2 = IR(ireqns=ir2_eqns, in_irtree=split_out_irtree, out_irtree=ir.out_irtree)

    return ir1, ir2


def merge_ir(ir1: IR, ir2: IR) -> IR:
    """Merge two IRs by concatenating them sequentially.

    The output of ir1 must have the same tree structure as the input of ir2.
    The resulting IR takes ir1's input and produces ir2's output.

    Args:
        ir1: The first IR (executed first).
        ir2: The second IR (executed second, receives ir1's output).

    Returns:
        A merged IR that is equivalent to ir2(ir1(input)).

    Raises:
        ValueError: If ir1's output structure doesn't match ir2's input structure.

    Example:
        >>> import autoform as af
        >>> def step1(x):
        ...     return af.concat("Step1: ", x)
        >>> def step2(x):
        ...     return af.concat(x, " -> Step2")
        >>> ir1 = af.build_ir(step1, "test")
        >>> ir2 = af.build_ir(step2, "test")
        >>> merged = af.merge_ir(ir1, ir2)
        >>> af.run_ir(merged, "input")
        'Step1: input -> Step2'
    """
    assert isinstance(ir1, IR), f"{type(ir1)=} is not an IR instance."
    assert isinstance(ir2, IR), f"{type(ir2)=} is not an IR instance."

    out1_structure = treelib.structure(ir1.out_irtree)
    in2_structure = treelib.structure(ir2.in_irtree)

    assert out1_structure == in2_structure, (
        f"IR output/input structure mismatch: "
        f"ir1 output structure {out1_structure} != ir2 input structure {in2_structure}"
    )

    # NOTE(asem): the key idea here is that ir2 might reference ir1 vars
    # so we need to bridge them. one way to do it is to map the end of ir1 to the start of ir2
    # and then remap ir2's input to ir1's output whenever ir2 references ir1 vars. the rest of
    # the code is simply walking over ir2 and remapping its input to ir1's output if needed

    out1_leaves = treelib.leaves(ir1.out_irtree)
    in2_leaves = treelib.leaves(ir2.in_irtree)

    var_mapping: dict[IRVar, tp.Any] = {}
    for out_atom, in_atom in zip(out1_leaves, in2_leaves, strict=True):
        # NOTE(asem): bridge ir1's output to ir2's input
        if is_irvar(in_atom):
            var_mapping[in_atom] = out_atom

    def remap_atom(atom):
        # NOTE(asem): act as identiy passing atoms of ir2
        # exceot if ir2 references ir1 vars then bridge them
        if is_irvar(atom) and atom in var_mapping:
            return var_mapping[atom]
        return atom

    remapped_eqns = []
    for ireqn in ir2.ireqns:
        remapped_in = treelib.map(remap_atom, ireqn.in_irtree)
        from autoform.core import IREqn

        remapped_eqns.append(IREqn(ireqn.prim, remapped_in, ireqn.out_irtree, ireqn.params))

    merged_eqns = list(ir1.ireqns) + remapped_eqns
    remapped_out = treelib.map(remap_atom, ir2.out_irtree)

    return IR(ireqns=merged_eqns, in_irtree=ir1.in_irtree, out_irtree=remapped_out)
