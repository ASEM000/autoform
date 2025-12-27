"""Harvest primitives - sow/reap/plant for effect handling."""

from __future__ import annotations

import functools as ft
import typing as tp
from autoform.core import Interpreter, get_interp, using_interp
from autoform.core import IR, EvalType
from autoform.core import (
    Primitive,
    batch_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree
from autoform.core import call_ir

# ==================================================================================================
# SOW
# ==================================================================================================

sow_p = Primitive("sow", tag="core")


def sow(in_tree: Tree, /, *, tag: tp.Hashable, name: tp.Hashable) -> Tree:
    """Tag a value with a category and name for later collection.

    `sow` marks a value with a `tag` (category) and `name` (unique identifier)
    that can be collected by `run_and_reap`. It acts as an identity operation in
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
        >>> ir = af.build_ir(program)("test")
        >>> result, reaped = af.reap_ir(ir, tag="debug")("What is 6*7?")
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
    primal: Tree, tangent: Tree, *, tag: tp.Hashable, name: tp.Hashable
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
    in_residuals: Tree, out_cotangent: Tree, *, tag: tp.Hashable, name: tp.Hashable
) -> Tree:
    del in_residuals
    return sow(out_cotangent, tag=(tag, "cotangent"), name=name)


@ft.partial(batch_rules.def_rule, sow_p)
def batch_sow(
    _: int, in_batched: Tree, x: Tree, *, tag: tp.Hashable, name: tp.Hashable
) -> tuple[Tree, Tree]:
    return sow(x, tag=(tag, "batch"), name=name), in_batched


# ==================================================================================================
# REAP
# ==================================================================================================

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


def reap_ir[**P, R](ir: IR, *, tag: tp.Hashable) -> tp.Callable[P, tuple[R, Reaped]]:
    """Create a reaping executor for an IR.

    Args:
        ir: The intermediate representation to run.
        tag: The tag to filter sown values by.

    Returns:
        A callable that executes the IR and returns (result, reaped_dict).

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prompt = af.sow(af.format("Q: {}", x), tag="debug", name="prompt")
        ...     return af.concat(prompt, " A: 42")
        >>> ir = af.build_ir(program)("test")
        >>> result, reaped = af.reap_ir(ir, tag="debug")("What?")
        >>> result
        'Q: What? A: 42'
        >>> reaped
        {'prompt': 'Q: What?'}
    """

    def execute(*args: P.args, **kwargs: P.kwargs) -> tuple[R, Reaped]:
        with using_interp(ReapInterpreter(tag=tag)) as reaper:
            result = call_ir(ir)(*args, **kwargs)
        return result, reaper.reaped

    return execute


# ==================================================================================================
# PLANT
# ==================================================================================================


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


def plant_ir[**P, R](ir: IR, plants: Reaped, *, tag: tp.Hashable) -> tp.Callable[P, R]:
    """Create a planting executor for an IR.

    Args:
        ir: The intermediate representation to run.
        plants: Dictionary mapping sow names to values to inject.
        tag: The tag to filter sow locations by.

    Returns:
        A callable that executes the IR with planted values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.sow(af.concat("Hello, ", x), tag="cache", name="greeting")
        >>> ir = af.build_ir(program)("test")
        >>> af.plant_ir(ir, {"greeting": "CACHED"}, tag="cache")("World")
        'CACHED'
    """

    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        with using_interp(PlantInterpreter(tag=tag, plants=plants)):
            return call_ir(ir)(*args, **kwargs)

    return execute
