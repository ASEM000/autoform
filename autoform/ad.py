"""Automatic differentiation over text"""

from __future__ import annotations

import asyncio
import functools as ft
from collections import defaultdict
from collections.abc import Callable
from operator import setitem
from typing import Any

from autoform.core import (
    IR,
    Interpreter,
    IRAtom,
    IREqn,
    IRLit,
    IRVar,
    Primitive,
    TransformationTag,
    Value,
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
from autoform.utils import Tree, batch_index, batch_spec, batch_transpose, lru_cache, treelib


class ADTag(TransformationTag): ...


zero_registry: dict[type, Any] = {}
zero_registry[str] = ""


def zero_cotangent(example):
    assert type(example) in zero_registry, f"No zero cotangent seted for type {type(example)}"
    return zero_registry[type(example)]


def zero_tangent(example):
    assert type(example) in zero_registry, f"No zero tangent seted for type {type(example)}"
    return zero_registry[type(example)]


# ==================================================================================================
# PUSHFORWARD
# ==================================================================================================

pushforward_call_p = Primitive("pushforward_call", tag={ADTag})


class PushforwardInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return push_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return await push_rules.aget(prim)(in_tree, **params)


@ft.partial(lru_cache, maxsize=256)
def pushforward(ir: IR, /) -> IR:
    """Transform an IR to compute primals and tangents (forward-mode AD).

    Creates a new IR that propagates tangent (perturbation) vectors alongside
    primal values. Useful for computing Jacobian-vector products (JVPs).

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(primals, tangents) -> (out_primalputs, out_tangentputs)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.trace(program)("a", "b")
        >>> pf_ir = af.pushforward(ir)
        >>> primals, tangents = call(pf_ir)((("Hello", " World"), ("dx", "dy")))
        >>> primals
        'Hello World'
        >>> tangents
        'dxdy'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom: IRAtom):
        return IRVar.fresh(type=atom.type, source=atom) if is_irvar(atom) else atom

    def make_t(atom: IRAtom):
        return (
            IRVar.fresh(type=atom.type, source=atom)
            if is_irvar(atom)
            else IRLit(zero_tangent(atom.value))
        )

    p_in_irtree = treelib.map(make_p, ir.in_irtree)
    t_in_irtree = treelib.map(make_t, ir.in_irtree)
    in_irtree = (p_in_irtree, t_in_irtree)
    out_p_irtree = treelib.map(make_p, ir.out_irtree)
    out_t_irtree = treelib.map(make_t, ir.out_irtree)
    out_irtree = (out_p_irtree, out_t_irtree)
    # NOTE(asem): effect on the wrapper IREqn is unused at execution time.
    # impl_pushforward_call never reads active_effect, and no EffectInterpreter
    # handler targets pushforward_call_p. inner IREqns carry their own effects
    # and restore them via ireqn.bind().
    effect = active_effect.get()
    ireqn = IREqn(pushforward_call_p, effect, in_irtree, out_irtree, dict(ir=ir))
    return IR([ireqn], in_irtree, out_irtree)


def impl_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, t_in_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    t_env: dict[IRVar, Value] = {}

    def write_p(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def write_t(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(t_env, atom, value)

    def read_p(atom: IRAtom) -> Value:
        return p_env[atom] if is_irvar(atom) else atom.value

    def read_t(atom: IRAtom) -> Value:
        return t_env[atom] if is_irvar(atom) else zero_tangent(atom.value)

    treelib.map(write_p, ir.in_irtree, p_in_tree)
    treelib.map(write_t, ir.in_irtree, t_in_tree)

    with using_interpreter(PushforwardInterpreter()):
        for ireqn in ir.ireqns:
            p_in_ireqn = treelib.map(read_p, ireqn.in_irtree)
            t_in_ireqn = treelib.map(read_t, ireqn.in_irtree)
            in_tree = (p_in_ireqn, t_in_ireqn)
            out_p_ireqn, out_t_ireqn = ireqn.bind(in_tree, **ireqn.params)
            treelib.map(write_p, ireqn.out_irtree, out_p_ireqn)
            treelib.map(write_t, ireqn.out_irtree, out_t_ireqn)

    out_p_tree = treelib.map(read_p, ir.out_irtree)
    out_t_tree = treelib.map(read_t, ir.out_irtree)
    return out_p_tree, out_t_tree


async def aimpl_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, t_in_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    t_env: dict[IRVar, Value] = {}

    def write_p(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def write_t(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(t_env, atom, value)

    def read_p(atom: IRAtom) -> Value:
        return p_env[atom] if is_irvar(atom) else atom.value

    def read_t(atom: IRAtom) -> Value:
        return t_env[atom] if is_irvar(atom) else zero_tangent(atom.value)

    treelib.map(write_p, ir.in_irtree, p_in_tree)
    treelib.map(write_t, ir.in_irtree, t_in_tree)

    with using_interpreter(PushforwardInterpreter()):
        for ireqn in ir.ireqns:
            p_in_ireqn = treelib.map(read_p, ireqn.in_irtree)
            t_in_ireqn = treelib.map(read_t, ireqn.in_irtree)
            in_tree = (p_in_ireqn, t_in_ireqn)
            out_p_ireqn, out_t_ireqn = await ireqn.abind(in_tree, **ireqn.params)
            treelib.map(write_p, ireqn.out_irtree, out_p_ireqn)
            treelib.map(write_t, ireqn.out_irtree, out_t_ireqn)

    out_p_tree = treelib.map(read_p, ir.out_irtree)
    out_t_tree = treelib.map(read_t, ir.out_irtree)
    return out_p_tree, out_t_tree


def eval_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    out = treelib.map(iratom_to_evaltype, ir.out_irtree)
    return out, out


def pushforward_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, t_in), (p_in_t, t_in_t) = primals, tangents
    pf_ir = pushforward(ir)
    p_out = call(pf_ir)((p_in, t_in))
    t_out = call(pf_ir)((p_in_t, t_in_t))
    return p_out, t_out


async def apushforward_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, t_in), (p_in_t, t_in_t) = primals, tangents
    pf_ir = pushforward(ir)
    p_out = await acall(pf_ir)((p_in, t_in))
    t_out = await acall(pf_ir)((p_in_t, t_in_t))
    return p_out, t_out


def pullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = call(pf_ir)((p_in, t_in))
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


async def apullback_fwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = await acall(pf_ir)((p_in, t_in))
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


def pullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    in_p, in_t = residuals
    out_c_p, out_c_t = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = call(pb_ir)((in_p, out_c_p))
    _, in_c_t = call(pb_ir)((in_p, out_c_t))
    return (in_c_p, in_c_t)


async def apullback_bwd_pushforward_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    in_p, _in_t = residuals
    out_c_p, out_c_t = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = await acall(pb_ir)((in_p, out_c_p))
    _, in_c_t = await acall(pb_ir)((in_p, out_c_t))
    return (in_c_p, in_c_t)


def batch_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if batch_spec(in_values, in_batched) is None:
        pf_ir = pushforward(ir)
        result = call(pf_ir)(in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = [call(pf_ir)((unbatch_p(b), unbatch_t(b))) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, pf_ir.out_irtree)
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pushforward_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    bs, in_batched, in_values = in_tree
    (p_cols, t_cols), (p_batched, t_batched) = in_values, in_batched

    if batch_spec(in_values, in_batched) is None:
        pf_ir = pushforward(ir)
        result = await acall(pf_ir)(in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_t = ft.partial(batch_index, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = await asyncio.gather(*[acall(pf_ir)((unbatch_p(b), unbatch_t(b))) for b in range(bs)])
    out_batched = treelib.map(lambda _: True, pf_ir.out_irtree)
    out_ib = batch_transpose(bs, out_batched, list(out_bi))
    return out_ib, out_batched


impl_rules.set(pushforward_call_p, impl_pushforward_call)
impl_rules.aset(pushforward_call_p, aimpl_pushforward_call)
eval_rules.set(pushforward_call_p, eval_pushforward_call)
push_rules.set(pushforward_call_p, pushforward_pushforward_call)
push_rules.aset(pushforward_call_p, apushforward_pushforward_call)
pull_fwd_rules.set(pushforward_call_p, pullback_fwd_pushforward_call)
pull_fwd_rules.aset(pushforward_call_p, apullback_fwd_pushforward_call)
pull_bwd_rules.set(pushforward_call_p, pullback_bwd_pushforward_call)
pull_bwd_rules.aset(pushforward_call_p, apullback_bwd_pushforward_call)
batch_rules.set(pushforward_call_p, batch_pushforward_call)
batch_rules.aset(pushforward_call_p, abatch_pushforward_call)


def dce_pushforward_call(ireqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    primals_used, tangents_used = out_used
    original_out_used = treelib.map(lambda p, t: p or t, primals_used, tangents_used)
    new_eqn = ireqn.using(ir=dce(ireqn.params["ir"], out_used=original_out_used))
    return default_dce(new_eqn, out_used)


dce_rules[pushforward_call_p] = dce_pushforward_call


# ==================================================================================================
# PULLBACK
# ==================================================================================================

pullback_call_p = Primitive("pullback_call", tag={ADTag})


cotangent_accumulators: dict[type, Callable[[list], Any]] = {}
cotangent_accumulators[str] = lambda cs: "".join(cs)


def accumulate_cotangents(cotangents: list):
    if not cotangents:
        return ""
    if len(cotangents) == 1:
        return cotangents[0]
    first, *_ = cotangents
    for typ, acc in cotangent_accumulators.items():
        if isinstance(first, typ):
            return acc(cotangents)
    return sum(cotangents[1:], cotangents[0])


class PullbackFwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return pull_fwd_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return await pull_fwd_rules.aget(prim)(in_tree, **params)


class PullbackBwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return pull_bwd_rules.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params):
        with using_interpreter(self.parent):
            return await pull_bwd_rules.aget(prim)(in_tree, **params)


@ft.partial(lru_cache, maxsize=256)
def pullback(ir: IR, /) -> IR:
    """Transform an IR to compute outputs and input cotangents (reverse-mode AD).

    Creates a new IR that computes gradients by backpropagating cotangent
    (adjoint) vectors. Useful for computing vector-Jacobian products (VJPs).

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(inputs, output_cotangents) -> (outputs, input_cotangents)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.trace(program)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> outputs, cotangents = call(pb_ir)((("Hello", " World"), "feedback"))
        >>> outputs
        'Hello World'
        >>> cotangents  # Gradient flows back to both inputs
        ('feedback', 'feedback')
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        return IRVar.fresh(type=atom.type, source=atom) if is_irvar(atom) else atom

    def make_c(atom):
        return (
            IRVar.fresh(type=atom.type, source=atom)
            if is_irvar(atom)
            else IRLit(zero_cotangent(atom.value))
        )

    in_p = treelib.map(make_p, ir.in_irtree)
    out_c = treelib.map(make_c, ir.out_irtree)
    in_irtree = (in_p, out_c)
    out_p = treelib.map(make_p, ir.out_irtree)
    in_c = treelib.map(make_c, ir.in_irtree)
    out_irtree = (out_p, in_c)
    # NOTE(asem): effect on the wrapper IREqn is unused at execution time.
    # impl_pullback_call never reads active_effect, and no EffectInterpreter
    # handler targets pullback_call_p. inner IREqns carry their own effects
    # and restore them via ireqn.bind().
    effect = active_effect.get()
    ireqn = IREqn(pullback_call_p, effect, in_irtree, out_irtree, dict(ir=ir))
    return IR([ireqn], in_irtree, out_irtree)


def impl_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, out_c_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    res_env: dict[int, Tree] = {}
    c_env: defaultdict[IRVar, list] = defaultdict(list)

    def write_p(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def read_p(atom: IRAtom) -> Value:
        return p_env[atom] if is_irvar(atom) else atom.value

    def write_c(atom: IRAtom, value: Value):
        is_irvar(atom) and c_env[atom].append(value)

    def read_c(atom: IRAtom) -> Value:
        return accumulate_cotangents(c_env[atom]) if is_irvar(atom) else zero_cotangent(atom.value)

    treelib.map(write_p, ir.in_irtree, p_in_tree)

    with using_interpreter(PullbackFwdInterpreter()):
        for i, eqn in enumerate(ir.ireqns):
            p_in_ireqn = treelib.map(read_p, eqn.in_irtree)
            out_p_ireqn, residuals = eqn.bind(p_in_ireqn, **eqn.params)
            res_env[i] = residuals
            treelib.map(write_p, eqn.out_irtree, out_p_ireqn)

    treelib.map(write_c, ir.out_irtree, out_c_tree)

    with using_interpreter(PullbackBwdInterpreter()):
        for i, eqn in enumerate(reversed(ir.ireqns)):
            idx = len(ir.ireqns) - 1 - i
            residuals = res_env[idx]
            out_c_ireqn = treelib.map(read_c, eqn.out_irtree)
            c_in_ireqn = eqn.bind((residuals, out_c_ireqn), **eqn.params)
            treelib.map(write_c, eqn.in_irtree, c_in_ireqn)

    out_p_tree = treelib.map(read_p, ir.out_irtree)
    in_c_tree = treelib.map(read_c, ir.in_irtree)
    return out_p_tree, in_c_tree


async def aimpl_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, out_c_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    res_env: dict[int, Tree] = {}
    c_env: defaultdict[IRVar, list] = defaultdict(list)

    def write_p(atom: IRAtom, value: Value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def read_p(atom: IRAtom) -> Value:
        return p_env[atom] if is_irvar(atom) else atom.value

    def write_c(atom: IRAtom, value: Value):
        is_irvar(atom) and c_env[atom].append(value)

    def read_c(atom):
        return accumulate_cotangents(c_env[atom]) if is_irvar(atom) else zero_cotangent(atom.value)

    treelib.map(write_p, ir.in_irtree, p_in_tree)

    with using_interpreter(PullbackFwdInterpreter()):
        for i, eqn in enumerate(ir.ireqns):
            p_in_ireqn = treelib.map(read_p, eqn.in_irtree)
            out_p_ireqn, residuals = await eqn.abind(p_in_ireqn, **eqn.params)
            res_env[i] = residuals
            treelib.map(write_p, eqn.out_irtree, out_p_ireqn)

    treelib.map(write_c, ir.out_irtree, out_c_tree)

    with using_interpreter(PullbackBwdInterpreter()):
        for i, eqn in enumerate(reversed(ir.ireqns)):
            idx = len(ir.ireqns) - 1 - i
            residuals = res_env[idx]
            out_c_ireqn = treelib.map(read_c, eqn.out_irtree)
            c_in_ireqn = await eqn.abind((residuals, out_c_ireqn), **eqn.params)
            treelib.map(write_c, eqn.in_irtree, c_in_ireqn)

    out_p_tree = treelib.map(read_p, ir.out_irtree)
    in_c_tree = treelib.map(read_c, ir.in_irtree)
    return out_p_tree, in_c_tree


def eval_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    out_p = treelib.map(iratom_to_evaltype, ir.out_irtree)
    in_c = treelib.map(iratom_to_evaltype, ir.in_irtree)
    return out_p, in_c


def pushforward_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = primals, tangents
    pb_ir = pullback(ir)
    out_p, in_c = call(pb_ir)((p_in, c_out))
    t_out_p, t_in_c = call(pb_ir)((t_p_in, t_c_out))
    return (out_p, in_c), (t_out_p, t_in_c)


async def apushforward_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    (p_in, c_out), (t_p_in, t_c_out) = primals, tangents
    pb_ir = pullback(ir)
    out_p, in_c = await acall(pb_ir)((p_in, c_out))
    t_out_p, t_in_c = await acall(pb_ir)((t_p_in, t_c_out))
    return (out_p, in_c), (t_out_p, t_in_c)


def pullback_fwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = call(pb_ir)((p_in, c_out))
    residuals = (p_in, c_out, out_p, in_c)
    return (out_p, in_c), residuals


async def apullback_fwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = await acall(pb_ir)((p_in, c_out))
    residuals = (p_in, c_out, out_p, in_c)
    return (out_p, in_c), residuals


def pullback_bwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    p_in, c_out, _, _ = residuals
    out_c_p, in_c_c = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = call(pb_ir)((p_in, out_c_p))
    pf_ir = pushforward(ir)
    _, in_c_cout = call(pf_ir)((p_in, in_c_c))
    return (in_c_p, in_c_cout)


async def apullback_bwd_pullback_call(in_tree: Tree, /, *, ir: IR) -> Tree:
    residuals, out_cotangent = in_tree
    p_in, c_out, _, _ = residuals
    out_c_p, in_c_c = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = await acall(pb_ir)((p_in, out_c_p))
    pf_ir = pushforward(ir)
    _, in_c_cout = await acall(pf_ir)((p_in, in_c_c))
    return (in_c_p, in_c_cout)


def batch_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    size, in_batched, in_values = in_tree
    (p_cols, out_c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = call(pb_ir)(in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(batch_index, out_c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = [call(pb_ir)((unbatch_p(b), unbatch_c(b))) for b in range(size)]
    out_batched = treelib.map(lambda _: True, pb_ir.out_irtree)
    out_ib = batch_transpose(size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_pullback_call(in_tree: Tree, /, *, ir: IR) -> tuple[Tree, Tree]:
    size, in_batched, in_values = in_tree
    (p_cols, out_c_cols) = in_values
    (p_batched, c_batched) = in_batched

    if batch_spec(in_values, in_batched) is None:
        pb_ir = pullback(ir)
        result = await acall(pb_ir)(in_values)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch_p = ft.partial(batch_index, p_cols, p_batched)
    unbatch_c = ft.partial(batch_index, out_c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = await asyncio.gather(*[
        acall(pb_ir)((unbatch_p(b), unbatch_c(b))) for b in range(size)
    ])
    out_batched = treelib.map(lambda _: True, pb_ir.out_irtree)
    out_ib = batch_transpose(size, out_batched, list(out_bi))
    return out_ib, out_batched


impl_rules.set(pullback_call_p, impl_pullback_call)
impl_rules.aset(pullback_call_p, aimpl_pullback_call)
eval_rules.set(pullback_call_p, eval_pullback_call)
push_rules.set(pullback_call_p, pushforward_pullback_call)
push_rules.aset(pullback_call_p, apushforward_pullback_call)
pull_fwd_rules.set(pullback_call_p, pullback_fwd_pullback_call)
pull_fwd_rules.aset(pullback_call_p, apullback_fwd_pullback_call)
pull_bwd_rules.set(pullback_call_p, pullback_bwd_pullback_call)
pull_bwd_rules.aset(pullback_call_p, apullback_bwd_pullback_call)
batch_rules.set(pullback_call_p, batch_pullback_call)
batch_rules.aset(pullback_call_p, abatch_pullback_call)


def dce_pullback_call(ireqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    # TODO(asem): take another look here
    outputs_used, _ = out_used
    new_eqn = ireqn.using(ir=dce(ireqn.params["ir"], out_used=outputs_used))
    return default_dce(new_eqn, out_used)


dce_rules[pullback_call_p] = dce_pullback_call
