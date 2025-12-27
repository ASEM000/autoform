"""Automatic differentiation over text"""

from __future__ import annotations

import functools as ft
import typing as tp
from collections import defaultdict
from collections.abc import Callable
from operator import setitem

from autoform.core import call
from autoform.core import (
    IR,
    IREqn,
    IRLit,
    IRVar,
    IRZero,
    Interpreter,
    Value,
    Var,
    get_interp,
    is_irvar,
    using_interp,
)
from autoform.core import (
    Primitive,
    batch_rules,
    dce_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, unbatch_at, lru_cache, treelib, transpose_batch
from autoform.optims import default_dce, dce

# ==================================================================================================
# PUSHFORWARD
# ==================================================================================================

pushforward_call_p = Primitive("pushforward_call", tag="transformation")


class PushforwardInterpreter(Interpreter):
    def __init__(self):
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params):
        with using_interp(self.parent):
            in_primal, in_tangent = in_tree
            return push_rules[prim](in_primal, in_tangent, **params)


@ft.partial(lru_cache, maxsize=256)
def pushforward(ir: IR) -> IR:
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
        >>> ir = af.build_ir(program)("a", "b")
        >>> pf_ir = af.pushforward(ir)
        >>> primals, tangents = call(pf_ir)((("Hello", " World"), ("dx", "dy")))
        >>> primals
        'Hello World'
        >>> tangents
        'dxdy'
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        return IRPVar.fresh(source=atom) if is_irvar(atom) else atom

    def make_t(atom):
        return IRTVar.fresh(source=atom) if is_irvar(atom) else IRZero(atom)

    p_in_irtree = treelib.map(make_p, ir.in_irtree)
    t_in_irtree = treelib.map(make_t, ir.in_irtree)
    in_irtree = (p_in_irtree, t_in_irtree)
    out_p_irtree = treelib.map(make_p, ir.out_irtree)
    out_t_irtree = treelib.map(make_t, ir.out_irtree)
    out_irtree = (out_p_irtree, out_t_irtree)
    eqn = IREqn(pushforward_call_p, in_irtree, out_irtree, dict(ir=ir))
    return IR([eqn], in_irtree, out_irtree)


class IRPVar(IRVar): ...


class IRTVar(IRVar): ...


@ft.partial(impl_rules.def_rule, pushforward_call_p)
def impl_pushforward_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, t_in_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    t_env: dict[IRVar, Value] = {}

    def write_p(atom, value: Value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def write_t(atom, value: Value):
        is_irvar(atom) and setitem(t_env, atom, value)

    def read_p(atom) -> Value:
        return p_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    def read_t(atom) -> Value:
        return t_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    treelib.map(write_p, ir.in_irtree, p_in_tree)
    treelib.map(write_t, ir.in_irtree, t_in_tree)

    with using_interp(PushforwardInterpreter()):
        for ireqn in ir.ireqns:
            p_in_ireqn = treelib.map(read_p, ireqn.in_irtree)
            t_in_ireqn = treelib.map(read_t, ireqn.in_irtree)
            in_tree = (p_in_ireqn, t_in_ireqn)
            out_p_ireqn, out_t_ireqn = ireqn.prim.bind(in_tree, **ireqn.params)
            treelib.map(write_p, ireqn.out_irtree, out_p_ireqn)
            treelib.map(write_t, ireqn.out_irtree, out_t_ireqn)

    out_p_tree = treelib.map(read_p, ir.out_irtree)
    out_t_tree = treelib.map(read_t, ir.out_irtree)
    return out_p_tree, out_t_tree


@ft.partial(eval_rules.def_rule, pushforward_call_p)
def eval_pushforward_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    del ir
    p_tree, t_tree = in_tree
    p_out = treelib.map(lambda _: Var(), p_tree, is_leaf=lambda x: isinstance(x, Var))
    t_out = treelib.map(lambda _: Var(), t_tree, is_leaf=lambda x: isinstance(x, Var))
    return p_out, t_out


@ft.partial(push_rules.def_rule, pushforward_call_p)
def pushforward_pushforward_call(primals: Tree, tangents: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, t_in), (p_in_t, t_in_t) = primals, tangents
    pf_ir = pushforward(ir)
    p_out = call(pf_ir)((p_in, t_in))
    t_out = call(pf_ir)((p_in_t, t_in_t))
    return p_out, t_out


@ft.partial(pull_fwd_rules.def_rule, pushforward_call_p)
def pullback_fwd_pushforward_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, t_in) = in_tree
    pf_ir = pushforward(ir)
    p_out, t_out = call(pf_ir)((p_in, t_in))
    residuals = (p_in, t_in)
    return (p_out, t_out), residuals


@ft.partial(pull_bwd_rules.def_rule, pushforward_call_p)
def pullback_bwd_pushforward_call(residuals: Tree, out_cotangent: Tree, *, ir: IR) -> Tree:
    in_p, in_t = residuals
    out_c_p, out_c_t = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = call(pb_ir)((in_p, out_c_p))
    _, in_c_t = call(pb_ir)((in_t, out_c_t))
    return (in_c_p, in_c_t)


@ft.partial(batch_rules.def_rule, pushforward_call_p)
def batch_pushforward_call(
    batch_size: int, in_batched: Tree[bool], in_tree: Tree, *, ir: IR
) -> tuple[Tree, Tree]:
    (p_cols, t_cols), (p_batched, t_batched) = in_tree, in_batched
    unbatch_p = ft.partial(unbatch_at, p_cols, p_batched)
    unbatch_t = ft.partial(unbatch_at, t_cols, t_batched)
    pf_ir = pushforward(ir)
    out_bi = [call(pf_ir)((unbatch_p(b), unbatch_t(b))) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, pf_ir.out_irtree)
    out_ib = transpose_batch(batch_size, out_batched, out_bi)
    return out_ib, out_batched


@ft.partial(dce_rules.def_rule, pushforward_call_p)
def dce_pushforward_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    dced_ir = dce(ireqn.params["ir"])
    new_eqn = ireqn.using(ir=dced_ir)
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn


# ==================================================================================================
# PULLBACK
# ==================================================================================================

pullback_call_p = Primitive("pullback_call", tag="transformation")


zero_cotangents_map: dict[type, Callable[[], tp.Any]] = {}
cotangent_accumulators: dict[type, Callable[[list], tp.Any]] = {}


def register_cotangent_rules(
    typ: type,
    zero: Callable[[], tp.Any],
    accumulator: Callable[[list], tp.Any],
):
    zero_cotangents_map[typ] = zero
    cotangent_accumulators[typ] = accumulator


register_cotangent_rules(str, lambda: "", lambda cs: "".join(cs))


def zero_cotangent(example: tp.Any = None) -> tp.Any:
    if example is not None and (zero_func := zero_cotangents_map.get(type(example))):
        return zero_func()
    return ""


def accumulate_cotangents(cotangents: list) -> tp.Any:
    if not cotangents:
        return zero_cotangent()
    if len(cotangents) == 1:
        return cotangents[0]
    first, *_ = cotangents
    for typ, acc in cotangent_accumulators.items():
        if isinstance(first, typ):
            return acc(cotangents)
    return sum(cotangents[1:], cotangents[0])


class PullbackFwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params):
        with using_interp(self.parent):
            return pull_fwd_rules[prim](in_tree, **params)


class PullbackBwdInterpreter(Interpreter):
    def __init__(self):
        self.parent = get_interp()

    def process(self, prim: Primitive, in_tree: Tree, **params):
        with using_interp(self.parent):
            in_residual, out_cotangent = in_tree
            return pull_bwd_rules[prim](in_residual, out_cotangent, **params)


@ft.partial(lru_cache, maxsize=256)
def pullback(ir: IR) -> IR:
    """Transform an IR to compute outputs and input cotangents (reverse-mode AD).

    Creates a new IR that computes gradients by backpropagating cotangent
    (adjoint) vectors. Useful for computing vector-Jacobian products (VJPs).

    Args:
        ir: The IR to transform.

    Returns:
        A new IR: `(primals, output_cotangent) -> (outputs, input_cotangents)`

    Example:
        >>> import autoform as af
        >>> def program(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.build_ir(program)("a", "b")
        >>> pb_ir = af.pullback(ir)
        >>> outputs, cotangents = call(pb_ir)((("Hello", " World"), "feedback"))
        >>> outputs
        'Hello World'
        >>> cotangents  # Gradient flows back to both inputs
        ('feedback', 'feedback')
    """
    assert isinstance(ir, IR), f"Expected IR, got {type(ir)}"

    def make_p(atom):
        return IRPVar.fresh(source=atom) if is_irvar(atom) else atom

    def make_c(atom):
        return IRCVar.fresh(source=atom) if is_irvar(atom) else IRZero(atom)

    in_p = treelib.map(make_p, ir.in_irtree)
    out_c = treelib.map(make_c, ir.out_irtree)
    in_irtree = (in_p, out_c)
    out_p = treelib.map(make_p, ir.out_irtree)
    in_c = treelib.map(make_c, ir.in_irtree)
    out_irtree = (out_p, in_c)
    eqn = IREqn(pullback_call_p, in_irtree, out_irtree, dict(ir=ir))
    return IR([eqn], in_irtree, out_irtree)


class IRCVar(IRVar): ...


@ft.partial(impl_rules.def_rule, pullback_call_p)
def impl_pullback_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in_tree, out_c_tree) = in_tree

    p_env: dict[IRVar, Value] = {}
    res_env: dict[int, Tree] = {}
    c_env: defaultdict[IRVar, list] = defaultdict(list)

    def write_p(atom, value):
        is_irvar(atom) and setitem(p_env, atom, value)

    def read_p(atom):
        return p_env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    def write_c(atom, value):
        if is_irvar(atom):
            value = zero_cotangent(value.value) if isinstance(value, IRZero) else value
            c_env[atom].append(value)

    def read_c(atom):
        return accumulate_cotangents(c_env[atom]) if is_irvar(atom) else ""

    treelib.map(write_p, ir.in_irtree, p_in_tree)

    with using_interp(PullbackFwdInterpreter()):
        for i, eqn in enumerate(ir.ireqns):
            p_in_ireqn = treelib.map(read_p, eqn.in_irtree)
            out_p_ireqn, residuals = eqn.prim.bind(p_in_ireqn, **eqn.params)
            res_env[i] = residuals
            treelib.map(write_p, eqn.out_irtree, out_p_ireqn)

    treelib.map(write_c, ir.out_irtree, out_c_tree)

    with using_interp(PullbackBwdInterpreter()):
        for i, eqn in enumerate(reversed(ir.ireqns)):
            idx = len(ir.ireqns) - 1 - i
            residuals = res_env[idx]
            out_c_ireqn = treelib.map(read_c, eqn.out_irtree)
            c_in_ireqn = eqn.prim.bind((residuals, out_c_ireqn), **eqn.params)
            treelib.map(write_c, eqn.in_irtree, c_in_ireqn)

    out_p_tree = treelib.map(read_p, ir.out_irtree)
    in_c_tree = treelib.map(read_c, ir.in_irtree)
    return out_p_tree, in_c_tree


@ft.partial(eval_rules.def_rule, pullback_call_p)
def eval_pullback_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    del ir
    in_p, out_c = in_tree
    is_var = lambda x: isinstance(x, Var)
    out_p = treelib.map(lambda _: Var(), out_c, is_leaf=is_var)
    in_c = treelib.map(lambda _: Var(), in_p, is_leaf=is_var)
    return out_p, in_c


@ft.partial(push_rules.def_rule, pullback_call_p)
def pushforward_pullback_call(primals: Tree, tangents: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out), (t_p_in, t_c_out) = primals, tangents
    pb_ir = pullback(ir)
    out_p, in_c = call(pb_ir)((p_in, c_out))
    t_out_p, t_in_c = call(pb_ir)((t_p_in, t_c_out))
    return (out_p, in_c), (t_out_p, t_in_c)


@ft.partial(pull_fwd_rules.def_rule, pullback_call_p)
def pullback_fwd_pullback_call(in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_in, c_out) = in_tree
    pb_ir = pullback(ir)
    out_p, in_c = call(pb_ir)((p_in, c_out))
    residuals = (p_in, c_out, out_p, in_c)
    return (out_p, in_c), residuals


@ft.partial(pull_bwd_rules.def_rule, pullback_call_p)
def pullback_bwd_pullback_call(residuals: Tree, out_cotangent: Tree, *, ir: IR) -> Tree:
    p_in, c_out, _, _ = residuals
    out_c_p, in_c_c = out_cotangent
    pb_ir = pullback(ir)
    _, in_c_p = call(pb_ir)((p_in, out_c_p))
    _, in_c_c = call(pb_ir)((p_in, in_c_c))
    return (in_c_p, in_c_c)


@ft.partial(batch_rules.def_rule, pullback_call_p)
def batch_pullback_call(size: int, in_batched: Tree, in_tree: Tree, *, ir: IR) -> tuple[Tree, Tree]:
    (p_cols, out_c_cols) = in_tree
    (p_batched, c_batched) = in_batched
    unbatch_p = ft.partial(unbatch_at, p_cols, p_batched)
    unbatch_c = ft.partial(unbatch_at, out_c_cols, c_batched)
    pb_ir = pullback(ir)
    out_bi = [call(pb_ir)((unbatch_p(b), unbatch_c(b))) for b in range(size)]
    out_batched = treelib.map(lambda _: True, pb_ir.out_irtree)
    out_ib = transpose_batch(size, out_batched, out_bi)
    return out_ib, out_batched


@ft.partial(dce_rules.def_rule, pullback_call_p)
def dce_pullback_call(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    dced_ir = dce(ireqn.params["ir"])
    new_eqn = ireqn.using(ir=dced_ir)
    can_axe, used_ins, _ = default_dce(ireqn, active_irvars)
    return can_axe, used_ins, new_eqn
