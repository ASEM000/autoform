"""Scheduling concurrent execution"""

from __future__ import annotations

import asyncio
import functools as ft
from collections import defaultdict, deque
from collections.abc import Callable
from operator import setitem

from autoform.ad import pullback, pushforward, zero_cotangent
from autoform.batch import batch
from autoform.core import (
    IR,
    IRAtom,
    IREqn,
    IRVar,
    Primitive,
    PrimitiveTag,
    acall,
    batch_rules,
    call,
    eval_rules,
    impl_rules,
    iratom_to_evaltype,
    is_irvar,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.dce import dce, dce_rules, default_dce
from autoform.utils import Tree, asyncify, batch_spec, lru_cache, treelib


class SchedulingTag(PrimitiveTag): ...


# ==================================================================================================
# GATHER
# ==================================================================================================

gather_p = Primitive("gather", tag={SchedulingTag})


def gather(ir_input_pairs: list[tuple[IR, Tree]], /) -> list[Tree]:
    """Run multiple IRs, concurrently when using acall().

    When executed with call(), runs sequentially.
    When executed with acall(), runs concurrently via asyncio.gather.

    Args:
        ir_input_pairs: pairs of (ir, inputs) to execute.

    Returns:
        List of results in same order as inputs.

    Example:
        >>> import autoform as af
        >>> ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        >>> ir2 = af.trace(lambda x: af.format("<{}>", x))("a")
        >>> result = af.gather([(ir1, "A"), (ir2, "B")])
        >>> result
        ['[A]', '<B>']
    """
    assert len(ir_input_pairs), "gather requires at least one (ir, inputs) pair"
    ins, irs = [], []
    for pair in ir_input_pairs:
        match pair:
            case (IR(), _):
                ir, inp = pair
                irs.append(ir)
                ins.append(inp)
            case _:
                raise TypeError(f"Expected (ir, inputs) tuple, got {pair}")

    return gather_p.bind(ins, irs=irs)


def impl_gather(in_tree: list[Tree], /, *, irs: list[IR]) -> list[Tree]:
    assert len(in_tree) == len(irs)
    return [call(ir)(inp) for ir, inp in zip(irs, in_tree, strict=True)]


async def aimpl_gather(in_tree: list[Tree], /, *, irs: list[IR]) -> list[Tree]:
    assert len(in_tree) == len(irs)

    if len(irs) == 1:
        [ir], [inputs] = irs, in_tree
        return [await acall(ir)(inputs)]

    async def run(pair):
        ir, inp = pair
        return await acall(ir)(inp)

    return list(await asyncio.gather(*[run(pair) for pair in zip(irs, in_tree, strict=True)]))


def eval_gather(in_tree: list[Tree], /, *, irs: list[IR]) -> list[Tree]:
    return [treelib.map(iratom_to_evaltype, ir.out_irtree) for ir in irs]


def push_gather(
    in_tree: tuple[list[Tree], list[Tree]], /, *, irs: list[IR]
) -> tuple[list[Tree], list[Tree]]:
    primals, tangents = in_tree
    pf_irs = [pushforward(ir) for ir in irs]
    pf_inputs = [(p, t) for p, t in zip(primals, tangents, strict=True)]
    results = impl_gather(pf_inputs, irs=pf_irs)
    p_outs, t_outs = zip(*results)
    return list(p_outs), list(t_outs)


async def apush_gather(
    in_tree: tuple[list[Tree], list[Tree]], /, *, irs: list[IR]
) -> tuple[list[Tree], list[Tree]]:
    primals, tangents = in_tree
    pf_irs = [pushforward(ir) for ir in irs]
    pf_inputs = [(p, t) for p, t in zip(primals, tangents, strict=True)]
    results = await aimpl_gather(pf_inputs, irs=pf_irs)
    p_outs, t_outs = zip(*results)
    return list(p_outs), list(t_outs)


def pull_fwd_gather(
    in_tree: list[Tree], /, *, irs: list[IR]
) -> tuple[list[Tree], tuple[list[Tree], list[IR]]]:
    results = impl_gather(in_tree, irs=irs)
    residuals = (in_tree, irs)
    return results, residuals


async def apull_fwd_gather(
    in_tree: list[Tree], /, *, irs: list[IR]
) -> tuple[list[Tree], tuple[list[Tree], list[IR]]]:
    results = await aimpl_gather(in_tree, irs=irs)
    residuals = (in_tree, irs)
    return results, residuals


def pull_bwd_gather(in_tree: Tree, /, *, irs: list[IR]) -> list[Tree]:
    residuals, out_cotangent = in_tree
    inputs, _ = residuals
    pb_irs = [pullback(ir) for ir in irs]
    pb_inputs = [(inp, cot) for inp, cot in zip(inputs, out_cotangent, strict=True)]
    results = impl_gather(pb_inputs, irs=pb_irs)
    return [cot for _, cot in results]


async def apull_bwd_gather(in_tree: Tree, /, *, irs: list[IR]) -> list[Tree]:
    residuals, out_cotangent = in_tree
    inputs, _ = residuals
    pb_irs = [pullback(ir) for ir in irs]
    pb_inputs = [(inp, cot) for inp, cot in zip(inputs, out_cotangent, strict=True)]
    results = await aimpl_gather(pb_inputs, irs=pb_irs)
    return [cot for _, cot in results]


def batch_gather(
    in_tree: tuple[int, list[bool], list[Tree]], /, *, irs: list[IR]
) -> tuple[list[Tree], list[Tree[bool]]]:
    batch_size, in_batched, inputs = in_tree

    results: list[Tree] = []
    out_batched: list[Tree[bool]] = []

    for ir, inp, inp_batched in zip(irs, inputs, in_batched, strict=True):
        if batch_spec(inp, inp_batched) is None:
            results.append(call(ir)(inp))
            out_batched.append(treelib.map(lambda _: False, ir.out_irtree))
        else:
            batched_ir = batch(ir, in_axes=inp_batched)
            results.append(call(batched_ir)(inp))
            out_batched.append(treelib.map(lambda _: True, ir.out_irtree))

    return results, out_batched


async def abatch_gather(
    in_tree: tuple[int, list[bool], list[Tree]], /, *, irs: list[IR]
) -> tuple[list[Tree], list[Tree[bool]]]:
    batch_size, in_batched, inputs = in_tree

    results: list[Tree] = []
    out_batched: list[Tree[bool]] = []

    for ir, inp, inp_batched in zip(irs, inputs, in_batched, strict=True):
        if batch_spec(inp, inp_batched) is None:
            results.append(await acall(ir)(inp))
            out_batched.append(treelib.map(lambda _: False, ir.out_irtree))
        else:
            batched_ir = batch(ir, in_axes=inp_batched)
            results.append(await acall(batched_ir)(inp))
            out_batched.append(treelib.map(lambda _: True, ir.out_irtree))

    return results, out_batched


impl_rules.set(gather_p, impl_gather)
impl_rules.aset(gather_p, aimpl_gather)
eval_rules.set(gather_p, eval_gather)
push_rules.set(gather_p, push_gather)
push_rules.aset(gather_p, apush_gather)
pull_fwd_rules.set(gather_p, pull_fwd_gather)
pull_fwd_rules.aset(gather_p, apull_fwd_gather)
pull_bwd_rules.set(gather_p, pull_bwd_gather)
pull_bwd_rules.aset(gather_p, apull_bwd_gather)
batch_rules.set(gather_p, batch_gather)
batch_rules.aset(gather_p, abatch_gather)


def dce_gather(ireqn: IREqn, out_used: list[bool], /) -> tuple[IREqn, list[bool]]:
    irs = ireqn.params["irs"]
    new_irs = [dce(ir, out_used=ou) for ir, ou in zip(irs, out_used, strict=True)]
    new_eqn = ireqn.using(irs=new_irs)
    return default_dce(new_eqn, out_used)


dce_rules[gather_p] = dce_gather


# ==================================================================================================
# TOPOSORT LEVELS
# ==================================================================================================


@ft.lru_cache(maxsize=256)
def toposort_levels(ir: IR, /) -> list[list[IREqn]]:
    """Group IR equations into dependency levels."""

    # NOTE(asem): equations form a dag where edges are defined by shared irvars.
    # if equation a produces $x and equation b uses $x, then a -> b.
    # this function groups equations into levels where:
    # 1. equations in the same level are independent (can run in parallel)
    # 2. level n must complete before level n+1 starts

    # NOTE(asem): three-step process:
    # 1. map each irvar to its creator equation
    # 2. build adjacency list (parent -> children) from irvar flow
    # 3. topological sort into levels using kahn's algorithm

    # NOTE(asem): step 1: map irvar -> creator equation
    irvar_to_parent: dict[IRVar, IREqn] = {}
    for ireqn in ir.ireqns:
        for out_iratom in treelib.leaves(ireqn.out_irtree):
            is_irvar(out_iratom) and setitem(irvar_to_parent, out_iratom, ireqn)

    # NOTE(asem): step 2: build adjacency list (parent -> children) and in-degree count
    adjacency_list = defaultdict(list)
    in_degree = defaultdict(lambda: 0)

    def has_parent(iratom: IRAtom) -> bool:
        return is_irvar(iratom) and (iratom in irvar_to_parent)

    for ireqn in ir.ireqns:
        # NOTE(asem): avoid adding the same parent multiple times if the input is repeated
        seen_parents: set[IREqn] = set()
        for in_irvar in (x for x in treelib.leaves(ireqn.in_irtree) if has_parent(x)):
            # NOTE(asem): consider `concat($1, $1)` this would repeat the same equation
            if (parent := irvar_to_parent[in_irvar]) not in seen_parents:
                adjacency_list[parent].append(ireqn)
                in_degree[ireqn] += 1
                seen_parents.add(parent)

    # NOTE(asem): step 3: kahn's algorithm
    # basically prune nodes with 0 indegree then update the children indegree
    queue = deque(ireqn for ireqn in ir.ireqns if in_degree[ireqn] == 0)
    levels = []

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node)
            for child in adjacency_list[node]:
                in_degree[child] -= 1
                in_degree[child] == 0 and queue.append(child)
        levels.append(level)
    return levels


# ==================================================================================================
# DEPENDS
# ==================================================================================================

depends_p = Primitive("depends", tag={SchedulingTag})

type DependsType[T] = tuple[T, tuple[Tree, ...]]


def depends[T](value: T, /, *deps) -> T:
    """Annotate that `value` depends on the evaluation of `deps`.

    This primitive is used to enforce execution order without affecting
    the actual data flow. It ensures that all `deps` are evaluated before
    `value` is returned.

    Args:
        value: The main value to return.
        *deps: Values that `value` depends on.
    Returns:
        The original `value`, after ensuring all `deps` are evaluated.
    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     a = af.format("First: {}", x)
        ...     b = af.format("Second: {}", x)
        ...     return af.depends(b, a)  # ensure 'a' is evaluated before returning 'b'
    """
    return depends_p.bind((value, deps))


def impl_depends[T](in_tree: DependsType[T], /) -> T:
    value, _ = in_tree
    return value


def eval_depends(in_tree: DependsType[Tree], /) -> Tree:
    value, _ = in_tree
    return value


def push_depends(in_tree: tuple[DependsType[Tree], DependsType[Tree]], /) -> tuple[Tree, Tree]:
    (primal_value, primal_deps), (tangent_value, tangent_deps) = in_tree
    return depends(primal_value, *primal_deps), depends(tangent_value, *tangent_deps)


def pull_fwd_depends(in_tree: DependsType[Tree], /) -> tuple[Tree, DependsType[Tree]]:
    value, deps = in_tree
    return depends(value, *deps), in_tree


def pull_bwd_depends(in_tree: tuple[DependsType[Tree], Tree], /) -> DependsType[Tree]:
    (_, deps), out_cotangent = in_tree
    return out_cotangent, treelib.map(zero_cotangent, deps)


def batch_depends(
    in_tree: tuple[int, tuple[bool, tuple[bool, ...]], DependsType[Tree]], /
) -> tuple[Tree, bool]:
    _, (value_batched, _), (value, deps) = in_tree
    return depends(value, *deps), value_batched


impl_rules.set(depends_p, impl_depends)
impl_rules.aset(depends_p, asyncify(impl_depends))
eval_rules.set(depends_p, eval_depends)
push_rules.set(depends_p, push_depends)
push_rules.aset(depends_p, asyncify(push_depends))
pull_fwd_rules.set(depends_p, pull_fwd_depends)
pull_fwd_rules.aset(depends_p, asyncify(pull_fwd_depends))
pull_bwd_rules.set(depends_p, pull_bwd_depends)
pull_bwd_rules.aset(depends_p, asyncify(pull_bwd_depends))
batch_rules.set(depends_p, batch_depends)
batch_rules.aset(depends_p, asyncify(batch_depends))

# ==================================================================================================
# SCHED
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def sched[**P, R](ir: IR[P, R], /, *, cond: Callable[[IREqn], bool] | None = None) -> IR[P, R]:
    """Schedule independent operations for parallel execution using gather.

    Args:
        ir: The IR to schedule.
        cond: Predicate that takes an IR Equation and returns True if the
              equation should be parallelized. If None, all operations are
              candidates for parallelization.

    Returns:
        A new IR with gather operations.

    Example:
        >>> import autoform as af
        >>> import asyncio
        >>>
        >>> def parallel_calls(x):
        ...     msg1 = [dict(role="user", content=af.format("Q1: {}", x))]
        ...     msg2 = [dict(role="user", content=af.format("Q2: {}", x))]
        ...     a = af.lm_call(msg1, model="gpt-4o-mini")
        ...     b = af.lm_call(msg2, model="gpt-4o-mini")
        ...     return af.concat(a, b)
        >>>
        >>> ir = af.trace(parallel_calls)("input")
        >>> scheduled = af.sched(ir)
        >>>
        >>> # sync execution (sequential)
        >>> result = af.call(scheduled)("hello")
        >>>
        >>> # async execution (concurrent via asyncio.gather)
        >>> result = asyncio.run(af.acall(scheduled)("hello"))
    """
    levels: list[list[IREqn]] = toposort_levels(ir)
    out_ireqns: list[IREqn] = []
    cond = (lambda _: True) if cond is None else cond

    def recurse(leaf):
        return sched(leaf, cond=cond) if isinstance(leaf, IR) else leaf

    def make_gather(ireqns: list[IREqn]) -> IREqn:
        irs = [IR([ireqn], ireqn.in_irtree, ireqn.out_irtree) for ireqn in ireqns]
        in_irtree = [ireqn.in_irtree for ireqn in ireqns]
        out_irtree = [ireqn.out_irtree for ireqn in ireqns]
        return IREqn(gather_p, None, in_irtree, out_irtree, dict(irs=irs))

    for level in levels:
        ireqns = [ireqn.using(**treelib.map(recurse, ireqn.params)) for ireqn in level]
        seq_ireqns = [ireqn for ireqn in ireqns if not cond(ireqn)]
        par_ireqns = [ireqn for ireqn in ireqns if cond(ireqn)]
        out_ireqns.extend([make_gather(par_ireqns)] if len(par_ireqns) > 1 else par_ireqns)
        out_ireqns.extend(seq_ireqns)

    return IR(out_ireqns, ir.in_irtree, ir.out_irtree)
