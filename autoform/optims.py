"""IR optimization passes"""

from __future__ import annotations

import functools as ft
from collections import deque
from operator import setitem

from autoform.core import IR, IREqn, IRLit, IRVar, dce_rules, impl_rules, is_irvar
from autoform.utils import Tree, lru_cache, treelib

# ==================================================================================================
# DEAD CODE ELIMINATION
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def dce(ir: IR) -> IR:
    """Remove dead code from an IR.

    Performs backward pass to identify which equations contribute to output.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     dead = af.concat(x, " dead")  # unused
        ...     live = af.concat(x, " live")  # returned
        ...     return live
        >>> ir = af.build_ir(program)("test")
        >>> len(ir.ireqns)
        2
        >>> dced = af.dce(ir)
        >>> len(dced.ireqns)
        1
    """
    active_irvars: set[IRVar] = set(x for x in treelib.leaves(ir.out_irtree) if is_irvar(x))
    active_ireqns: deque[IREqn] = deque()

    for ireqn in reversed(ir.ireqns):
        # NOTE(asem): dce_rule of hop handles nested IRs
        can_axe, cur_active, new_eqn = dce_rules[ireqn.prim](ireqn, active_irvars)

        if not can_axe:
            active_ireqns.appendleft(new_eqn)
            active_irvars |= cur_active

    return IR(list(active_ireqns), in_irtree=ir.in_irtree, out_irtree=ir.out_irtree)


# ==================================================================================================
# CONSTANT FOLDING
# ==================================================================================================


@ft.partial(lru_cache, maxsize=256)
def fold(ir: IR) -> IR:
    """Evaluate constant IR subexpressions.

    Replaces equations with all-literal inputs with their computed values.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     constant = af.format("{}, {}", "a", "b")
        ...     return af.concat(constant, x)
        >>> ir = af.build_ir(program)("test")
        >>> len(ir.ireqns)
        2
        >>> folded = af.fold(ir)
        >>> len(folded.ireqns)
        1
    """

    def is_const_irtree(irtree: Tree) -> bool:
        leaves = treelib.leaves(irtree)
        return all(isinstance(leaf, IRLit) for leaf in leaves)

    def run_const_eqn(ireqn: IREqn, in_irtree: Tree):
        in_ireqn_tree = treelib.map(lambda x: x.value, in_irtree)
        out_ireqn_tree = impl_rules[ireqn.prim](in_ireqn_tree, **ireqn.params)
        return treelib.map(IRLit, out_ireqn_tree)

    env: dict[IRVar, IRVar | IRLit] = {}
    eqns = []

    def write(atom, value):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom):
        return env[atom] if is_irvar(atom) else atom

    treelib.map(write, ir.in_irtree, ir.in_irtree)

    for ireqn in ir.ireqns:
        # NOTE(asem): recursively fold nested IRs in HOP params
        params = {k: fold(v) if isinstance(v, IR) else v for k, v in ireqn.params.items()}
        ireqn = IREqn(ireqn.prim, ireqn.in_irtree, ireqn.out_irtree, ireqn.effect, params)

        in_irtree = treelib.map(read, ireqn.in_irtree)

        if is_const_irtree(in_irtree):
            # NOTE(asem): constant inputs denotes all inputs are literals,
            # equation can be evaluated without at compile time.
            const_out_irtree = run_const_eqn(ireqn, in_irtree)
            treelib.map(write, ireqn.out_irtree, const_out_irtree)
        else:
            # NOTE(asem): non-constant inputs denotes at least one input is a variable,
            # equation must be evaluated at runtime.
            treelib.map(write, ireqn.out_irtree, ireqn.out_irtree)
            eqns.append(IREqn(ireqn.prim, in_irtree, ireqn.out_irtree, ireqn.effect, ireqn.params))

    out_irtree = treelib.map(read, ir.out_irtree)

    return IR(eqns, ir.in_irtree, out_irtree)
