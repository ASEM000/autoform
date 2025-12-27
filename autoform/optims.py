"""IR optimization passes"""

from __future__ import annotations

from collections import deque
from operator import setitem

from autoform.core import IR, IREqn, IRLit, IRVar, is_irvar
from autoform.core import dce_rules, impl_rules
from autoform.utils import Tree, treelib

# ==================================================================================================
# DEAD CODE ELIMINATION
# ==================================================================================================


def default_dce(ireqn: IREqn, active_irvars: set[IRVar]) -> tuple[bool, set[IRVar], IREqn]:
    out_vars = set(x for x in treelib.leaves(ireqn.out_irtree) if is_irvar(x))
    if out_vars.isdisjoint(active_irvars):
        return True, set(), ireqn  # axe (equation returned but unused)
    in_vars = set(x for x in treelib.leaves(ireqn.in_irtree) if is_irvar(x))
    return False, in_vars, ireqn  # (equation unchanged)


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
        dce_rule = dce_rules[ireqn.prim] if ireqn.prim in dce_rules else default_dce
        can_axe, cur_active, new_eqn = dce_rule(ireqn, active_irvars)

        if not can_axe:
            active_ireqns.appendleft(new_eqn)
            active_irvars |= cur_active

    return IR(list(active_ireqns), in_irtree=ir.in_irtree, out_irtree=ir.out_irtree)


# ==================================================================================================
# CONSTANT FOLDING
# ==================================================================================================


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
        in_irtree = treelib.map(read, ireqn.in_irtree)
        if is_const_irtree(in_irtree):
            out_irtree = run_const_eqn(ireqn, in_irtree)
            treelib.map(write, ireqn.out_irtree, out_irtree)
        else:
            treelib.map(write, ireqn.out_irtree, ireqn.out_irtree)
            eqns.append(IREqn(ireqn.prim, in_irtree, ireqn.out_irtree, ireqn.params))

    out_irtree = treelib.map(read, ir.out_irtree)

    return IR(eqns, ir.in_irtree, out_irtree)
