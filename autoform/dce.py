"""Dead code elimination"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from autoform.core import IR, IRAtom, IREqn, IRLit, IRVar, Primitive, is_irvar
from autoform.utils import Tree, treelib

# ==================================================================================================
# DEAD CODE ELIMINATION
# ==================================================================================================


def default_dce(ireqn: IREqn, out_used: Tree[bool]) -> tuple[IREqn, Tree[bool]]:
    # NOTE(asem): out_used is a pytree of bool matching the ireqn output pytree that
    # denotes which output is used. the return is a another IREqn (mostly for edited HOP IR)
    # and a out_used
    should_use = treelib.any(out_used)
    in_used = treelib.map(lambda _: should_use, ireqn.in_irtree)
    return ireqn, in_used


type DCERule = Callable[[IREqn, Tree[bool]], tuple[IREqn, Tree[bool]]]

dce_rules: dict[Primitive, DCERule] = {}


def dce[**P, R](
    ir: IR[P, R], /, *, out_used: Tree[bool] | None = None, keep_effects: bool = True
) -> IR[P, R]:
    """Remove dead code from an IR.

    Performs backward pass to identify which equations contribute to output.

    Args:
        ir: The IR to optimize.
        out_used: A pytree of bool matching the ir output pytree that denotes which output is used.
        keep_effects: Whether to keep equations with side effects even if their outputs are not used.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     dead = af.concat(x, " dead")  # unused
        ...     live = af.concat(x, " live")  # returned
        ...     return live
        >>> ir = af.trace(program)("test")
        >>> len(ir.ireqns)
        2
        >>> dced = af.dce(ir)
        >>> len(dced.ireqns)
        1
    """

    if out_used is None:
        user_out_used = treelib.map(lambda _: True, ir.out_irtree)
    else:
        assert treelib.all(isinstance(leaf, bool) for leaf in treelib.leaves(out_used))
        assert treelib.structure(out_used) == treelib.structure(ir.out_irtree)
        user_out_used = out_used

    def collect_used_irvars(tree: Tree[IRAtom], used: Tree[bool]) -> set[IRVar]:
        active_irvars: set[IRVar] = set()
        flat_tree, flat_used = treelib.leaves(tree), treelib.leaves(used)
        for iratom, keep in zip(flat_tree, flat_used, strict=True):
            if keep and is_irvar(iratom):
                active_irvars.add(iratom)
        return active_irvars

    active_irvars: set[IRVar] = collect_used_irvars(ir.out_irtree, user_out_used)
    active_ireqns: deque[IREqn] = deque()

    def is_active_node(node: IRAtom) -> bool:
        return is_irvar(node) and (node in active_irvars)

    for ireqn in reversed(ir.ireqns):
        # NOTE(asem): walk backwards and feed dce rules the appropriate
        # out_used tree. if any output is used, keep the equation. and
        # add the irvars corresponding to the used outputs to the active set.
        ireqn_out_used = treelib.map(is_active_node, ireqn.out_irtree)
        new_ireqn, in_used = dce_rules.get(ireqn.prim, default_dce)(ireqn, ireqn_out_used)
        assert treelib.structure(in_used) == treelib.structure(ireqn.in_irtree)

        if ireqn.effect and keep_effects:
            active_ireqns.appendleft(new_ireqn)
            active_irvars |= set(x for x in treelib.leaves(ireqn.in_irtree) if is_irvar(x))

        elif treelib.any(in_used):
            active_ireqns.appendleft(new_ireqn)
            active_irvars |= collect_used_irvars(ireqn.in_irtree, in_used)

    # NOTE(asem): output sanitization step
    # `call(ir)` always reads `ir.out_irtree`, even if a caller provided an `out_used` mask.
    # so after DCE removes equations, `out_irtree` may contain IRVars that are no longer
    # defined ("dangling"), which would crash at runtime when the interpreter tries to
    # read them.
    in_vars = set(x for x in treelib.leaves(ir.in_irtree) if is_irvar(x))
    defined_vars: set[IRVar] = set(in_vars)
    for kept in active_ireqns:
        for atom in treelib.leaves(kept.out_irtree):
            is_irvar(atom) and defined_vars.add(atom)

    def sanitize_out_leaf(atom: IRAtom, used: bool) -> IRAtom:
        if not is_irvar(atom):
            # NOTE(asem): leaf is already a literal, nothing to sanitize.
            # >>> def program(x):
            # ...     return (x, "const")
            return atom
        if atom in defined_vars:
            # NOTE(asem): defined output var (either an input var or produced by a kept eqn).
            # >>> def program(x):
            # ...     y = af.concat(x, "!")
            # ...     return y
            # y's IRVar is in `defined_vars` and stays as-is.
            return atom
        if not used:
            # NOTE(asem): unused-but-dangling output slot (typically from partial `out_used`).
            # >>> def program(x):
            # ...   a=af.concat(x,"a")
            # ...   b=af.concat(x,"b")
            # ...   return (a,b)
            # >>> af.dce(ir, out_used=(True, False))
            # drops eqn for b, but keeps a 2-tuple output.
            # the second leaf becomes IRLit(None).
            return IRLit(None)
        # NOTE(asem): this should be unreachable for well-behaved primitives/rules.
        assert False, (
            "DCE produced an invalid IR: a used output IRVar is not defined by inputs or kept equations. "
            "This typically indicates inconsistent `out_used` or a bug in a DCE rule for a primitive."
        )

    out_irtree = treelib.map(sanitize_out_leaf, ir.out_irtree, user_out_used)
    return IR(list(active_ireqns), in_irtree=ir.in_irtree, out_irtree=out_irtree)
