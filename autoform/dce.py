# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dead code elimination"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from autoform.core import IR, IREqn, IRLit, IRVal, IRVar, Prim, is_irvar
from autoform.utils import Tree, treelib

# ==================================================================================================
# DEAD CODE ELIMINATION
# ==================================================================================================


def default_dce(ir_eqn: IREqn, out_used: Tree[bool]) -> tuple[IREqn, Tree[bool]]:
    # NOTE(asem): out_used is a pytree of bool matching the ir_eqn output pytree that
    # denotes which output is used. the return is a another IREqn (mostly for edited HOP IR)
    # and a out_used
    should_use = treelib.any(out_used)
    in_used = treelib.map(lambda _: should_use, ir_eqn.in_ir_tree)
    return ir_eqn, in_used


type DCERule = Callable[[IREqn, Tree[bool]], tuple[IREqn, Tree[bool]]]

dce_rules: dict[Prim, DCERule] = {}


def dce[**P, R](
    ir: IR[P, R], /, *, out_used: Tree[bool] | None = None, keep_effects: bool = True
) -> IR[P, R]:
    """Remove dead code from an IR.

    Performs backward pass to identify which equations contribute to output.

    Args:
        ir: The IR to optimize.
        out_used: A pytree of bool matching the ir output pytree that denotes which output is used.
        keep_effects: keep equations with effects (e.g. checkpoint) even if their outputs are not used.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     dead = af.concat(x, " dead")  # unused
        ...     live = af.concat(x, " live")  # returned
        ...     return live
        >>> ir = af.trace(program)("test")
        >>> len(ir.ir_eqns)
        2
        >>> dced = af.dce(ir)
        >>> len(dced.ir_eqns)
        1
    """

    if out_used is None:
        user_out_used = treelib.map(lambda _: True, ir.out_ir_tree)
    else:
        assert treelib.all(isinstance(leaf, bool) for leaf in treelib.leaves(out_used))
        assert treelib.structure(out_used) == treelib.structure(ir.out_ir_tree)
        user_out_used = out_used

    def collect_used_irvars(tree: Tree[IRVal], used: Tree[bool]) -> set[IRVar]:
        active_irvars: set[IRVar] = set()
        flat_tree, flat_used = treelib.leaves(tree), treelib.leaves(used)
        for iratom, keep in zip(flat_tree, flat_used, strict=True):
            if keep and is_irvar(iratom):
                active_irvars.add(iratom)
        return active_irvars

    active_irvars: set[IRVar] = collect_used_irvars(ir.out_ir_tree, user_out_used)
    active_ir_eqns: deque[IREqn] = deque()

    def is_active_node(node: IRVal) -> bool:
        return is_irvar(node) and (node in active_irvars)

    for ir_eqn in reversed(ir.ir_eqns):
        # NOTE(asem): walk backwards and feed dce rules the appropriate
        # out_used tree. if any output is used, keep the equation. and
        # add the irvars corresponding to the used outputs to the active set.
        ir_eqn_out_used = treelib.map(is_active_node, ir_eqn.out_ir_tree)
        new_ir_eqn, in_used = dce_rules.get(ir_eqn.prim, default_dce)(ir_eqn, ir_eqn_out_used)
        assert treelib.structure(in_used) == treelib.structure(ir_eqn.in_ir_tree)

        if ir_eqn.effect and keep_effects:
            active_ir_eqns.appendleft(new_ir_eqn)
            active_irvars |= set(x for x in treelib.leaves(ir_eqn.in_ir_tree) if is_irvar(x))

        elif treelib.any(in_used):
            active_ir_eqns.appendleft(new_ir_eqn)
            active_irvars |= collect_used_irvars(ir_eqn.in_ir_tree, in_used)

    # NOTE(asem): output sanitization step
    # `call(ir)` always reads `ir.out_ir_tree`, even if a caller provided an `out_used` mask.
    # so after DCE removes equations, `out_ir_tree` may contain IRVars that are no longer
    # defined ("dangling"), which would crash at runtime when the interpreter tries to
    # read them.
    in_vars = set(x for x in treelib.leaves(ir.in_ir_tree) if is_irvar(x))
    defined_vars: set[IRVar] = set(in_vars)
    for kept in active_ir_eqns:
        for atom in treelib.leaves(kept.out_ir_tree):
            is_irvar(atom) and defined_vars.add(atom)

    def sanitize_out_leaf(atom: IRVal, used: bool) -> IRVal:
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

    out_ir_tree = treelib.map(sanitize_out_leaf, ir.out_ir_tree, user_out_used)
    return IR(list(active_ir_eqns), in_ir_tree=ir.in_ir_tree, out_ir_tree=out_ir_tree)
