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

import autoform.analysis as analysis
import autoform.core as core
import autoform.utils as utils

type Tree[T] = utils.Tree[T]
type UsedTree = Tree[bool]
type DCEResult = tuple[core.Eqn, UsedTree]

# ==================================================================================================
# DEAD CODE ELIMINATION
# ==================================================================================================


def default_dce(eqn: core.Eqn, out_used: UsedTree) -> DCEResult:
    # NOTE(asem): out_used is a pytree of bool matching the eqn output pytree that
    # denotes which output is used. the return is a another Eqn (mostly for edited HOP IR)
    # and a out_used
    should_use = utils.tree.any(out_used)
    in_used = utils.tree.map(lambda _: should_use, eqn.in_ir_tree)
    return eqn, in_used


type DCERule = Callable[[core.Eqn, UsedTree], DCEResult]

dce_rules: dict[core.Prim, DCERule] = {}
non_dce_primitives: set[core.Prim] = set()


def dce[*A, R](ir: core.IR[*A, R], /, *, out_used: UsedTree | None = None) -> core.IR[*A, R]:
    """Remove dead code from an IR.

    Performs backward pass to identify which equations contribute to output.

    Args:
        ir: The IR to optimize.
        out_used: A pytree of bool matching the ir output pytree that denotes which output is used.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     dead = af.concat(x, " dead")  # unused
        ...     live = af.concat(x, " live")  # returned
        ...     return live
        >>> ir = af.trace(program)("test")
        >>> len(ir.eqns)
        2
        >>> dced = af.dce(ir)
        >>> len(dced.eqns)
        1
    """

    if out_used is None:
        user_out_used = utils.tree.map(lambda _: True, ir.out_ir_tree)
    else:
        assert utils.tree.all(isinstance(leaf, bool) for leaf in utils.tree.leaves(out_used))
        assert utils.tree.structure(out_used) == utils.tree.structure(ir.out_ir_tree)
        user_out_used = out_used

    live_boundaries: analysis.Liveness = analysis.ir_liveness(ir, out_used=user_out_used)
    active_ir_vars: set[core.Var] = set(live_boundaries[-1])
    active_eqns: deque[core.Eqn] = deque()

    def is_active_node(node) -> bool:
        return core.is_irvar(node) and (node in active_ir_vars)

    for eqn in reversed(ir.eqns):
        is_non_dce = eqn.prim in non_dce_primitives
        # NOTE(asem): walk backwards and feed dce rules the appropriate
        # out_used tree. if any output is used, keep the equation. and
        # add the irvars corresponding to the used outputs to the active set.
        eqn_out_used = utils.tree.map(is_active_node, eqn.out_ir_tree)
        new_eqn, in_used = dce_rules.get(eqn.prim, default_dce)(eqn, eqn_out_used)
        assert utils.tree.structure(in_used) == utils.tree.structure(eqn.in_ir_tree)

        if is_non_dce:
            active_eqns.appendleft(new_eqn)
            active_ir_vars |= set(analysis.ir_var_leaves(eqn.in_ir_tree))

        elif utils.tree.any(in_used):
            active_eqns.appendleft(new_eqn)
            active_ir_vars |= set(analysis.ir_var_leaves(utils.mask(eqn.in_ir_tree, in_used)))

    # NOTE(asem): output sanitization step
    # `call(ir)` always reads `ir.out_ir_tree`, even if a caller provided an `out_used` mask.
    # so after DCE removes equations, `out_ir_tree` may contain Vars that are no longer
    # defined ("dangling"), which would crash at runtime when the interpreter tries to
    # read them.
    in_vars = set(analysis.ir_var_leaves(ir.in_ir_tree))
    defined_vars: set[core.Var] = set(in_vars)
    for kept in active_eqns:
        for atom in utils.tree.leaves(kept.out_ir_tree):
            core.is_irvar(atom) and defined_vars.add(atom)

    def sanitize_out_leaf(atom, used: bool):
        if not core.is_irvar(atom):
            # NOTE(asem): leaf is already a literal, nothing to sanitize.
            # >>> def program(x):
            # ...     return (x, "const")
            return atom
        if atom in defined_vars:
            # NOTE(asem): defined output var (either an input var or produced by a kept eqn).
            # >>> def program(x):
            # ...     y = af.concat(x, "!")
            # ...     return y
            # y's Var is in `defined_vars` and stays as-is.
            return atom
        if not used:
            # NOTE(asem): unused-but-dangling output slot (typically from partial `out_used`).
            # >>> def program(x):
            # ...   a=af.concat(x,"a")
            # ...   b=af.concat(x,"b")
            # ...   return (a,b)
            # >>> af.dce(ir, out_used=(True, False))
            # drops eqn for b, but keeps a 2-tuple output.
            # the second leaf becomes None.
            return None
        # NOTE(asem): this should be unreachable for well-behaved primitives/rules.
        assert False, (
            "DCE produced an invalid IR: a used output Var is not defined by inputs or kept equations. "
            "This typically indicates inconsistent `out_used` or a bug in a DCE rule for a primitive."
        )

    out_ir_tree = utils.tree.map(sanitize_out_leaf, ir.out_ir_tree, user_out_used)
    return core.IR(list(active_eqns), in_ir_tree=ir.in_ir_tree, out_ir_tree=out_ir_tree)
