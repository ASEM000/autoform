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

import autoform.abstract as abstract
import autoform.analysis as analysis
import autoform.core as core
import autoform.utils as utils

__all__ = ["dce"]

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
    in_used = utils.tree.map(lambda _: should_use, eqn.in_tree)
    return eqn, in_used


type DCERule = Callable[[core.Eqn, UsedTree], DCEResult]

dce_rules: dict[core.Prim, DCERule] = {}
non_dce_primitives: set[core.Prim] = set()


def is_non_dce(eqn: core.Eqn, /) -> bool:
    # NOTE(asem): recursively check if any nested irs contains non-dce prim
    # for example
    # >>> branch = af.trace(lambda x: af.checkpoint(x, key="save"))("x")
    # >>> def program(x):
    # ...     value = x + "!"
    # ...     af.switch("a", {"a": branch}, value)
    # ...     return x
    # the switch result is unused, but dropping it would also drop the checkpoint.
    # value must stay live because the checkpoint still needs it.
    def func(leaf):
        return isinstance(leaf, core.IR) and any(is_non_dce(eqn) for eqn in leaf.eqns)

    return eqn.prim in non_dce_primitives or utils.tree.any(utils.tree.map(func, eqn.params))


def update_eqn_out(eqn: core.Eqn, active_vars: set[core.Var], /) -> core.Eqn:
    # NOTE(asem): an inner IR may lose outputs while its wrapper still runs.
    # >>> def save(x):
    # ...     af.checkpoint(x, key="save")
    # ...     return x + "!"
    # >>> ir = af.batch(af.trace(save)("x"))
    # >>> dced = af.dce(ir, out_used=False)
    # the DCE pass removes concat but ir.out_tree needs to be updated
    def keep_var(atom, out):
        if isinstance(out, abstract.AVal):
            # NOTE(asem): assure DCE does not change avals
            assert core.is_var(atom) and atom.aval == out
            return atom
        assert not (core.is_var(atom) and atom in active_vars)
        return out

    in_avals = utils.tree.map(core.aval_if_var, eqn.in_tree)
    out_avals = core.abstract_rules.get(eqn.prim)(in_avals, **eqn.params)
    out_tree = utils.tree.map(keep_var, eqn.out_tree, out_avals)
    return core.Eqn(eqn.prim, eqn.in_tree, out_tree, eqn.params, eqn.tags)


def sanitize_out(ir: core.IR, eqns: list[core.Eqn], out_used: UsedTree, /) -> Tree:
    # NOTE(asem): output sanitization step
    # `call(ir)` always reads `ir.out_tree`, even if a caller provided an `out_used` mask.
    # so after DCE removes equations, `out_tree` may contain Vars that are no longer
    # defined ("dangling"), which would crash at runtime when the interpreter tries to
    # read them.
    # >>> def program(x):
    # ...     a = x + "!"
    # ...     b = x + "?"
    # ...     return x, a, b
    # >>> ir = af.trace(program)("x")
    # >>> af.dce(ir, out_used=(True, True, False)).call("x")
    # ('x', 'x!', None)
    # eqns contains only the kept equations: x is an input, a is still produced,
    # and b has no producer left, so only b's output slot becomes None.
    in_vars = set(analysis.var_leaves(ir.in_tree))
    defined_vars: set[core.Var] = set(in_vars)
    for kept in eqns:
        for atom in utils.tree.leaves(kept.out_tree):
            core.is_var(atom) and defined_vars.add(atom)

    def sanitize_out_leaf(atom, used: bool):
        if not core.is_var(atom):
            # NOTE(asem): leaf is already a literal, nothing to sanitize.
            # >>> def program(x):
            # ...     return (x, "const")
            return atom
        if atom in defined_vars:
            # NOTE(asem): defined output var (either an input var or produced by a kept eqn).
            # >>> def program(x):
            # ...     y = x + "!"
            # ...     return y
            # y's Var is in `defined_vars` and stays as-is.
            return atom
        if not used:
            # NOTE(asem): unused-but-dangling output slot (typically from partial `out_used`).
            # >>> def program(x):
            # ...   a=(x + "a")
            # ...   b=(x + "b")
            # ...   return (a, b)
            # >>> af.dce(ir, out_used=(True, False))
            # drops eqn for b, but keeps a 2-tuple output.
            # the second leaf becomes None.
            return None
        # NOTE(asem): this should be unreachable for well-behaved primitives/rules.
        assert False, (
            "DCE produced an invalid IR: a used output Var is not defined by inputs or kept equations. "
            "This typically indicates inconsistent `out_used` or a bug in a DCE rule for a primitive."
        )

    return utils.tree.map(sanitize_out_leaf, ir.out_tree, out_used)


def dce[*A, R](ir: core.IR[*A, R], /, *, out_used: UsedTree | None = None) -> core.IR[*A, R]:
    """Remove dead code from an IR.

    Performs backward pass to identify which equations contribute to output.

    Args:
        ir: The IR to optimize.
        out_used: A pytree of bool matching the ir output pytree that denotes which output is used.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     dead = x + " dead"  # unused
        ...     live = x + " live"  # returned
        ...     return live
        >>> ir = af.trace(program)("test")
        >>> len(ir.eqns)
        2
        >>> dced = af.dce(ir)
        >>> len(dced.eqns)
        1
    """

    if out_used is None:
        user_out_used = utils.tree.map(lambda _: True, ir.out_tree)
    else:
        assert utils.tree.all(isinstance(leaf, bool) for leaf in utils.tree.leaves(out_used))
        assert utils.tree.structure(out_used) == utils.tree.structure(ir.out_tree)
        user_out_used = out_used

    live_boundaries: analysis.Liveness = analysis.ir_liveness(ir, out_used=user_out_used)
    active_vars: set[core.Var] = set(live_boundaries[-1])
    active_eqns: deque[core.Eqn] = deque()

    def is_active_node(node) -> bool:
        return core.is_var(node) and (node in active_vars)

    for eqn in reversed(ir.eqns):
        # NOTE(asem): walk backwards and feed dce rules the appropriate
        # out_used tree. if any output is used, keep the equation. and
        # add the irvars corresponding to the used outputs to the active set.
        protected = is_non_dce(eqn)
        eqn_out_used: Tree[bool] = utils.tree.map(is_active_node, eqn.out_tree)
        new_eqn, in_used = dce_rules.get(eqn.prim, default_dce)(eqn, eqn_out_used)
        assert utils.tree.structure(in_used) == utils.tree.structure(eqn.in_tree)

        changed = new_eqn is not eqn
        used = utils.tree.any(eqn_out_used)
        keep = protected or used
        new_eqn = update_eqn_out(new_eqn, active_vars) if changed and keep else new_eqn

        if protected:
            active_eqns.appendleft(new_eqn)
            active_vars |= set(analysis.var_leaves(eqn.in_tree))

        elif used:
            active_eqns.appendleft(new_eqn)
            active_vars |= set(analysis.var_leaves(utils.mask(eqn.in_tree, in_used)))

    eqns = list(active_eqns)
    out_tree = sanitize_out(ir, eqns, user_out_used)
    return core.IR(eqns, in_tree=ir.in_tree, out_tree=out_tree)
