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

"""Shared IR analysis utilities."""

from __future__ import annotations

from typing import cast

import autoform.core as core
import autoform.utils as utils

type Tree[T] = utils.Tree[T]
type UsedTree = Tree[bool]
type LiveSet = set[core.IRVar]
type Liveness = list[tuple[LiveSet, LiveSet]]

__all__ = [
    "ir_tree_ir_vars",
    "ir_var_producers",
    "ir_eqn_graph",
    "ir_liveness",
    "ir_tree_used_ir_vars",
]


def ir_tree_ir_vars(ir_tree: Tree, /) -> tuple[core.IRVar, ...]:
    """Return IRVars from an IR tree in leaf order."""

    return tuple(cast(core.IRVar, x) for x in utils.tree.leaves(ir_tree) if core.is_irvar(x))


def ir_var_producers(ir: core.IR, /) -> dict[core.IRVar, core.IREqn]:
    """Return the top-level producer equation for each IRVar defined by ``ir``."""

    producers: dict[core.IRVar, core.IREqn] = {}
    for ir_eqn in ir.ir_eqns:
        for ir_var in ir_tree_ir_vars(ir_eqn.out_ir_tree):
            assert producers.get(ir_var) is None
            producers[ir_var] = ir_eqn
    return producers


def ir_eqn_graph(ir: core.IR, /) -> dict[core.IREqn, list[core.IREqn]]:
    """Return top-level equation dependencies as parent -> children adjacency."""

    ir_var_to_parent = ir_var_producers(ir)
    adjacency_list: dict[core.IREqn, list[core.IREqn]] = {ir_eqn: [] for ir_eqn in ir.ir_eqns}
    for ir_eqn in ir.ir_eqns:
        seen_parents: set[core.IREqn] = set()
        for in_ir_var in ir_tree_ir_vars(ir_eqn.in_ir_tree):
            if (p := ir_var_to_parent.get(in_ir_var)) is not None and p not in seen_parents:
                adjacency_list[p].append(ir_eqn)
                seen_parents.add(p)

    return adjacency_list


def ir_tree_used_ir_vars(ir_tree: Tree, used: UsedTree, /) -> LiveSet:
    # NOTE(asem): this helper reads a pytree of IR leaves together with a pytree mask of bools
    # and returns exactly the IRVars whose corresponding mask entry is True.
    # >>> tree = (IRVar(id=1), IRVar(id=2))
    # >>> used = (True, False)
    # >>> ir_tree_used_ir_vars(tree, used)
    # {IRVar(id=1)}
    assert utils.tree.structure(ir_tree) == utils.tree.structure(used)
    assert utils.tree.all(isinstance(leaf, bool) for leaf in utils.tree.leaves(used))
    used_ir_vars: LiveSet = set()
    flat_tree, flat_used = utils.tree.leaves(ir_tree), utils.tree.leaves(used)
    for ir_atom, keep in zip(flat_tree, flat_used, strict=True):
        keep and core.is_irvar(ir_atom) and used_ir_vars.add(ir_atom)
    return used_ir_vars


def ir_liveness(ir: core.IR, /, *, out_used: UsedTree = None) -> Liveness:
    """Return per-equation liveness as ``(before, after)`` pairs."""

    # NOTE(asem): liveness is a backward dataflow analysis that computes, for each equation, the set
    # IRVars that are live (used by later equations or outputs) at the boundary before that equation.

    if out_used is None:
        live_after = set(ir_tree_ir_vars(ir.out_ir_tree))
    else:
        assert utils.tree.all(isinstance(leaf, bool) for leaf in utils.tree.leaves(out_used))
        assert utils.tree.structure(out_used) == utils.tree.structure(ir.out_ir_tree)
        # NOTE(asem): with a partial output mask, only the selected output IRVars are live.
        # >>> def program(x):
        # ...     a = af.concat(x, "!")
        # ...     b = af.concat(x, "?")
        # ...     return a, b
        # >>> ir_liveness(ir, out_used=(True, False))[-1][1]
        # {a}
        live_after = ir_tree_used_ir_vars(ir.out_ir_tree, out_used)

    liveness: Liveness = [None] * len(ir.ir_eqns)

    for i in reversed(range(len(ir.ir_eqns))):
        ir_eqn = ir.ir_eqns[i]
        uses = set(ir_tree_ir_vars(ir_eqn.in_ir_tree))
        defs = set(ir_tree_ir_vars(ir_eqn.out_ir_tree))
        live_before = uses | (live_after - defs)
        liveness[i] = (live_before, live_after)
        live_after = live_before

    return liveness
