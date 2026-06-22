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
type LiveSet = set[core.Var]
type Liveness = list[LiveSet]

__all__ = ["var_leaves", "var_producers", "eqn_graph", "ir_liveness"]


def var_leaves(ir_tree: Tree, /) -> list[core.Var]:
    """Return Vars from an IR tree in leaf order."""

    return [cast(core.Var, x) for x in utils.tree.leaves(ir_tree) if core.is_irvar(x)]


def var_producers(ir: core.IR, /) -> dict[core.Var, core.Eqn]:
    """Return the top-level producer equation for each Var defined by ``ir``."""

    producers: dict[core.Var, core.Eqn] = {}
    for eqn in ir.eqns:
        for var in var_leaves(eqn.out_ir_tree):
            assert producers.get(var) is None
            producers[var] = eqn
    return producers


def eqn_graph(ir: core.IR, /) -> dict[core.Eqn, list[core.Eqn]]:
    """Return top-level equation dependencies as parent -> children adjacency."""

    var_to_parent = var_producers(ir)
    adjacency_list: dict[core.Eqn, list[core.Eqn]] = {eqn: [] for eqn in ir.eqns}
    for eqn in ir.eqns:
        seen_parents: set[core.Eqn] = set()
        for in_var in var_leaves(eqn.in_ir_tree):
            if (p := var_to_parent.get(in_var)) is not None and p not in seen_parents:
                adjacency_list[p].append(eqn)
                seen_parents.add(p)

    return adjacency_list


def ir_liveness(ir: core.IR, /, *, out_used: UsedTree | None = None) -> Liveness:
    """Return live Vars at each IR boundary."""

    # NOTE(asem): liveness is a backward dataflow analysis that computes Vars live
    # at each boundary. The result length is len(ir.eqns) + 1: the first item is
    # the live input boundary, and the last item is the selected output boundary.

    if out_used is None:
        live_after = set(var_leaves(ir.out_ir_tree))
    else:
        assert utils.tree.all(isinstance(leaf, bool) for leaf in utils.tree.leaves(out_used))
        assert utils.tree.structure(out_used) == utils.tree.structure(ir.out_ir_tree)
        # NOTE(asem): with a partial output mask, only the selected output Vars are live.
        # >>> def program(x):
        # ...     a = af.concat(x, "!")
        # ...     b = af.concat(x, "?")
        # ...     return a, b
        # >>> ir_liveness(ir, out_used=(True, False))[-1]
        # {a}
        live_after = set(var_leaves(utils.mask(ir.out_ir_tree, out_used)))

    liveness: Liveness = [set() for _ in range(len(ir.eqns) + 1)]
    liveness[-1] = live_after

    for i, eqn in reversed(tuple(enumerate(ir.eqns))):
        # NOTE(asem): move in reversed order of equation list starting from the output Vars
        # with each step up the live before is basically all the live Vars used + live after
        # without the Vars defined by the current equation.
        uses: LiveSet = set(var_leaves(eqn.in_ir_tree))
        defs: LiveSet = set(var_leaves(eqn.out_ir_tree))
        live_before = uses | (live_after - defs)
        liveness[i] = live_before
        live_after = live_before

    return liveness
