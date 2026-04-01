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

from autoform.core import IR, IREqn, IRVal, IRVar, is_irvar
from autoform.utils import Tree, treelib

__all__ = ["ir_tree_ir_vars", "ir_var_producers", "ir_eqn_graph"]


def ir_tree_ir_vars(tree: Tree[IRVal], /) -> tuple[IRVar, ...]:
    """Return IRVars from an IR tree in leaf order."""

    return tuple(cast(IRVar, x) for x in treelib.leaves(tree) if is_irvar(x))


def ir_var_producers(ir: IR, /) -> dict[IRVar, IREqn]:
    """Return the top-level producer equation for each IRVar defined by ``ir``."""

    producers: dict[IRVar, IREqn] = {}
    for ir_eqn in ir.ir_eqns:
        for ir_var in ir_tree_ir_vars(ir_eqn.out_ir_tree):
            assert (prior := producers.get(ir_var)) is None, (
                f"IRVar {ir_var!r} is produced by multiple equations: {prior!r}, {ir_eqn!r}"
            )
            producers[ir_var] = ir_eqn
    return producers


def ir_eqn_graph(ir: IR, /) -> dict[IREqn, list[IREqn]]:
    """Return top-level equation dependencies as parent -> children adjacency."""

    ir_var_to_parent = ir_var_producers(ir)
    adjacency_list: dict[IREqn, list[IREqn]] = {ir_eqn: [] for ir_eqn in ir.ir_eqns}
    for ir_eqn in ir.ir_eqns:
        seen_parents: set[IREqn] = set()
        for in_ir_var in ir_tree_ir_vars(ir_eqn.in_ir_tree):
            if (p := ir_var_to_parent.get(in_ir_var)) is not None and p not in seen_parents:
                adjacency_list[p].append(ir_eqn)
                seen_parents.add(p)

    return adjacency_list
