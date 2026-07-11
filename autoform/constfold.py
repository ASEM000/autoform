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

"""Constant folding"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import autoform.core as core
import autoform.utils as utils

__all__ = ["constfold"]

type Tree[T] = utils.Tree[T]


def constfold[*A, R](
    ir: core.IR[*A, R], /, *, cond: Callable[[core.Eqn], bool] | None = None
) -> core.IR[*A, R]:
    """Evaluate IR equations whose inputs are concrete literals.

    Args:
        ir: The IR to fold.
        cond: Optional predicate that takes an equation and returns ``True`` if
            the equation may be evaluated when its inputs are concrete. If
            ``None``, all concrete equations are candidates.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     prefix = af.concat("hello", " ")
        ...     return af.concat(prefix, x)
        >>> ir = af.trace(program)("seed")
        >>> folded = af.constfold(ir)
        >>> len(folded.eqns)
        1
    """

    assert isinstance(ir, core.IR), f"Expected IR, got {type(ir)}"
    cond = (lambda _: True) if cond is None else cond

    def has_var(value: Tree) -> bool:
        return any(core.is_var(leaf) for leaf in utils.tree.leaves(value))

    def fold_param(value):
        return constfold(value, cond=cond) if isinstance(value, core.IR) else value

    env: dict[core.Var, Any] = {}
    out_eqns: list[core.Eqn] = []

    def read(ir_val):
        return env.get(ir_val, ir_val) if core.is_var(ir_val) else ir_val

    def write(ir_val, value):
        if core.is_var(ir_val):
            env[ir_val] = value

    for eqn in ir.eqns:
        in_tree = utils.tree.map(read, eqn.in_tree)
        params = utils.tree.map(fold_param, eqn.params)
        eqn = core.Eqn(eqn.prim, in_tree, eqn.out_tree, params, eqn.tags)
        if has_var(in_tree) or not cond(eqn):
            out_eqns.append(eqn)
            continue

        with core.using_interpreter(core.EvalInterpreter()):
            out_tree = eqn.bind(in_tree, **eqn.params)
        utils.tree.map(write, eqn.out_tree, out_tree)

    out_tree = utils.tree.map(read, ir.out_tree)
    return core.IR(out_eqns, ir.in_tree, out_tree)
