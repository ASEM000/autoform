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

from autoform.core import IRVal, IRVar, is_irvar
from autoform.utils import Tree, treelib

__all__ = ["ir_tree_ir_vars"]


def ir_tree_ir_vars(tree: Tree[IRVal], /) -> tuple[IRVar, ...]:
    """Return IRVars from an IR tree in leaf order."""

    return tuple(cast(IRVar, x) for x in treelib.leaves(tree) if is_irvar(x))
