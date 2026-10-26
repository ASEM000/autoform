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

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import autoform.src.core as core

type Tree[T] = Any


class ImplInterp(core.Interp):
    __slots__ = []

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        return core.r_impl.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        return await core.r_aimpl.get(prim)(in_tree, **params)


core.active_interp = ContextVar("active_interp", default=ImplInterp())
