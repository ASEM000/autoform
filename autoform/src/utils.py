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

"""Utility functions for autoform"""

from __future__ import annotations

import builtins
import functools as ft
from collections.abc import Awaitable, Callable
from typing import Any, cast

import autoform.src.tree as tree

type Tree[T] = Any


def strict_zip(*iterables):
    return builtins.zip(*iterables, strict=True)


zip = strict_zip


# ==================================================================================================
# ASYNC UTILITIES
# ==================================================================================================


def asyncify[**P, R](func: Callable[P, R], /) -> Callable[P, Awaitable[R]]:
    @ft.wraps(func)
    async def afunc(*args, **kwargs):
        return func(*args, **kwargs)

    return afunc


def dispatch[R](default: Callable[..., R] | None = None):
    # NOTE(asem): dispatch using optree tree spec with leaves exact type. unlike single dispatch
    # this dispatch on exact type, allow pytrees. however its a bit slower.
    rules: dict[tuple[tree.PyTreeSpec, tuple[type, ...]], Callable[..., R]] = {}

    def register[**P](types: Tree[type], rule: Callable[P, R], /) -> Callable[P, R]:
        leaves, spec = tree.flatten(types)
        rules[spec, tuple(leaves)] = rule
        return rule

    def call(values: Tree, /, **params: Any) -> R:
        leaves, spec = tree.flatten(values)
        key = spec, tuple(type(x) for x in leaves)
        if (rule := rules.get(key, default)) is None:
            raise TypeError(f"No dispatch rule for {key!r}")
        return rule(values, **params)

    if default is not None:
        call = ft.wraps(default)(call)
    call.register = register
    call.rules = rules
    return call


def lru_cache[**P, R](func: Callable[P, R], maxsize: int = 256) -> Callable[P, R]:
    return cast(Callable[P, R], ft.lru_cache(maxsize=maxsize)(func))


def index(node: Tree, b: int, /) -> Tree:
    # NOTE(asem): index a struct without indexing support
    # useful to deal with arbitrary pytrees
    children = tree.leaves(node, is_leaf=lambda x: x is None or x is not node)
    return children[b]
