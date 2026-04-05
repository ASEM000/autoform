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

"""Memoize"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from optree import PyTreeSpec

from autoform.checkpoint import checkpoint_p
from autoform.core import (
    Interpreter,
    Prim,
    active_intercept,
    active_interpreter,
    using_interpreter,
)
from autoform.utils import Tree, treelib

type CacheKey = tuple[Prim, tuple[Tree, ...], PyTreeSpec]
non_memoizable_primitives: set[Prim] = {checkpoint_p}


def make_key(prim: Prim, in_tree: Tree, /, **params) -> CacheKey:
    flat, struct = treelib.flatten((in_tree, params))
    return (prim, tuple(flat), struct)


class MemoizingInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()
        self.cache: dict[CacheKey, Tree] = {}

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        if active_intercept.get() is not None or prim in non_memoizable_primitives:
            return self.parent.interpret(prim, in_tree, **params)
        if (key := make_key(prim, in_tree, **params)) not in self.cache:
            self.cache[key] = self.parent.interpret(prim, in_tree, **params)
        return self.cache[key]

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        if active_intercept.get() is not None or prim in non_memoizable_primitives:
            return await self.parent.ainterpret(prim, in_tree, **params)
        if (key := make_key(prim, in_tree, **params)) not in self.cache:
            self.cache[key] = await self.parent.ainterpret(prim, in_tree, **params)
        return self.cache[key]


@contextmanager
def memoize() -> Generator[None, None, None]:
    """Cache primitive results within the context.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     a = af.concat(x, "!")
        ...     b = af.concat(x, "!")  # same call, will be cached
        ...     return af.concat(a, b)
        >>> ir = af.trace(program)("test")
        >>> with af.memoize():
        ...     result = ir.call("hello")
        >>> result
        'hello!hello!'

    Tracing a program with `memoize` will act as compile-time deduplication of
    identical primitive calls (including stochastic primitives like :func:`lm_call`).
    intercepted and non-memoizable primitives are not memoized.

    Example:
        >>> def program(x):
        ...     with af.memoize():
        ...         a = af.concat(x, "!")
        ...         b = af.concat(x, "!")  # same call, will be cached
        ...         return a, b
        >>> ir = af.trace(program)("test")
        >>> len(ir.ir_eqns)
        1
    """
    with using_interpreter(MemoizingInterpreter()):
        yield
