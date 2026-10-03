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

import asyncio
from collections.abc import Generator
from contextlib import contextmanager

from optree import PyTreeSpec

import autoform.core as core
import autoform.intercept as intercept
import autoform.stage as stage
import autoform.utils as utils

__all__ = ["memoize"]

type Tree[T] = utils.Tree[T]
type CacheKey = tuple[core.Prim, tuple[Tree, ...], PyTreeSpec]
non_memoizable_primitives: set[core.Prim] = {intercept.checkpoint_p}


def is_non_memo(eqn: stage.Eqn, /) -> bool:
    # NOTE(asem): input is eqn to match `is_non_dce` signature, even though the
    # memoizing interpreter works on primitve level. The following is an example of
    # nested non-memo prim (checkpoint inside siwtch branch)
    # >>> branch = af.trace(lambda x: af.checkpoint(x, key="save"))("x")
    # >>> def program(x):
    # ...     with af.memoize():
    # ...         a = af.switch("a", {"a": branch}, x)
    # ...         b = af.switch("a", {"a": branch}, x)
    # ...         return a, b
    # without checking nested IR, the second switch would not fire.
    def func(leaf):
        # NOTE(asem): check if any non-memo prim is in nested IR too.
        return isinstance(leaf, stage.IR) and any(is_non_memo(eqn) for eqn in leaf.eqns)

    return eqn.prim in non_memoizable_primitives or utils.tree.any(utils.tree.map(func, eqn.params))


def make_key(prim: core.Prim, in_tree: Tree, /, **params) -> CacheKey:
    flat, struct = utils.tree.flatten((in_tree, params))
    return (prim, tuple(flat), struct)


class MemoizingInterpreter(core.Interpreter):
    __slots__ = ["parent", "cache", "pending"]

    def __init__(self):
        self.parent = core.active_interpreter.get()
        self.cache: dict[CacheKey, Tree] = {}
        self.pending: dict[CacheKey, asyncio.Future] = {}

    def interpret(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        # NOTE(asem): constructing Eqn here is simply to make is_non_memo accepts Eqn
        # as its counter part in `is_non_dce`. a bit more work but more uniform impl.
        if is_non_memo(stage.Eqn(prim, in_tree, None, params)):
            return self.parent.interpret(prim, in_tree, **params)
        if (key := make_key(prim, in_tree, **params)) not in self.cache:
            self.cache[key] = self.parent.interpret(prim, in_tree, **params)
        return self.cache[key]

    async def ainterpret(self, prim: core.Prim, in_tree: Tree, /, **params) -> Tree:
        # NOTE(asem): the async case needs a bit more handling, in this example
        # >>> async def example(ir):
        # ...     with af.memoize():
        # ...         a = asyncio.create_task(ir.acall("x"))
        # ...         b = asyncio.create_task(ir.acall("x"))
        # ...         await asyncio.gather(a, b, return_exceptions=True)
        # ...         c = await ir.acall("x")
        #
        # for a,b unlike sequential sync case, 1) cache is not enough to mark some work is already
        # running, as a, b both can launch concurrently without cache hit. moreover,
        # error/cancellation needs a bit of care, as a can fail or be cancelled before completion
        # while b is still waiting for its result.

        if is_non_memo(stage.Eqn(prim, in_tree, None, params)):
            return await self.parent.ainterpret(prim, in_tree, **params)
        key = make_key(prim, in_tree, **params)
        if key in self.cache:
            # NOTE(asem): case 1) same call is launched and completed
            return self.cache[key]
        if key in self.pending:
            # NOTE(asem): case 2) shield prevents cancelling b from cancelling the shared future.
            # if not shield, then cancelling a-b future, when a completes future.set_result will
            # fail.
            return await asyncio.shield(self.pending[key])

        # NOTE(asem) case 3) primitive is memoizable with no cache/pending result
        future = asyncio.get_running_loop().create_future()

        try:
            # NOTE(asem): case 3a) task a is launched
            self.pending[key] = future
            result = await self.parent.ainterpret(prim, in_tree, **params)
        except Exception as error:
            # NOTE(asem): case 3b) task a raises an error, box the future with exception
            # and raise error in task a.
            future.set_exception(error)
            # NOTE(asem): in case of a single task, the exception is never unboxed, thus asyncio
            # may print an error message.
            future.exception()
            raise
        except asyncio.CancelledError:
            # NOTE(asem): case 3c) task a is canelled, future is cancelled and b recieves
            # canellation error
            future.cancel()
            raise
        else:
            # NOTE(asem): case 3a) task a computation is ready, box it in future.
            future.set_result(result)
            self.cache[key] = result
            return result
        finally:
            # NOTE(asem): remove pending key for all cases.
            del self.pending[key]


@contextmanager
def memoize() -> Generator[None, None, None]:
    """Cache primitive results within the context.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     a = x + "!"
        ...     b = x + "!"  # same call, will be cached
        ...     return a + b
        >>> ir = af.trace(program)("test")
        >>> with af.memoize():
        ...     result = ir.call("hello")
        >>> result
        'hello!hello!'

    Tracing a program with :func:`memoize` will act as compile-time deduplication of
    identical primitive calls (including stochastic primitives like :func:`autoform.lm.fill`).
    Non-memoizable primitives are not memoized.

    Example:
        >>> def program(x):
        ...     with af.memoize():
        ...         a = x + "!"
        ...         b = x + "!"  # same call, will be cached
        ...         return a, b
        >>> ir = af.trace(program)("test")
        >>> len(ir.eqns)
        1
    """
    with core.using_interpreter(MemoizingInterpreter()):
        yield
