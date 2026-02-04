"""Memoize"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from optree import PyTreeSpec

from autoform.core import Interpreter, Primitive, active_interpreter, using_interpreter
from autoform.utils import Tree, treelib

CacheKey = tuple[Primitive, tuple[Tree, ...], PyTreeSpec]


def make_key(prim: Primitive, in_tree: Tree, /, **params) -> CacheKey:
    flat, struct = treelib.flatten((in_tree, params))
    return (prim, tuple(flat), struct)


class MemoizingInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()
        self.cache: dict[CacheKey, Tree] = {}

    def interpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree:
        if (key := make_key(prim, in_tree, **params)) in self.cache:
            return self.cache[key]
        result = self.parent.interpret(prim, in_tree, **params)
        self.cache[key] = result
        return result

    async def ainterpret(self, prim: Primitive, in_tree: Tree, /, **params) -> Tree:
        if (key := make_key(prim, in_tree, **params)) in self.cache:
            return self.cache[key]
        result = await self.parent.ainterpret(prim, in_tree, **params)
        self.cache[key] = result
        return result


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
        ...     result = af.call(ir)("hello")
        >>> result
        'hello!hello!'

    Tracing a program with `memoize` will act as compile-time deduplication of
    identical primitive calls.

    Example:
        >>> def program(x):
        ...     with af.memoize():
        ...         a = af.concat(x, "!")
        ...         b = af.concat(x, "!")  # same call, will be cached
        ...         return a, b
        >>> ir = af.trace(program)("test")
        >>> len(ir.ireqns)
        1
    """
    with using_interpreter(MemoizingInterpreter()):
        yield
