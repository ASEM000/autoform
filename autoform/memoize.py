"""Memoize"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from optree import PyTreeSpec

from autoform.core import Effect, EffectInterpreter, Primitive, using_interpreter
from autoform.utils import Tree, treelib


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

    cache: dict[tuple[Primitive, Effect | None, tuple[Tree, ...], PyTreeSpec], Tree] = {}

    def make_key(prim, effect: Effect | None, in_tree: Any, /, **params):
        flat, struct = treelib.flatten((in_tree, params))
        return (prim, effect, tuple(flat), struct)

    def handler(prim, effect, in_tree, /, **params):
        key = make_key(prim, effect, in_tree, **params)
        if key in cache:
            return cache[key]
        out_tree = yield in_tree
        cache[key] = out_tree
        return out_tree

    with using_interpreter(EffectInterpreter(default=handler)):
        yield
