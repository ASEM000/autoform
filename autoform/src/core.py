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

"""Core."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Self

import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]


# ==================================================================================================
# AVAL
# ==================================================================================================


aval_types: dict[type, Callable[[Any], AVal]] = {}


class AVal:
    """Base class for abstract values used by traced programs.

    Abstract values carry trace-time information about runtime values. Extension
    domains subclass ``AVal`` to describe the information primitive abstract
    rules need, such as shape, dtype, schema, or other static metadata.
    By default, equality and hashing depend only on the AVal class. Domains with
    metadata should override both methods to include that metadata.

    Example:
        >>> import autoform.extend as afe
        >>> class AValArray(afe.AVal):
        ...     def __init__(self, shape, dtype):
        ...         self.shape = shape
        ...         self.dtype = dtype
    """

    __slots__ = []

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))


def avalof(value, /) -> AVal:
    if (rule := aval_types.get(type(value))) is None:
        raise TypeError(f"No aval rule registered for {value!r}")
    aval = rule(value)
    assert isinstance(aval, AVal), f"Aval rule returned {aval!r}"
    return aval


# ==================================================================================================
# ALGEBRA
# ==================================================================================================


class Algebra:
    __slots__ = ["aval"]

    def __init__(self, aval: AVal, /):
        self.aval = aval

    def zero(self):
        raise NotImplementedError

    def accum(self, values: list[Any], /):
        raise NotImplementedError


algebra_rules: dict[type[AVal], Callable[[AVal], Algebra]] = {}


def algebraof(aval: AVal, /) -> Algebra:
    """Look up algebra behavior by exact AVal type, without inheritance."""
    if (rule := algebra_rules.get(type(aval))) is None:
        raise TypeError(f"No algebra registered for {aval!r}")
    algebra = rule(aval)
    assert isinstance(algebra, Algebra), f"Algebra rule returned {algebra!r}"
    return algebra


# ==================================================================================================
# DOMAIN
# ==================================================================================================


class Domain:
    __slots__ = ["aval"]

    def __init__(self, aval: AVal, /):
        self.aval = aval

    def base(self) -> AVal:
        return self.aval

    def accepts(self, value, /) -> bool:
        actual = avalof(value)
        if self.aval == actual:
            return True
        base = domainof(actual).base()
        return base != actual and self.accepts(base)

    def join(self, other: AVal, /) -> AVal:
        if self.accepts(other):
            return self.aval
        domain = domainof(other)
        if domain.accepts(self.aval):
            return other
        left, right = self.base(), domain.base()
        if left != self.aval or right != other:
            return domainof(left).join(right)
        raise TypeError(f"No common domain for {self.aval!r} and {other!r}")


domain_rules: dict[type[AVal], Callable[[AVal], Domain]] = {}


def domainof(aval: AVal, /) -> Domain:
    """Return domain of input abstract value"""
    if (rule := domain_rules.get(type(aval))) is None:
        return Domain(aval)
    domain = rule(aval)
    assert isinstance(domain, Domain), f"Domain rule returned {domain!r}"
    return domain


# ==================================================================================================
# SPACES
# ==================================================================================================


class Space:
    __slots__ = ["name", "rules"]

    def __init__(self, name: str, /):
        assert isinstance(name, str), f"Expected str, got {name!r}"
        self.name = name
        self.rules: dict[type[AVal], Callable[[AVal], AVal]] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def set[R](self, value: type[AVal], rule: R, /, *, replace: bool = False) -> R:
        assert isinstance(value, type), f"Expected type, got {value!r}"
        assert issubclass(value, AVal), f"Expected AVal type, got {value!r}"
        assert callable(rule), f"Expected callable, got {rule!r}"
        assert isinstance(replace, bool), f"Expected bool for replace, got {type(replace)}"
        assert replace or value not in self.rules, f"Rule for {value} already defined"
        self.rules[value] = rule
        return rule

    def map(self, value: AVal, /) -> AVal:
        """Return the abstract value of ``value`` in this space."""
        assert isinstance(value, AVal), f"Expected AVal, got {value!r}"
        if (rule := self.rules.get(type(value))) is None:
            raise TypeError(f"No {self.name} aval rule registered for {value!r}")
        aval = rule(value)
        assert isinstance(aval, AVal), f"{self.name.capitalize()} aval rule returned {aval!r}"
        return aval


class Zero[T: AVal]:
    """Symbolic zero for an abstract value."""

    __slots__ = ["aval"]

    def __init__(self, aval: T, /):
        assert isinstance(aval, AVal), f"Expected AVal, got {aval!r}"
        self.aval = aval

    def __repr__(self):
        return f"Zero({self.aval!r})"

    def __eq__(self, other):
        return isinstance(other, Zero) and self.aval == other.aval

    def __hash__(self):
        return hash((type(self), self.aval))


s_primal = Space("primal")
s_tangent = Space("tangent")
s_cotangent = Space("cotangent")

aval_types[Zero] = lambda value: value.aval

# ==================================================================================================
# PRIMITIVE
# ==================================================================================================


class Prim:
    """Primitive operation key used by interpreter rule registries.

    A primitive has no behavior by itself. Runtime, abstract, batching, and AD
    behavior are attached by registering rules keyed by the ``Prim`` instance.

    Args:
        name: The name of the primitive.

    Example:
        >>> import autoform.extend as afe
        >>> p_add = afe.Prim("add")
    """

    __slots__ = ["name"]

    def __init__(self, name: str):
        assert isinstance(name, str), f"Invalid name type: {type(name)=}"
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def bind(self, value: Tree, /, **params):
        return active_interpreter.get().interpret(self, value, **params)

    async def abind(self, value: Tree, /, **params):
        return await active_interpreter.get().ainterpret(self, value, **params)


# ==================================================================================================
# RULES
# ==================================================================================================


class Rule[T]:
    __slots__ = ["name", "base", "active"]

    def __init__(self, name: str):
        assert isinstance(name, str), f"Invalid name type: {type(name)=}"
        self.name = name
        self.base: ChainMap[Prim, Callable[..., T]] = ChainMap()
        self.active = ContextVar(f"{name} rules", default=self.base)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def set[R: Callable[..., T]](self, prim: Prim, rule: R, /) -> R:
        assert isinstance(prim, Prim)
        assert callable(rule)
        self.base[prim] = rule
        return rule

    def get(self, prim: Prim, /) -> Callable[..., T]:
        return self.active.get()[prim]

    @contextmanager
    def override(self, *overrides: tuple[Prim, Callable[..., T]]) -> Generator[Self, None, None]:
        for prim, rule in overrides:
            assert isinstance(prim, Prim)
            assert callable(rule)
        token = self.active.set(self.active.get().new_child(dict(overrides)))
        try:
            yield self
        finally:
            self.active.reset(token)


r_impl: Rule[Tree] = Rule("impl")
r_aimpl: Rule[Awaitable[Tree]] = Rule("aimpl")
r_abstract: Rule[Tree] = Rule("abstract")
r_batch: Rule[tuple[Tree, Tree[bool]]] = Rule("batch")
r_abatch: Rule[Awaitable[tuple[Tree, Tree[bool]]]] = Rule("abatch")
r_push: Rule[tuple[Tree, Tree]] = Rule("pushforward")
r_apush: Rule[Awaitable[tuple[Tree, Tree]]] = Rule("apushforward")
r_pull_fwd: Rule[tuple[Tree, Tree]] = Rule("pullback_fwd")
r_apull_fwd: Rule[Awaitable[tuple[Tree, Tree]]] = Rule("apullback_fwd")
r_pull_bwd: Rule[Tree] = Rule("pullback_bwd")
r_apull_bwd: Rule[Awaitable[Tree]] = Rule("apullback_bwd")


# ==================================================================================================
# INTERPRETER
# ==================================================================================================


class Interpreter:
    __slots__ = []

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        raise NotImplementedError

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        raise NotImplementedError


@contextmanager
def using_interpreter[T: Interpreter](interpreter: T) -> Generator[T, None, None]:
    """Run primitive dispatch through an interpreter inside the context."""

    token = active_interpreter.set(interpreter)
    try:
        yield interpreter
    finally:
        active_interpreter.reset(token)


# ==================================================================================================
# EVAL
# ==================================================================================================


class EvalInterpreter(Interpreter):
    __slots__ = []

    def interpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return r_impl.get(prim)(in_tree, **params)

    async def ainterpret(self, prim: Prim, in_tree: Tree, /, **params) -> Tree:
        return await r_aimpl.get(prim)(in_tree, **params)


active_interpreter = ContextVar[Interpreter]("active_interpreter", default=EvalInterpreter())
