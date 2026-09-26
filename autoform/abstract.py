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

"""Abstract values and spaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import autoform.utils as utils

type Tree[T] = utils.Tree[T]

# ==================================================================================================
# BASE TYPES
# ==================================================================================================


class AVal:
    """Base class for abstract values used by traced programs.

    Abstract values carry trace-time information about runtime values. Extension
    domains subclass ``AVal`` to describe the information primitive abstract
    rules need, such as shape, dtype, schema, or other static metadata.

    Example:
        >>> import autoform.extend as afe
        >>> class ArrayAVal(afe.AVal):
        ...     def __init__(self, shape, dtype):
        ...         self.shape = shape
        ...         self.dtype = dtype
    """

    __slots__ = []

    def zero(self):
        """Construct a concrete zero with this abstract value."""
        assert False, f"No concrete zero defined for {self!r}"

    def accumulate(self, cotangents: Tree, /):
        """Combine nonzero cotangent contributions with this abstract value."""
        assert False, f"No cotangent accumulation defined for {self!r}"


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


def materialize_zeros(x: Tree, /) -> Tree:
    """Replace each Zero leaf in a pytree with its concrete zero value.

    ``materialize_zeros`` is useful inside transform rules before calling primitives
    that expect real runtime values instead of symbolic zeros.

    Args:
        x: Pytree that may contain ``Zero`` leaves.

    Returns:
        A pytree with the same structure as ``x`` where each symbolic zero has
        been replaced by the concrete zero returned by its AVal.

    Raises:
        AssertionError: If a ``Zero`` has a type with no concrete
            zero (e.g. ``Zero(BoolAVal())``). This indicates an invalid gradient
            path through a non-differentiable type.
    """

    def map_func(x):
        if not isinstance(x, Zero):
            return x
        return x.aval.zero()

    return utils.tree.map(map_func, x)


type Val = str | int | float | bool

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

    def set[R: Callable[[AVal], AVal]](
        self, value_type: type[AVal], rule: R, /, *, replace: bool = False
    ) -> R:
        assert isinstance(value_type, type), f"Expected type, got {value_type!r}"
        assert issubclass(value_type, AVal), f"Expected AVal type, got {value_type!r}"
        assert callable(rule), f"Expected callable, got {rule!r}"
        assert isinstance(replace, bool), f"Expected bool for replace, got {type(replace)}"
        assert replace or value_type not in self.rules, f"Rule for {value_type} already defined"
        self.rules[value_type] = rule
        return rule

    def map(self, value: AVal, /) -> AVal:
        """Return the abstract value of ``value`` in this space."""
        assert isinstance(value, AVal), f"Expected AVal, got {value!r}"
        if (rule := self.rules.get(type(value))) is None:
            raise TypeError(f"No {self.name} aval rule registered for {value!r}")
        aval = rule(value)
        assert isinstance(aval, AVal), f"{self.name.capitalize()} aval rule returned {aval!r}"
        return aval


primal_s = Space("primal")
tangent_s = Space("tangent")
cotangent_s = Space("cotangent")

aval_types: dict[type, Callable[[Any], AVal]] = {}


def avalof(value, /) -> AVal:
    if (rule := aval_types.get(type(value))) is None:
        raise TypeError(f"No aval rule registered for {value!r}")
    aval = rule(value)
    assert isinstance(aval, AVal), f"Aval rule returned {aval!r}"
    return aval


aval_types[Zero] = lambda value: value.aval
