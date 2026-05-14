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

"""Extension API for AutoForm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from autoform.ad import (
    Zero,
    cot_acc_rules,
    is_zero,
    materialize,
    zero_rules,
    zeroof,
)
from autoform.core import (
    AVal,
    BoolAVal,
    EvalType,
    FloatAVal,
    IntAVal,
    Prim,
    StrAVal,
    TraceRule,
    abstract_rules,
    aval_rules,
    avalof,
    batch_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    trace_add_rules,
    trace_eq_rules,
    trace_matmul_rules,
    trace_mul_rules,
    trace_sub_rules,
    trace_truediv_rules,
    trace_types,
)
from autoform.utils import Tree, batch_index, batch_spec

__all__ = [
    "AVal",
    "StrAVal",
    "IntAVal",
    "FloatAVal",
    "BoolAVal",
    "EvalType",
    "Prim",
    "TraceRule",
    "Tree",
    "Zero",
    "register_trace_type",
    "register_zero",
    "register_cotangent_accumulator",
    "register_add",
    "register_sub",
    "register_mul",
    "register_div",
    "register_matmul",
    "register_eq",
    "impl_rules",
    "abstract_rules",
    "push_rules",
    "pull_fwd_rules",
    "pull_bwd_rules",
    "batch_rules",
    "avalof",
    "zeroof",
    "materialize",
    "is_zero",
    "batch_index",
    "batch_spec",
]

type AValRule = Callable[[Any], AVal]
type ZeroRule = Callable[[AVal], Any]
type CotAccRule = Callable[[list[Any], AVal], Any]


def register_trace_type(type: type, aval_rule: AValRule, /) -> AValRule:
    """Register a Python type as a traceable input type.

    :func:`autoform.trace` treats registered Python types as dynamic leaves. During
    tracing, each concrete value is passed to ``aval_rule`` and replaced by an
    :class:`AVal` that carries the abstract information needed by primitive rules.

    Registering a trace type only teaches AutoForm how to abstract concrete
    inputs. It does not define concrete execution, AD behavior, batching, or
    Python operator syntax. Those are registered separately through primitive
    rules and the other helpers in this module.

    Args:
        type: Concrete Python type accepted as a dynamic input leaf.
        aval_rule: Function from a concrete value to its abstract value.

    Returns:
        The registered rule.

    Example:
        >>> import functools as ft
        >>> import autoform.extend as afe
        >>> class Token: ...
        >>> class TokenAVal(afe.AVal): ...
        >>> @ft.partial(afe.register_trace_type, Token)
        ... def token_aval(value):
        ...     return TokenAVal()
    """
    trace_types.add(type)
    aval_rules[type] = aval_rule
    return aval_rule


def register_zero(aval_type: type[AVal], rule: ZeroRule, /) -> ZeroRule:
    """Register the concrete zero value for an abstract value type.

    Reverse-mode transforms use :class:`Zero` to represent a missing or blocked
    cotangent. :func:`autoform.extend.materialize` later turns that symbolic zero
    into a concrete runtime value by looking up this rule.

    Register this for differentiable value domains whose cotangents may need to
    flow through primitives that materialize zeros, such as
    :func:`autoform.pushforward` or custom pullback rules.

    Args:
        aval_type: Abstract value type this zero rule handles.
        rule: Function from an abstract value instance to a concrete zero.

    Returns:
        The registered rule.

    Example:
        >>> import functools as ft
        >>> import autoform.extend as afe
        >>> class TokenAVal(afe.AVal): ...
        >>> @ft.partial(afe.register_zero, TokenAVal)
        ... def zero_token(aval):
        ...     return ""
    """
    zero_rules[aval_type] = rule
    return rule


def register_cotangent_accumulator(aval_type: type[AVal], rule: CotAccRule, /) -> CotAccRule:
    """Register how to combine cotangents for an abstract value type.

    :func:`autoform.pullback` can produce multiple cotangent contributions for the same input.
    AutoForm groups those contributions and calls the accumulator for the
    corresponding leaf :class:`AVal`.

    The rule receives the non-zero cotangents and the abstract value of the
    leaf being accumulated. Zeros are filtered before the rule is called.

    Args:
        aval_type: Abstract value type this accumulator handles.
        rule: Function of ``(cotangents, aval)`` returning one cotangent.

    Returns:
        The registered rule.

    Example:
        >>> import functools as ft
        >>> import autoform.extend as afe
        >>> class ScoreAVal(afe.AVal): ...
        >>> @ft.partial(afe.register_cotangent_accumulator, ScoreAVal)
        ... def add_scores(cotangents, aval):
        ...     return sum(cotangents)
    """
    cot_acc_rules[aval_type] = rule
    return rule


def register_add(aval_type: type[AVal], rule: TraceRule, /) -> TraceRule:
    """Register tracing dispatch for ``+`` on traced values with this aval.

    This treats ``+`` as staged syntax while tracing. The rule is called during
    tracing, not during normal execution, and should usually bind an AutoForm
    primitive that implements the operation.

    Args:
        aval_type: Abstract value type of the left traced operand.
        rule: Function called as ``rule(left, right)`` for ``left + right`` and
            ``rule(right, left)`` for reflected ``right + left``.

    Returns:
        The registered rule.
    """
    trace_add_rules[aval_type] = rule
    return rule


def register_sub(aval_type: type[AVal], rule: TraceRule, /) -> TraceRule:
    """Register tracing dispatch for ``-`` on traced values with this aval.

    This treats ``-`` as staged syntax while tracing. The rule should normally
    bind the primitive that represents subtraction for the extension domain.

    Args:
        aval_type: Abstract value type of the traced operand that dispatches.
        rule: Function called as ``rule(left, right)`` for ``left - right`` and
            ``rule(right, left)`` for reflected ``right - left``.

    Returns:
        The registered rule.
    """
    trace_sub_rules[aval_type] = rule
    return rule


def register_mul(aval_type: type[AVal], rule: TraceRule, /) -> TraceRule:
    """Register tracing dispatch for ``*`` on traced values with this aval.

    This treats ``*`` as staged syntax while tracing, instead of evaluating the
    operation with Python.

    Args:
        aval_type: Abstract value type of the traced operand that dispatches.
        rule: Function called as ``rule(left, right)`` for ``left * right`` and
            ``rule(right, left)`` for reflected ``right * left``.

    Returns:
        The registered rule.
    """
    trace_mul_rules[aval_type] = rule
    return rule


def register_div(aval_type: type[AVal], rule: TraceRule, /) -> TraceRule:
    """Register tracing dispatch for ``/`` on traced values with this aval.

    This treats true division as staged syntax while tracing.

    Args:
        aval_type: Abstract value type of the traced operand that dispatches.
        rule: Function called as ``rule(left, right)`` for ``left / right`` and
            ``rule(right, left)`` for reflected ``right / left``.

    Returns:
        The registered rule, so the helper can be used as a decorator.
    """
    trace_truediv_rules[aval_type] = rule
    return rule


def register_matmul(aval_type: type[AVal], rule: TraceRule, /) -> TraceRule:
    """Register tracing dispatch for ``@`` on traced values with this aval.

    This treats matrix multiplication as staged syntax while tracing. It is
    intended for domains such as arrays, matrices, or tensors where
    matrix multiplication should stage a primitive into the IR.

    Args:
        aval_type: Abstract value type of the traced operand that dispatches.
        rule: Function called as ``rule(left, right)`` for ``left @ right`` and
            ``rule(right, left)`` for reflected ``right @ left``.

    Returns:
        The registered rule, so the helper can be used as a decorator.
    """
    trace_matmul_rules[aval_type] = rule
    return rule


def register_eq(aval_type: type[AVal], rule: TraceRule, /) -> TraceRule:
    """Register tracing dispatch for ``==`` on traced values with this aval.

    Python equality on a traced value cannot be evaluated concretely during
    tracing. Register this when equality should become a staged primitive.

    Args:
        aval_type: Abstract value type of the left traced operand.
        rule: Function called as ``rule(left, right)`` for ``left == right``.

    Returns:
        The registered rule, so the helper can be used as a decorator.
    """
    trace_eq_rules[aval_type] = rule
    return rule
