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

import autoform.ad as ad
import autoform.axis as axis
import autoform.control as control
import autoform.core as core
import autoform.dead as dead
import autoform.intercept as intercept
import autoform.lm as lm
import autoform.memo as memo
import autoform.order as order
import autoform.path as path
import autoform.string as string
import autoform.utils as utils

# ==================================================================================================
# TYPES
# ==================================================================================================

AVal = core.AVal
StrAVal = core.StrAVal
IntAVal = core.IntAVal
FloatAVal = core.FloatAVal
BoolAVal = core.BoolAVal
Space = core.Space
primal_s = core.primal_s
tangent_s = core.tangent_s
cotangent_s = core.cotangent_s
Prim = core.Prim
Zero = ad.Zero
Interpreter = core.Interpreter
IR = core.IR
Eqn = core.Eqn
Var = core.Var

# ==================================================================================================
# RULE REGISTRIES
# ==================================================================================================

impl_rules = core.impl_rules
abstract_rules = core.abstract_rules
push_rules = core.push_rules
pull_fwd_rules = core.pull_fwd_rules
pull_bwd_rules = core.pull_bwd_rules
batch_rules = core.batch_rules

# ==================================================================================================
# HELPERS
# ==================================================================================================

zeroof = ad.zeroof
tangent_zeroof = ad.tangent_zeroof
cotangent_zeroof = ad.cotangent_zeroof
materialize = ad.materialize
is_zero = ad.is_zero
batch_index = utils.batch_index
batch_spec = utils.batch_spec
batch_transpose = utils.batch_transpose
using_interpreter = core.using_interpreter
serial_fanout = order.serial_fanout
active_interpreter = core.active_interpreter
active_tags = core.active_tags
is_var = core.is_var
aval_if_var = core.aval_if_var
active_client = lm.active_client

# ==================================================================================================
# PRIMITIVE KEYS
# ==================================================================================================

format_p = string.format_p
concat_p = string.concat_p
match_p = string.match_p
lm_call_p = lm.lm_call_p
lm_schema_call_p = lm.lm_schema_call_p
factor_p = path.factor_p
weight_call_p = path.weight_call_p
checkpoint_p = intercept.checkpoint_p
stop_gradient_p = control.stop_gradient_p
switch_p = control.switch_p
while_loop_p = control.while_loop_p
fixpoint_p = control.fixpoint_p
fanout_p = order.fanout_p
depends_p = order.depends_p
batch_call_p = axis.batch_call_p
pushforward_call_p = ad.pushforward_call_p
pullback_call_p = ad.pullback_call_p

__all__ = [
    "AVal",
    "StrAVal",
    "IntAVal",
    "FloatAVal",
    "BoolAVal",
    "Space",
    "primal_s",
    "tangent_s",
    "cotangent_s",
    "Prim",
    "Zero",
    "Interpreter",
    "IR",
    "Eqn",
    "Var",
    "register_trace_type",
    "register_zero",
    "register_cotangent_accumulator",
    "register_non_dce",
    "register_non_memoizable",
    "register_add",
    "register_neg",
    "register_sub",
    "register_mul",
    "register_div",
    "register_pow",
    "register_matmul",
    "register_eq",
    "register_ne",
    "register_lt",
    "register_le",
    "register_gt",
    "register_ge",
    "impl_rules",
    "abstract_rules",
    "push_rules",
    "pull_fwd_rules",
    "pull_bwd_rules",
    "batch_rules",
    "zeroof",
    "tangent_zeroof",
    "cotangent_zeroof",
    "materialize",
    "is_zero",
    "batch_index",
    "batch_spec",
    "batch_transpose",
    "using_interpreter",
    "serial_fanout",
    "active_interpreter",
    "active_tags",
    "is_var",
    "aval_if_var",
    "active_client",
    "format_p",
    "concat_p",
    "match_p",
    "lm_call_p",
    "lm_schema_call_p",
    "factor_p",
    "weight_call_p",
    "checkpoint_p",
    "stop_gradient_p",
    "switch_p",
    "while_loop_p",
    "fixpoint_p",
    "fanout_p",
    "depends_p",
    "batch_call_p",
    "pushforward_call_p",
    "pullback_call_p",
]

type AValRule = Callable[[Any], AVal]
type ZeroRule = Callable[[AVal], Any]
type CotAccRule = Callable[[list[Any], AVal], Any]

# ==================================================================================================
# REGISTRATION
# ==================================================================================================


def register_trace_type[T: AValRule](type: type, aval_rule: T, /) -> T:
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
    core.trace_types.add(type)
    core.primal_s.set(type, aval_rule)
    return aval_rule


def register_zero[T: ZeroRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register the concrete zero value for an abstract value type.

    Differentiation transforms use :class:`Zero` to represent a missing or blocked
    tangent or cotangent. :func:`autoform.extend.materialize` later turns that
    symbolic zero into a concrete runtime value by looking up this rule.

    Register this for differentiable value domains whose tangents or cotangents may
    need to flow through primitives that materialize zeros, such as custom
    pushforward or pullback rules.

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
    ad.zero_rules[aval_type] = rule
    return rule


def register_cotangent_accumulator[T: CotAccRule](aval_type: type[AVal], rule: T, /) -> T:
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
    ad.cot_acc_rules[aval_type] = rule
    return rule


def register_non_dce[T: Prim](prim: T, /) -> T:
    """Register a primitive as preserved during dead-code elimination.

    Marks an extension primitive as semantically relevant even when its output is unused,
    such as a scoring, logging, or collection boundary.

    Args:
        prim: Primitive preserved by :func:`autoform.dce`.

    Returns:
        The registered primitive.
    """
    rules = dead.non_dce_primitives
    assert prim not in rules, f"Primitive {prim} is already registered as non-DCE."
    rules.add(prim)
    return prim


def register_non_memoizable[T: Prim](prim: T, /) -> T:
    """Register a primitive as excluded from :func:`autoform.memoize`.

    Marks an extension primitive as requiring repeated execution, such as stochastic sampling,
    scoring, logging, or calls that observe runtime state.

    Args:
        prim: Primitive excluded from memoization.

    Returns:
        The registered primitive.
    """
    rules = memo.non_memoizable_primitives
    assert prim not in rules, f"Primitive {prim} is already registered as non-memoizable."
    rules.add(prim)
    return prim


# ==================================================================================================
# OPERATOR REGISTRATION
# ==================================================================================================


def register_neg[T: core.TraceUnaryRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for unary ``-`` on traced values with this aval."""
    rules = core.trace_neg_rules
    assert aval_type not in rules, f"Negation for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_add[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
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
    rules = core.trace_add_rules
    assert aval_type not in rules, f"Addition for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_sub[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
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
    rules = core.trace_sub_rules
    assert aval_type not in rules, f"Subtraction for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_mul[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
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
    rules = core.trace_mul_rules
    assert aval_type not in rules, f"Multiplication for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_div[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``/`` on traced values with this aval.

    This treats true division as staged syntax while tracing.

    Args:
        aval_type: Abstract value type of the traced operand that dispatches.
        rule: Function called as ``rule(left, right)`` for ``left / right`` and
            ``rule(right, left)`` for reflected ``right / left``.

    Returns:
        The registered rule, so the helper can be used as a decorator.
    """
    rules = core.trace_truediv_rules
    assert aval_type not in rules, f"True division for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_pow[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``**`` on traced values with this aval."""
    rules = core.trace_pow_rules
    assert aval_type not in rules, f"Power for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_matmul[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
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
    rules = core.trace_matmul_rules
    assert aval_type not in rules, f"Matrix multiplication for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_eq[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``==`` on traced values with this aval.

    Python equality on a traced value cannot be evaluated concretely during
    tracing. Register this when equality should become a staged primitive.

    Args:
        aval_type: Abstract value type of the left traced operand.
        rule: Function called as ``rule(left, right)`` for ``left == right``.

    Returns:
        The registered rule, so the helper can be used as a decorator.
    """
    rules = core.trace_eq_rules
    assert aval_type not in rules, f"Equality for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_ne[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``!=`` on traced values with this aval."""
    rules = core.trace_ne_rules
    assert aval_type not in rules, f"Comparison for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_lt[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``<`` on traced values with this aval."""
    rules = core.trace_lt_rules
    assert aval_type not in rules, f"Comparison for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_le[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``<=`` on traced values with this aval."""
    rules = core.trace_le_rules
    assert aval_type not in rules, f"Comparison for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_gt[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``>`` on traced values with this aval."""
    rules = core.trace_gt_rules
    assert aval_type not in rules, f"Comparison for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule


def register_ge[T: core.TraceRule](aval_type: type[AVal], rule: T, /) -> T:
    """Register tracing dispatch for ``>=`` on traced values with this aval."""
    rules = core.trace_ge_rules
    assert aval_type not in rules, f"Comparison for {aval_type} is already registered."
    rules[aval_type] = rule
    return rule
