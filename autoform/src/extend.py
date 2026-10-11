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

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.control as control
import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.intercept as intercept
import autoform.src.lm as lm
import autoform.src.memo as memo
import autoform.src.numeric as numeric
import autoform.src.order as order
import autoform.src.path as path
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils

# ==================================================================================================
# TYPES
# ==================================================================================================

AVal = core.AVal
AValStr = string.AValStr
AValInt = numeric.AValInt
AValFloat = numeric.AValFloat
AValBool = numeric.AValBool
Space = core.Space
avalof = core.avalof
s_primal = core.s_primal
s_tangent = core.s_tangent
s_cotangent = core.s_cotangent
Prim = core.Prim
Rule = core.Rule
Dunder = stage.Dunder
Zero = core.Zero
Interpreter = core.Interpreter
IR = stage.IR
Eqn = stage.Eqn
Var = stage.Var

# ==================================================================================================
# HELPERS
# ==================================================================================================

materialize_zeros = core.materialize_zeros
batch_index = utils.batch_index
batch_spec = utils.batch_spec
batch_transpose = utils.batch_transpose
using_interpreter = core.using_interpreter
serial_fanout = order.serial_fanout
active_interpreter = core.active_interpreter
active_tags = stage.active_tags
is_var = stage.is_var
aval_if_var = stage.aval_if_var
active_client = lm.active_client

# ==================================================================================================
# PRIMITIVE KEYS
# ==================================================================================================

p_concat = string.p_concat
p_match = string.p_match
p_neg = numeric.p_neg
p_add = numeric.p_add
p_sub = numeric.p_sub
p_mul = numeric.p_mul
p_div = numeric.p_div
p_eq = numeric.p_eq
p_ne = numeric.p_ne
p_lt = numeric.p_lt
p_le = numeric.p_le
p_gt = numeric.p_gt
p_ge = numeric.p_ge
p_fill = lm.p_fill
p_factor = path.p_factor
p_weight_call = path.p_weight_call
p_checkpoint = intercept.p_checkpoint
p_stop_gradient = control.p_stop_gradient
p_switch = control.p_switch
p_while_loop = control.p_while_loop
p_fixpoint = control.p_fixpoint
p_fanout = order.p_fanout
p_depends = order.p_depends
p_batch_call = axis.p_batch_call
p_pushforward_call = ad.p_pushforward_call
p_pullback_call = ad.p_pullback_call


type AValRule = Callable[[Any], AVal]

# ==================================================================================================
# REGISTRATION
# ==================================================================================================


def register_impl[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.impl_rules.set(prim, rule)


def register_aimpl[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.aimpl_rules.set(prim, rule)


def register_abstract[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.abstract_rules.set(prim, rule)


def register_batch[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.batch_rules.set(prim, rule)


def register_abatch[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.abatch_rules.set(prim, rule)


def register_pushforward[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.push_rules.set(prim, rule)


def register_apushforward[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.apush_rules.set(prim, rule)


def register_pullback_fwd[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.pull_fwd_rules.set(prim, rule)


def register_apullback_fwd[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.apull_fwd_rules.set(prim, rule)


def register_pullback_bwd[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.pull_bwd_rules.set(prim, rule)


def register_apullback_bwd[R: Callable[..., Any]](prim: Prim, rule: R, /) -> R:
    return core.apull_bwd_rules.set(prim, rule)


def register_trace_type[T: AValRule](type: type, aval_rule: T, /) -> T:
    """Register a Python type as a traceable input type.

    :func:`autoform.trace` treats registered Python types as dynamic leaves. During
    tracing, each concrete value is passed to ``aval_rule`` and replaced by an
    :class:`AVal` that carries the abstract information needed by primitive rules.

    Args:
        type: Concrete Python type accepted as a dynamic input leaf.
        aval_rule: Function from a concrete value to its abstract value.

    Returns:
        The registered rule.

    Example:
        >>> import functools as ft
        >>> import autoform.extend as afe
        >>> class Token: ...
        >>> class AValToken(afe.AVal): ...
        >>> @ft.partial(afe.register_trace_type, Token)
        ... def token_aval(value):
        ...     return AValToken()
    """
    core.aval_types[type] = aval_rule
    stage.trace_types.add(type)
    return aval_rule


def register_dce[R: dead.DCERule](prim: Prim, rule: R, /) -> R:
    """Register a dead-code elimination rule for a primitive."""
    assert isinstance(prim, Prim)
    dead.dce_rules[prim] = rule
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
    assert isinstance(prim, Prim)
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
    assert isinstance(prim, Prim)
    rules = memo.non_memoizable_primitives
    assert prim not in rules, f"Primitive {prim} is already registered as non-memoizable."
    rules.add(prim)
    return prim


# ==================================================================================================
# DUNDER REGISTRATION
# ==================================================================================================


def register_dunder[T: Callable[..., Any]](
    dunder: Dunder,
    aval_type: type[AVal],
    rule: T,
    /,
    *,
    replace: bool = False,
) -> T:
    """Register Python :class:`autoform.extend.Dunder` behavior for a traced abstract value type.

    Args:
        dunder: :class:`autoform.extend.Dunder` being staged.
        aval_type: Abstract value type that selects the rule.
        rule: Callable implementing the trace-time dunder behavior.
        replace: Whether to replace an existing rule explicitly.

    Returns:
        The registered rule.
    """
    assert isinstance(dunder, Dunder), f"Expected Dunder, got {dunder!r}"
    assert issubclass(aval_type, AVal), f"Expected AVal type, got {aval_type!r}"
    assert callable(rule), f"Expected callable, got {rule!r}"
    assert isinstance(replace, bool), f"Expected bool for replace, got {type(replace)}"
    key = dunder, aval_type
    assert replace or key not in stage.dunder_rules, f"Dunder rule is already defined"
    stage.dunder_rules[key] = rule
    return rule
