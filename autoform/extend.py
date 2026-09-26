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
import autoform.numeric as numeric
import autoform.order as order
import autoform.path as path
import autoform.string as string
import autoform.utils as utils

# ==================================================================================================
# TYPES
# ==================================================================================================

AVal = core.AVal
StrAVal = string.StrAVal
IntAVal = numeric.IntAVal
FloatAVal = numeric.FloatAVal
BoolAVal = numeric.BoolAVal
Space = core.Space
avalof = core.avalof
primal_s = core.primal_s
tangent_s = core.tangent_s
cotangent_s = core.cotangent_s
Prim = core.Prim
Dunder = core.Dunder
Zero = core.Zero
Interpreter = core.Interpreter
Box = core.Box
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

materialize_zeros = core.materialize_zeros
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

concat_p = string.concat_p
match_p = string.match_p
neg_p = numeric.neg_p
add_p = numeric.add_p
sub_p = numeric.sub_p
mul_p = numeric.mul_p
div_p = numeric.div_p
eq_p = numeric.eq_p
ne_p = numeric.ne_p
lt_p = numeric.lt_p
le_p = numeric.le_p
gt_p = numeric.gt_p
ge_p = numeric.ge_p
complete_p = lm.complete_p
generate_p = lm.generate_p
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
    "avalof",
    "primal_s",
    "tangent_s",
    "cotangent_s",
    "Prim",
    "Dunder",
    "Zero",
    "Interpreter",
    "Box",
    "IR",
    "Eqn",
    "Var",
    "register_trace_type",
    "register_non_dce",
    "register_non_memoizable",
    "register_dunder",
    "impl_rules",
    "abstract_rules",
    "push_rules",
    "pull_fwd_rules",
    "pull_bwd_rules",
    "batch_rules",
    "materialize_zeros",
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
    "concat_p",
    "match_p",
    "neg_p",
    "add_p",
    "sub_p",
    "mul_p",
    "div_p",
    "eq_p",
    "ne_p",
    "lt_p",
    "le_p",
    "gt_p",
    "ge_p",
    "complete_p",
    "generate_p",
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

# ==================================================================================================
# REGISTRATION
# ==================================================================================================


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
        >>> class TokenAVal(afe.AVal): ...
        >>> @ft.partial(afe.register_trace_type, Token)
        ... def token_aval(value):
        ...     return TokenAVal()
    """
    core.aval_types[type] = aval_rule
    core.trace_types.add(type)
    return aval_rule


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
    assert replace or key not in core.dunder_rules, f"Dunder rule is already defined"
    core.dunder_rules[key] = rule
    return rule
