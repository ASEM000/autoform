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

from autoform.src.extend import IR as IR
from autoform.src.extend import AVal as AVal
from autoform.src.extend import BoolAVal as BoolAVal
from autoform.src.extend import Dunder as Dunder
from autoform.src.extend import Eqn as Eqn
from autoform.src.extend import FloatAVal as FloatAVal
from autoform.src.extend import IntAVal as IntAVal
from autoform.src.extend import Interpreter as Interpreter
from autoform.src.extend import Prim as Prim
from autoform.src.extend import Rule as Rule
from autoform.src.extend import Space as Space
from autoform.src.extend import StrAVal as StrAVal
from autoform.src.extend import Var as Var
from autoform.src.extend import Zero as Zero
from autoform.src.extend import active_client as active_client
from autoform.src.extend import active_interpreter as active_interpreter
from autoform.src.extend import active_tags as active_tags
from autoform.src.extend import add_p as add_p
from autoform.src.extend import aval_if_var as aval_if_var
from autoform.src.extend import avalof as avalof
from autoform.src.extend import batch_call_p as batch_call_p
from autoform.src.extend import batch_index as batch_index
from autoform.src.extend import batch_spec as batch_spec
from autoform.src.extend import batch_transpose as batch_transpose
from autoform.src.extend import checkpoint_p as checkpoint_p
from autoform.src.extend import concat_p as concat_p
from autoform.src.extend import cotangent_s as cotangent_s
from autoform.src.extend import depends_p as depends_p
from autoform.src.extend import div_p as div_p
from autoform.src.extend import eq_p as eq_p
from autoform.src.extend import factor_p as factor_p
from autoform.src.extend import fanout_p as fanout_p
from autoform.src.extend import fill_p as fill_p
from autoform.src.extend import fixpoint_p as fixpoint_p
from autoform.src.extend import ge_p as ge_p
from autoform.src.extend import gt_p as gt_p
from autoform.src.extend import is_var as is_var
from autoform.src.extend import le_p as le_p
from autoform.src.extend import lt_p as lt_p
from autoform.src.extend import match_p as match_p
from autoform.src.extend import materialize_zeros as materialize_zeros
from autoform.src.extend import mul_p as mul_p
from autoform.src.extend import ne_p as ne_p
from autoform.src.extend import neg_p as neg_p
from autoform.src.extend import primal_s as primal_s
from autoform.src.extend import pullback_call_p as pullback_call_p
from autoform.src.extend import pushforward_call_p as pushforward_call_p
from autoform.src.extend import register_abatch as register_abatch
from autoform.src.extend import register_abstract as register_abstract
from autoform.src.extend import register_aimpl as register_aimpl
from autoform.src.extend import register_apullback_bwd as register_apullback_bwd
from autoform.src.extend import register_apullback_fwd as register_apullback_fwd
from autoform.src.extend import register_apushforward as register_apushforward
from autoform.src.extend import register_batch as register_batch
from autoform.src.extend import register_dce as register_dce
from autoform.src.extend import register_dunder as register_dunder
from autoform.src.extend import register_impl as register_impl
from autoform.src.extend import register_non_dce as register_non_dce
from autoform.src.extend import register_non_memoizable as register_non_memoizable
from autoform.src.extend import register_pullback_bwd as register_pullback_bwd
from autoform.src.extend import register_pullback_fwd as register_pullback_fwd
from autoform.src.extend import register_pushforward as register_pushforward
from autoform.src.extend import register_trace_type as register_trace_type
from autoform.src.extend import serial_fanout as serial_fanout
from autoform.src.extend import stop_gradient_p as stop_gradient_p
from autoform.src.extend import sub_p as sub_p
from autoform.src.extend import switch_p as switch_p
from autoform.src.extend import tangent_s as tangent_s
from autoform.src.extend import using_interpreter as using_interpreter
from autoform.src.extend import weight_call_p as weight_call_p
from autoform.src.extend import while_loop_p as while_loop_p
