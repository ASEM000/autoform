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

from autoform.src.ad import p_pullback_bwd_call as p_pullback_bwd_call
from autoform.src.ad import p_pullback_fwd_call as p_pullback_fwd_call
from autoform.src.ad import p_pushforward_call as p_pushforward_call
from autoform.src.axis import ABatch as ABatch
from autoform.src.axis import p_batch_call as p_batch_call
from autoform.src.control import p_fixpoint as p_fixpoint
from autoform.src.control import p_stop_gradient as p_stop_gradient
from autoform.src.control import p_switch as p_switch
from autoform.src.control import p_while_loop as p_while_loop
from autoform.src.core import AVal as AVal
from autoform.src.core import Interp as Interp
from autoform.src.core import Prim as Prim
from autoform.src.core import Rule as Rule
from autoform.src.core import Space as Space
from autoform.src.core import Zero as Zero
from autoform.src.core import active_interp as active_interp
from autoform.src.core import avalof as avalof
from autoform.src.core import instantiate as instantiate
from autoform.src.core import r_abatch as r_abatch
from autoform.src.core import r_abstract as r_abstract
from autoform.src.core import r_aimpl as r_aimpl
from autoform.src.core import r_apull_bwd as r_apull_bwd
from autoform.src.core import r_apull_fwd as r_apull_fwd
from autoform.src.core import r_apush as r_apush
from autoform.src.core import r_batch as r_batch
from autoform.src.core import r_impl as r_impl
from autoform.src.core import r_pull_bwd as r_pull_bwd
from autoform.src.core import r_pull_fwd as r_pull_fwd
from autoform.src.core import r_push as r_push
from autoform.src.core import s_cotangent as s_cotangent
from autoform.src.core import s_primal as s_primal
from autoform.src.core import s_tangent as s_tangent
from autoform.src.core import using_interp as using_interp
from autoform.src.dead import register_dce as register_dce
from autoform.src.dead import register_non_dce as register_non_dce
from autoform.src.intercept import p_checkpoint as p_checkpoint
from autoform.src.lm import active_client as active_client
from autoform.src.lm import p_fill as p_fill
from autoform.src.memo import register_non_memoizable as register_non_memoizable
from autoform.src.numeric import ABool as ABool
from autoform.src.numeric import AFloat as AFloat
from autoform.src.numeric import AInt as AInt
from autoform.src.numeric import p_add as p_add
from autoform.src.numeric import p_div as p_div
from autoform.src.numeric import p_eq as p_eq
from autoform.src.numeric import p_ge as p_ge
from autoform.src.numeric import p_gt as p_gt
from autoform.src.numeric import p_le as p_le
from autoform.src.numeric import p_lt as p_lt
from autoform.src.numeric import p_mul as p_mul
from autoform.src.numeric import p_ne as p_ne
from autoform.src.numeric import p_neg as p_neg
from autoform.src.numeric import p_sub as p_sub
from autoform.src.order import p_depends as p_depends
from autoform.src.order import p_fanout as p_fanout
from autoform.src.order import serial_fanout as serial_fanout
from autoform.src.path import p_factor as p_factor
from autoform.src.path import p_weight_call as p_weight_call
from autoform.src.stage import IR as IR
from autoform.src.stage import AOpaque as AOpaque
from autoform.src.stage import Dunder as Dunder
from autoform.src.stage import Eqn as Eqn
from autoform.src.stage import Opaque as Opaque
from autoform.src.stage import Var as Var
from autoform.src.stage import active_tags as active_tags
from autoform.src.stage import aval_if_var as aval_if_var
from autoform.src.stage import is_var as is_var
from autoform.src.stage import register_dunder as register_dunder
from autoform.src.stage import register_trace_type as register_trace_type
from autoform.src.string import AStr as AStr
from autoform.src.string import p_match as p_match
from autoform.src.string import p_render as p_render
from autoform.src.utils import batch_index as batch_index
from autoform.src.utils import batch_spec as batch_spec
from autoform.src.utils import batch_transpose as batch_transpose
