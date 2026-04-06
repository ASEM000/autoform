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

"""Autoform: Composable function transformations for LLM programs."""

from autoform.ad import pullback, pushforward
from autoform.batch import batch
from autoform.checkpoint import checkpoint, collect, inject
from autoform.control import stop_gradient, switch, while_loop
from autoform.core import acall, call, trace
from autoform.dce import dce
from autoform.inference import factor, weight
from autoform.lm import lm_call, struct_lm_call, using_router
from autoform.memoize import memoize
from autoform.scheduling import depends, gather, sched
from autoform.string import concat, format, match
from autoform.utils import PYTREE_NAMESPACE, Struct

__all__ = [
    # core
    "trace",
    # execution
    "call",
    "acall",
    "using_router",
    # transformations
    "pushforward",
    "pullback",
    "batch",
    "collect",
    "inject",
    "dce",
    "sched",
    "memoize",
    "weight",
    # primitives
    "format",
    "concat",
    "match",
    "lm_call",
    "struct_lm_call",
    "stop_gradient",
    "checkpoint",
    "switch",
    "while_loop",
    "gather",
    "depends",
    "factor",
    # types
    "Struct",
    "PYTREE_NAMESPACE",
]
