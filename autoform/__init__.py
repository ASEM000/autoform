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

"""Autoform: Composable function transformations for text-space programs."""

# ==================================================================================================
# IMPORTS
# ==================================================================================================

import autoform.ad as ad
import autoform.axis as axis
import autoform.boundary as boundary
import autoform.control as control
import autoform.core as core
import autoform.dead as dead
import autoform.extend as extend
import autoform.intercept as intercept
import autoform.lm as lm
import autoform.memo as memo
import autoform.order as order
import autoform.path as path
import autoform.schemas as schemas
import autoform.string as string
import autoform.utils as utils

# ==================================================================================================
# METADATA
# ==================================================================================================

__version__ = "0.3.0"

# ==================================================================================================
# CORE
# ==================================================================================================

trace = core.trace
fold = core.fold
tag = core.tag

# ==================================================================================================
# TRANSFORMS
# ==================================================================================================

pushforward = ad.pushforward
pullback = ad.pullback
batch = axis.batch
custom = boundary.custom
collect = intercept.collect
inject = intercept.inject
checkpoint = intercept.checkpoint
dce = dead.dce
sched = axis.sched
memoize = memo.memoize
weighted = path.weighted

# ==================================================================================================
# PRIMITIVES
# ==================================================================================================

format = string.format
concat = string.concat
match = string.match
lm_call = lm.lm_call
lm_schema_call = lm.lm_schema_call
lm_client = lm.lm_client
stop_gradient = control.stop_gradient
switch = control.switch
while_loop = control.while_loop
fixpoint = control.fixpoint
depends = order.depends
factor = path.factor

# ==================================================================================================
# SCHEMAS
# ==================================================================================================

Bool = schemas.Bool
Doc = schemas.Doc
Enum = schemas.Enum
Float = schemas.Float
Int = schemas.Int
Str = schemas.Str

# ==================================================================================================
# TYPES
# ==================================================================================================

PYTREE_NAMESPACE = utils.PYTREE_NAMESPACE

# ==================================================================================================
# EXPORTS
# ==================================================================================================

__all__ = [
    "__version__",
    # core
    "trace",
    "fold",
    # execution
    "lm_client",
    "tag",
    # transformations
    "pushforward",
    "pullback",
    "custom",
    "batch",
    "collect",
    "inject",
    "dce",
    "sched",
    "memoize",
    "weighted",
    # primitives
    "format",
    "concat",
    "match",
    "lm_call",
    "lm_schema_call",
    "stop_gradient",
    "checkpoint",
    "switch",
    "while_loop",
    "fixpoint",
    "depends",
    "factor",
    # schemas
    "Bool",
    "Doc",
    "Enum",
    "Float",
    "Int",
    "Str",
    # types
    "PYTREE_NAMESPACE",
    # modules
    "extend",
    "path",
]
