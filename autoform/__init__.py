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

import autoform.ad as ad
import autoform.batch as batch
import autoform.checkpoint as checkpoint
import autoform.control as control
import autoform.core as core
import autoform.custom as custom
import autoform.dce as dce
import autoform.lm as lm
import autoform.memoize as memoize
import autoform.scheduling as scheduling
import autoform.schemas as schemas
import autoform.string as string
import autoform.utils as utils

trace = core.trace
fold = core.fold
tag = core.tag

pushforward = ad.pushforward
pullback = ad.pullback
batch = batch.batch
custom = custom.custom
collect = checkpoint.collect
inject = checkpoint.inject
checkpoint = checkpoint.checkpoint
dce = dce.dce
sched = scheduling.sched
memoize = memoize.memoize

format = string.format
concat = string.concat
match = string.match
lm_call = lm.lm_call
lm_schema_call = lm.lm_schema_call
lm_client = lm.lm_client
stop_gradient = control.stop_gradient
switch = control.switch
while_loop = control.while_loop
depends = scheduling.depends

Bool = schemas.Bool
Doc = schemas.Doc
Enum = schemas.Enum
Float = schemas.Float
Int = schemas.Int
Str = schemas.Str

PYTREE_NAMESPACE = utils.PYTREE_NAMESPACE

__all__ = [
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
    "depends",
    # schemas
    "Bool",
    "Doc",
    "Enum",
    "Float",
    "Int",
    "Str",
    # types
    "PYTREE_NAMESPACE",
]
