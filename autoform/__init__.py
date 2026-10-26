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

from autoform import extend as extend
from autoform import lm as lm
from autoform import numeric as numeric
from autoform import text as text
from autoform import tree as tree
from autoform.src.ad import fwd_bwd as fwd_bwd
from autoform.src.ad import pullback as pullback
from autoform.src.ad import pushforward as pushforward
from autoform.src.axis import batch as batch
from autoform.src.boundary import custom as custom
from autoform.src.control import fixpoint as fixpoint
from autoform.src.control import stop_gradient as stop_gradient
from autoform.src.control import switch as switch
from autoform.src.control import while_loop as while_loop
from autoform.src.dead import dce as dce
from autoform.src.intercept import checkpoint as checkpoint
from autoform.src.intercept import collect as collect
from autoform.src.intercept import inject as inject
from autoform.src.memo import memoize as memoize
from autoform.src.order import depends as depends
from autoform.src.order import sched as sched
from autoform.src.path import factor as factor
from autoform.src.path import weight as weight
from autoform.src.stage import fold as fold
from autoform.src.stage import tag as tag
from autoform.src.stage import trace as trace
from autoform.src.tree import NAMESPACE as NAMESPACE
from autoform.src.untyped import untyped as untyped

__version__ = "0.3.0"
