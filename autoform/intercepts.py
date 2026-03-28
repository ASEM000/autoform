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

"""Intercepts"""

# NOTE(asem): intercepts provide a way to customize primitive evaluation
# with custom behavior, without defining new primitives or rules. the interceptor
# intercepts during execution, not during transformations (pushforward,
# pullback, batch) - those still use standard rules.
#
# an interceptor communicates with the interpreter using a generator pattern:
#   yield in_tree  -> invoke the primitive, receive result
#   return result  -> return final value to caller
#
# this enables: logging, caching, multi-shot continuations value injection, and other
# runtime behaviors hard to achieve with standard rules.

from __future__ import annotations

import functools as ft

from autoform.core import (
    Prim,
    PrimTag,
    abstract_rules,
    batch_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import asyncify, batch_index, batch_spec, batch_transpose

# ==================================================================================================
# INTERCEPT
# ==================================================================================================


class InterceptTag(PrimTag): ...


intercept_p = Prim("intercept", tag={InterceptTag})


def intercept(x, /, **p):
    return intercept_p.bind(x, **p)


def impl_intercept(x, /, **_):
    return x


def abstract_intercept(x, /, **_):
    return x


def push_intercept(in_tree, /, **params):
    primal, tangent = in_tree
    return intercept_p.bind(primal, **params), intercept_p.bind(tangent, **params)


def pull_fwd_intercept(x, /, **params):
    return intercept_p.bind(x, **params), None


def pull_bwd_intercept(in_tree, /, **params):
    _, cotangent = in_tree
    return intercept_p.bind(cotangent, **params)


def batch_intercept(in_tree, /, **params):
    batch_size, in_batched, x = in_tree

    if batch_spec(x, in_batched) is None:
        return intercept_p.bind(x, **params), False

    unbatch = ft.partial(batch_index, x, in_batched)
    out_bi = [intercept_p.bind(unbatch(b), **params) for b in range(batch_size)]
    out_batched = in_batched
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_intercept(in_tree, /, **params):
    batch_size, in_batched, x = in_tree

    if batch_spec(x, in_batched) is None:
        return await intercept_p.abind(x, **params), False

    unbatch = ft.partial(batch_index, x, in_batched)
    out_bi = [await intercept_p.abind(unbatch(b), **params) for b in range(batch_size)]
    out_batched = in_batched
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


impl_rules.set(intercept_p, impl_intercept)
impl_rules.aset(intercept_p, asyncify(impl_intercept))
abstract_rules.set(intercept_p, abstract_intercept)
push_rules.set(intercept_p, push_intercept)
push_rules.aset(intercept_p, asyncify(push_intercept))
pull_fwd_rules.set(intercept_p, pull_fwd_intercept)
pull_fwd_rules.aset(intercept_p, asyncify(pull_fwd_intercept))
pull_bwd_rules.set(intercept_p, pull_bwd_intercept)
pull_bwd_rules.aset(intercept_p, asyncify(pull_bwd_intercept))
batch_rules.set(intercept_p, batch_intercept)
batch_rules.aset(intercept_p, abatch_intercept)
