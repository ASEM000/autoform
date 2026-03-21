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

"""Effects"""

# NOTE(asem): effects provide a way to intercept primitive evaluation
# with custom behavior, without defining new primitives or rules. the handler
# intercepts during execution, not during transformations (pushforward,
# pullback, batch) - those still use standard rules.
#
# a handler communicates with the interpreter using a generator pattern:
#   yield in_tree  -> invoke the primitive, receive result
#   return result  -> return final value to caller
#
# this enables: logging, caching, multi-shot continuations value injection, and other
# runtime behaviors hard to achieve with standard rules.

from __future__ import annotations

import functools as ft

from autoform.core import (
    Primitive,
    PrimitiveTag,
    batch_rules,
    abstract_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import asyncify, batch_index, batch_spec, batch_transpose

# ==================================================================================================
# EFFECT
# ==================================================================================================


class EffectTag(PrimitiveTag): ...


effect_p = Primitive("effect", tag={EffectTag})


def effect(x, /, **p):
    return effect_p.bind(x, **p)


def impl_effect(x, /, **_):
    return x


def abstract_effect(x, /, **_):
    return x


def push_effect(in_tree, /, **params):
    primal, tangent = in_tree
    return effect_p.bind(primal, **params), effect_p.bind(tangent, **params)


def pull_fwd_effect(x, /, **params):
    return effect_p.bind(x, **params), None


def pull_bwd_effect(in_tree, /, **params):
    _, cotangent = in_tree
    return effect_p.bind(cotangent, **params)


def batch_effect(in_tree, /, **params):
    batch_size, in_batched, x = in_tree

    if batch_spec(x, in_batched) is None:
        return effect_p.bind(x, **params), False

    unbatch = ft.partial(batch_index, x, in_batched)
    out_bi = [effect_p.bind(unbatch(b), **params) for b in range(batch_size)]
    out_batched = in_batched
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


async def abatch_effect(in_tree, /, **params):
    batch_size, in_batched, x = in_tree

    if batch_spec(x, in_batched) is None:
        return await effect_p.abind(x, **params), False

    unbatch = ft.partial(batch_index, x, in_batched)
    out_bi = [await effect_p.abind(unbatch(b), **params) for b in range(batch_size)]
    out_batched = in_batched
    out_ib = batch_transpose(batch_size, out_batched, out_bi)
    return out_ib, out_batched


impl_rules.set(effect_p, impl_effect)
impl_rules.aset(effect_p, asyncify(impl_effect))
abstract_rules.set(effect_p, abstract_effect)
push_rules.set(effect_p, push_effect)
push_rules.aset(effect_p, asyncify(push_effect))
pull_fwd_rules.set(effect_p, pull_fwd_effect)
pull_fwd_rules.aset(effect_p, asyncify(pull_fwd_effect))
pull_bwd_rules.set(effect_p, pull_bwd_effect)
pull_bwd_rules.aset(effect_p, asyncify(pull_bwd_effect))
batch_rules.set(effect_p, batch_effect)
batch_rules.aset(effect_p, abatch_effect)
