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

"""Typing"""

from __future__ import annotations

import autoform.core as core
import autoform.utils as utils

type Tree[T] = utils.Tree[T]


# ==================================================================================================
# TYPE CHECK
# ==================================================================================================


typecheck_p = core.Prim("typecheck")


def typecheck(value, aval, /):
    return typecheck_p.bind(value, aval=aval)


def impl_check(value: Tree, /, *, aval: core.AVal) -> Tree:
    aval.check(value)
    return value


def pushforward_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    primal, tangent = in_tree
    primal = typecheck_p.bind(primal, aval=core.primal_s.map(aval))
    tangent = typecheck_p.bind(tangent, aval=core.tangent_s.map(aval))
    return primal, tangent


async def apushforward_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    primal, tangent = in_tree
    primal = await typecheck_p.abind(primal, aval=core.primal_s.map(aval))
    tangent = await typecheck_p.abind(tangent, aval=core.tangent_s.map(aval))
    return primal, tangent


def pullback_fwd_check(value: Tree, /, *, aval: core.AVal) -> tuple[Tree, None]:
    return typecheck_p.bind(value, aval=core.primal_s.map(aval)), None


async def apullback_fwd_check(value: Tree, /, *, aval: core.AVal) -> tuple[Tree, None]:
    return await typecheck_p.abind(value, aval=core.primal_s.map(aval)), None


def pullback_bwd_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    _, cotangent = in_tree
    return typecheck_p.bind(cotangent, aval=core.cotangent_s.map(aval))


async def apullback_bwd_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    _, cotangent = in_tree
    return await typecheck_p.abind(cotangent, aval=core.cotangent_s.map(aval))


def batch_check(in_tree: Tree, /, *, aval: core.AVal) -> tuple[Tree, Tree]:
    size, batched, value = in_tree
    if (spec := utils.batch_spec(value, batched)) is None:
        return typecheck_p.bind(value, aval=aval), batched
    if spec.is_leaf() or spec.num_children != size:
        raise TypeError(f"Expected a batch container with {size} items, got {value!r}")
    if not size:
        return value, batched
    checked = [
        typecheck_p.bind(utils.batch_index(value, batched, b), aval=aval) for b in range(size)
    ]
    out = utils.batch_transpose(size, batched, spec.unflatten(checked))
    return utils.tree.map(lambda b, x: x if b else utils.index(x, 0), batched, out), batched


async def abatch_check(in_tree: Tree, /, *, aval: core.AVal) -> tuple[Tree, Tree]:
    size, batched, value = in_tree
    if (spec := utils.batch_spec(value, batched)) is None:
        return await typecheck_p.abind(value, aval=aval), batched
    if spec.is_leaf() or spec.num_children != size:
        raise TypeError(f"Expected a batch container with {size} items, got {value!r}")
    if not size:
        return value, batched
    checked = [
        await typecheck_p.abind(utils.batch_index(value, batched, b), aval=aval)
        for b in range(size)
    ]
    out = utils.batch_transpose(size, batched, spec.unflatten(checked))
    return utils.tree.map(lambda b, x: x if b else utils.index(x, 0), batched, out), batched


core.impl_rules.set(typecheck_p, impl_check)
core.aimpl_rules.set(typecheck_p, utils.asyncify(impl_check))
core.abstract_rules.set(typecheck_p, impl_check)
core.push_rules.set(typecheck_p, pushforward_check)
core.apush_rules.set(typecheck_p, apushforward_check)
core.pull_fwd_rules.set(typecheck_p, pullback_fwd_check)
core.apull_fwd_rules.set(typecheck_p, apullback_fwd_check)
core.pull_bwd_rules.set(typecheck_p, pullback_bwd_check)
core.apull_bwd_rules.set(typecheck_p, apullback_bwd_check)
core.batch_rules.set(typecheck_p, batch_check)
core.abatch_rules.set(typecheck_p, abatch_check)
