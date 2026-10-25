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

from typing import Any

import autoform.src.core as core
import autoform.src.tree as tree
import autoform.src.utils as utils

type Tree[T] = Any


# ==================================================================================================
# TYPE CHECK
# ==================================================================================================


p_typecheck = core.Prim("typecheck")


def typecheck(value, aval, /):
    return p_typecheck.bind(value, aval=aval)


def impl_check(value: Tree, /, *, aval: core.AVal) -> Tree:
    if aval.join(value) != aval:
        raise TypeError(f"Expected {aval!r}, got {value!r}")
    return value


def pushforward_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    p, t = in_tree
    p = p_typecheck.bind(p, aval=core.s_primal.map(aval))
    t = p_typecheck.bind(t, aval=core.s_tangent.map(aval))
    return p, t


async def apushforward_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    p, t = in_tree
    p = await p_typecheck.abind(p, aval=core.s_primal.map(aval))
    t = await p_typecheck.abind(t, aval=core.s_tangent.map(aval))
    return p, t


def pullback_fwd_check(value: Tree, /, *, aval: core.AVal) -> tuple[Tree, None]:
    return p_typecheck.bind(value, aval=core.s_primal.map(aval)), None


async def apullback_fwd_check(value: Tree, /, *, aval: core.AVal) -> tuple[Tree, None]:
    return await p_typecheck.abind(value, aval=core.s_primal.map(aval)), None


def pullback_bwd_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    _, c = in_tree
    return p_typecheck.bind(c, aval=core.s_cotangent.map(aval))


async def apullback_bwd_check(in_tree: Tree, /, *, aval: core.AVal) -> Tree:
    _, c = in_tree
    return await p_typecheck.abind(c, aval=core.s_cotangent.map(aval))


def batch_check(in_tree: Tree, /, *, aval: core.AVal) -> tuple[Tree, Tree]:
    spec, in_axes, value = in_tree

    def validate(use, x):
        if use:
            actual = tree.structure(x, is_leaf=lambda node: node is not x)
            if actual != spec.tree_spec or actual.is_leaf():
                raise TypeError(f"Expected a batch container matching {spec}, got {x!r}")

    tree.map(validate, in_axes, value)
    if not spec.size:
        return value, in_axes
    items = spec.unstack(in_axes, value)
    checked = [p_typecheck.bind(x, aval=aval) for x in items]
    out = spec.stack(in_axes, checked)
    get = spec.tree_spec.accessors()[0]
    return tree.map(lambda b, x: x if b else get(x), in_axes, out), in_axes


async def abatch_check(in_tree: Tree, /, *, aval: core.AVal) -> tuple[Tree, Tree]:
    spec, in_axes, value = in_tree

    def validate(use, x):
        if use:
            actual = tree.structure(x, is_leaf=lambda node: node is not x)
            if actual != spec.tree_spec or actual.is_leaf():
                raise TypeError(f"Expected a batch container matching {spec}, got {x!r}")

    tree.map(validate, in_axes, value)
    if not spec.size:
        return value, in_axes
    items = spec.unstack(in_axes, value)
    checked = [await p_typecheck.abind(x, aval=aval) for x in items]
    out = spec.stack(in_axes, checked)
    get = spec.tree_spec.accessors()[0]
    return tree.map(lambda b, x: x if b else get(x), in_axes, out), in_axes


core.r_impl.set(p_typecheck, impl_check)
core.r_aimpl.set(p_typecheck, utils.asyncify(impl_check))
core.r_abstract.set(p_typecheck, impl_check)
core.r_push.set(p_typecheck, pushforward_check)
core.r_apush.set(p_typecheck, apushforward_check)
core.r_pull_fwd.set(p_typecheck, pullback_fwd_check)
core.r_apull_fwd.set(p_typecheck, apullback_fwd_check)
core.r_pull_bwd.set(p_typecheck, pullback_bwd_check)
core.r_apull_bwd.set(p_typecheck, apullback_bwd_check)
core.r_batch.set(p_typecheck, batch_check)
core.r_abatch.set(p_typecheck, abatch_check)
