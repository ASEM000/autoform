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

"""encode/decode json"""

from __future__ import annotations

import functools as ft
import json as jsonlib
from collections.abc import Iterable
from typing import Any

from optree import PyTreeSpec

import autoform.core as core
import autoform.stage as stage
import autoform.utils as utils

__all__ = ["Json", "JsonAVal", "encode", "decode"]

zip = utils.strict_zip
type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]


class JsonAVal(core.AVal):
    __slots__ = ["spec", "avals"]

    def __init__(self, spec: PyTreeSpec, avals: Iterable[core.AVal]):
        self.spec = spec
        self.avals = tuple(avals)
        assert spec.num_leaves == len(self.avals)
        assert all(isinstance(aval, core.AVal) for aval in self.avals)

    def __repr__(self):
        return f"JsonAVal({self.spec!r}, {self.avals!r})"

    def __eq__(self, other):
        return type(self) is type(other) and (self.spec, self.avals) == (other.spec, other.avals)

    def __hash__(self):
        return hash((type(self), self.spec, self.avals))

    def zero(self):
        return encode(self.spec.unflatten(aval.zero() for aval in self.avals))

    def accumulate(self, cotangents):
        import autoform.ad as ad

        if any(core.avalof(value) != self for value in cotangents):
            raise TypeError("JSON cotangents must have matching specs and leaf types")

        def accumulate_leaf(*values):
            return ad.cot_acc(list(values))

        values = [decode(value) for value in cotangents]
        return encode(utils.tree.map(accumulate_leaf, *values))


class Json:
    __slots__ = ["text", "aval"]

    def __init__(self, text: str, aval: JsonAVal):
        assert isinstance(text, str), f"Expected JSON text, got {text!r}"
        assert isinstance(aval, JsonAVal), f"Expected JsonAVal, got {aval!r}"
        self.text = text
        self.aval = aval

    def __repr__(self):
        return f"Json({self.text!r}, {self.aval!r})"

    def __eq__(self, other):
        return type(self) is type(other) and (self.text, self.aval) == (other.text, other.aval)

    def __hash__(self):
        return hash((type(self), self.text, self.aval))


def map_json_aval(space: core.Space, aval: JsonAVal) -> JsonAVal:
    return JsonAVal(aval.spec, map(space.map, aval.avals))


core.aval_types[Json] = lambda value: value.aval
stage.trace_types.add(Json)
core.primal_s.set(JsonAVal, ft.partial(map_json_aval, core.primal_s))
core.tangent_s.set(JsonAVal, ft.partial(map_json_aval, core.tangent_s))
core.cotangent_s.set(JsonAVal, ft.partial(map_json_aval, core.cotangent_s))


def json_property_names(entries: Iterable[Any]) -> tuple[str, ...]:
    names = []
    used = set()
    for entry in entries:
        name = str(entry)
        while name in used:
            name += "_"
        names.append(name)
        used.add(name)
    return tuple(names)


def json_value(tree: Tree) -> Any:
    if utils.tree.is_leaf(tree):
        return tree
    children, spec = utils.tree.flatten(tree, is_leaf=lambda x: id(x) != id(tree))
    return {
        name: json_value(child)
        for name, child in zip(json_property_names(spec.entries()), children)
        if utils.tree.leaves(child)
    }


def parse_value(aval_tree: Tree, value: Any) -> Tree:
    if utils.tree.is_leaf(aval_tree):
        if core.avalof(value) != aval_tree:
            raise TypeError(f"Expected {aval_tree!r}, got {value!r}")
        return value
    children, spec = utils.tree.flatten(aval_tree, is_leaf=lambda x: id(x) != id(aval_tree))
    names = json_property_names(spec.entries())
    present = [bool(utils.tree.leaves(child)) for child in children]
    expected = {name for name, keep in zip(names, present) if keep}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Expected JSON object with keys {expected!r}, got {value!r}")
    return spec.unflatten(
        parse_value(child, value[name]) if keep else child
        for name, child, keep in zip(names, children, present)
    )


# ==================================================================================================
# ENCODE AND DECODE
# ==================================================================================================

encode_p = core.Prim("json_encode")
decode_p = core.Prim("json_decode")


def encode(tree: Tree, /) -> Json:
    """Encode tree

    Example:
        >>> import autoform as af
        >>> value = af.json.encode({"score": 0.5, "text": "hello"})
        >>> value.text
        '{"score": 0.5, "text": "hello"}'
        >>> af.json.decode(value)
        {'score': 0.5, 'text': 'hello'}
    """
    return encode_p.bind(tree)


def decode(value: Json, /) -> Tree:
    """Decode a typed JSON value."""
    return decode_p.bind(value)


def abstract_encode(in_tree: Tree, /) -> JsonAVal:
    flat, spec = utils.tree.flatten(in_tree)
    avals = [x if isinstance(x, core.AVal) else core.avalof(x) for x in flat]
    return JsonAVal(spec, avals)


def impl_encode(in_tree: Tree, /) -> Json:
    aval = abstract_encode(in_tree)
    flat = utils.tree.leaves(in_tree)
    if flat and all(isinstance(x, core.Zero) for x in flat):
        return core.Zero(aval)
    value = json_value(core.materialize_zeros(in_tree))
    return Json(jsonlib.dumps(value, allow_nan=False), aval)


def abstract_decode(value: Any, /) -> Tree:
    aval = value if isinstance(value, core.AVal) else core.avalof(value)
    if not isinstance(aval, JsonAVal):
        raise TypeError(f"Expected JsonAVal, got {aval!r}")
    return aval.spec.unflatten(aval.avals)


def impl_decode(value: Json, /) -> Tree:
    aval_tree = abstract_decode(value)
    if isinstance(value, core.Zero):
        return utils.tree.map(core.Zero, aval_tree)

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return parse_value(aval_tree, jsonlib.loads(value.text, parse_constant=reject_constant))


def pushforward_encode(in_tree: TreePair, /) -> TreePair:
    p_in, t_in = in_tree
    if utils.tree.structure(p_in) != utils.tree.structure(t_in):
        raise ValueError("Primal and tangent inputs must have identical pytree specs")
    return encode(p_in), encode(t_in)


def pushforward_decode(in_tree: TreePair, /) -> TreePair:
    p_in, t_in = in_tree
    return decode(p_in), decode(t_in)


def pullback_fwd_codec(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    out = prim.bind(in_tree)
    return out, None


def pullback_bwd_encode(in_tree: TreePair, /) -> Tree:
    _, out_cotangent = in_tree
    return decode(out_cotangent)


def pullback_bwd_decode(in_tree: TreePair, /) -> Json:
    _, out_cotangent = in_tree
    return encode(out_cotangent)


def batch_codec(prim: core.Prim, in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, in_values = in_tree
    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        out = prim.bind(in_values)
        return out, utils.tree.map(lambda _: False, out)
    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    results = [prim.bind(unbatch(i)) for i in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, results[0])
    out = utils.batch_transpose(batch_size, out_batched, spec.unflatten(results))
    return out, out_batched


core.impl_rules.set(encode_p, impl_encode)
core.aimpl_rules.set(encode_p, utils.asyncify(impl_encode))
core.abstract_rules.set(encode_p, abstract_encode)
core.batch_rules.set(encode_p, ft.partial(batch_codec, encode_p))
core.abatch_rules.set(encode_p, utils.asyncify(ft.partial(batch_codec, encode_p)))
core.push_rules.set(encode_p, pushforward_encode)
core.apush_rules.set(encode_p, utils.asyncify(pushforward_encode))
core.pull_fwd_rules.set(encode_p, ft.partial(pullback_fwd_codec, encode_p))
core.apull_fwd_rules.set(encode_p, utils.asyncify(ft.partial(pullback_fwd_codec, encode_p)))
core.pull_bwd_rules.set(encode_p, pullback_bwd_encode)
core.apull_bwd_rules.set(encode_p, utils.asyncify(pullback_bwd_encode))

core.impl_rules.set(decode_p, impl_decode)
core.aimpl_rules.set(decode_p, utils.asyncify(impl_decode))
core.abstract_rules.set(decode_p, abstract_decode)
core.batch_rules.set(decode_p, ft.partial(batch_codec, decode_p))
core.abatch_rules.set(decode_p, utils.asyncify(ft.partial(batch_codec, decode_p)))
core.push_rules.set(decode_p, pushforward_decode)
core.apush_rules.set(decode_p, utils.asyncify(pushforward_decode))
core.pull_fwd_rules.set(decode_p, ft.partial(pullback_fwd_codec, decode_p))
core.apull_fwd_rules.set(decode_p, utils.asyncify(ft.partial(pullback_fwd_codec, decode_p)))
core.pull_bwd_rules.set(decode_p, pullback_bwd_decode)
core.apull_bwd_rules.set(decode_p, utils.asyncify(pullback_bwd_decode))
