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

"""LM (Language Model) primitives"""

from __future__ import annotations

import functools as ft
import json as jsonlib
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from litellm import ModelResponse, acompletion, completion

import autoform.control as control
import autoform.core as core
import autoform.json as json
import autoform.order as order
import autoform.schemas as schemas
import autoform.stage as stage
import autoform.utils as utils

__all__ = [
    "Client",
    "LiteLLMClient",
    "client",
    "fill",
    "describe",
    "parse",
]


zip = utils.strict_zip

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type ClientType = ModelResponse

describe = json.describe
parse = json.parse


@runtime_checkable
class Client(Protocol):
    def completion(self, *, messages: list[dict], model: str, **kwargs) -> ClientType: ...
    async def acompletion(self, *, messages: list[dict], model: str, **kwargs) -> ClientType: ...


class LiteLLMClient:
    __slots__ = []

    def completion(self, *, messages: list[dict], model: str, **kwargs) -> ClientType:
        return completion(messages=messages, model=model, **kwargs)

    async def acompletion(self, *, messages: list[dict], model: str, **kwargs) -> ClientType:
        return await acompletion(messages=messages, model=model, **kwargs)


active_client: ContextVar[Client] = ContextVar("active_client", default=LiteLLMClient())


@contextmanager
def client(client: Client) -> Generator[Client, None, None]:
    """Set the LM client for all lm primitives.

    The client must expose ``.completion()`` and ``.acompletion()`` matching
    LiteLLM's chat completion signature.

    Acceptable clients include the default direct LiteLLM adapter, a configured
    ``litellm.Router``, or any wrapper object that forwards those two methods
    while preserving the LiteLLM request and response shapes.

    Example:
        >>> import autoform as af
        >>> from litellm import Router  # doctest: +SKIP
        >>> client = Router(  # doctest: +SKIP
        ...     model_list=[
        ...         dict(model_name="gpt-4", litellm_params=dict(model="gpt-5.5")),
        ...     ],
        ...     max_parallel_requests=10,
        ... )
        >>> with af.lm.client(client):  # doctest: +SKIP
        ...     ir.call(inputs)
    """
    assert isinstance(client, Client), f"Expected LMClient instance, got {type(client)}"
    token = active_client.set(client)
    try:
        yield client
    finally:
        active_client.reset(token)


# ==================================================================================================
# HELPERS
# ==================================================================================================

PUSH_SYSTEM_PROMPT = "Translate an input change into the corresponding output change. "
PUSH_PROMPT = """INPUT: {input} INPUT CHANGE: {in_tangent}"""
GRAD_SYSTEM_PROMPT = "Translate output feedback into corresponding input feedback."
GRAD_PROMPT = """INPUT: {input} OUTPUT: {output} OUTPUT FEEDBACK: {out_cotangent}"""


def schema_content(value: Tree, schema: Tree) -> str:
    value = dict(values=json.json_value(value), schema=describe(schema))
    return jsonlib.dumps(value, allow_nan=False)


def json_content(value: json.Json) -> str:
    aval = core.avalof(value)
    schema = aval.spec.unflatten(map(schemas.aval_to_schema, aval.avals))
    content = dict(values=jsonlib.loads(value.text), schema=describe(schema))
    return jsonlib.dumps(content, allow_nan=False)


def schema_completion(in_tree: Tree, /, *, schema: Any) -> Any:
    messages, model = in_tree
    json_schema = describe(schema)
    if json_schema is None:
        return parse(schema, None)
    resp = active_client.get().completion(
        messages=messages,
        model=model,
        response_format=dict(
            type="json_schema",
            json_schema=dict(
                name="autoform_schema",
                strict=True,
                schema=json_schema,
            ),
        ),
    )
    return parse(schema, jsonlib.loads(resp.choices[0].message.content))


async def aschema_completion(in_tree: Tree, /, *, schema: Any) -> Any:
    messages, model = in_tree
    json_schema = describe(schema)
    if json_schema is None:
        return parse(schema, None)
    resp = await active_client.get().acompletion(
        messages=messages,
        model=model,
        response_format=dict(
            type="json_schema",
            json_schema=dict(
                name="autoform_schema",
                strict=True,
                schema=json_schema,
            ),
        ),
    )
    return parse(schema, jsonlib.loads(resp.choices[0].message.content))


def schema_abstract_tree(schema: Any) -> Tree:
    def abstract(x: Any) -> Any:
        if schemas.is_schema(x):
            return core.avalof(x)
        if not stage.is_traceable(x):
            raise TypeError(f"Static schema leaf must be traceable, got {x!r}")
        return x

    return utils.tree.map(abstract, schema, is_leaf=schemas.is_schema)


def fill_context(context: Tree, model: str, /, *, schema: Tree) -> Tree:
    schm_tree, _ = utils.partition(schemas.is_schema, schema, is_leaf=schemas.is_schema)
    encoded, holes = prepare_fill(dict(context=context, output=schm_tree))
    out = fill_p.bind((encoded, model), schema=holes)
    return merge_fill(schema, out["output"])


async def afill_context(context: Tree, model: str, /, *, schema: Tree) -> Tree:
    schm_tree, _ = utils.partition(schemas.is_schema, schema, is_leaf=schemas.is_schema)
    encoded, holes = prepare_fill(dict(context=context, output=schm_tree))
    out = await fill_p.abind((encoded, model), schema=holes)
    return merge_fill(schema, out["output"])


def feedback_schema(tree: Tree) -> Tree:
    def make_schema(path, value):
        aval = core.cotangent_s.map(core.avalof(value))
        schema = schemas.aval_to_schema(aval)
        schema.desc = f"Input cotangent at {path}, original value {value!r}."
        return schema

    return utils.tree.map_with_path(make_schema, tree)


def pullback_fwd_lm(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    context, model = in_tree
    out = prim.bind(in_tree, **params)
    residuals = (context, model, out)
    return out, residuals


async def apull_fwd_lm(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    context, model = in_tree
    out = await prim.abind(in_tree, **params)
    residuals = (context, model, out)
    return out, residuals


def batch_lm(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        result = prim.bind(in_values, **params)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    bind = ft.partial(prim.bind, **params)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, spec.unflatten(results))
    return out_ib, out_batched


async def abatch_lm(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        result = await prim.abind(in_values, **params)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    inputs = [(unbatch(b),) for b in range(batch_size)]
    in0, *_ = inputs
    ir = stage.trace(ft.partial(prim.bind, **params))(*in0)
    results = await order.fanout_p.abind(inputs, irs=[ir] * batch_size)
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, spec.unflatten(results))
    return out_ib, out_batched


# ==================================================================================================
# FILL
# ==================================================================================================

fill_p = core.Prim("fill")


def fill(tree: Tree, /, *, model: str) -> Tree:
    """Fill schema nodes in a pytree with generated values.

    Args:
        tree: A pytree containing context leaves with registered AVal schemas and static schema nodes.
        model: The model name or active client model alias to use.

    Returns:
        ``tree`` with each schema node replaced by a generated value.
    """

    def check_context(value):
        schemas.aval_to_schema(core.avalof(value))

    utils.tree.map(check_context, tree)
    if not any(map(schemas.is_schema, utils.tree.leaves(tree, is_leaf=schemas.is_schema))):
        return tree
    assert core.avalof(model) == core.avalof(""), f"Expected string model: {model!r}"
    encoded, schm_tree = prepare_fill(tree)
    out = fill_p.bind((encoded, control.stop_gradient(model)), schema=schm_tree)
    return merge_fill(tree, out)


def prepare_fill(tree: Tree) -> TreePair:
    schm_tree, lit_tree = utils.partition(schemas.is_schema, tree, is_leaf=schemas.is_schema)
    return json.encode(lit_tree), schm_tree


def merge_fill(tree: Tree, generated: Tree) -> Tree:
    def generated_field(node, value):
        return value if schemas.is_schema(node) else node

    return utils.tree.map(generated_field, tree, generated, is_leaf=schemas.is_schema)


def fill_request(in_tree: Tree, /) -> Tree:
    encoded, model = in_tree
    content = json_content(encoded)
    return [dict(role="user", content=content)], model


def impl_fill(in_tree: Tree, /, *, schema: Tree) -> Tree:
    return schema_completion(fill_request(in_tree), schema=schema)


async def aimpl_fill(in_tree: Tree, /, *, schema: Tree) -> Tree:
    return await aschema_completion(fill_request(in_tree), schema=schema)


def abstract_fill(in_tree: Tree, /, *, schema: Tree) -> Tree:
    encoded, model = in_tree
    aval = core.avalof("")
    assert type(model) in (str, type(aval)), f"Expected string model: {model!r}"
    context_aval = encoded if isinstance(encoded, core.AVal) else core.avalof(encoded)
    if not isinstance(context_aval, json.JsonAVal):
        raise TypeError(f"Expected typed JSON context, got {context_aval!r}")
    return schema_abstract_tree(schema)


def fill_pushforward_request(in_tree: Tree, /, *, schema: Tree) -> TreePair:
    (p_context, p_model), (t_context, _) = core.materialize_zeros(in_tree)
    expected = core.tangent_s.map(core.avalof(p_context))
    if core.avalof(t_context) != expected:
        raise TypeError(f"Expected {expected!r} tangent, got {core.avalof(t_context)!r}")

    prompt = PUSH_PROMPT.format(
        input=json_content(p_context),
        in_tangent=json_content(t_context),
    )
    context = dict(
        instruction=PUSH_SYSTEM_PROMPT,
        request=prompt,
        output_schema=jsonlib.dumps(describe(schema), allow_nan=False),
    )

    def tangent_field(x):
        if not schemas.is_schema(x):
            return x
        return schemas.aval_to_schema(core.tangent_s.map(core.avalof(x)))

    t_schema = utils.tree.map(tangent_field, schema, is_leaf=schemas.is_schema)
    return (context, p_model), t_schema


def pushforward_fill(in_tree: Tree, /, *, schema: Tree) -> TreePair:
    p_in, _ = in_tree
    (context, model), t_schema = fill_pushforward_request(in_tree, schema=schema)
    p_out = fill_p.bind(p_in, schema=schema)
    t_out = fill_context(context, model, schema=t_schema)
    return p_out, t_out


async def apush_fill(in_tree: Tree, /, *, schema: Tree) -> TreePair:
    p_in, _ = in_tree
    (context, model), t_schema = fill_pushforward_request(in_tree, schema=schema)
    p_ir = stage.trace(ft.partial(fill_p.bind, schema=schema))(p_in)
    t_ir = stage.trace(ft.partial(fill_context, schema=t_schema))(context, model)
    return await order.fanout_p.abind([(p_in,), (context, model)], irs=[p_ir, t_ir])


def fill_pullback_request(in_tree: Tree, /, *, schema: Tree) -> TreePair | None:
    residuals, out_cotangent = in_tree
    encoded, model, out = residuals

    def check_cotangent(p_leaf, c_leaf):
        aval = core.cotangent_s.map(core.avalof(p_leaf))
        if core.avalof(c_leaf) != aval:
            raise TypeError(f"Expected {aval!r} cotangent, got {c_leaf!r}")

    utils.tree.map(check_cotangent, out, out_cotangent)

    if all(isinstance(x, core.Zero) for x in utils.tree.leaves(out_cotangent)):
        return None

    lit_tree = json.decode(encoded)

    def to_schema(x):
        return schemas.aval_to_schema(core.avalof(x))

    context_schema = utils.tree.map(to_schema, lit_tree)

    def to_cotangent_schema(x):
        if not schemas.is_schema(x):
            return x
        return schemas.aval_to_schema(core.cotangent_s.map(core.avalof(x)))

    cotangent_schema = utils.tree.map(to_cotangent_schema, schema, is_leaf=schemas.is_schema)
    out_cotangent = core.materialize_zeros(out_cotangent)
    prompt = GRAD_PROMPT.format(
        input=schema_content((lit_tree, model), (context_schema, schemas.Str())),
        output=schema_content(out, schema),
        out_cotangent=schema_content(out_cotangent, cotangent_schema),
    )

    context = dict(instruction=GRAD_SYSTEM_PROMPT, request=prompt)
    in_schema = feedback_schema((lit_tree, model))
    return (context, model), in_schema


def pullback_bwd_fill(in_tree: Tree, /, *, schema: Tree) -> Tree:
    def zero_input(x):
        return core.Zero(core.cotangent_s.map(core.avalof(x)))

    request = fill_pullback_request(in_tree, schema=schema)
    if request is None:
        (encoded, model, _), _ = in_tree
        return utils.tree.map(zero_input, (encoded, model))
    (context, model), in_schema = request
    feedback, model_feedback = fill_context(context, model, schema=in_schema)
    return json.encode(feedback), model_feedback


async def apull_bwd_fill(in_tree: Tree, /, *, schema: Tree) -> Tree:
    def zero_input(x):
        return core.Zero(core.cotangent_s.map(core.avalof(x)))

    request = fill_pullback_request(in_tree, schema=schema)
    if request is None:
        (encoded, model, _), _ = in_tree
        return utils.tree.map(zero_input, (encoded, model))
    (context, model), in_schema = request
    feedback, model_feedback = await afill_context(context, model, schema=in_schema)
    return json.encode(feedback), model_feedback


core.impl_rules.set(fill_p, impl_fill)
core.aimpl_rules.set(fill_p, aimpl_fill)
core.abstract_rules.set(fill_p, abstract_fill)
core.batch_rules.set(fill_p, ft.partial(batch_lm, fill_p))
core.abatch_rules.set(fill_p, ft.partial(abatch_lm, fill_p))
core.push_rules.set(fill_p, pushforward_fill)
core.apush_rules.set(fill_p, apush_fill)
core.pull_fwd_rules.set(fill_p, ft.partial(pullback_fwd_lm, fill_p))
core.apull_fwd_rules.set(fill_p, ft.partial(apull_fwd_lm, fill_p))
core.pull_bwd_rules.set(fill_p, pullback_bwd_fill)
core.apull_bwd_rules.set(fill_p, apull_bwd_fill)
