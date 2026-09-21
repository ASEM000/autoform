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

import asyncio
import functools as ft
import json
import re
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from litellm import ModelResponse, acompletion, completion

import autoform.core as core
import autoform.schemas as schemas
import autoform.utils as utils

__all__ = [
    "Client",
    "LiteLLMClient",
    "EchoClient",
    "client",
    "complete",
    "generate",
    "emit_json_schema",
    "parse_json",
]

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type Messages = list[dict[str, str]]
type Roles = list[str]
type JsonSchema = dict[str, Any]
type JsonSchemaRule = Callable[[Any], JsonSchema | None]
type JsonValueRule = Callable[[Any, Any], Any]
type ClientType = ModelResponse


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


def echo_messages(messages: Messages) -> str:
    return "\n".join(f"<{message['role']}> {message['content']}" for message in messages)


class EchoClient:
    """Echoes messages passed to lm calls without provider calls.

    Mainly for debugging and demonstration.

    Args:
        render: A synchronous callable that receives all messages and returns response text.

    Example:
        >>> import autoform as af
        >>> with af.lm.client(af.lm.EchoClient()):
        ...     msg1 = dict(role="system", content="Translate to Korean.")
        ...     msg2 = dict(role="user", content="Hello!")
        ...     print(af.lm.complete([msg1, msg2], model="echo"))
        <system> Translate to Korean.
        <user> Hello!

    Example with a custom renderer:
        >>> client = af.lm.EchoClient(render=lambda messages: messages[-1]["content"])
        >>> with af.lm.client(client):
        ...     af.lm.complete([dict(role="user", content="Hello!")], model="echo")
        'Hello!'
    """

    __slots__ = ["render"]

    def __init__(self, render: Callable[[Messages], str] = echo_messages):
        self.render = render

    def completion(self, *, messages: list[dict], model: str, **kwargs) -> ClientType:
        content = self.render(messages)
        assert isinstance(content, str), f"`EchoClient` renderer must return strings."
        return ModelResponse(choices=[dict(message=dict(role="assistant", content=content))])

    async def acompletion(self, *, messages: list[dict], model: str, **kwargs) -> ClientType:
        return self.completion(messages=messages, model=model, **kwargs)


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
# COMPLETE
# ==================================================================================================

complete_p = core.Prim("complete")


def complete(messages: Messages, /, *, model: str) -> str:
    """Complete a conversation with text.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The model name or active client model alias to use (e.g., "gpt-5.5").

    Returns:
        The response text.

    Use :func:`client` to configure provider-specific settings like ``max_tokens``.

    Example:
        >>> import autoform as af
        >>> def program(name: str) -> str:
        ...     greeting = "Hello, " + name + "!"
        ...     sys = dict(role="system", content="translate the greeting to Korean")
        ...     usr = dict(role="user", content=greeting)
        ...     greeting = af.lm.complete([sys, usr], model="gpt-5.5")
        ...     return greeting
        >>> ir = af.trace(program)("World") # doctest: +SKIP
        >>> result = ir.call("x0") # doctest: +SKIP

    Example with :func:`client`:
        >>> import autoform as af
        >>> from litellm import Router  # doctest: +SKIP
        >>> params_1024 = dict(model="gpt-5.5", max_tokens=1024)
        >>> params_512 = dict(model="gpt-5.5", max_tokens=512)
        >>> model_list = [
        ...     dict(model_name="gpt-5.5-1024", litellm_params=params_1024),
        ...     dict(model_name="gpt-5.5-512", litellm_params=params_512),
        ... ]
        >>> router = Router(model_list=model_list)  # doctest: +SKIP
        >>> def program(text: str, model: str):
        ...     msg = [{"role": "user", "content": ("Explain " + text + " in one line.")}]
        ...     answer = af.lm.complete(msg, model=model)
        ...     return "Answer: " + answer
        >>> ir = af.trace(program)("topic", "model")
        >>> model_names = ["gpt-5.5-1024", "gpt-5.5-512"]
        >>> with af.lm.client(router):  # doctest: +SKIP
        ...     result = af.batch(ir, in_axes=(False, True)).call("AI", model_names)
    """
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles, contents = [m["role"] for m in messages], [m["content"] for m in messages]
    return complete_p.bind((contents, model), roles=roles)


# TODO(asem): take a look into this
GRAD_PROMPT = """Given this LLM interaction:

INPUT: {content}
OUTPUT: {out}
FEEDBACK ON OUTPUT: {out_cotangent}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""


def impl_complete(in_tree: Tree, /, *, roles: Roles) -> str:
    contents, model = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    response = active_client.get().completion(messages=messages, model=model)
    return response.choices[0].message.content


async def aimpl_complete(in_tree: Tree, /, *, roles: Roles) -> str:
    contents, model = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    response = await active_client.get().acompletion(messages=messages, model=model)
    return response.choices[0].message.content


def abstract_complete(in_tree: Tree, /, *, roles: Roles) -> core.EvalType:
    contents, model = in_tree
    assert all(type(x) in (str, core.StrAVal) for x in contents), f"Expected strings: {contents!r}"
    assert type(model) in (str, core.StrAVal), f"Expected string model: {model!r}"
    return core.StrAVal()


def pushforward_complete(in_tree: Tree, /, *, roles: Roles) -> TreePair:
    import autoform.ad as ad

    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model)
    p_resp = complete_p.bind(p_tree, roles=roles)
    t_tree = (ad.materialize(tangent_contents), primal_model)
    t_resp = complete_p.bind(t_tree, roles=roles)
    return p_resp, t_resp


async def apush_complete(in_tree: Tree, /, *, roles: Roles) -> TreePair:
    import autoform.ad as ad

    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(complete_p.abind, roles=roles)
    p_tree = (primal_contents, primal_model)
    t_tree = (ad.materialize(tangent_contents), primal_model)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_complete(in_tree: Tree, /, *, roles: Roles) -> TreePair:
    contents, model = in_tree
    out = complete_p.bind((contents, model), roles=roles)
    residuals = (contents, model, out)
    return out, residuals


async def apull_fwd_complete(in_tree: Tree, /, *, roles: Roles) -> TreePair:
    contents, model = in_tree
    out = await complete_p.abind((contents, model), roles=roles)
    residuals = (contents, model, out)
    return out, residuals


def pullback_bwd_complete(in_tree: Tree, /, *, roles: Roles) -> Tree:
    import autoform.ad as ad

    residuals, out_cotangent = in_tree
    out_cotangent = ad.materialize(out_cotangent)
    contents, model, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grad_out = complete_p.bind(([grad_prompt], model), roles=["user"])
        grads.append(grad_out)
    return grads, ad.cotangent_zeroof(model)


async def apull_bwd_complete(in_tree: Tree, /, *, roles: Roles) -> Tree:
    import autoform.ad as ad

    residuals, out_cotangent = in_tree
    out_cotangent = ad.materialize(out_cotangent)
    contents, model, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        grad_out = complete_p.abind(([prompt], model), roles=["user"])
        return await grad_out

    return (await asyncio.gather(*[grad(c) for c in contents]), ad.cotangent_zeroof(model))


def batch_complete(in_tree: Tree, /, *, roles: Roles) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return complete_p.bind(in_values, roles=roles), False

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    results = [complete_p.bind(unbatch(b), roles=roles) for b in range(batch_size)]
    out_tree = spec.unflatten(results)
    return out_tree, True


async def abatch_complete(in_tree: Tree, /, *, roles: Roles) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return await complete_p.abind(in_values, roles=roles), False

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    abind = ft.partial(complete_p.abind, roles=roles)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_tree = spec.unflatten(results)
    return out_tree, True


core.impl_rules.set(complete_p, impl_complete)
core.impl_rules.aset(complete_p, aimpl_complete)
core.abstract_rules.set(complete_p, abstract_complete)
core.push_rules.set(complete_p, pushforward_complete)
core.push_rules.aset(complete_p, apush_complete)
core.pull_fwd_rules.set(complete_p, pullback_fwd_complete)
core.pull_fwd_rules.aset(complete_p, apull_fwd_complete)
core.pull_bwd_rules.set(complete_p, pullback_bwd_complete)
core.pull_bwd_rules.aset(complete_p, apull_bwd_complete)
core.batch_rules.set(complete_p, batch_complete)
core.batch_rules.aset(complete_p, abatch_complete)

# ==================================================================================================
# GENERATE
# ==================================================================================================

generate_p = core.Prim("generate")


def generate(messages: Messages, /, *, model: str, schema: Any) -> Any:
    """Generate a value matching the supplied schema.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The model name or active client model alias to use (e.g., "gpt-5.5").
        schema: An autoform schema tree describing the output.

    Returns:
        A value with the same pytree structure as the schema.

    Use :func:`client` to configure provider-specific settings like ``max_tokens``.

    Example with a registered pytree:
        >>> import optree
        >>> import autoform as af
        >>> @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
        ... class Answer:
        ...     answer: float
        ...     reasoning: str
        >>> schema = Answer(
        ...     answer=af.Float() @ af.Doc("The numeric answer."),
        ...     reasoning=af.Str() @ af.Doc("The reasoning behind the answer."),
        ... )
        >>> msgs = [dict(role="user", content="1 + 1?")]
        >>> output = af.lm.generate(  # doctest: +SKIP
        ...     msgs,
        ...     model="openai/gpt-5.5",
        ...     schema=schema,
        ... )
        >>> output  # doctest: +SKIP
        Answer(answer=2.0, reasoning='Adding 1 and 1 gives 2.')
    """
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles, contents = [m["role"] for m in messages], [m["content"] for m in messages]

    return generate_p.bind((contents, model), roles=roles, schema=schema)


SCHEMA_GRAD_PROMPT = """Given this LLM interaction:

INPUT: {content}
STRUCTURED OUTPUT FEEDBACK:
{feedback}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise.

- Each field is one leaf of the generated output.
- Path locates the field from the root output object.
- Value is the generated value.
- Feedback is natural-language feedback for that field; empty feedback means no change.
"""


json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}

json_schema_rules: dict[type[Any], JsonSchemaRule] = {}


def string_json_schema(s: schemas.Str) -> JsonSchema:
    schema: JsonSchema = dict(type="string")
    if s.min is not None:
        schema["minLength"] = s.min
    if s.max is not None:
        schema["maxLength"] = s.max
    if s.pattern is not None:
        schema["pattern"] = s.pattern
    return schema


def integer_json_schema(s: schemas.Int) -> JsonSchema:
    schema: JsonSchema = dict(type="integer")
    if s.min is not None:
        schema["minimum"] = s.min
    if s.max is not None:
        schema["maximum"] = s.max
    return schema


def number_json_schema(s: schemas.Float) -> JsonSchema:
    schema: JsonSchema = dict(type="number")
    if s.min is not None:
        schema["minimum"] = s.min
    if s.max is not None:
        schema["maximum"] = s.max
    return schema


def boolean_json_schema(_: schemas.Bool) -> JsonSchema:
    return dict(type="boolean")


def enum_json_schema(s: schemas.Enum) -> JsonSchema:
    value_type = type(s.values[0])
    if value_type not in json_types:
        raise TypeError("Enum values must be str, int, float, or bool")
    return dict(type=json_types[value_type], enum=list(s.values))


def docd_json_schema(docd: schemas.Docd[Any]) -> JsonSchema | None:
    if (schema := emit_json_schema(docd.value)) is None:
        return None
    return schema | dict(description=docd.text)


json_schema_rules[schemas.Str] = string_json_schema
json_schema_rules[schemas.Int] = integer_json_schema
json_schema_rules[schemas.Float] = number_json_schema
json_schema_rules[schemas.Bool] = boolean_json_schema
json_schema_rules[schemas.Enum] = enum_json_schema
json_schema_rules[schemas.Docd] = docd_json_schema


def emit_json_schema(node: Any) -> JsonSchema | None:
    if rule := json_schema_rules.get(type(node)):
        return rule(node)
    if type(node) not in json_schema_rules and utils.tree.is_leaf(node):
        return None

    children, spec = utils.tree.flatten(node, is_leaf=lambda x: id(x) != id(node))
    properties = OrderedDict()
    for entry, child in zip(spec.entries(), children, strict=True):
        property_name = str(entry)
        if (child_schema := emit_json_schema(child)) is not None:
            if property_name in properties:
                raise TypeError(f"Duplicate object entries {(property_name,)!r}")
            properties[property_name] = child_schema

    if not properties:
        return None

    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


json_value_rules: dict[type[Any], JsonValueRule] = {}


def string_json_value(s: schemas.Str, value: str) -> str:
    if type(value) is not str:
        raise ValueError("Expected string")
    if s.min is not None and len(value) < s.min:
        raise ValueError(f"Expected string with length >= {s.min}")
    if s.max is not None and len(value) > s.max:
        raise ValueError(f"Expected string with length <= {s.max}")
    if s.pattern is not None and not re.search(s.pattern, value):
        raise ValueError(f"Expected string matching {s.pattern!r}")
    return value


def integer_json_value(s: schemas.Int, value: int) -> int:
    if type(value) is not int:
        raise ValueError("Expected integer")
    if s.min is not None and value < s.min:
        raise ValueError(f"Expected integer >= {s.min}")
    if s.max is not None and value > s.max:
        raise ValueError(f"Expected integer <= {s.max}")
    return value


def number_json_value(s: schemas.Float, value: int | float) -> float:
    if type(value) not in (int, float):
        raise ValueError("Expected number")
    if s.min is not None and value < s.min:
        raise ValueError(f"Expected number >= {s.min}")
    if s.max is not None and value > s.max:
        raise ValueError(f"Expected number <= {s.max}")
    return float(value)


def boolean_json_value(_: schemas.Bool, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Expected boolean")
    return value


def enum_json_value(s: schemas.Enum, value: Any) -> Any:
    if value not in s:
        raise ValueError(f"Expected one of {s.values!r}")
    return value


def docd_json_value(s: schemas.Docd[Any], value: Any) -> Any:
    return parse_json(s.value, value)


json_value_rules[schemas.Str] = string_json_value
json_value_rules[schemas.Int] = integer_json_value
json_value_rules[schemas.Float] = number_json_value
json_value_rules[schemas.Bool] = boolean_json_value
json_value_rules[schemas.Enum] = enum_json_value
json_value_rules[schemas.Docd] = docd_json_value


def parse_json(schema: Any, value: Any) -> Any:
    if rule := json_value_rules.get(type(schema)):
        return rule(schema, value)
    if type(schema) not in json_schema_rules and utils.tree.is_leaf(schema):
        return schema

    flat_schemas, spec_schema = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))
    flat_values, spec_value = utils.tree.flatten(value, is_leaf=lambda x: id(x) != id(value))
    schema_keys = [str(entry) for entry in spec_schema.entries()]
    emitted = [emit_json_schema(child) is not None for child in flat_schemas]
    expected_keys = [k for k, e in zip(schema_keys, emitted, strict=True) if e]
    value_keys = [str(entry) for entry in spec_value.entries()]

    if len(expected_keys) != len(value_keys) or set(expected_keys) != set(value_keys):
        raise ValueError(f"Key mismatch: expected entries {expected_keys!r}, got {value_keys!r}")

    out_pos = {k: i for i, k in enumerate(value_keys)}
    values = (
        parse_json(child, flat_values[out_pos[key]] if emit else None)
        for key, child, emit in zip(schema_keys, flat_schemas, emitted, strict=True)
    )
    return spec_schema.unflatten(values)


def impl_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> Any:
    contents, model = in_tree
    json_schema = emit_json_schema(schema)
    if json_schema is None:
        return parse_json(schema, None)
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
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
    return parse_json(schema, json.loads(resp.choices[0].message.content))


async def aimpl_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> Any:
    contents, model = in_tree
    json_schema = emit_json_schema(schema)
    if json_schema is None:
        return parse_json(schema, None)
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
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
    return parse_json(schema, json.loads(resp.choices[0].message.content))


def string_schema_abstract(_: schemas.Str) -> core.StrAVal:
    return core.StrAVal()


def integer_schema_abstract(_: schemas.Int) -> core.IntAVal:
    return core.IntAVal()


def number_schema_abstract(_: schemas.Float) -> core.FloatAVal:
    return core.FloatAVal()


def boolean_schema_abstract(_: schemas.Bool) -> core.BoolAVal:
    return core.BoolAVal()


def enum_schema_abstract(s: schemas.Enum) -> core.AVal:
    return core.primal_s.avalof(s.values[0])


def docd_schema_abstract(s: schemas.Docd[Any]) -> Tree:
    return schema_abstract_tree(s.value)


schema_abstract_rules = {}
schema_abstract_rules[schemas.Str] = string_schema_abstract
schema_abstract_rules[schemas.Int] = integer_schema_abstract
schema_abstract_rules[schemas.Float] = number_schema_abstract
schema_abstract_rules[schemas.Bool] = boolean_schema_abstract
schema_abstract_rules[schemas.Enum] = enum_schema_abstract
schema_abstract_rules[schemas.Docd] = docd_schema_abstract


def schema_abstract_tree(schema: Any) -> Tree:
    def abstract(x: Any) -> Any:
        if rule := schema_abstract_rules.get(type(x)):
            return rule(x)
        if not core.is_traceable(x):
            raise TypeError(f"Static schema leaf must be traceable, got {x!r}")
        return x

    return utils.tree.map(abstract, schema, is_leaf=lambda x: type(x) in schema_abstract_rules)


def abstract_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> Tree:
    contents, model = in_tree
    assert all(type(x) in (str, core.StrAVal) for x in contents), f"Expected strings: {contents!r}"
    assert type(model) in (str, core.StrAVal), f"Expected string model: {model!r}"
    return schema_abstract_tree(schema)


def pushforward_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> TreePair:
    import autoform.ad as ad

    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model)
    t_tree = (ad.materialize(tangent_contents), primal_model)
    p_resp = generate_p.bind(p_tree, roles=roles, schema=schema)
    t_resp = generate_p.bind(t_tree, roles=roles, schema=schema)
    return p_resp, t_resp


async def apush_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> TreePair:
    import autoform.ad as ad

    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(generate_p.abind, roles=roles, schema=schema)
    p_tree = (primal_contents, primal_model)
    t_tree = (ad.materialize(tangent_contents), primal_model)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> TreePair:
    contents, model = in_tree
    out = generate_p.bind(in_tree, roles=roles, schema=schema)
    residuals = (contents, model, out)
    return out, residuals


async def apull_fwd_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> TreePair:
    contents, model = in_tree
    out = await generate_p.abind(in_tree, roles=roles, schema=schema)
    residuals = (contents, model, out)
    return out, residuals


def build_cotangent_schema_summary(out: Tree, cotangent: Tree) -> str:
    import autoform.ad as ad

    def validate_schema_feedback(path: str, feedback: Any) -> str:
        if ad.is_zero(feedback):
            return "No feedback"
        if type(feedback) is str:
            return feedback
        raise TypeError(f"{path}: schema output cotangent leaves must be text, got {feedback!r}")

    out_leaves, out_spec = utils.tree.flatten(out)
    cotangents = out_spec.flatten_up_to(cotangent)
    lines = ["Fields:"]

    for accessor, value, feedback in zip(out_spec.accessors(), out_leaves, cotangents, strict=True):
        feedback = validate_schema_feedback(accessor.codify("$"), feedback)
        lines.append(accessor.codify("$"))
        lines.append(f"\tvalue: {value!r}")
        lines.append(f"\tfeedback: {feedback!r}")
    return "\n".join(lines).expandtabs(2)


def pullback_bwd_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> Tree:
    import autoform.ad as ad

    residuals, out_cotangent = in_tree
    contents, model, out = residuals
    feedback = build_cotangent_schema_summary(out, out_cotangent)
    grads = []
    for content in contents:
        grad_prompt = SCHEMA_GRAD_PROMPT.format(content=content, feedback=feedback)
        grad_out = complete_p.bind(([grad_prompt], model), roles=["user"])
        grads.append(grad_out)
    return grads, ad.cotangent_zeroof(model)


async def apull_bwd_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> Tree:
    import autoform.ad as ad

    residuals, out_cotangent = in_tree
    contents, model, out = residuals
    feedback = build_cotangent_schema_summary(out, out_cotangent)

    async def grad(c):
        prompt = SCHEMA_GRAD_PROMPT.format(content=c, feedback=feedback)
        grad_out = complete_p.abind(([prompt], model), roles=["user"])
        return await grad_out

    return (await asyncio.gather(*[grad(c) for c in contents]), ad.cotangent_zeroof(model))


def batch_generate(in_tree: Tree, /, *, roles: Roles, schema: Any) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if utils.batch_spec(in_values, in_batched) is None:
        result = generate_p.bind(in_values, roles=roles, schema=schema)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    bind = ft.partial(generate_p.bind, roles=roles, schema=schema)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, results)
    return out_ib, out_batched


async def abatch_generate(in_tree: Tree, /, *, roles: Roles, schema: Tree) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if utils.batch_spec(in_values, in_batched) is None:
        result = await generate_p.abind(in_values, roles=roles, schema=schema)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    abind = ft.partial(generate_p.abind, roles=roles, schema=schema)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, list(results))
    return out_ib, out_batched


core.impl_rules.set(generate_p, impl_generate)
core.impl_rules.aset(generate_p, aimpl_generate)
core.abstract_rules.set(generate_p, abstract_generate)
core.push_rules.set(generate_p, pushforward_generate)
core.push_rules.aset(generate_p, apush_generate)
core.pull_fwd_rules.set(generate_p, pullback_fwd_generate)
core.pull_fwd_rules.aset(generate_p, apull_fwd_generate)
core.pull_bwd_rules.set(generate_p, pullback_bwd_generate)
core.pull_bwd_rules.aset(generate_p, apull_bwd_generate)
core.batch_rules.set(generate_p, batch_generate)
core.batch_rules.aset(generate_p, abatch_generate)
