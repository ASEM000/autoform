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
import json
import re
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from litellm import ModelResponse, acompletion, completion

import autoform.control as control
import autoform.core as core
import autoform.order as order
import autoform.schemas as schemas
import autoform.stage as stage
import autoform.utils as utils

__all__ = [
    "Client",
    "LiteLLMClient",
    "EchoClient",
    "client",
    "complete",
    "generate",
    "emit_json_schema",
    "parse_json_value",
]


zip = utils.strict_zip

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type Messages = list[dict[str, str]]
type JsonSchema = dict[str, Any]
type EmitJsonSchemaRule = Callable[[Any], JsonSchema | None]
type ParseJsonValueRule = Callable[[Any, Any], Any]
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
    # NOTE(asem): emit a single stop_gradient not to pollute the IR with sg for each role
    roles, model = control.stop_gradient((roles, model))
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    return complete_p.bind((messages, model))


def impl_complete(in_tree: Tree, /) -> str:
    messages, model = in_tree
    response = active_client.get().completion(messages=messages, model=model)
    return response.choices[0].message.content


async def aimpl_complete(in_tree: Tree, /) -> str:
    messages, model = in_tree
    response = await active_client.get().acompletion(messages=messages, model=model)
    return response.choices[0].message.content


def abstract_complete(in_tree: Tree, /) -> Any:
    messages, model = in_tree
    aval = core.avalof("")
    fields = [m[key] for m in messages for key in ("role", "content")]
    assert all(type(x) in (str, type(aval)) for x in fields), f"Expected strings: {messages!r}"
    assert type(model) in (str, type(aval)), f"Expected string model: {model!r}"
    return aval


def pushforward_complete(in_tree: Tree, /) -> TreePair:
    p_in, t_in = in_tree
    t_in = core.materialize_zeros(t_in)
    p_messages, p_model = p_in
    t_messages, *_ = t_in
    t_request = [dict(role=p["role"], content=t["content"]) for p, t in zip(p_messages, t_messages)]
    t_tree = (t_request, p_model)
    p_resp = complete_p.bind(p_in)
    t_resp = complete_p.bind(t_tree)
    return p_resp, t_resp


async def apush_complete(in_tree: Tree, /) -> TreePair:
    p_in, t_in = in_tree
    t_in = core.materialize_zeros(t_in)
    p_messages, p_model = p_in
    t_messages, *_ = t_in
    t_request = [dict(role=p["role"], content=t["content"]) for p, t in zip(p_messages, t_messages)]
    t_tree = (t_request, p_model)
    ir = stage.trace(complete_p.bind)(p_in)
    p_resp, t_resp = await order.fanout_p.abind([(p_in,), (t_tree,)], irs=[ir, ir])
    return p_resp, t_resp


def pullback_fwd_complete(in_tree: Tree, /) -> TreePair:
    messages, model = in_tree
    out = complete_p.bind(in_tree)
    residuals = (messages, model, out)
    return out, residuals


async def apull_fwd_complete(in_tree: Tree, /) -> TreePair:
    messages, model = in_tree
    out = await complete_p.abind(in_tree)
    residuals = (messages, model, out)
    return out, residuals


GRAD_SYSTEM_PROMPT = "Translate output feedback into feedback on the corresponding input fields."
GRAD_PROMPT = """INPUT: {input} OUTPUT: {output} OUTPUT FEEDBACK: {out_cotangent}"""


def pullback_bwd_complete(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = core.materialize_zeros(out_cotangent)
    messages, model, out = residuals
    prompt = GRAD_PROMPT.format(input=(messages, model), output=out, out_cotangent=out_cotangent)

    def make_schema(path, value):
        return schemas.Str(desc=f"Feedback for input at {path}: {value!r}.")

    in_schema = utils.tree.map_with_path(make_schema, (messages, model))
    system_request = dict(role="system", content=GRAD_SYSTEM_PROMPT)
    user_request = dict(role="user", content=prompt)
    return generate_p.bind(([system_request, user_request], model), schema=in_schema)


async def apull_bwd_complete(in_tree: Tree, /) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = core.materialize_zeros(out_cotangent)
    messages, model, out = residuals
    prompt = GRAD_PROMPT.format(input=(messages, model), output=out, out_cotangent=out_cotangent)

    def make_schema(path, value):
        return schemas.Str(desc=f"Feedback for input at {path}: {value!r}.")

    in_schema = utils.tree.map_with_path(make_schema, (messages, model))
    system_request = dict(role="system", content=GRAD_SYSTEM_PROMPT)
    user_request = dict(role="user", content=prompt)
    return await generate_p.abind(([system_request, user_request], model), schema=in_schema)


def batch_complete(in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return complete_p.bind(in_values), False

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    results = [complete_p.bind(unbatch(b)) for b in range(batch_size)]
    out_tree = spec.unflatten(results)
    return out_tree, True


async def abatch_complete(in_tree: Tree, /) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        return await complete_p.abind(in_values), False

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    inputs = [(unbatch(b),) for b in range(batch_size)]
    in0, *_ = inputs
    ir = stage.trace(complete_p.bind)(*in0)
    results = await order.fanout_p.abind(inputs, irs=[ir] * batch_size)
    out_tree = spec.unflatten(results)
    return out_tree, True


core.impl_rules.set(complete_p, impl_complete)
core.aimpl_rules.set(complete_p, aimpl_complete)
core.abstract_rules.set(complete_p, abstract_complete)
core.batch_rules.set(complete_p, batch_complete)
core.abatch_rules.set(complete_p, abatch_complete)
core.push_rules.set(complete_p, pushforward_complete)
core.apush_rules.set(complete_p, apush_complete)
core.pull_fwd_rules.set(complete_p, pullback_fwd_complete)
core.apull_fwd_rules.set(complete_p, apull_fwd_complete)
core.pull_bwd_rules.set(complete_p, pullback_bwd_complete)
core.apull_bwd_rules.set(complete_p, apull_bwd_complete)


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
        ...     answer=af.Float(desc="The numeric answer."),
        ...     reasoning=af.Str(desc="The reasoning behind the answer."),
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
    # NOTE(asem): emit a single stop_gradient not to pollute the IR with sg for each role
    roles, model = control.stop_gradient((roles, model))
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    return generate_p.bind((messages, model), schema=schema)


json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}

emit_json_schema_rules: dict[type[Any], EmitJsonSchemaRule] = {}


def emit_string_json_schema(schema: schemas.Str) -> JsonSchema:
    json_schema: JsonSchema = dict(type="string")
    if schema.min is not None:
        json_schema["minLength"] = schema.min
    if schema.max is not None:
        json_schema["maxLength"] = schema.max
    if schema.pattern is not None:
        json_schema["pattern"] = schema.pattern
    return json_schema


def emit_integer_json_schema(schema: schemas.Int) -> JsonSchema:
    json_schema: JsonSchema = dict(type="integer")
    if schema.min is not None:
        json_schema["minimum"] = schema.min
    if schema.max is not None:
        json_schema["maximum"] = schema.max
    return json_schema


def emit_number_json_schema(schema: schemas.Float) -> JsonSchema:
    json_schema: JsonSchema = dict(type="number")
    if schema.min is not None:
        json_schema["minimum"] = schema.min
    if schema.max is not None:
        json_schema["maximum"] = schema.max
    return json_schema


def emit_boolean_json_schema(schema: schemas.Bool) -> JsonSchema:
    return dict(type="boolean")


def emit_enum_json_schema(schema: schemas.Enum) -> JsonSchema:
    value_type = type(schema.values[0])
    if value_type not in json_types:
        raise TypeError("Enum values must be str, int, float, or bool")
    return dict(type=json_types[value_type], enum=list(schema.values))


emit_json_schema_rules[schemas.Str] = emit_string_json_schema
emit_json_schema_rules[schemas.Int] = emit_integer_json_schema
emit_json_schema_rules[schemas.Float] = emit_number_json_schema
emit_json_schema_rules[schemas.Bool] = emit_boolean_json_schema
emit_json_schema_rules[schemas.Enum] = emit_enum_json_schema


def emit_json_schema(schema: Any, *, value: Any = ...) -> Any:
    # NOTE(asem): internal function to emit json based on the following rules
    # - A literal in the schema will not be generated in the schema.
    # - Emission rules use rules registry that can be extended.
    # - With value, emit typed JSON data using the same structure. Primal bounds
    #   and enum choices do not constrain cotangents.

    # Example:
    #     >>> import json
    #     >>> import autoform as af
    #     >>> schema = {
    #     ...     "name": af.Str(min=1, desc="Name slot."),
    #     ...     "source": "fixed",
    #     ... }
    #     >>> print(json.dumps(af.lm.emit_json_schema(schema), indent=2))
    #     {
    #       "type": "object",
    #       "properties": {
    #         "name": {
    #           "type": "string",
    #           "minLength": 1,
    #           "description": "Name slot."
    #         }
    #       },
    #       "required": [
    #         "name"
    #       ],
    #       "additionalProperties": false
    #     }
    # here only name is emitted, while literal value fixed is omitted.
    if rule := emit_json_schema_rules.get(type(schema)):
        if value is ...:
            json_schema = rule(schema)
            if schema.desc is not None:
                json_schema["description"] = schema.desc
            return json_schema
        aval = core.avalof(schema)
        if core.avalof(value) != aval:
            raise TypeError(f"Expected {aval!r}, got {value!r}")
        return value

    # NOTE(asem): literal leaf case
    if type(schema) not in emit_json_schema_rules and utils.tree.is_leaf(schema):
        return None

    children, spec = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))
    values = [...] * len(children) if value is ... else spec.flatten_up_to(value)
    properties = OrderedDict()
    for entry, child, v in zip(spec.entries(), children, values):
        property_name = str(entry)
        if (child_schema := emit_json_schema(child, value=v)) is not None:
            if property_name in properties:
                raise TypeError(f"Duplicate object entries {(property_name,)!r}")
            properties[property_name] = child_schema

    if not properties:
        # NOTE(asem): all tree is literals
        # >>> dict(key="k", value=1)
        return None

    if value is not ...:
        return properties

    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


parse_json_value_rules: dict[type[Any], ParseJsonValueRule] = {}


def parse_string_json_value(schema: schemas.Str, value: Any) -> str:
    if type(value) is not str:
        raise ValueError("Expected string")
    if schema.min is not None and len(value) < schema.min:
        raise ValueError(f"Expected string with length >= {schema.min}")
    if schema.max is not None and len(value) > schema.max:
        raise ValueError(f"Expected string with length <= {schema.max}")
    if schema.pattern is not None and not re.search(schema.pattern, value):
        raise ValueError(f"Expected string matching {schema.pattern!r}")
    return value


def parse_integer_json_value(schema: schemas.Int, value: Any) -> int:
    if type(value) is not int:
        raise ValueError("Expected integer")
    if schema.min is not None and value < schema.min:
        raise ValueError(f"Expected integer >= {schema.min}")
    if schema.max is not None and value > schema.max:
        raise ValueError(f"Expected integer <= {schema.max}")
    return value


def parse_number_json_value(schema: schemas.Float, value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("Expected number")
    if schema.min is not None and value < schema.min:
        raise ValueError(f"Expected number >= {schema.min}")
    if schema.max is not None and value > schema.max:
        raise ValueError(f"Expected number <= {schema.max}")
    return float(value)


def parse_boolean_json_value(schema: schemas.Bool, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Expected boolean")
    return value


def parse_enum_json_value(schema: schemas.Enum, value: Any) -> Any:
    if value not in schema:
        raise ValueError(f"Expected one of {schema.values!r}")
    return value


parse_json_value_rules[schemas.Str] = parse_string_json_value
parse_json_value_rules[schemas.Int] = parse_integer_json_value
parse_json_value_rules[schemas.Float] = parse_number_json_value
parse_json_value_rules[schemas.Bool] = parse_boolean_json_value
parse_json_value_rules[schemas.Enum] = parse_enum_json_value


def parse_json_value(schema: Any, value: Any) -> Any:
    # NOTE(asem): internal function to
    # 1. Rebuild json to original tree .
    # 2. Validate its values.
    # Here this function needs the original schema tree formed by schema nodes and/or literals
    # and the output json value dict. The dict is being rebuilt and validated against the schema
    # tree. For example
    # >>> class Struct(NamedTuple):
    # ...
    # >>> reference_schema = {
    # ...     "name": af.Str(min=1),
    # ...     "source": "literal",
    # ...     "details": None,
    # ... }
    # >>> model_json_output = {"name": "x"}
    # >>> parse_json_value(reference_schema, model_json_output)
    # {"name": "x", "source": "literal", "details": None}
    # 3 cases are handled here
    # 1. Registered schema type with parsing rule (e.g. af.Str())
    # 2. Literal leaf (e.g. "literal")
    # 3. Pytree of made of registred schema nodes or literals.

    # NOTE(asem): case 1: in case a parsing rule exists use it.
    if rule := parse_json_value_rules.get(type(schema)):
        return rule(schema, value)

    # NOTE(asem): case 2: literal node case.
    if type(schema) not in emit_json_schema_rules and utils.tree.is_leaf(schema):
        return schema

    # NOTE(asem): case 3: flatten the schema one level to rebuild its original container.
    flat_schemas, spec_schema = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))

    schema_keys = [str(entry) for entry in spec_schema.entries()]
    is_emitted: list[bool] = [emit_json_schema(child) is not None for child in flat_schemas]
    expected_keys = [k for k, e in zip(schema_keys, is_emitted) if e]
    if expected_keys or value is not None:
        if not isinstance(value, dict):
            raise ValueError("Expected object")
        if len(expected_keys) != len(value) or set(expected_keys) != value.keys():
            raise ValueError(
                f"Key mismatch: expected entries {expected_keys!r}, got {list(value)!r}"
            )

    return spec_schema.unflatten(
        parse_json_value(child, value[key] if emit else None)
        for key, child, emit in zip(schema_keys, flat_schemas, is_emitted)
    )


def impl_generate(in_tree: Tree, /, *, schema: Any) -> Any:
    messages, model = in_tree
    json_schema = emit_json_schema(schema)
    if json_schema is None:
        return parse_json_value(schema, None)
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
    return parse_json_value(schema, json.loads(resp.choices[0].message.content))


async def aimpl_generate(in_tree: Tree, /, *, schema: Any) -> Any:
    messages, model = in_tree
    json_schema = emit_json_schema(schema)
    if json_schema is None:
        return parse_json_value(schema, None)
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
    return parse_json_value(schema, json.loads(resp.choices[0].message.content))


def abstract_generate(in_tree: Tree, /, *, schema: Any) -> Tree:
    messages, model = in_tree
    aval = core.avalof("")
    fields = [m[key] for m in messages for key in ("role", "content")]
    assert all(type(x) in (str, type(aval)) for x in fields), f"Expected strings: {messages!r}"
    assert type(model) in (str, type(aval)), f"Expected string model: {model!r}"

    def abstract(x: Any) -> Any:
        if schemas.is_schema(x):
            return core.avalof(x)
        if not stage.is_traceable(x):
            raise TypeError(f"Static schema leaf must be traceable, got {x!r}")
        return x

    return utils.tree.map(abstract, schema, is_leaf=schemas.is_schema)


def pushforward_generate(in_tree: Tree, /, *, schema: Any) -> TreePair:
    p_in, t_in = in_tree
    t_in = core.materialize_zeros(t_in)
    p_messages, p_model = p_in
    t_messages, *_ = t_in
    t_request = [dict(role=p["role"], content=t["content"]) for p, t in zip(p_messages, t_messages)]
    t_tree = (t_request, p_model)
    p_resp = generate_p.bind(p_in, schema=schema)
    t_resp = generate_p.bind(t_tree, schema=schema)
    return p_resp, t_resp


async def apush_generate(in_tree: Tree, /, *, schema: Any) -> TreePair:
    p_in, t_in = in_tree
    t_in = core.materialize_zeros(t_in)
    p_messages, p_model = p_in
    t_messages, *_ = t_in
    t_request = [dict(role=p["role"], content=t["content"]) for p, t in zip(p_messages, t_messages)]
    t_tree = (t_request, p_model)
    ir = stage.trace(ft.partial(generate_p.bind, schema=schema))(p_in)
    p_resp, t_resp = await order.fanout_p.abind([(p_in,), (t_tree,)], irs=[ir, ir])
    return p_resp, t_resp


def pullback_fwd_generate(in_tree: Tree, /, *, schema: Any) -> TreePair:
    messages, model = in_tree
    out = generate_p.bind(in_tree, schema=schema)
    residuals = (messages, model, out)
    return out, residuals


async def apull_fwd_generate(in_tree: Tree, /, *, schema: Any) -> TreePair:
    messages, model = in_tree
    out = await generate_p.abind(in_tree, schema=schema)
    residuals = (messages, model, out)
    return out, residuals


def pullback_bwd_generate(in_tree: Tree, /, *, schema: Any) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = core.materialize_zeros(out_cotangent)
    messages, model, out = residuals
    out = json.dumps(emit_json_schema(schema, value=out), allow_nan=False)
    out_cotangent = json.dumps(emit_json_schema(schema, value=out_cotangent), allow_nan=False)
    prompt = GRAD_PROMPT.format(input=(messages, model), output=out, out_cotangent=out_cotangent)

    def make_schema(path, value):
        return schemas.Str(desc=f"Feedback for input at {path}: {value!r}.")

    in_schema = utils.tree.map_with_path(make_schema, (messages, model))
    system_request = dict(role="system", content=GRAD_SYSTEM_PROMPT)
    user_request = dict(role="user", content=prompt)
    return generate_p.bind(([system_request, user_request], model), schema=in_schema)


async def apull_bwd_generate(in_tree: Tree, /, *, schema: Any) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = core.materialize_zeros(out_cotangent)
    messages, model, out = residuals
    out = json.dumps(emit_json_schema(schema, value=out), allow_nan=False)
    out_cotangent = json.dumps(emit_json_schema(schema, value=out_cotangent), allow_nan=False)
    prompt = GRAD_PROMPT.format(input=(messages, model), output=out, out_cotangent=out_cotangent)

    def make_schema(path, value):
        return schemas.Str(desc=f"Feedback for input at {path}: {value!r}.")

    in_schema = utils.tree.map_with_path(make_schema, (messages, model))
    system_request = dict(role="system", content=GRAD_SYSTEM_PROMPT)
    user_request = dict(role="user", content=prompt)
    return await generate_p.abind(([system_request, user_request], model), schema=in_schema)


def batch_generate(in_tree: Tree, /, *, schema: Any) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        result = generate_p.bind(in_values, schema=schema)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    bind = ft.partial(generate_p.bind, schema=schema)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, spec.unflatten(results))
    return out_ib, out_batched


async def abatch_generate(in_tree: Tree, /, *, schema: Tree) -> TreePair:
    batch_size, in_batched, in_values = in_tree

    if (spec := utils.batch_spec(in_values, in_batched)) is None:
        result = await generate_p.abind(in_values, schema=schema)
        out_batched = utils.tree.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(utils.batch_index, in_values, in_batched)
    inputs = [(unbatch(b),) for b in range(batch_size)]
    in0, *_ = inputs
    ir = stage.trace(ft.partial(generate_p.bind, schema=schema))(*in0)
    results = await order.fanout_p.abind(inputs, irs=[ir] * batch_size)
    out_batched = utils.tree.map(lambda _: True, results[0])
    out_ib = utils.batch_transpose(batch_size, out_batched, spec.unflatten(results))
    return out_ib, out_batched


core.impl_rules.set(generate_p, impl_generate)
core.aimpl_rules.set(generate_p, aimpl_generate)
core.abstract_rules.set(generate_p, abstract_generate)
core.batch_rules.set(generate_p, batch_generate)
core.abatch_rules.set(generate_p, abatch_generate)
core.push_rules.set(generate_p, pushforward_generate)
core.apush_rules.set(generate_p, apush_generate)
core.pull_fwd_rules.set(generate_p, pullback_fwd_generate)
core.apull_fwd_rules.set(generate_p, apull_fwd_generate)
core.pull_bwd_rules.set(generate_p, pullback_bwd_generate)
core.apull_bwd_rules.set(generate_p, apull_bwd_generate)
