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

import copy
import functools as ft
import json as jsonlib
import math
import re
from collections.abc import Callable, Generator, Hashable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from litellm import ResponsesAPIResponse, aresponses, responses

import autoform.control as control
import autoform.core as core
import autoform.numeric as numeric
import autoform.order as order
import autoform.stage as stage
import autoform.string as string
import autoform.utils as utils

__all__ = [
    "Client",
    "LiteLLMClient",
    "client",
    "fill",
    "Bool",
    "Enum",
    "Float",
    "Int",
    "Str",
    "describe",
    "parse",
]


zip = utils.strict_zip

type Tree[T] = utils.Tree[T]
type TreePair = tuple[Tree, Tree]
type ClientType = ResponsesAPIResponse


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


# ==================================================================================================
# DESCRIBE AND PARSE
# ==================================================================================================

type JsonSchema = dict[str, Any]
type DescribeRule = Callable[[Any], JsonSchema]
type ParseRule = Callable[[Any, Any], Any]

describe_rules: dict[type, DescribeRule] = {}
parse_rules: dict[type, ParseRule] = {}
missing = object()


def is_describe_node(node: Any) -> bool:
    return type(node) in describe_rules


def is_parse_node(node: Any) -> bool:
    return type(node) in parse_rules


def describe(schema: Tree, /) -> JsonSchema | None:
    """Describe a tree using registered description rules, omitting literal fields."""
    schm_tree, _ = utils.partition(
        is_describe_node,
        schema,
        is_leaf=is_describe_node,
        fillvalue=missing,
    )
    return describe_node(schm_tree)


def describe_node(schema: Tree, /) -> JsonSchema | None:
    if schema is missing:
        return None
    if rule := describe_rules.get(type(schema)):
        return rule(schema)

    flat, spec = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))
    property_names = json_property_names(spec.entries())
    properties = {}
    for name, child in zip(property_names, flat):
        if (child_schema := describe_node(child)) is not None:
            properties[name] = child_schema
    if not properties:
        return None

    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


def project_value(schema: Tree, value: Tree, /) -> Any:
    def select_field(node, value):
        if not is_describe_node(node):
            return None
        aval = core.avalof(node)
        if core.avalof(value) != aval:
            raise TypeError(f"Expected {aval!r}, got {value!r}")
        return value

    generated = utils.tree.map(select_field, schema, value, is_leaf=is_describe_node)
    return json_value(generated) if utils.tree.leaves(generated) else None


def parse_node(schema: Tree, value: Any, /) -> Tree:
    if schema is missing:
        return missing
    if rule := parse_rules.get(type(schema)):
        return rule(schema, value)

    def has_parse_node(node):
        return any(map(is_parse_node, utils.tree.leaves(node, is_leaf=is_parse_node)))

    flat, spec = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))
    schema_keys = json_property_names(spec.entries())
    properties = {key: child for key, child in zip(schema_keys, flat) if has_parse_node(child)}
    if not properties:
        return schema
    expected_spec = utils.tree.structure(properties, is_leaf=lambda x: id(x) != id(properties))
    values = dict(zip(expected_spec.entries(), expected_spec.flatten_up_to(value)))
    children = (parse_node(child, values.get(key)) for key, child in zip(schema_keys, flat))
    return spec.unflatten(children)


def parse(schema: Tree, value: Any, /) -> Tree:
    """Parse a tree of registered nodes with their constraints and preserve literals."""

    def generated_field(literal, generated):
        return generated if literal is missing else literal

    schm_tree, lit_tree = utils.partition(
        is_parse_node,
        schema,
        is_leaf=is_parse_node,
        fillvalue=missing,
    )
    generated = parse_node(schm_tree, value)
    return utils.tree.map(generated_field, lit_tree, generated)


# ==================================================================================================
# USER SCHEMA NODES
# ==================================================================================================


def slotted_values(node: Any) -> tuple[Any, ...]:
    return tuple(getattr(node, name) for name in type(node).__slots__)


class Spec(Hashable):
    __slots__ = ["desc"]

    def __init__(self, *, desc: str | None = None) -> None:
        if desc is not None and core.avalof(desc) != core.avalof(""):
            raise TypeError(f"desc must be a string, got {desc!r}")
        self.desc = desc

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def flatten(node):
            return (node.desc,), slotted_values(node), ("desc",)

        def unflatten(metadata, children):
            node = object.__new__(cls)
            (node.desc,) = children
            for name, value in utils.strict_zip(cls.__slots__, metadata):
                setattr(node, name, value)
            return node

        utils.tree.register_node(cls, flatten, unflatten)

    def __matmul__(self, desc: str) -> Spec:
        aval = desc if isinstance(desc, core.AVal) else core.avalof(desc)
        if aval != core.avalof(""):
            raise TypeError(f"desc must be a string, got {desc!r}")
        spec = copy.copy(self)
        spec.desc = desc
        return spec

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return (self.desc, slotted_values(self)) == (other.desc, slotted_values(other))

    def __hash__(self) -> int:
        return hash((type(self), self.desc, slotted_values(self)))

    def __repr__(self) -> str:
        fields = (f"{name}={getattr(self, name)!r}" for name in (*type(self).__slots__, "desc"))
        return f"{type(self).__name__}({', '.join(fields)})"


class Str(Spec):
    """String schema node with optional length and pattern constraints.

    Args:
        desc: Optional generation guidance.
        min: Optional minimum length of the string.
        max: Optional maximum length of the string.
        pattern: Optional regular expression pattern that the string must match.

    Example:
        >>> import autoform as af
        >>> spec = af.lm.Str(min=1, max=80, pattern=r"^[A-Za-z ]+$") @ "description"
    """

    __slots__ = ["min", "max", "pattern"]

    def __init__(
        self,
        *,
        desc: str | None = None,
        min: int | None = None,
        max: int | None = None,
        pattern: str | None = None,
    ) -> None:
        super().__init__(desc=desc)
        if min is not None and type(min) is not int:
            raise TypeError(f"min must be an int, got {min!r}")
        if max is not None and type(max) is not int:
            raise TypeError(f"max must be an int, got {max!r}")
        if min is not None and min < 0:
            raise ValueError(f"min must be >= 0, got {min!r}")
        if max is not None and max < 0:
            raise ValueError(f"max must be >= 0, got {max!r}")
        if pattern is not None and type(pattern) is not str:
            raise TypeError(f"pattern must be a string, got {pattern!r}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min must be <= max, got min={min!r}, max={max!r}")
        if pattern is not None:
            re.compile(pattern)
        self.min = min
        self.max = max
        self.pattern = pattern


class Int(Spec):
    """Integer schema node with optional range constraints.

    Args:
        desc: Optional generation guidance.
        min: Optional minimum value.
        max: Optional maximum value.

    Example:
        >>> import autoform as af
        >>> count = af.lm.Int(min=0, max=10)
    """

    __slots__ = ["min", "max"]

    def __init__(
        self,
        *,
        desc: str | None = None,
        min: int | None = None,
        max: int | None = None,
    ) -> None:
        super().__init__(desc=desc)
        if min is not None and type(min) is not int:
            raise TypeError(f"min must be an int, got {min!r}")
        if max is not None and type(max) is not int:
            raise TypeError(f"max must be an int, got {max!r}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min must be <= max, got min={min!r}, max={max!r}")
        self.min = min
        self.max = max


class Float(Spec):
    """Number schema node with optional range constraints.

    Args:
        desc: Optional generation guidance.
        min: Optional minimum value.
        max: Optional maximum value.

    Example:
        >>> import autoform as af
        >>> score = af.lm.Float(min=0, max=1)
    """

    __slots__ = ["min", "max"]

    def __init__(
        self,
        *,
        desc: str | None = None,
        min: int | float | None = None,
        max: int | float | None = None,
    ) -> None:
        super().__init__(desc=desc)
        if min is not None and type(min) not in (int, float):
            raise TypeError(f"min must be a number, got {min!r}")
        if max is not None and type(max) not in (int, float):
            raise TypeError(f"max must be a number, got {max!r}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min must be <= max, got min={min!r}, max={max!r}")
        self.min = min
        self.max = max


class Bool(Spec):
    """Boolean schema node.

    Args:
        desc: Optional generation guidance.

    Example:
        >>> import autoform as af
        >>> ok = af.lm.Bool()
    """

    __slots__ = []


class Enum(Spec):
    """Enum schema node with a fixed set of allowed values.

    Args:
        desc: Optional generation guidance.
        *values: Allowed values. Values must be non-empty and share one type.

    Example:
        >>> import autoform as af
        >>> kind = af.lm.Enum("summary", "definition")
    """

    __slots__ = ["values"]

    def __init__(self, *values: Any, desc: str | None = None) -> None:
        super().__init__(desc=desc)
        if not values:
            raise TypeError("Enum must have at least one value")
        value_types = {type(value) for value in values}
        if len(value_types) != 1:
            raise TypeError(f"Enum values must share one type, got {value_types!r}")
        self.values = values

    def __contains__(self, value: Any) -> bool:
        return type(value) is type(self.values[0]) and value in self.values


# ==================================================================================================
# JSON DESCRIPTION RULES
# ==================================================================================================


def with_description(schema: Spec, value: JsonSchema) -> JsonSchema:
    if schema.desc is not None:
        value["description"] = schema.desc
    return value


def describe_str(schema: Str) -> JsonSchema:
    json_schema: JsonSchema = dict(type="string")
    if schema.min is not None:
        json_schema["minLength"] = schema.min
    if schema.max is not None:
        json_schema["maxLength"] = schema.max
    if schema.pattern is not None:
        json_schema["pattern"] = schema.pattern
    return with_description(schema, json_schema)


def describe_int(schema: Int) -> JsonSchema:
    json_schema: JsonSchema = dict(type="integer")
    if schema.min is not None:
        json_schema["minimum"] = schema.min
    if schema.max is not None:
        json_schema["maximum"] = schema.max
    return with_description(schema, json_schema)


def describe_float(schema: Float) -> JsonSchema:
    json_schema: JsonSchema = dict(type="number")
    if schema.min is not None:
        json_schema["minimum"] = schema.min
    if schema.max is not None:
        json_schema["maximum"] = schema.max
    return with_description(schema, json_schema)


def describe_bool(schema: Bool) -> JsonSchema:
    return with_description(schema, dict(type="boolean"))


def describe_enum(schema: Enum) -> JsonSchema:
    json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if (value_type := type(schema.values[0])) not in json_types:
        raise TypeError("Enum values must be str, int, float, or bool")
    if value_type is float and not all(math.isfinite(value) for value in schema.values):
        raise ValueError("Enum values must be finite")
    json_schema = dict(type=json_types[value_type], enum=list(schema.values))
    return with_description(schema, json_schema)


describe_rules[Str] = describe_str
describe_rules[Int] = describe_int
describe_rules[Float] = describe_float
describe_rules[Bool] = describe_bool
describe_rules[Enum] = describe_enum


def is_schema(node: Any) -> bool:
    return isinstance(node, Spec)


# ==================================================================================================
# JSON PARSING RULES
# ==================================================================================================


def parse_str(schema: Str, value: Any) -> str:
    if type(value) is not str:
        raise ValueError("Expected string")
    if schema.min is not None and len(value) < schema.min:
        raise ValueError(f"Expected string with length >= {schema.min}")
    if schema.max is not None and len(value) > schema.max:
        raise ValueError(f"Expected string with length <= {schema.max}")
    if schema.pattern is not None and not re.search(schema.pattern, value):
        raise ValueError(f"Expected string matching {schema.pattern!r}")
    return value


def parse_int(schema: Int, value: Any) -> int:
    if type(value) is not int:
        raise ValueError("Expected integer")
    if schema.min is not None and value < schema.min:
        raise ValueError(f"Expected integer >= {schema.min}")
    if schema.max is not None and value > schema.max:
        raise ValueError(f"Expected integer <= {schema.max}")
    return value


def parse_float(schema: Float, value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("Expected number")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Expected finite number")
    if schema.min is not None and value < schema.min:
        raise ValueError(f"Expected number >= {schema.min}")
    if schema.max is not None and value > schema.max:
        raise ValueError(f"Expected number <= {schema.max}")
    return float(value)


def parse_bool(schema: Bool, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Expected boolean")
    return value


def parse_enum(schema: Enum, value: Any) -> Any:
    if value not in schema:
        raise ValueError(f"Expected one of {schema.values!r}")
    return value


parse_rules[Str] = parse_str
parse_rules[Int] = parse_int
parse_rules[Float] = parse_float
parse_rules[Bool] = parse_bool
parse_rules[Enum] = parse_enum


# ==================================================================================================
# ABSTRACT
# ==================================================================================================


core.aval_types[Str] = lambda _: core.avalof("")
core.aval_types[Int] = lambda _: core.avalof(0)
core.aval_types[Float] = lambda _: core.avalof(0.0)
core.aval_types[Bool] = lambda _: core.avalof(False)
core.aval_types[Enum] = lambda schema: core.avalof(schema.values[0])


type AValSchemaRule = Callable[[core.AVal], Spec]

aval_schema_rules: dict[type[core.AVal], AValSchemaRule] = {}
aval_schema_rules[string.StrAVal] = lambda _: Str()
aval_schema_rules[numeric.IntAVal] = lambda _: Int()
aval_schema_rules[numeric.FloatAVal] = lambda _: Float()
aval_schema_rules[numeric.BoolAVal] = lambda _: Bool()


def aval_to_schema(aval: core.AVal) -> Spec:
    if rule := aval_schema_rules.get(type(aval)):
        schema = rule(aval)
        assert is_schema(schema), f"AVal schema rule returned {schema!r}"
        return schema
    raise TypeError(f"No schema rule registered for {aval!r}")


# ==================================================================================================
# CLIENTS
# ==================================================================================================


@runtime_checkable
class Client(Protocol):
    def responses(self, *, input: str, model: str, **kwargs) -> ClientType: ...
    async def aresponses(self, *, input: str, model: str, **kwargs) -> ClientType: ...


class LiteLLMClient:
    __slots__ = []

    def responses(self, *, input: str, model: str, **kwargs) -> ClientType:
        return responses(input=input, model=model, **kwargs)

    async def aresponses(self, *, input: str, model: str, **kwargs) -> ClientType:
        return await aresponses(input=input, model=model, **kwargs)


active_client: ContextVar[Client] = ContextVar("active_client", default=LiteLLMClient())


@contextmanager
def client(client: Client) -> Generator[Client, None, None]:
    """Set the LM client for all lm primitives.

    The client must expose ``.responses()`` and ``.aresponses()`` matching
    LiteLLM's Responses signature.

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
        ...     default_max_parallel_requests=10,
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
    value = dict(values=json_value(core.materialize_zeros(value)), schema=describe(schema))
    return jsonlib.dumps(value, allow_nan=False)


def literal_content(lit_tree: Tree) -> str:
    def to_schema(value):
        return aval_to_schema(core.avalof(value))

    schema = utils.tree.map(to_schema, lit_tree)
    return schema_content(lit_tree, schema)


def schema_response(in_tree: Tree, /, *, schema: Any) -> Any:
    input, model = in_tree
    json_schema = describe(schema)
    if json_schema is None:
        return parse(schema, None)

    fmt = dict(type="json_schema", name="autoform_schema", strict=True, schema=json_schema)
    out = active_client.get().responses(input=input, model=model, text=dict(format=fmt))
    return parse(schema, jsonlib.loads(out.output_text))


async def aschema_response(in_tree: Tree, /, *, schema: Any) -> Any:
    input, model = in_tree
    json_schema = describe(schema)
    if json_schema is None:
        return parse(schema, None)
    fmt = dict(type="json_schema", name="autoform_schema", strict=True, schema=json_schema)
    out = await active_client.get().aresponses(input=input, model=model, text=dict(format=fmt))
    return parse(schema, jsonlib.loads(out.output_text))


def schema_abstract_tree(schema: Any) -> Tree:
    def abstract(x: Any) -> Any:
        if is_schema(x):
            return core.avalof(x)
        if not stage.is_traceable(x):
            raise TypeError(f"Static schema leaf must be traceable, got {x!r}")
        return x

    return utils.tree.map(abstract, schema, is_leaf=is_schema)


def fill_context(context: Tree, model: str, /, *, schema: Tree) -> Tree:
    schm_tree, _ = utils.partition(is_schema, schema, is_leaf=is_schema)
    lit_tree, holes = prepare_fill(dict(context=context, output=schm_tree))
    in_tree, static_tree = fill_input((lit_tree, holes, model))
    out = fill_p.bind(in_tree, static_tree=static_tree)
    return merge_fill(schema, out["output"])


async def afill_context(context: Tree, model: str, /, *, schema: Tree) -> Tree:
    schm_tree, _ = utils.partition(is_schema, schema, is_leaf=is_schema)
    lit_tree, holes = prepare_fill(dict(context=context, output=schm_tree))
    in_tree, static_tree = fill_input((lit_tree, holes, model))
    out = await fill_p.abind(in_tree, static_tree=static_tree)
    return merge_fill(schema, out["output"])


def feedback_schema(tree: Tree) -> Tree:
    def make_schema(path, value):
        aval = core.cotangent_s.map(core.avalof(value))
        schema = aval_to_schema(aval)
        schema.desc = f"Input cotangent at {path}, original value {value!r}."
        return schema

    return utils.tree.map_with_path(make_schema, tree)


def pullback_fwd_lm(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    out = prim.bind(in_tree, **params)
    residuals = (*in_tree, out)
    return out, residuals


async def apull_fwd_lm(prim: core.Prim, in_tree: Tree, /, **params) -> TreePair:
    out = await prim.abind(in_tree, **params)
    residuals = (*in_tree, out)
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
        tree: A pytree containing context leaves with registered AVal schemas and schema nodes.
        model: The model name or active client model alias to use.

    Returns:
        ``tree`` with each schema node replaced by a generated value.

    Example:
        >>> import autoform as af
        >>> spec = af.lm.Str() @ "description"
        >>> out = af.lm.fill(dict(result=spec), model="model-name")  # doctest: +SKIP
    """

    def check_context(value):
        aval_to_schema(core.avalof(value))

    utils.tree.map(check_context, tree)
    if not any(map(is_schema, utils.tree.leaves(tree, is_leaf=is_schema))):
        return tree
    assert core.avalof(model) == core.avalof(""), f"Expected string model: {model!r}"
    lit_tree, schm_tree = prepare_fill(tree)
    in_tree, static_tree = fill_input((lit_tree, schm_tree, control.stop_gradient(model)))
    out = fill_p.bind(in_tree, static_tree=static_tree)
    return merge_fill(tree, out)


def prepare_fill(tree: Tree) -> TreePair:
    schm_tree, lit_tree = utils.partition(is_schema, tree, is_leaf=is_schema)
    return lit_tree, schm_tree


def merge_fill(tree: Tree, generated: Tree) -> Tree:
    def generated_field(node, value):
        return value if is_schema(node) else node

    return utils.tree.map(generated_field, tree, generated, is_leaf=is_schema)


def fill_input(in_tree: Tree) -> TreePair:
    lit_tree, schm_tree, model = in_tree

    def is_dynamic(node):
        return not is_schema(node)

    dynamic_tree, static_tree = utils.partition(is_dynamic, schm_tree)
    return (lit_tree, dynamic_tree, model), static_tree


def reconstruct_schema(dynamic_tree: Tree, static_tree: Tree) -> Tree:
    def merge_field(static, dynamic):
        return dynamic if static is None else static

    def is_none(node):
        return node is None

    return utils.tree.map(merge_field, static_tree, dynamic_tree, is_leaf=is_none)


def impl_fill(in_tree: Tree, /, *, static_tree: Tree) -> Tree:
    lit_tree, dynamic_tree, model = in_tree
    schema = reconstruct_schema(dynamic_tree, static_tree)
    return schema_response((literal_content(lit_tree), model), schema=schema)


async def aimpl_fill(in_tree: Tree, /, *, static_tree: Tree) -> Tree:
    lit_tree, dynamic_tree, model = in_tree
    schema = reconstruct_schema(dynamic_tree, static_tree)
    return await aschema_response((literal_content(lit_tree), model), schema=schema)


def abstract_fill(in_tree: Tree, /, *, static_tree: Tree) -> Tree:
    lit_tree, dynamic_tree, model = in_tree
    schema = reconstruct_schema(dynamic_tree, static_tree)
    aval = core.avalof("")
    assert type(model) in (str, type(aval)), f"Expected string model: {model!r}"

    def check_literal(value):
        aval = value if isinstance(value, core.AVal) else core.avalof(value)
        aval_to_schema(aval)

    def check_description(value):
        aval = value if isinstance(value, core.AVal) else core.avalof(value)
        if aval != core.avalof(""):
            raise TypeError(f"desc must be a string, got {value!r}")

    utils.tree.map(check_literal, lit_tree)
    utils.tree.map(check_description, dynamic_tree)
    return schema_abstract_tree(schema)


def schema_description(node: Spec) -> str | None:
    return node.desc


def fill_pushforward_request(in_tree: Tree, /, *, static_tree: Tree) -> TreePair:
    (p_lit_tree, p_dynamic_tree, p_model), (t_lit_tree, t_dynamic_tree, _) = core.materialize_zeros(
        in_tree
    )
    schema = reconstruct_schema(p_dynamic_tree, static_tree)
    t_schema = reconstruct_schema(t_dynamic_tree, static_tree)
    if utils.tree.structure(p_lit_tree) != utils.tree.structure(t_lit_tree):
        raise ValueError("Primal and tangent literals must have identical pytree specs")
    if utils.tree.structure(p_dynamic_tree) != utils.tree.structure(t_dynamic_tree):
        raise ValueError("Primal and tangent schemas must have identical pytree specs")

    def check_tangent(value, tangent):
        expected = core.tangent_s.map(core.avalof(value))
        if core.avalof(tangent) != expected:
            raise TypeError(f"Expected {expected!r} tangent, got {tangent!r}")

    utils.tree.map(check_tangent, (p_lit_tree, p_dynamic_tree), (t_lit_tree, t_dynamic_tree))

    prompt = PUSH_PROMPT.format(
        input=literal_content(p_lit_tree),
        in_tangent=literal_content(t_lit_tree),
    )
    desc_tree = utils.tree.map(schema_description, t_schema, is_leaf=is_schema)
    context = dict(
        instruction=PUSH_SYSTEM_PROMPT,
        request=prompt,
        output_schema=jsonlib.dumps(describe(schema), allow_nan=False),
        desc_change=jsonlib.dumps(json_value(desc_tree), allow_nan=False),
    )

    def tangent_field(x):
        if not is_schema(x):
            return x
        return aval_to_schema(core.tangent_s.map(core.avalof(x)))

    t_schema = utils.tree.map(tangent_field, schema, is_leaf=is_schema)
    return (context, p_model), t_schema


def pushforward_fill(in_tree: Tree, /, *, static_tree: Tree) -> TreePair:
    p_in, _ = in_tree
    (context, model), t_schema = fill_pushforward_request(in_tree, static_tree=static_tree)
    p_out = fill_p.bind(p_in, static_tree=static_tree)
    t_out = fill_context(context, model, schema=t_schema)
    return p_out, t_out


async def apush_fill(in_tree: Tree, /, *, static_tree: Tree) -> TreePair:
    p_in, _ = in_tree
    (context, model), t_schema = fill_pushforward_request(in_tree, static_tree=static_tree)
    p_ir = stage.trace(ft.partial(fill_p.bind, static_tree=static_tree))(p_in)
    t_ir = stage.trace(ft.partial(fill_context, schema=t_schema))(context, model)
    return await order.fanout_p.abind([(p_in,), (context, model)], irs=[p_ir, t_ir])


def fill_pullback_request(in_tree: Tree, /, *, static_tree: Tree) -> TreePair | None:
    residuals, out_cotangent = in_tree
    lit_tree, dynamic_tree, model, out = residuals
    schema = reconstruct_schema(dynamic_tree, static_tree)

    def check_cotangent(p_leaf, c_leaf):
        aval = core.cotangent_s.map(core.avalof(p_leaf))
        if core.avalof(c_leaf) != aval:
            raise TypeError(f"Expected {aval!r} cotangent, got {c_leaf!r}")

    utils.tree.map(check_cotangent, out, out_cotangent)

    if all(isinstance(x, core.Zero) for x in utils.tree.leaves(out_cotangent)):
        return None

    def to_schema(x):
        return aval_to_schema(core.avalof(x))

    context_schema = utils.tree.map(to_schema, lit_tree)

    def to_cotangent_schema(x):
        if not is_schema(x):
            return x
        return aval_to_schema(core.cotangent_s.map(core.avalof(x)))

    cotangent_schema = utils.tree.map(to_cotangent_schema, schema, is_leaf=is_schema)
    out_cotangent = core.materialize_zeros(out_cotangent)
    desc_tree = utils.tree.map(schema_description, schema, is_leaf=is_schema)
    desc_schema = utils.tree.map(to_schema, desc_tree)
    prompt = GRAD_PROMPT.format(
        input=schema_content((lit_tree, model, desc_tree), (context_schema, Str(), desc_schema)),
        output=schema_content(out, schema),
        out_cotangent=schema_content(out_cotangent, cotangent_schema),
    )

    context = dict(instruction=GRAD_SYSTEM_PROMPT, request=prompt)
    in_schema = feedback_schema((lit_tree, model, desc_tree))
    return (context, model), in_schema


def pullback_bwd_fill(in_tree: Tree, /, *, static_tree: Tree) -> Tree:
    def zero_input(x):
        return core.Zero(core.cotangent_s.map(core.avalof(x)))

    request = fill_pullback_request(in_tree, static_tree=static_tree)
    if request is None:
        (lit_tree, dynamic_tree, model, _), _ = in_tree
        return utils.tree.map(zero_input, (lit_tree, dynamic_tree, model))
    (context, model), in_schema = request
    feedback, model_feedback, desc_feedback = fill_context(context, model, schema=in_schema)
    (_, dynamic_tree, _, _), _ = in_tree
    dynamic_feedback = utils.tree.map(
        cotangent_schema, dynamic_tree, desc_feedback, is_leaf=is_schema
    )
    return feedback, dynamic_feedback, model_feedback


async def apull_bwd_fill(in_tree: Tree, /, *, static_tree: Tree) -> Tree:
    def zero_input(x):
        return core.Zero(core.cotangent_s.map(core.avalof(x)))

    request = fill_pullback_request(in_tree, static_tree=static_tree)
    if request is None:
        (lit_tree, dynamic_tree, model, _), _ = in_tree
        return utils.tree.map(zero_input, (lit_tree, dynamic_tree, model))
    (context, model), in_schema = request
    feedback, model_feedback, desc_feedback = await afill_context(context, model, schema=in_schema)
    (_, dynamic_tree, _, _), _ = in_tree
    dynamic_feedback = utils.tree.map(
        cotangent_schema, dynamic_tree, desc_feedback, is_leaf=is_schema
    )
    return feedback, dynamic_feedback, model_feedback


def cotangent_schema(node: Spec, feedback: str | None) -> Spec:
    if node.desc is None:
        return node
    return node @ feedback


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
