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

"""Schema DSL.

There are two ways to think about structured output.

The first way is type-first. A class describes what should be generated, and the
same class is also the return type:
    class Answer(BaseModel):
        name: str
        score: float

That works, but it is not a great fit for autoform. A type is a recipe, not the
value that flows through the program. Tracing a type means inspecting
annotations and rebuilding the result from that type later.

The second way is instance-first. The schema is already a value with the shape
we want back:
    >>> import autoform as af
    >>> answer = {"name": af.Str(), "score": af.Float(min=0, max=1)}

This fits autoform better. The schema is an ordinary pytree.

Descriptions attach directly to schema nodes and guide generation:
    >>> answer = {
    ...     "name": af.Str(desc="Subject name."),
    ...     "kind": af.Enum("summary", "definition", desc="Answer kind."),
    ...     "score": af.Float(min=0, max=1, desc="Confidence score."),
    ... }

Any registered pytree can carry the schema:
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
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import autoform.core as core
import autoform.utils as utils

__all__ = [
    "Bool",
    "Enum",
    "Float",
    "Int",
    "Str",
    "emit_json_schema",
    "parse_json_value",
]

# ==================================================================================================
# USER SCHEMA NODES
# ==================================================================================================


def slotted_values(node: Any) -> tuple[Any, ...]:
    return tuple(getattr(node, name) for name in (*type(node).__slots__, "desc"))


class Spec(Hashable):
    __slots__ = ["desc"]

    def __init__(self, *, desc: str | None = None) -> None:
        if desc is not None and type(desc) is not str:
            raise TypeError(f"desc must be a string, got {desc!r}")
        self.desc = desc

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        utils.tree.register_node(cls, lambda node: ((), node), lambda node, _: node)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))

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
        >>> name = af.Str(min=1, max=80, pattern=r"^[A-Za-z ]+$")
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
        >>> count = af.Int(min=0, max=10)
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
        >>> score = af.Float(min=0, max=1)
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
        >>> ok = af.Bool()
    """

    __slots__ = []


class Enum(Spec):
    """Enum schema node with a fixed set of allowed values.

    Args:
        desc: Optional generation guidance.
        *values: Allowed values. Values must be non-empty and share one type.

    Example:
        >>> import autoform as af
        >>> kind = af.Enum("summary", "definition")
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
# JSON EMISSION
# ==================================================================================================

type JsonSchema = dict[str, Any]
type EmitJsonSchemaRule = Callable[[Any], JsonSchema]

zip = utils.strict_zip


missing = object()


def json_property_names(entries: Iterable[Any]) -> tuple[str, ...]:
    # NOTE(asem): avoid stringified name collision
    names = []
    used = set()
    for entry in entries:
        name = str(entry)
        while name in used:
            name += "_"
        names.append(name)
        used.add(name)
    return tuple(names)


def emit_string_json_schema(schema: Str) -> JsonSchema:
    json_schema: JsonSchema = dict(type="string")
    if schema.min is not None:
        json_schema["minLength"] = schema.min
    if schema.max is not None:
        json_schema["maxLength"] = schema.max
    if schema.pattern is not None:
        json_schema["pattern"] = schema.pattern
    return json_schema


def emit_integer_json_schema(schema: Int) -> JsonSchema:
    json_schema: JsonSchema = dict(type="integer")
    if schema.min is not None:
        json_schema["minimum"] = schema.min
    if schema.max is not None:
        json_schema["maximum"] = schema.max
    return json_schema


def emit_number_json_schema(schema: Float) -> JsonSchema:
    json_schema: JsonSchema = dict(type="number")
    if schema.min is not None:
        json_schema["minimum"] = schema.min
    if schema.max is not None:
        json_schema["maximum"] = schema.max
    return json_schema


def emit_boolean_json_schema(schema: Bool) -> JsonSchema:
    return dict(type="boolean")


def emit_enum_json_schema(schema: Enum) -> JsonSchema:
    json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if (value_type := type(schema.values[0])) not in json_types:
        raise TypeError("Enum values must be str, int, float, or bool")
    if value_type is float and not all(math.isfinite(value) for value in schema.values):
        raise ValueError("Enum values must be finite")
    return dict(type=json_types[value_type], enum=list(schema.values))


emit_json_schema_rules: dict[type[Spec], EmitJsonSchemaRule] = {}
emit_json_schema_rules[Str] = emit_string_json_schema
emit_json_schema_rules[Int] = emit_integer_json_schema
emit_json_schema_rules[Float] = emit_number_json_schema
emit_json_schema_rules[Bool] = emit_boolean_json_schema
emit_json_schema_rules[Enum] = emit_enum_json_schema


def is_schema(node: Any) -> bool:
    return isinstance(node, Spec)


def partition_schema(schema: Any) -> tuple[Any, Any]:
    lhs = utils.tree.map(lambda x: missing if is_schema(x) else x, schema, is_leaf=is_schema)
    rhs = utils.tree.map(lambda x: x if is_schema(x) else missing, schema, is_leaf=is_schema)
    return lhs, rhs


def emit_json_schema(schema: Any) -> JsonSchema | None:
    # NOTE(asem): key idea here is the only schema nodes qualify for json emission
    # >>> schema = {
    # ...     "name": af.Str(min=1, desc="doc text"),
    # ...     "source": "fixed",
    # ... }
    # >>> emit_json_schema(schema)
    # {
    #   "type": "object",
    #   "properties": {
    #     "name": {
    #       "type": "string",
    #       "minLength": 1,
    #       "description": "doc text"
    #     }
    #   },
    #   "required": [
    #     "name"
    #   ],
    #   "additionalProperties": false
    # }
    # here "fixed" is a literal and excluded from the schema
    _, schema_tree = partition_schema(schema)
    return emit_json_tree(schema_tree)


def emit_json_tree(schema: Any) -> JsonSchema | None:
    if schema is missing:
        return None

    if is_schema(schema):
        json_schema = emit_json_schema_rules[type(schema)](schema)
        if schema.desc is not None:
            json_schema["description"] = schema.desc
        return json_schema

    flat, spec = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))
    property_names = json_property_names(spec.entries())
    properties = {}
    for name, child in zip(property_names, flat):
        if (child_schema := emit_json_tree(child)) is not None:
            properties[name] = child_schema
    if not properties:
        return None

    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


# ==================================================================================================
# JSON PARSING
# ==================================================================================================

type ParseJsonValueRule = Callable[[Any, Any], Any]


def parse_string_json_value(schema: Str, value: Any) -> str:
    if type(value) is not str:
        raise ValueError("Expected string")
    if schema.min is not None and len(value) < schema.min:
        raise ValueError(f"Expected string with length >= {schema.min}")
    if schema.max is not None and len(value) > schema.max:
        raise ValueError(f"Expected string with length <= {schema.max}")
    if schema.pattern is not None and not re.search(schema.pattern, value):
        raise ValueError(f"Expected string matching {schema.pattern!r}")
    return value


def parse_integer_json_value(schema: Int, value: Any) -> int:
    if type(value) is not int:
        raise ValueError("Expected integer")
    if schema.min is not None and value < schema.min:
        raise ValueError(f"Expected integer >= {schema.min}")
    if schema.max is not None and value > schema.max:
        raise ValueError(f"Expected integer <= {schema.max}")
    return value


def parse_number_json_value(schema: Float, value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("Expected number")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Expected finite number")
    if schema.min is not None and value < schema.min:
        raise ValueError(f"Expected number >= {schema.min}")
    if schema.max is not None and value > schema.max:
        raise ValueError(f"Expected number <= {schema.max}")
    return float(value)


def parse_boolean_json_value(schema: Bool, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Expected boolean")
    return value


def parse_enum_json_value(schema: Enum, value: Any) -> Any:
    if value not in schema:
        raise ValueError(f"Expected one of {schema.values!r}")
    return value


parse_json_value_rules: dict[type[Spec], ParseJsonValueRule] = {}
parse_json_value_rules[Str] = parse_string_json_value
parse_json_value_rules[Int] = parse_integer_json_value
parse_json_value_rules[Float] = parse_number_json_value
parse_json_value_rules[Bool] = parse_boolean_json_value
parse_json_value_rules[Enum] = parse_enum_json_value


def parse_json_tree(schema: Any, value: Any) -> Any:
    # NOTE(asem): schema leaves are missing or spec nodes
    if schema is missing:
        return missing

    # NOTE(asem): case 1: in case a parsing rule exists use it.
    if is_schema(schema):
        return parse_json_value_rules[type(schema)](schema, value)

    # NOTE(asem): case 2: flatten the schema one level to rebuild its original container.
    flat, spec = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))

    schema_keys = json_property_names(spec.entries())
    properties = {k: c for k, c in zip(schema_keys, flat) if emit_json_tree(c) is not None}
    if not properties:
        return schema
    expected_spec = utils.tree.structure(properties, is_leaf=lambda x: id(x) != id(properties))
    values = dict(zip(expected_spec.entries(), expected_spec.flatten_up_to(value)))
    children = (parse_json_tree(child, values.get(key)) for key, child in zip(schema_keys, flat))
    return spec.unflatten(children)


def parse_json_value(schema: Any, value: Any) -> Any:
    literal_tree, schema_tree = partition_schema(schema)
    generated_tree = parse_json_tree(schema_tree, value)
    return utils.tree.map(lambda l, r: r if l is missing else l, literal_tree, generated_tree)


# ==================================================================================================
# ABSTRACT
# ==================================================================================================


core.aval_types[Str] = lambda _: core.avalof("")
core.aval_types[Int] = lambda _: core.avalof(0)
core.aval_types[Float] = lambda _: core.avalof(0.0)
core.aval_types[Bool] = lambda _: core.avalof(False)
core.aval_types[Enum] = lambda schema: core.avalof(schema.values[0])
