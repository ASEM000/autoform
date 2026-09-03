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

Docs attach to the thing they describe and are used to guide the generation process.
The same form works for a leaf or for arbitrary nested structures:

    >>> answer = {
    ...     "name": af.Str() @ af.Doc("Subject name."),
    ...     "kind": af.Enum("summary", "definition") @ af.Doc("Answer kind."),
    ...     "score": af.Float(min=0, max=1) @ af.Doc("Confidence score."),
    ... } @ af.Doc("Answer object.")

Any registered pytree can carry the schema:

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
    >>> output = af.lm_schema_call(  # doctest: +SKIP
    ...     msgs,
    ...     model="openai/gpt-5.5",
    ...     schema=schema,
    ... )
    >>> output  # doctest: +SKIP
    Answer(answer=2.0, reasoning="Adding 1 and 1 gives 2.")

"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any, NoReturn

from optree import GetAttrEntry

import autoform.utils as utils

__all__ = ["Bool", "Doc", "Enum", "Float", "Int", "Str", "make_json_schema_and_parser"]

# ==================================================================================================
# TYPES
# ==================================================================================================


type JsonSchema = dict[str, Any]
type Parser = Callable[[Any], Any]
type SchemaRule = Callable[[Any], JsonSchema | None]
type ValidRule = Callable[[Any, Any], Any]


# ==================================================================================================
# USER SCHEMA NODES
# ==================================================================================================


def slotted_values(node: Any) -> tuple[Any, ...]:
    return tuple(getattr(node, name) for name in type(node).__slots__)


class Spec(Hashable):
    __slots__ = []

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in type(self).__slots__)
        return f"{type(self).__name__}({fields})"


class Str(Spec):
    """String schema node with optional length and pattern constraints.

    Use this node in schema trees passed to :func:`autoform.lm_schema_call`.

    Args:
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
        min: int | None = None,
        max: int | None = None,
        pattern: str | None = None,
    ) -> None:
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

    Use this node in schema trees passed to :func:`autoform.lm_schema_call`.

    Args:
        min: Optional minimum value.
        max: Optional maximum value.

    Example:
        >>> import autoform as af
        >>> count = af.Int(min=0, max=10)
    """

    __slots__ = ["min", "max"]

    def __init__(self, *, min: int | None = None, max: int | None = None) -> None:
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

    Use this node in schema trees passed to :func:`autoform.lm_schema_call`.

    Args:
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
        min: int | float | None = None,
        max: int | float | None = None,
    ) -> None:
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

    Use this node in schema trees passed to :func:`autoform.lm_schema_call`.

    Example:
        >>> import autoform as af
        >>> ok = af.Bool()
    """

    __slots__ = []


class Enum(Spec):
    """Enum schema node with a fixed set of allowed values.

    Use this node in schema trees passed to :func:`autoform.lm_schema_call`.

    Args:
        *values: Allowed values. Values must be non-empty, share one type, and
            be JSON scalar values.

    Example:
        >>> import autoform as af
        >>> kind = af.Enum("summary", "definition")
    """

    __slots__ = ["values"]

    def __init__(self, *values: Any) -> None:
        if not values:
            raise TypeError("Enum must have at least one value")
        value_types = {type(value) for value in values}
        if len(value_types) != 1:
            raise TypeError(f"Enum values must share one type, got {value_types!r}")
        if next(iter(value_types)) not in json_type:
            raise TypeError("Enum values must be str, int, float, or bool")
        self.values = values

    def __contains__(self, value: Any) -> bool:
        return type(value) is type(self.values[0]) and value in self.values


class Docd[T]:
    __slots__ = ["value", "text"]

    def __init__(self, value: T, text: str, /) -> None:
        self.value = value
        assert type(text) is str, f"description must be a string, got {text!r}"
        self.text = text

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))

    def __repr__(self) -> str:
        return f"Docd({self.value!r}, text={self.text!r})"


class Doc:
    """Description node for attaching JSON Schema descriptions.

    Use this node in schema trees passed to :func:`autoform.lm_schema_call`.

    Args:
        text: Description text.

    Example:
        >>> import autoform as af
        >>> name = af.Str() @ af.Doc("Subject name.")
    """

    __slots__ = ["text"]

    def __init__(self, text: str, /) -> None:
        if not isinstance(text, str):
            raise TypeError(f"description must be a string, got {text!r}")
        self.text = text

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))

    def __rmatmul__[T](self, value: T) -> Docd[T]:
        return Docd(value, self.text)

    def __repr__(self) -> str:
        return f"Doc({self.text!r})"


utils.tree.register_node(
    Docd,
    lambda node: ((node.value,), node.text, ("value",)),
    lambda text, children: Docd(children[0], text),
    path_entry_type=GetAttrEntry,
)


# ==================================================================================================
# JSON
# ==================================================================================================

json_type = {str: "string", int: "integer", float: "number", bool: "boolean"}

schema_rules: dict[type[Any], SchemaRule] = {}


def string_schema(s: Str) -> JsonSchema:
    schema: JsonSchema = {"type": "string"}
    if s.min is not None:
        schema["minLength"] = s.min
    if s.max is not None:
        schema["maxLength"] = s.max
    if s.pattern is not None:
        schema["pattern"] = s.pattern
    return schema


def integer_schema(s: Int) -> JsonSchema:
    schema: JsonSchema = {"type": "integer"}
    if s.min is not None:
        schema["minimum"] = s.min
    if s.max is not None:
        schema["maximum"] = s.max
    return schema


def number_schema(s: Float) -> JsonSchema:
    schema: JsonSchema = {"type": "number"}
    if s.min is not None:
        schema["minimum"] = s.min
    if s.max is not None:
        schema["maximum"] = s.max
    return schema


def docd_schema(docd: Docd[Any]) -> JsonSchema | None:
    if (schema := schema_build(docd.value)) is None:
        return None
    return schema | {"description": docd.text}


schema_rules[Str] = string_schema
schema_rules[Int] = integer_schema
schema_rules[Float] = number_schema
schema_rules[Bool] = lambda _: {"type": "boolean"}
schema_rules[Enum] = lambda s: {"type": json_type[type(s.values[0])], "enum": list(s.values)}
schema_rules[Docd] = docd_schema


def is_static_schema_leaf(schema_node: Any) -> bool:
    spec_type = type(schema_node) in schema_rules
    leaf_type = utils.tree.is_leaf(schema_node)
    return not spec_type and leaf_type


def schema_build(node: Any) -> JsonSchema | None:
    if rule := schema_rules.get(type(node)):
        return rule(node)
    if is_static_schema_leaf(node):
        return None

    children, spec = utils.tree.flatten(node, is_leaf=lambda node: id(node) != id(node))
    properties = OrderedDict()
    for entry, child in zip(spec.entries(), children, strict=True):
        property_name = str(entry)
        if (child_schema := schema_build(child)) is not None:
            if property_name in properties:
                raise TypeError(f"Duplicate object entries {(property_name,)!r}")
            properties[property_name] = child_schema

    if not properties:
        return None

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def error(expected: Any) -> NoReturn:
    raise ValueError(f"Expected {expected}")


valid_rules: dict[type[Any], ValidRule] = {}


def string_value(s: Str, value: str) -> str:
    if type(value) is not str:
        error("string")
    if s.min is not None and len(value) < s.min:
        error(f"string with length >= {s.min}")
    if s.max is not None and len(value) > s.max:
        error(f"string with length <= {s.max}")
    if s.pattern is not None and not re.search(s.pattern, value):
        error(f"string matching {s.pattern!r}")
    return value


def integer_value(s: Int, value: int) -> int:
    if type(value) is not int:
        error("integer")
    if s.min is not None and value < s.min:
        error(f"integer >= {s.min}")
    if s.max is not None and value > s.max:
        error(f"integer <= {s.max}")
    return value


def number_value(s: Float, value: int | float) -> float:
    if type(value) not in (int, float):
        error("number")
    if s.min is not None and value < s.min:
        error(f"number >= {s.min}")
    if s.max is not None and value > s.max:
        error(f"number <= {s.max}")
    return float(value)


valid_rules[Str] = string_value
valid_rules[Int] = integer_value
valid_rules[Float] = number_value
valid_rules[Bool] = lambda _, value: value if type(value) is bool else error("boolean")
valid_rules[Enum] = lambda s, value: value if value in s else error(f"one of {s.values!r}")
valid_rules[Docd] = lambda s, value: tree_parse(s.value, value)


def tree_parse(schema: Any, value: Any) -> Any:
    if rule := valid_rules.get(type(schema)):
        return rule(schema, value)
    if is_static_schema_leaf(schema):
        return schema

    flat_schemas, spec_schema = utils.tree.flatten(schema, is_leaf=lambda x: id(x) != id(schema))
    flat_values, spec_value = utils.tree.flatten(value, is_leaf=lambda x: id(x) != id(value))
    schema_keys = [str(entry) for entry in spec_schema.entries()]
    emitted = [schema_build(child) is not None for child in flat_schemas]
    expected_keys = [k for k, e in zip(schema_keys, emitted, strict=True) if e]
    value_keys = [str(entry) for entry in spec_value.entries()]

    if len(expected_keys) != len(value_keys) or set(expected_keys) != set(value_keys):
        raise ValueError(f"Key mismatch: expected entries {expected_keys!r}, got {value_keys!r}")

    out_pos = {k: i for i, k in enumerate(value_keys)}
    values = (
        tree_parse(child, flat_values[out_pos[key]] if emit else None)
        for key, child, emit in zip(schema_keys, flat_schemas, emitted, strict=True)
    )
    return spec_schema.unflatten(values)


def make_json_schema_and_parser(schema: Any) -> tuple[JsonSchema | None, Parser]:
    def parse(out_json: Any) -> Any:
        return tree_parse(schema, out_json)

    return schema_build(schema), parse
