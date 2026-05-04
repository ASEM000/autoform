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
    ...     model="openai/gpt-5.2",
    ...     schema=schema,
    ... )
    >>> output  # doctest: +SKIP
    Answer(answer=2.0, reasoning="Adding 1 and 1 gives 2.")

"""

from __future__ import annotations

import functools as ft
import re
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any, NoReturn, TypeGuard

from optree import GetAttrEntry, PyTreeAccessor, PyTreeSpec

from autoform.utils import Tree, treelib

__all__ = ["Bool", "Doc", "Enum", "Float", "Int", "Str", "make_json_schema_and_parser"]

json_type = {str: "string", int: "integer", float: "number", bool: "boolean"}

# ==================================================================================================
# TYPES
# ==================================================================================================


type JsonSchema = dict[str, Any]
type Parser[T] = Callable[[Any], T]
type SchemaRule = Callable[[Any], JsonSchema]
type ValidRule = Callable[[Any, Any, PyTreeAccessor, PyTreeAccessor], Any]
type FlattenedSchema = tuple[tuple[Spec, ...], PyTreeSpec]


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


class Scalar[T](Spec):
    __slots__ = []


class Str(Scalar[str]):
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


class Int(Scalar[int]):
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


class Float(Scalar[float]):
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


class Bool(Scalar[bool]):
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


treelib.register_node(
    Docd,
    lambda node: ((node.value,), node.text, ("value",)),
    lambda text, children: Docd(children[0], text),
    path_entry_type=GetAttrEntry,
)


SCHEMA_MSG = "Expected a pytree containing Str(), Int(), Float(), Bool(), Enum(...)"


# ==================================================================================================
# SCHEMA BUILD
# ==================================================================================================


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


schema_rules[Str] = string_schema
schema_rules[Int] = integer_schema
schema_rules[Float] = number_schema
schema_rules[Bool] = lambda s: {"type": "boolean"}
schema_rules[Enum] = lambda s: {"type": json_type[type(s.values[0])], "enum": list(s.values)}
schema_rules[Docd] = lambda s: schema_build(s.value) | {"description": s.text}


def schema_build(tree: Any) -> JsonSchema:
    if rule := schema_rules.get(type(tree)):
        return rule(tree)
    if treelib.is_leaf(tree):
        raise TypeError(f"{SCHEMA_MSG}, got {tree!r}")

    children, spec = treelib.flatten(
        tree,
        is_leaf=lambda node: id(node) != id(tree),
        none_is_leaf=True,
    )
    properties = OrderedDict()
    for entry, child in zip(spec.entries(), children, strict=True):
        property_name = str(entry)
        if property_name in properties:
            raise TypeError(f"{SCHEMA_MSG}; duplicate object entries {(property_name,)!r}")
        properties[property_name] = schema_build(child)

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def is_schema_spec(node: Any) -> TypeGuard[Spec]:
    return isinstance(node, Spec)


# ==================================================================================================
# PARSING
# ==================================================================================================


def error(
    in_acc: PyTreeAccessor,
    out_acc: PyTreeAccessor,
    expected: Any,
) -> NoReturn:
    source, target = out_acc.codify("$"), in_acc.codify("$")
    raise ValueError(f"json {source} -> tree {target}: expected {expected}")


valid_rules: dict[type[Any], ValidRule] = {}


def string_value(
    s: Str,
    value: str,
    in_acc: PyTreeAccessor,
    out_acc: PyTreeAccessor,
) -> str:
    if type(value) is not str:
        error(in_acc, out_acc, "string")
    if s.min is not None and len(value) < s.min:
        error(in_acc, out_acc, f"string with length >= {s.min}")
    if s.max is not None and len(value) > s.max:
        error(in_acc, out_acc, f"string with length <= {s.max}")
    if s.pattern is not None and not re.search(s.pattern, value):
        error(in_acc, out_acc, f"string matching {s.pattern!r}")
    return value


def integer_value(
    s: Int,
    value: int,
    in_acc: PyTreeAccessor,
    out_acc: PyTreeAccessor,
) -> int:
    if type(value) is not int:
        error(in_acc, out_acc, "integer")
    if s.min is not None and value < s.min:
        error(in_acc, out_acc, f"integer >= {s.min}")
    if s.max is not None and value > s.max:
        error(in_acc, out_acc, f"integer <= {s.max}")
    return value


def number_value(
    s: Float,
    value: int | float,
    in_acc: PyTreeAccessor,
    out_acc: PyTreeAccessor,
) -> float:
    if type(value) not in (int, float):
        error(in_acc, out_acc, "number")
    if s.min is not None and value < s.min:
        error(in_acc, out_acc, f"number >= {s.min}")
    if s.max is not None and value > s.max:
        error(in_acc, out_acc, f"number <= {s.max}")
    return float(value)


valid_rules[Str] = string_value
valid_rules[Int] = integer_value
valid_rules[Float] = number_value
valid_rules[Bool] = lambda _, v, i, o: v if type(v) is bool else error(i, o, "boolean")
valid_rules[Enum] = lambda s, v, i, o: v if v in s else error(i, o, f"one of {s.values!r}")
valid_rules[Docd] = lambda s, v, i, o: tree_parse(s.value, v, i, o)


def tree_parse[T: Tree[Spec | Docd[Spec]]](
    schema_tree: T,
    value_tree: Any,
    schema_acc: PyTreeAccessor,
    value_acc: PyTreeAccessor,
) -> T:
    # NOTE(asem): recursively validate value_tree against schema_tree,
    # using the accessors to track the path for error messages.
    # while the value tree is the json output with a dict structure, this code does not assume
    # dicts.
    if rule := valid_rules.get(type(schema_tree)):
        return rule(schema_tree, value_tree, schema_acc, value_acc)
    if treelib.is_leaf(schema_tree):
        raise TypeError(f"{schema_acc.codify('$')}: {SCHEMA_MSG}, got {schema_tree!r}")

    flat_schema, spec_schema = treelib.flatten(
        schema_tree,
        is_leaf=lambda node: id(node) != id(schema_tree),
        none_is_leaf=True,
    )

    flat_value, spec_value = treelib.flatten(
        value_tree,
        is_leaf=lambda node: id(node) != id(value_tree),
        none_is_leaf=True,
    )

    schema_keys = [str(entry) for entry in spec_schema.entries()]
    value_keys = [str(entry) for entry in spec_value.entries()]

    if len(schema_keys) != len(value_keys) or set(schema_keys) != set(value_keys):
        raise ValueError(f"Key mismatch: expected entries {schema_keys!r}, got {value_keys!r}")

    out_pos = {key: i for i, key in enumerate(value_keys)}
    out_accessors = spec_value.accessors()
    values = (
        tree_parse(
            in_child,
            flat_value[out_pos[in_key]],
            schema_acc + in_accessor,
            value_acc + out_accessors[out_pos[in_key]],
        )
        for in_key, in_child, in_accessor in zip(
            schema_keys,
            flat_schema,
            spec_schema.accessors(),
            strict=True,
        )
    )
    return spec_schema.unflatten(values)


# ==================================================================================================
# CACHED BUILD
# ==================================================================================================


@ft.lru_cache(maxsize=256)
def schema_from_flat_and_spec(schema: FlattenedSchema) -> JsonSchema:
    leaves, treespec = schema
    return schema_build(treespec.unflatten(leaves))


@ft.lru_cache(maxsize=256)
def parser_from_flat_and_spec[T](flattened_schema: FlattenedSchema) -> Parser[T]:
    schema_leaves, treespec = flattened_schema
    in_schema = treespec.unflatten(schema_leaves)

    def parse(out_json: Any) -> T:
        return tree_parse(in_schema, out_json, PyTreeAccessor(), PyTreeAccessor())

    return parse


def make_json_schema_and_parser[T](schema: T) -> tuple[JsonSchema, Parser[T]]:
    leaves, treespec = treelib.flatten(schema, is_leaf=is_schema_spec, none_is_leaf=True)
    flat_and_spec = (tuple(leaves), treespec)
    json_schema = schema_from_flat_and_spec(flat_and_spec)
    parser = parser_from_flat_and_spec(flat_and_spec)
    return json_schema, parser
