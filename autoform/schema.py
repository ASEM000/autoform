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

"""Schema DSL"""

from __future__ import annotations

import functools as ft
import re
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from typing import Any, NoReturn

import optree

from autoform.utils import treelib

__all__ = ["Bool", "Doc", "Enum", "Float", "Int", "Str", "build"]

json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}

# ==================================================================================================
# TYPES
# ==================================================================================================


type Schema = Any
type SchemaRule = Callable[[Any], dict[str, Any]]
type ValueRule = Callable[[Any, Any, optree.PyTreeAccessor, str], Any]
type Path = tuple[Any, ...]
type ObjectProperties = OrderedDict[str, Any]
type SchemaLeaf = Spec
type SchemaKey = tuple[tuple[SchemaLeaf, ...], optree.PyTreeSpec]
type TreeCache = tuple[
    optree.PyTreeSpec,
    optree.PyTreeSpec,
    tuple[tuple[SchemaLeaf, ValueRule, optree.PyTreeAccessor], ...],
]


# ==================================================================================================
# USER SCHEMA NODES
# ==================================================================================================


class Spec:
    __slots__ = []


class Scalar[T](Spec):
    __slots__ = []

    def __init_subclass__(cls, *, schema: str):
        super().__init_subclass__()
        cls.schema = schema


class Str(Scalar[str], schema="string"):
    """ "String schema node with optional length and pattern constraints.

    Args:
        min: Optional minimum length of the string.
        max: Optional maximum length of the string.
        pattern: Optional regular expression pattern that the string must match.
    """

    __slots__ = ["min", "max", "pattern"]

    def __init__(
        self,
        *,
        min: int | None = None,
        max: int | None = None,
        pattern: str | None = None,
    ):
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


class Int(Scalar[int], schema="integer"):
    """Integer schema node with optional range constraints.

    Args:
        min: Optional minimum value.
        max: Optional maximum value.
    """

    __slots__ = ["min", "max"]

    def __init__(self, *, min: int | None = None, max: int | None = None):
        if min is not None and type(min) is not int:
            raise TypeError(f"min must be an int, got {min!r}")
        if max is not None and type(max) is not int:
            raise TypeError(f"max must be an int, got {max!r}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min must be <= max, got min={min!r}, max={max!r}")
        self.min = min
        self.max = max


class Float(Scalar[float], schema="number"):
    """Number schema node with optional range constraints.

    Args:
        min: Optional minimum value.
        max: Optional maximum value.
    """

    __slots__ = ["min", "max"]

    def __init__(self, *, min: int | float | None = None, max: int | float | None = None):
        if min is not None and type(min) not in (int, float):
            raise TypeError(f"min must be a number, got {min!r}")
        if max is not None and type(max) not in (int, float):
            raise TypeError(f"max must be a number, got {max!r}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min must be <= max, got min={min!r}, max={max!r}")
        self.min = min
        self.max = max


class Bool(Scalar[bool], schema="boolean"):
    __slots__ = []


class Enum(Spec):
    __slots__ = ["values"]

    def __init__(self, *values: Any):
        if not values:
            raise TypeError("Enum must have at least one value")
        value_types = {type(value) for value in values}
        if len(value_types) != 1:
            raise TypeError(f"Enum values must share one type, got {value_types!r}")
        if next(iter(value_types)) not in json_types:
            raise TypeError("Enum values must be str, int, float, or bool")
        self.values = values

    def __contains__(self, value: Any) -> bool:
        return type(value) is type(self.values[0]) and value in self.values

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and isinstance(other, Enum) and self.values == other.values

    def __hash__(self) -> int:
        return hash((type(self), self.values))

    def __repr__(self) -> str:
        return f"Enum{self.values!r}"


class Documented[T]:
    __slots__ = ["value", "text"]

    def __init__(self, value: T, text: str):
        self.value = value
        self.text = text

    def __repr__(self) -> str:
        return f"{self.value!r} @ {self.text!r}"


class Doc:
    __slots__ = ["text"]

    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError(f"description must be a string, got {text!r}")
        self.text = text

    def __rmatmul__(self, value: Any):
        return Documented(value, self.text)

    def __repr__(self) -> str:
        return f"Doc({self.text!r})"


treelib.register_node(
    Documented,
    lambda node: ((node.value,), node.text, ("value",)),
    lambda text, children: Documented(children[0], text),
    path_entry_type=optree.GetAttrEntry,
)


schema_message = "Expected a pytree containing Str(), Int(), Float(), Bool(), Enum(...)"


# ==================================================================================================
# SCHEMA RULES
# ==================================================================================================


schema_rules: dict[Any, SchemaRule] = {}
schema_node_rules = defaultdict(lambda: default_schema_node)
value_node_rules = defaultdict(lambda: lambda x: x)


def doc_schema(node: Documented) -> tuple[Path, Any]:
    path, value = node.value
    *path, _ = path
    return tuple(path), value | {"description": node.text}


def string_schema(s: Str) -> dict[str, Any]:
    schema = {"type": s.schema}
    if s.min is not None:
        schema["minLength"] = s.min
    if s.max is not None:
        schema["maxLength"] = s.max
    if s.pattern is not None:
        schema["pattern"] = s.pattern
    return schema


def number_schema(s: Int | Float) -> dict[str, Any]:
    schema = {"type": s.schema}
    if s.min is not None:
        schema["minimum"] = s.min
    if s.max is not None:
        schema["maximum"] = s.max
    return schema


schema_rules[Str] = string_schema
schema_rules[Int] = number_schema
schema_rules[Float] = number_schema
schema_rules[Bool] = lambda s: {"type": s.schema}
schema_rules[Enum] = lambda s: {"type": json_types[type(s.values[0])], "enum": list(s.values)}
schema_node_rules[Documented] = doc_schema
value_node_rules[Documented] = lambda node: node.value

# ==================================================================================================
# PARSE RULES
# ==================================================================================================


def error(accessor: Any, path: str, expected: Any) -> NoReturn:
    raise ValueError(f"{accessor.codify(path)}: expected {getattr(expected, 'schema', expected)}")


value_rules: dict[object, ValueRule] = {}
transport_node_rules = defaultdict(lambda: default_transport_node)


def string_value(s: Str, value: Any, accessor: optree.PyTreeAccessor, path: str) -> str:
    if type(value) is not str:
        error(accessor, path, s)
    if s.min is not None and len(value) < s.min:
        error(accessor, path, f"string with length >= {s.min}")
    if s.max is not None and len(value) > s.max:
        error(accessor, path, f"string with length <= {s.max}")
    if s.pattern is not None and not re.search(s.pattern, value):
        error(accessor, path, f"string matching {s.pattern!r}")
    return value


def integer_value(s: Int, value: Any, accessor: optree.PyTreeAccessor, path: str) -> int:
    if type(value) is not int:
        error(accessor, path, s)
    if s.min is not None and value < s.min:
        error(accessor, path, f"integer >= {s.min}")
    if s.max is not None and value > s.max:
        error(accessor, path, f"integer <= {s.max}")
    return value


def number_value(s: Float, value: Any, accessor: optree.PyTreeAccessor, path: str) -> float:
    if type(value) not in (int, float):
        error(accessor, path, s)
    if s.min is not None and value < s.min:
        error(accessor, path, f"number >= {s.min}")
    if s.max is not None and value > s.max:
        error(accessor, path, f"number <= {s.max}")
    return float(value)


value_rules[Str] = string_value
value_rules[Int] = integer_value
value_rules[Float] = number_value
value_rules[Bool] = lambda s, v, a, p: v if type(v) is bool else error(a, p, s)
value_rules[Enum] = lambda s, v, a, p: v if v in s else error(a, p, f"one of {s.values!r}")


def doc_transport(node: Documented) -> tuple[Path, Any]:
    path, value = node.value
    *path, _ = path
    return tuple(path), value


transport_node_rules[Documented] = doc_transport

# ==================================================================================================
# PUBLIC ENTRYPOINTS
# ==================================================================================================


def build(schema: Schema) -> tuple[dict[str, Any], Callable[[Any], Any]]:
    leaves, spec = treelib.flatten(schema, is_leaf=is_schema_spec, none_is_leaf=True)
    key = (tuple(leaves), spec)
    return build_schema_for_key(key), parser_for_key(key)


# ==================================================================================================
# SCHEMA TREE
# ==================================================================================================


@ft.lru_cache(maxsize=256)
def build_schema_for_key(key: SchemaKey) -> dict[str, Any]:
    leaves, spec = key
    _, schema = spec.traverse(
        zip(spec.paths(), spec.accessors(), leaves, strict=True),
        build_schema_node,
        build_schema_leaf,
    )
    return schema


def build_schema_node(node: Any):
    return schema_node_rules[type(node)](node)


def build_schema_leaf(node: Any) -> Any:
    path, accessor, leaf = node
    if not is_schema_spec(leaf):
        raise TypeError(f"{accessor.codify('$')}: {schema_message}, got {leaf!r}")
    return path, schema_rules[type(leaf)](leaf)


def default_schema_node(node: Any):
    return object_schema(node)


def object_schema(node: Any):
    children, _, _, _ = treelib.flatten_one_level(node)
    (child_path, _), *_ = children
    *path, _ = child_path
    properties = object_properties(children)
    return tuple(path), {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def object_properties(children: Iterable[tuple[Path, Any]]) -> ObjectProperties:
    properties = OrderedDict()
    for path, value in children:
        *_, entry = path
        property_name = str(entry)
        if property_name in properties:
            raise TypeError(f"{schema_message}; duplicate object entries {(property_name,)!r}")
        properties[property_name] = value
    return properties


def is_schema_spec(node: Any) -> bool:
    return type(node) in schema_rules


# ==================================================================================================
# VALUE BUILD
# ==================================================================================================


@ft.lru_cache(maxsize=256)
def parser_for_key(key: SchemaKey) -> Callable[[Any], Any]:
    cache = tree_cache_for_key(key)
    return lambda value: build_from_cache(cache, value)


@ft.lru_cache(maxsize=256)
def tree_cache_for_key(key: SchemaKey) -> TreeCache:
    leaves, spec = key
    _, transport = spec.traverse(
        zip(spec.paths(), spec.accessors(), leaves, strict=True),
        transport_node,
        transport_leaf,
    )
    transport_leaves, transport_spec = treelib.flatten(transport, is_leaf=is_schema_spec)
    marker = object()
    value_tree = spec.traverse(
        leaves,
        lambda node: value_node_rules[type(node)](node),
        lambda _: marker,
    )
    return (
        transport_spec,
        treelib.structure(value_tree, is_leaf=lambda node: node is marker),
        tuple(
            (leaf, value_rules[type(leaf)], accessor)
            for leaf, accessor in zip(transport_leaves, transport_spec.accessors(), strict=True)
        ),
    )


def transport_node(node: Any):
    return transport_node_rules[type(node)](node)


def transport_leaf(node: Any) -> tuple[Path, Any]:
    path, accessor, leaf = node
    if not is_schema_spec(leaf):
        raise TypeError(f"{accessor.codify('$')}: {schema_message}, got {leaf!r}")
    return path, leaf


def default_transport_node(node: Any):
    return transport_object(node)


def transport_object(node: Any) -> tuple[Path, ObjectProperties]:
    children, _, _, _ = treelib.flatten_one_level(node)
    (path, _), *_ = children
    *path, _ = path
    return tuple(path), object_properties(children)


def build_from_cache(cache: TreeCache, value: Any) -> Any:
    transport_spec, value_spec, rules = cache
    try:
        leaves = transport_spec.flatten_up_to(value)
    except ValueError:
        _, spec = treelib.flatten(value)
        raise ValueError(f"$: expected {transport_spec}, got {spec}") from None
    output = (rule(l, v, a, "$") for v, (l, rule, a) in zip(leaves, rules, strict=True))
    return value_spec.unflatten(output)
