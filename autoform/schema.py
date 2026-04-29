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
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from typing import Any, NoReturn

import optree

from autoform.utils import treelib

__all__ = ["Boolean", "Doc", "Enum", "Float", "Integer", "String", "build_schema", "build_value"]

json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}

# ==================================================================================================
# TYPES
# ==================================================================================================


type Schema = Any
type SchemaRule = Callable[[Any], dict[str, Any]]
type ValueRule = Callable[[Any, Any, optree.PyTreeAccessor, str], Any]
type Path = tuple[Any, ...]
type ObjectProperties = OrderedDict[str, Any]
type SchemaLeaf = ScalarSpec[Any] | EnumSpec[Any]
type SchemaKey = tuple[tuple[SchemaLeaf, ...], optree.PyTreeSpec]
type ValueCache = tuple[
    optree.PyTreeSpec,
    optree.PyTreeSpec,
    tuple[tuple[SchemaLeaf, ValueRule, optree.PyTreeAccessor], ...],
]

# ==================================================================================================
# USER SCHEMA NODES
# ==================================================================================================


class SchemaSpec:
    __slots__ = []


class ScalarSpec[T](SchemaSpec):
    __slots__ = ["name", "schema"]

    def __new__(cls, name: str, schema: str):
        self = object.__new__(cls)
        self.name = name
        self.schema = schema
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"use {self.name}, not {self.name}()")

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and isinstance(other, ScalarSpec)
            and self.name == other.name
            and self.schema == other.schema
        )

    def __hash__(self) -> int:
        return hash((type(self), self.name, self.schema))

    def __repr__(self) -> str:
        return self.name


class StringSpec(ScalarSpec[str]):
    __slots__ = []


class IntegerSpec(ScalarSpec[int]):
    __slots__ = []


class FloatSpec(ScalarSpec[float]):
    __slots__ = []


class BooleanSpec(ScalarSpec[bool]):
    __slots__ = []


class EnumSpec[T](SchemaSpec):
    __slots__ = ["values"]

    def __new__(cls, values: tuple[T, ...]):
        if not values:
            raise TypeError("Enum must have at least one value")
        value_types = {type(value) for value in values}
        if len(value_types) != 1:
            raise TypeError(f"Enum values must share one type, got {value_types!r}")
        if next(iter(value_types)) not in json_types:
            raise TypeError("Enum values must be str, int, float, or bool")
        self = object.__new__(cls)
        self.values = values
        return self

    def __contains__(self, value: Any) -> bool:
        return type(value) is type(self.values[0]) and value in self.values

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and isinstance(other, EnumSpec)
            and self.values == other.values
        )

    def __hash__(self) -> int:
        return hash((type(self), self.values))

    def __repr__(self) -> str:
        return f"Enum{self.values!r}"


class Enum:
    __slots__ = []

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("use Enum[...], not Enum(...)")

    def __class_getitem__(cls, values: Any):
        return EnumSpec(values if isinstance(values, tuple) else (values,))


class DocNode[T]:
    __slots__ = ["value", "text"]

    def __new__(cls, value: T, text: str):
        if not isinstance(text, str):
            raise TypeError(f"description must be a string, got {text!r}")
        self = object.__new__(cls)
        self.value = value
        self.text = text
        return self

    def __init__(self, value: T, text: str): ...

    def __matmul__(self, doc: Doc):
        if not isinstance(doc, Doc):
            raise TypeError(f"description must be Doc(...), got {doc!r}")
        return doc.__rmatmul__(self)

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and isinstance(other, DocNode)
            and self.value == other.value
            and self.text == other.text
        )

    def __hash__(self) -> int:
        return hash((type(self), self.value, self.text))

    def __repr__(self) -> str:
        return f"{self.value!r} @ {self.text!r}"


class Doc:
    __slots__ = ["text"]

    def __new__(cls, text: str):
        if not isinstance(text, str):
            raise TypeError(f"description must be a string, got {text!r}")
        self = object.__new__(cls)
        self.text = text
        return self

    def __rmatmul__(self, value: Any):
        return DocNode(value, self.text)

    def __repr__(self) -> str:
        return f"Doc({self.text!r})"


treelib.register_node(
    DocNode,
    lambda node: ((node.value,), node.text, ("doc",)),
    lambda text, children: DocNode(children[0], text),
    path_entry_type=optree.GetAttrEntry,
)


String = StringSpec(name="String", schema="string")
Integer = IntegerSpec(name="Integer", schema="integer")
Float = FloatSpec(name="Float", schema="number")
Boolean = BooleanSpec(name="Boolean", schema="boolean")

schema_message = (
    "expected pytree containing String, Integer, Float, Boolean, Enum[...], "
    "or schema nodes with @ Doc(...)"
)


# ==================================================================================================
# SCHEMA RULES
# ==================================================================================================


schema_rules: dict[Any, SchemaRule] = {}
schema_node_rules = defaultdict(lambda: default_schema_node)
value_node_rules = defaultdict(lambda: lambda x: x)


def enum_schema(s: EnumSpec) -> dict[str, Any]:
    return {"type": json_types[type(s.values[0])], "enum": list(s.values)}


def doc_schema(node: DocNode) -> tuple[Path, Any]:
    path, value = node.value
    *path, _ = path
    return tuple(path), value | {"description": node.text}


schema_rules[StringSpec] = lambda s: {"type": s.schema}
schema_rules[IntegerSpec] = lambda s: {"type": s.schema}
schema_rules[FloatSpec] = lambda s: {"type": s.schema}
schema_rules[BooleanSpec] = lambda s: {"type": s.schema}
schema_rules[EnumSpec] = enum_schema
schema_node_rules[DocNode] = doc_schema
value_node_rules[DocNode] = lambda node: node.value

# ==================================================================================================
# PARSE RULES
# ==================================================================================================


def error(accessor: Any, path: str, expected: Any) -> NoReturn:
    raise ValueError(f"{accessor.codify(path)}: expected {getattr(expected, 'schema', expected)}")


value_rules: dict[object, ValueRule] = {}
transport_node_rules = defaultdict(lambda: default_transport_node)


value_rules[StringSpec] = lambda s, v, a, p: v if type(v) is str else error(a, p, s)
value_rules[IntegerSpec] = lambda s, v, a, p: v if type(v) is int else error(a, p, s)
value_rules[FloatSpec] = lambda s, v, a, p: float(v) if type(v) in (int, float) else error(a, p, s)
value_rules[BooleanSpec] = lambda s, v, a, p: v if type(v) is bool else error(a, p, s)
value_rules[EnumSpec] = lambda s, v, a, p: v if v in s else error(a, p, f"one of {s.values!r}")


def doc_transport(node: DocNode) -> tuple[Path, Any]:
    path, value = node.value
    *path, _ = path
    return tuple(path), value


transport_node_rules[DocNode] = doc_transport

# ==================================================================================================
# PUBLIC ENTRYPOINTS
# ==================================================================================================


def build_schema(schema: Schema) -> dict[str, Any]:
    leaves, spec = treelib.flatten(schema, is_leaf=is_schema_spec, none_is_leaf=True)
    return build_schema_for_key((tuple(leaves), spec))


def build_value(schema: Schema, value: Any, path: str = "$") -> Any:
    leaves, spec = treelib.flatten(schema, is_leaf=is_schema_spec, none_is_leaf=True)
    return build_value_from_cache(value_cache_for_key((tuple(leaves), spec)), value, path)


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
def value_cache_for_key(key: SchemaKey) -> ValueCache:
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


def build_value_from_cache(cache: ValueCache, value: Any, path: str = "$") -> Any:
    transport_spec, value_spec, rules = cache
    try:
        leaves = transport_spec.flatten_up_to(value)
    except ValueError:
        _, spec = treelib.flatten(value)
        raise ValueError(f"{path}: expected {transport_spec}, got {spec}") from None
    output = (rule(l, v, a, path) for v, (l, rule, a) in zip(leaves, rules, strict=True))
    return value_spec.unflatten(output)
