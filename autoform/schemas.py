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
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from typing import Any, NoReturn, TypeGuard

import optree

from autoform.utils import treelib

__all__ = ["Bool", "Doc", "Enum", "Float", "Int", "Str", "build"]

json_type = {str: "string", int: "integer", float: "number", bool: "boolean"}

# ==================================================================================================
# TYPES
# ==================================================================================================


type Path = tuple[Any, ...]
type JsonSchema = dict[str, Any]
type ObjectProperties = OrderedDict[str, Any]
type Parser = Callable[[Any], Any]
type SchemaRule = Callable[[Any], JsonSchema]
type ValueNodeRule = Callable[[Any], Any]
type ValueRule = Callable[[Any, Any, optree.PyTreeAccessor, str], Any]
type TransportNodeRule = Callable[[Any], tuple[Path, Any]]
type SchemaKey = tuple[tuple[Spec, ...], optree.PyTreeSpec]
type TreeCache = tuple[
    optree.PyTreeSpec,
    optree.PyTreeSpec,
    tuple[tuple[Spec, ValueRule, optree.PyTreeAccessor], ...],
]


# ==================================================================================================
# USER SCHEMA NODES
# ==================================================================================================


def slotted_values(node: Any) -> tuple[Any, ...]:
    return tuple(getattr(node, name) for name in type(node).__slots__)


class Spec:
    __slots__ = []

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))


class Scalar[T](Spec):
    __slots__ = []

    def __init_subclass__(cls, *, schema: str) -> None:
        super().__init_subclass__()
        cls.schema = schema


class Str(Scalar[str], schema="string"):
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


class Int(Scalar[int], schema="integer"):
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


class Float(Scalar[float], schema="number"):
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


class Bool(Scalar[bool], schema="boolean"):
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

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and isinstance(other, Enum) and self.values == other.values

    def __hash__(self) -> int:
        return hash((type(self), self.values))

    def __repr__(self) -> str:
        return f"Enum{self.values!r}"


class Meta[T]:
    __slots__ = ["value", "text"]

    def __init__(self, value: T, text: str) -> None:
        self.value = value
        assert type(text) is str, f"description must be a string, got {text!r}"
        self.text = text

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))

    def __repr__(self) -> str:
        return f"Meta({self.value!r}, text={self.text!r})"


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

    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"description must be a string, got {text!r}")
        self.text = text

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and slotted_values(self) == slotted_values(other)

    def __hash__(self) -> int:
        return hash((type(self), slotted_values(self)))

    def __rmatmul__[T](self, value: T) -> Meta[T]:
        return Meta(value, self.text)

    def __repr__(self) -> str:
        return f"Doc({self.text!r})"


treelib.register_node(
    Meta,
    lambda node: ((node.value,), node.text, ("value",)),
    lambda text, children: Meta(children[0], text),
    path_entry_type=optree.GetAttrEntry,
)


schema_message = "Expected a pytree containing Str(), Int(), Float(), Bool(), Enum(...)"


# ==================================================================================================
# SCHEMA BUILD
# ==================================================================================================


schema_rules: dict[type[Any], SchemaRule] = {}


def document_schema(node: Meta[Any]) -> JsonSchema:
    return schema_build(node.value) | {"description": node.text}


def string_schema(s: Str) -> JsonSchema:
    schema: JsonSchema = {"type": s.schema}
    if s.min is not None:
        schema["minLength"] = s.min
    if s.max is not None:
        schema["maxLength"] = s.max
    if s.pattern is not None:
        schema["pattern"] = s.pattern
    return schema


def number_schema(s: Int | Float) -> JsonSchema:
    schema: JsonSchema = {"type": s.schema}
    if s.min is not None:
        schema["minimum"] = s.min
    if s.max is not None:
        schema["maximum"] = s.max
    return schema


schema_rules[Str] = string_schema
schema_rules[Int] = number_schema
schema_rules[Float] = number_schema
schema_rules[Bool] = lambda s: {"type": s.schema}
schema_rules[Enum] = lambda s: {"type": json_type[type(s.values[0])], "enum": list(s.values)}
schema_rules[Meta] = document_schema


@ft.lru_cache(maxsize=256)
def build_schema_for_key(key: SchemaKey) -> JsonSchema:
    leaves, spec = key
    return schema_build(spec.unflatten(leaves))


def schema_build(tree: Any) -> JsonSchema:
    if rule := schema_rules.get(type(tree)):
        return rule(tree)
    if treelib.is_leaf(tree):
        raise TypeError(f"{schema_message}, got {tree!r}")

    children, _, entries, _ = treelib.flatten_one_level(tree)
    properties = OrderedDict()
    for entry, child in zip(entries, children, strict=True):
        property_name = str(entry)
        if property_name in properties:
            raise TypeError(f"{schema_message}; duplicate object entries {(property_name,)!r}")
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
# VALUE VALIDATION
# ==================================================================================================


def error(accessor: Any, path: str, expected: Any) -> NoReturn:
    raise ValueError(f"{accessor.codify(path)}: expected {getattr(expected, 'schema', expected)}")


value_rules: dict[type[Spec], ValueRule] = {}


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


# ==================================================================================================
# PARSING
# ==================================================================================================


def transport_properties(children: Iterable[tuple[Path, Any]]) -> ObjectProperties:
    properties = OrderedDict()
    for path, value in children:
        *_, entry = path
        property_name = str(entry)
        if property_name in properties:
            raise TypeError(f"{schema_message}; duplicate object entries {(property_name,)!r}")
        properties[property_name] = value
    return properties


value_node_rules: defaultdict[type[Any], ValueNodeRule] = defaultdict(lambda: lambda x: x)
transport_node_rules = defaultdict(lambda: default_transport_node)


def doc_transport(node: Meta[tuple[Path, Any]]) -> tuple[Path, Any]:
    path, value = node.value
    *path, _ = path
    return tuple(path), value


value_node_rules[Meta] = lambda node: node.value
transport_node_rules[Meta] = doc_transport


@ft.lru_cache(maxsize=256)
def parser_for_key(key: SchemaKey) -> Parser:
    cache = tree_cache_for_key(key)
    return lambda value: build_from_cache(cache, value)


@ft.lru_cache(maxsize=256)
def tree_cache_for_key(key: SchemaKey) -> TreeCache:
    leaves, spec = key
    _, transport = spec.traverse(
        zip(spec.paths(), spec.accessors(), leaves, strict=True),
        transport_node_rule,
        transport_leaf,
    )
    transport_leaves, transport_spec = treelib.flatten(transport, is_leaf=is_schema_spec)
    marker = object()
    value_tree = spec.traverse(
        leaves,
        value_node_rule,
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


def value_node_rule(node: Any) -> Any:
    return value_node_rules[type(node)](node)


def transport_node_rule(node: Any) -> tuple[Path, Any]:
    return transport_node_rules[type(node)](node)


def transport_leaf(node: Any) -> tuple[Path, Any]:
    path, accessor, leaf = node
    if not is_schema_spec(leaf):
        raise TypeError(f"{accessor.codify('$')}: {schema_message}, got {leaf!r}")
    return path, leaf


def default_transport_node(node: Any) -> tuple[Path, ObjectProperties]:
    return transport_object(node)


def transport_object(node: Any) -> tuple[Path, ObjectProperties]:
    children, _, _, _ = treelib.flatten_one_level(node)
    (path, _), *_ = children
    *path, _ = path
    return tuple(path), transport_properties(children)


def build_from_cache(cache: TreeCache, value: Any) -> Any:
    transport_spec, value_spec, rules = cache
    try:
        leaves = transport_spec.flatten_up_to(value)
    except ValueError:
        _, spec = treelib.flatten(value)
        raise ValueError(f"$: expected {transport_spec}, got {spec}") from None
    output = (rule(l, v, a, "$") for v, (l, rule, a) in zip(leaves, rules, strict=True))
    return value_spec.unflatten(output)


def build(schema: Any) -> tuple[JsonSchema, Parser]:
    leaves, spec = treelib.flatten(schema, is_leaf=is_schema_spec, none_is_leaf=True)
    key = (tuple(leaves), spec)
    return build_schema_for_key(key), parser_for_key(key)
