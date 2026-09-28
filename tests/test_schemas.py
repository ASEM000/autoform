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

import optree
import pytest

import autoform as af


@pytest.mark.parametrize(
    "construct, error, message",
    [
        pytest.param(
            lambda: af.Str(minimum=0),
            TypeError,
            "unexpected keyword",
            id="str-unexpected-keyword",
        ),
        pytest.param(
            lambda: af.Str(pattern=1),
            TypeError,
            "pattern must be a string",
            id="str-pattern-must-be-a-string",
        ),
        pytest.param(
            lambda: af.Str(min=-1),
            ValueError,
            "min must be >= 0",
            id="str-min-must-be-0",
        ),
        pytest.param(
            lambda: af.Str(max=-1),
            ValueError,
            "max must be >= 0",
            id="str-max-must-be-0",
        ),
        pytest.param(
            lambda: af.Str(min=2, max=1),
            ValueError,
            "min must be <= max",
            id="str-min-must-be-max",
        ),
        pytest.param(
            lambda: af.Int(min=0.5),
            TypeError,
            "min must be an int",
            id="int-min-must-be-an-int",
        ),
        pytest.param(
            lambda: af.Int(min=2, max=1),
            ValueError,
            "min must be <= max",
            id="int-min-must-be-max",
        ),
        pytest.param(
            lambda: af.Float(min="0"),
            TypeError,
            "min must be a number",
            id="float-min-must-be-a-number",
        ),
        pytest.param(
            lambda: af.Float(min=2, max=1),
            ValueError,
            "min must be <= max",
            id="float-min-must-be-max",
        ),
        pytest.param(
            lambda: af.Enum(),
            TypeError,
            "Enum must have at least one value",
            id="enum-enum-must-have-at-least-one-value",
        ),
        pytest.param(
            lambda: af.Enum("summary", 1),
            TypeError,
            "Enum values must share one type",
            id="enum-enum-values-must-share-one-type",
        ),
        pytest.param(
            lambda: af.Str(desc=1),
            TypeError,
            "desc must be a string",
            id="desc-must-be-a-string",
        ),
    ],
)
def test_schema_dsl_rejects_invalid_forms(construct, error, message):
    with pytest.raises(error, match=message):
        construct()


@pytest.mark.parametrize(
    "left, right",
    [
        pytest.param(
            af.Str(min=1, max=3, pattern="x"),
            af.Str(min=1, max=3, pattern="x"),
            id="string",
        ),
        pytest.param(af.Int(min=0, max=10), af.Int(min=0, max=10), id="integer"),
        pytest.param(af.Float(min=0, max=1), af.Float(min=0, max=1), id="float"),
        pytest.param(af.Bool(), af.Bool(), id="boolean"),
        pytest.param(
            af.Enum("summary", "definition"),
            af.Enum("summary", "definition"),
            id="enum",
        ),
        pytest.param(
            af.Str(desc="Subject name."),
            af.Str(desc="Subject name."),
            id="described-string",
        ),
    ],
)
def test_schema_dsl_nodes_compare_by_value(left, right):
    assert left == right
    assert hash(left) == hash(right)


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(af.Str(min=1), id="string"),
        pytest.param(af.Int(min=0), id="integer"),
        pytest.param(af.Float(min=0, max=1), id="float"),
        pytest.param(af.Bool(), id="boolean"),
        pytest.param(af.Enum("yes", "no"), id="enum"),
    ],
)
def test_schema_specs_are_static_during_tracing(schema):
    ir = af.trace(lambda x, y: (x, y))(schema, "seed")
    assert af.utils.tree.leaves(schema) == []
    assert ir.in_tree[0] is schema
    assert ir.call(schema, "hello") == (schema, "hello")


def test_new_spec_subclasses_register_as_static_nodes():
    class CustomSpec(af.schemas.Spec):
        __slots__ = []

    schema = CustomSpec()
    leaves, spec = af.utils.tree.flatten(schema)
    assert leaves == []
    assert spec.unflatten(leaves) is schema
    assert af.trace(lambda x: x)(schema).call(schema) is schema


def test_json_mangles_duplicate_object_entries_before_omitting_literals():
    schema = {0: {"fixed": "value"}, "0": af.Str()}
    json_schema = af.schemas.emit_json_schema(schema)
    assert list(json_schema["properties"]) == ["0_"]
    assert af.schemas.parse_json_value(schema, {"0_": "generated"}) == {
        0: {"fixed": "value"},
        "0": "generated",
    }


@pytest.mark.parametrize(
    "schema", [af.Float(), af.Float(min=0, max=1)], ids=["unbounded", "bounded"]
)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_parse_float_rejects_nonfinite_values(schema, value):
    with pytest.raises(ValueError, match="Expected finite number"):
        af.schemas.parse_json_value(schema, value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_emit_enum_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="Enum values must be finite"):
        af.schemas.emit_json_schema(af.Enum(0.0, value))


def test_parse_uses_partitioned_schema():
    class CustomSpec(af.schemas.Spec):
        __slots__ = []

    calls = []

    def emit(schema):
        calls.append(schema)
        return dict(type="string")

    af.schemas.emit_json_schema_rules[CustomSpec] = emit
    af.schemas.parse_json_value_rules[CustomSpec] = lambda _, value: value
    try:
        schema = {
            "generated": {"text": af.Str(desc="Generated text.")},
            "literal": {"text": "fixed", "nothing": None},
            "custom": CustomSpec(),
        }
        value = {"generated": {"text": "x"}, "custom": "y"}
        assert af.schemas.parse_json_value(schema, value) == {
            "generated": {"text": "x"},
            "literal": {"text": "fixed", "nothing": None},
            "custom": "y",
        }
        assert calls == [schema["custom"]]
        assert af.schemas.emit_json_schema(schema)["required"] == ["custom", "generated"]
        assert calls == [schema["custom"], schema["custom"]]
    finally:
        del af.schemas.emit_json_schema_rules[CustomSpec]
        del af.schemas.parse_json_value_rules[CustomSpec]


def test_partition_and_parse_custom_pytree():
    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        score: object
        metadata: object
        reasoning: object

    schema = Answer(
        af.Float(min=0, max=1),
        {"source": "fixed"},
        af.Str(desc="Reasoning."),
    )
    literal_tree, schema_tree = af.schemas.partition_schema(schema)

    assert literal_tree == Answer(
        af.schemas.missing,
        {"source": "fixed"},
        af.schemas.missing,
    )
    assert schema_tree == Answer(
        af.Float(min=0, max=1),
        {"source": af.schemas.missing},
        af.Str(desc="Reasoning."),
    )
    assert af.schemas.emit_json_tree(schema_tree) == af.schemas.emit_json_schema(schema)
    generated_tree = af.schemas.parse_json_tree(
        schema_tree, {"score": 0.8, "reasoning": "Evidence agrees."}
    )
    assert generated_tree == Answer(
        0.8,
        {"source": af.schemas.missing},
        "Evidence agrees.",
    )
    assert af.schemas.parse_json_value(
        schema,
        {"score": 0.8, "reasoning": "Evidence agrees."},
    ) == Answer(
        0.8,
        {"source": "fixed"},
        "Evidence agrees.",
    )
