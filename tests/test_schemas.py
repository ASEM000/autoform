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
            lambda: af.lm.Str(minimum=0),
            TypeError,
            "unexpected keyword",
            id="str-unexpected-keyword",
        ),
        pytest.param(
            lambda: af.lm.Str(pattern=1),
            TypeError,
            "pattern must be a string",
            id="str-pattern-must-be-a-string",
        ),
        pytest.param(
            lambda: af.lm.Str(min=-1),
            ValueError,
            "min must be >= 0",
            id="str-min-must-be-0",
        ),
        pytest.param(
            lambda: af.lm.Str(max=-1),
            ValueError,
            "max must be >= 0",
            id="str-max-must-be-0",
        ),
        pytest.param(
            lambda: af.lm.Str(min=2, max=1),
            ValueError,
            "min must be <= max",
            id="str-min-must-be-max",
        ),
        pytest.param(
            lambda: af.lm.Int(min=0.5),
            TypeError,
            "min must be an int",
            id="int-min-must-be-an-int",
        ),
        pytest.param(
            lambda: af.lm.Int(min=2, max=1),
            ValueError,
            "min must be <= max",
            id="int-min-must-be-max",
        ),
        pytest.param(
            lambda: af.lm.Float(min="0"),
            TypeError,
            "min must be a number",
            id="float-min-must-be-a-number",
        ),
        pytest.param(
            lambda: af.lm.Float(min=2, max=1),
            ValueError,
            "min must be <= max",
            id="float-min-must-be-max",
        ),
        pytest.param(
            lambda: af.lm.Enum(),
            TypeError,
            "Enum must have at least one value",
            id="enum-enum-must-have-at-least-one-value",
        ),
        pytest.param(
            lambda: af.lm.Enum("summary", 1),
            TypeError,
            "Enum values must share one type",
            id="enum-enum-values-must-share-one-type",
        ),
        pytest.param(
            lambda: af.lm.Str(desc=1),
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
            af.lm.Str(min=1, max=3, pattern="x"),
            af.lm.Str(min=1, max=3, pattern="x"),
            id="string",
        ),
        pytest.param(af.lm.Int(min=0, max=10), af.lm.Int(min=0, max=10), id="integer"),
        pytest.param(af.lm.Float(min=0, max=1), af.lm.Float(min=0, max=1), id="float"),
        pytest.param(af.lm.Bool(), af.lm.Bool(), id="boolean"),
        pytest.param(
            af.lm.Enum("summary", "definition"),
            af.lm.Enum("summary", "definition"),
            id="enum",
        ),
        pytest.param(
            af.lm.Str(desc="Subject name."),
            af.lm.Str(desc="Subject name."),
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
        pytest.param(af.lm.Str(min=1), id="string"),
        pytest.param(af.lm.Int(min=0), id="integer"),
        pytest.param(af.lm.Float(min=0, max=1), id="float"),
        pytest.param(af.lm.Bool(), id="boolean"),
        pytest.param(af.lm.Enum("yes", "no"), id="enum"),
    ],
)
def test_schema_specs_are_static_during_tracing(schema):
    ir = af.trace(lambda x, y: (x, y))(schema, "seed")
    assert af.utils.tree.leaves(schema) == []
    assert ir.in_tree[0] is schema
    assert ir.call(schema, "hello") == (schema, "hello")


def test_new_spec_subclasses_register_as_static_nodes():
    class CustomSpec(af.lm.Spec):
        __slots__ = []

    schema = CustomSpec()
    leaves, spec = af.utils.tree.flatten(schema)
    assert leaves == []
    assert spec.unflatten(leaves) is schema
    assert af.trace(lambda x: x)(schema).call(schema) is schema


@pytest.mark.parametrize("operation", ["describe", "parse"])
def test_unregistered_schema_nodes_remain_static(operation):
    class CustomSpec(af.lm.Spec):
        __slots__ = []

    schema = CustomSpec()
    if operation == "describe":
        assert af.lm.describe(schema) is None
    else:
        assert af.lm.parse(schema, "value") is schema


@pytest.mark.parametrize(
    "schema, expected",
    [
        pytest.param(af.lm.Str(min=1, desc="Text"), dict(type="string", minLength=1), id="str"),
        pytest.param(af.lm.Int(min=0, desc="Text"), dict(type="integer", minimum=0), id="int"),
        pytest.param(af.lm.Float(max=1, desc="Text"), dict(type="number", maximum=1), id="float"),
        pytest.param(af.lm.Bool(desc="Text"), dict(type="boolean"), id="bool"),
        pytest.param(
            af.lm.Enum("yes", "no", desc="Text"),
            dict(type="string", enum=["yes", "no"]),
            id="enum",
        ),
    ],
)
def test_json_rules_own_schema_descriptions(schema, expected):
    expected = dict(expected, description="Text")
    assert af.lm.describe_rules[type(schema)](schema) == expected
    assert af.lm.describe(schema) == expected


def test_json_mangles_duplicate_object_entries_before_omitting_literals():
    schema = {0: {"fixed": "value"}, "0": af.lm.Str()}
    json_schema = af.lm.describe(schema)
    assert list(json_schema["properties"]) == ["0_"]
    assert af.lm.parse(schema, {"0_": "generated"}) == {
        0: {"fixed": "value"},
        "0": "generated",
    }


@pytest.mark.parametrize(
    "schema", [af.lm.Float(), af.lm.Float(min=0, max=1)], ids=["unbounded", "bounded"]
)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_parse_float_rejects_nonfinite_values(schema, value):
    with pytest.raises(ValueError, match="Expected finite number"):
        af.lm.parse(schema, value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_describe_enum_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="Enum values must be finite"):
        af.lm.describe(af.lm.Enum(0.0, value))


def test_parse_uses_partitioned_schema():
    class CustomSpec(af.lm.Spec):
        __slots__ = []

    calls = []

    def describe(schema):
        calls.append(schema)
        return dict(type="string")

    af.lm.describe_rules[CustomSpec] = describe
    af.lm.parse_rules[CustomSpec] = lambda _, value: value
    schema = {
        "generated": {"text": af.lm.Str(desc="Generated text.")},
        "literal": {"text": "fixed", "nothing": None},
        "custom": CustomSpec(),
    }
    value = {"generated": {"text": "x"}, "custom": "y"}
    assert af.lm.parse(schema, value) == {
        "generated": {"text": "x"},
        "literal": {"text": "fixed", "nothing": None},
        "custom": "y",
    }
    assert calls == []
    assert af.lm.describe(schema)["required"] == ["custom", "generated"]
    assert calls == [schema["custom"]]


def test_partition_and_parse_custom_pytree():
    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        score: object
        metadata: object
        reasoning: object

    schema = Answer(
        af.lm.Float(min=0, max=1),
        {"source": "fixed"},
        af.lm.Str(desc="Reasoning."),
    )
    schm_tree, lit_tree = af.utils.partition(
        af.lm.is_schema,
        schema,
        is_leaf=af.lm.is_schema,
        fillvalue=af.lm.missing,
    )

    assert lit_tree == Answer(
        af.lm.missing,
        {"source": "fixed"},
        af.lm.missing,
    )
    assert schm_tree == Answer(
        af.lm.Float(min=0, max=1),
        {"source": af.lm.missing},
        af.lm.Str(desc="Reasoning."),
    )
    assert af.lm.describe_node(schm_tree) == af.lm.describe(schema)
    generated_tree = af.lm.parse_node(
        schm_tree,
        {"score": 0.8, "reasoning": "Evidence agrees."},
    )
    assert generated_tree == Answer(
        0.8,
        {"source": af.lm.missing},
        "Evidence agrees.",
    )
    assert af.lm.parse(
        schema,
        {"score": 0.8, "reasoning": "Evidence agrees."},
    ) == Answer(
        0.8,
        {"source": "fixed"},
        "Evidence agrees.",
    )
