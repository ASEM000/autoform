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

import autoform.schema as schema


def test_schema_dsl_builds_described_schema():
    answer = {
        "name": schema.Str() @ schema.Doc("Subject name."),
        "kind": schema.Enum("summary", "definition") @ schema.Doc("Answer kind."),
        "score": schema.Float() @ schema.Doc("Confidence score."),
    } @ schema.Doc("Answer object.")

    json_schema, parse = schema.build(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["summary", "definition"],
                "description": "Answer kind.",
            },
            "name": {"type": "string", "description": "Subject name."},
            "score": {"type": "number", "description": "Confidence score."},
        },
        "required": ["kind", "name", "score"],
        "additionalProperties": False,
        "description": "Answer object.",
    }
    assert parse(
        {"name": "subject", "kind": "summary", "score": 1},
    ) == {
        "name": "subject",
        "kind": "summary",
        "score": 1.0,
    }


def test_schema_dsl_builds_tree():
    answer = {
        "name": schema.Str(),
        "count": schema.Int(),
        "score": schema.Float(),
        "ok": schema.Bool(),
        "kind": schema.Enum("summary", "definition"),
    }

    _, parse = schema.build(answer)

    assert parse(
        {"name": "hello", "count": 2, "score": 1, "ok": True, "kind": "summary"},
    ) == {
        "name": "hello",
        "count": 2,
        "score": 1.0,
        "ok": True,
        "kind": "summary",
    }


def test_schema_dsl_builds_string_constraints():
    json_schema, parse = schema.build({"name": schema.Str(min=2, max=4, pattern=r"^[a-z]+$")})

    assert json_schema == {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 2,
                "maxLength": 4,
                "pattern": r"^[a-z]+$",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert parse({"name": "okay"}) == {"name": "okay"}
    with pytest.raises(ValueError, match=r"\$\['name'\]: expected string with length >= 2"):
        parse({"name": "x"})
    with pytest.raises(ValueError, match=r"\$\['name'\]: expected string with length <= 4"):
        parse({"name": "hello"})
    with pytest.raises(ValueError, match=r"\$\['name'\]: expected string matching"):
        parse({"name": "OK"})


def test_schema_dsl_builds_custom_pytree_value():
    class Answer:
        __slots__ = ["text", "score"]

        def __init__(self, text, score):
            self.text = text
            self.score = score

        def __eq__(self, other):
            return (
                type(self) is type(other) and self.text == other.text and self.score == other.score
            )

    schema.treelib.register_node(
        Answer,
        lambda answer: ((answer.text, answer.score), None, ("text", "score")),
        lambda _, children: Answer(*children),
        path_entry_type=optree.GetAttrEntry,
    )

    answer = Answer(schema.Str(), schema.Float())

    json_schema, parse = schema.build(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["text", "score"],
        "additionalProperties": False,
    }
    assert parse({"text": "hello", "score": 2}) == Answer("hello", 2.0)


def test_schema_dsl_rejects_invalid_forms():
    with pytest.raises(TypeError, match="unexpected keyword"):
        schema.Str(minimum=0)
    with pytest.raises(TypeError, match="pattern must be a string"):
        schema.Str(pattern=1)
    with pytest.raises(ValueError, match="min must be >= 0"):
        schema.Str(min=-1)
    with pytest.raises(ValueError, match="max must be >= 0"):
        schema.Str(max=-1)
    with pytest.raises(ValueError, match="min must be <= max"):
        schema.Str(min=2, max=1)
    with pytest.raises(TypeError, match="Enum must have at least one value"):
        schema.Enum()
    with pytest.raises(TypeError, match="Enum values must share one type"):
        schema.Enum("summary", 1)
    with pytest.raises(TypeError, match="description must be a string"):
        schema.Doc(1)


def test_schema_dsl_reports_value_errors_by_path():
    _, parse_count = schema.build({"count": schema.Int()})
    _, parse_score = schema.build({"score": schema.Float()})

    with pytest.raises(ValueError, match=r"\$\['count'\]: expected integer"):
        parse_count({"count": True})
    with pytest.raises(ValueError, match=r"\$\['score'\]: expected number"):
        parse_score({"score": "bad"})
