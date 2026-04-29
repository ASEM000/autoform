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
        "name": schema.String @ schema.Doc("Subject name."),
        "kind": schema.Enum["summary", "definition"] @ schema.Doc("Answer kind."),
        "score": schema.Float @ schema.Doc("Confidence score."),
    } @ schema.Doc("Answer object.")

    assert schema.build_schema(answer) == {
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


def test_schema_dsl_builds_value_tree():
    answer = {
        "name": schema.String,
        "count": schema.Integer,
        "score": schema.Float,
        "ok": schema.Boolean,
        "kind": schema.Enum["summary", "definition"],
    }

    assert schema.build_value(
        answer,
        {"name": "hello", "count": 2, "score": 1, "ok": True, "kind": "summary"},
    ) == {
        "name": "hello",
        "count": 2,
        "score": 1.0,
        "ok": True,
        "kind": "summary",
    }


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

    answer = Answer(schema.String, schema.Float)

    assert schema.build_schema(answer) == {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["text", "score"],
        "additionalProperties": False,
    }
    assert schema.build_value(answer, {"text": "hello", "score": 2}) == Answer("hello", 2.0)


def test_schema_dsl_rejects_invalid_forms():
    with pytest.raises(TypeError, match="use String"):
        schema.String()
    with pytest.raises(TypeError, match=r"use Enum\[\.\.\.\]"):
        schema.Enum("summary", "definition")
    with pytest.raises(TypeError, match="Enum values must share one type"):
        schema.Enum["summary", 1]
    with pytest.raises(TypeError, match="description must be a string"):
        schema.Doc(1)


def test_schema_dsl_reports_value_errors_by_path():
    with pytest.raises(ValueError, match=r"\$\['count'\]: expected integer"):
        schema.build_value({"count": schema.Integer}, {"count": True})
    with pytest.raises(ValueError, match=r"\$\['score'\]: expected number"):
        schema.build_value({"score": schema.Float}, {"score": "bad"})
