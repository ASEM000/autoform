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
import autoform.schemas as schemas
from autoform.schemas import make_json_schema_and_parser
from autoform.utils import treelib


def test_schema_dsl_builds_described_schema():
    answer = {
        "name": af.Str() @ af.Doc("Subject name."),
        "kind": af.Enum("summary", "definition") @ af.Doc("Answer kind."),
        "score": af.Float() @ af.Doc("Confidence score."),
    } @ af.Doc("Answer object.")

    json_schema, parse = make_json_schema_and_parser(answer)

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
        "name": af.Str(),
        "count": af.Int(),
        "score": af.Float(),
        "ok": af.Bool(),
        "kind": af.Enum("summary", "definition"),
    }

    _, parse = make_json_schema_and_parser(answer)

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
    json_schema, parse = make_json_schema_and_parser({"name": af.Str(min=2, max=4, pattern=r"^[a-z]+$")})

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


def test_schema_dsl_builds_number_constraints():
    json_schema, parse = make_json_schema_and_parser({
        "count": af.Int(min=-2, max=2),
        "score": af.Float(min=0, max=1),
    })

    assert json_schema == {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": -2, "maximum": 2},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["count", "score"],
        "additionalProperties": False,
    }
    assert parse({"count": 0, "score": 1}) == {"count": 0, "score": 1.0}
    with pytest.raises(ValueError, match=r"\$\['count'\]: expected integer >= -2"):
        parse({"count": -3, "score": 0.5})
    with pytest.raises(ValueError, match=r"\$\['count'\]: expected integer <= 2"):
        parse({"count": 3, "score": 0.5})
    with pytest.raises(ValueError, match=r"\$\['score'\]: expected number >= 0"):
        parse({"count": 0, "score": -0.1})
    with pytest.raises(ValueError, match=r"\$\['score'\]: expected number <= 1"):
        parse({"count": 0, "score": 1.1})


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

    treelib.register_node(
        Answer,
        lambda answer: ((answer.text, answer.score), None, ("text", "score")),
        lambda _, children: Answer(*children),
        path_entry_type=optree.GetAttrEntry,
    )

    answer = Answer(af.Str(), af.Float())

    json_schema, parse = make_json_schema_and_parser(answer)

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
        af.Str(minimum=0)
    with pytest.raises(TypeError, match="pattern must be a string"):
        af.Str(pattern=1)
    with pytest.raises(ValueError, match="min must be >= 0"):
        af.Str(min=-1)
    with pytest.raises(ValueError, match="max must be >= 0"):
        af.Str(max=-1)
    with pytest.raises(ValueError, match="min must be <= max"):
        af.Str(min=2, max=1)
    with pytest.raises(TypeError, match="min must be an int"):
        af.Int(min=0.5)
    with pytest.raises(ValueError, match="min must be <= max"):
        af.Int(min=2, max=1)
    with pytest.raises(TypeError, match="min must be a number"):
        af.Float(min="0")
    with pytest.raises(ValueError, match="min must be <= max"):
        af.Float(min=2, max=1)
    with pytest.raises(TypeError, match="Enum must have at least one value"):
        af.Enum()
    with pytest.raises(TypeError, match="Enum values must share one type"):
        af.Enum("summary", 1)
    with pytest.raises(TypeError, match="description must be a string"):
        af.Doc(1)


def test_schema_dsl_reports_value_errors_by_path():
    _, parse_count = make_json_schema_and_parser({"count": af.Int()})
    _, parse_score = make_json_schema_and_parser({"score": af.Float()})

    with pytest.raises(ValueError, match=r"\$\['count'\]: expected integer"):
        parse_count({"count": True})
    with pytest.raises(ValueError, match=r"\$\['score'\]: expected number"):
        parse_score({"score": "bad"})


def test_schema_dsl_nodes_compare_by_value():
    pairs = [
        (af.Str(min=1, max=3, pattern="x"), af.Str(min=1, max=3, pattern="x")),
        (af.Int(min=0, max=10), af.Int(min=0, max=10)),
        (af.Float(min=0, max=1), af.Float(min=0, max=1)),
        (af.Bool(), af.Bool()),
        (af.Enum("summary", "definition"), af.Enum("summary", "definition")),
        (af.Doc("Subject name."), af.Doc("Subject name.")),
        (af.Str() @ af.Doc("Subject name."), af.Str() @ af.Doc("Subject name.")),
    ]

    for left, right in pairs:
        assert left == right
        assert hash(left) == hash(right)


def test_schema_dsl_reuses_cache_for_equal_schema_nodes():
    schemas.schema_from_flat_and_spec.cache_clear()
    schemas.parser_from_flat_and_spec.cache_clear()

    def answer():
        return {
            "name": af.Str(min=1, max=80),
            "count": af.Int(min=0, max=10),
            "score": af.Float(min=0, max=1),
            "ok": af.Bool(),
            "kind": af.Enum("summary", "definition"),
        } @ af.Doc("Answer object.")

    make_json_schema_and_parser(answer())
    make_json_schema_and_parser(answer())

    assert schemas.schema_from_flat_and_spec.cache_info().hits == 1
    assert schemas.parser_from_flat_and_spec.cache_info().hits == 1
