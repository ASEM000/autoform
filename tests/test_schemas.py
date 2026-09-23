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
            lambda: af.Doc(1),
            TypeError,
            "description must be a string",
            id="doc-description-must-be-a-string",
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
        pytest.param(af.Doc("Subject name."), af.Doc("Subject name."), id="description"),
        pytest.param(
            af.Str() @ af.Doc("Subject name."),
            af.Str() @ af.Doc("Subject name."),
            id="described-string",
        ),
    ],
)
def test_schema_dsl_nodes_compare_by_value(left, right):
    assert left == right
    assert hash(left) == hash(right)
