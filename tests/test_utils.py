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
from autoform.utils import batch_spec, batch_transpose, tree


@pytest.fixture
def point_type():
    @tree.dataclasses.dataclass
    class Point:
        x: int
        y: int

    return Point


@pytest.mark.parametrize(
    "columns, axes, expected",
    [
        pytest.param((["a", "b"], ["x", "y"]), True, 2, id="both-mapped"),
        pytest.param((["a", "b"], "single"), (True, False), 2, id="broadcast"),
        pytest.param(("a", "b"), (False, False), None, id="unmapped"),
    ],
)
def test_batch_size(columns, axes, expected):
    spec = batch_spec(columns, axes)
    assert (spec.num_children if spec is not None else None) == expected


@pytest.mark.parametrize(
    "right, expected",
    [
        pytest.param(("a", ["b", "c"]), True, id="equal"),
        pytest.param(("a", ("b", "c")), False, id="structure"),
        pytest.param(("a", ["b", "d"]), False, id="leaves"),
    ],
)
def test_tree_equal(right, expected):
    assert af.utils.tree_equal(("a", ["b", "c"]), right) is expected


@pytest.mark.parametrize(
    "rows, axes, expected",
    [
        pytest.param(
            [["a", "x"], ["b", "y"], ["c", "z"]],
            [True, True],
            [["a", "b", "c"], ["x", "y", "z"]],
            id="list",
        ),
        pytest.param([("a", "x"), ("b", "y")], (True, True), (["a", "b"], ["x", "y"]), id="tuple"),
        pytest.param(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            {"a": True, "b": True},
            {"a": [1, 3], "b": [2, 4]},
            id="dict",
        ),
    ],
)
def test_batch_transpose(rows, axes, expected):
    assert batch_transpose(len(rows), axes, rows) == expected


def test_batch_transpose_custom_pytree(point_type):
    point = point_type
    result = batch_transpose(2, point(True, True), [point(1, 2), point(3, 4)])
    assert result == point([1, 3], [2, 4])


@pytest.mark.parametrize(
    "columns, axes, values, expected",
    [
        pytest.param((["a", "b", "c"],), (True,), ["x", "y", "z"], ["x", "y", "z"], id="list"),
        pytest.param((("a", "b", "c"),), (True,), ["x", "y", "z"], ("x", "y", "z"), id="tuple"),
        pytest.param((("a", "b"), "broadcast"), (True, False), ["x", "y"], ("x", "y"), id="mixed"),
        pytest.param(
            ((("a", "b", "c"),),),
            ((True,),),
            ["x", "y", "z"],
            ("x", "y", "z"),
            id="nested",
        ),
    ],
)
def test_batch_spec_container(columns, axes, values, expected):
    assert batch_spec(columns, axes).unflatten(values) == expected


def test_batch_spec_custom_container(point_type):
    point = point_type
    columns = ([point(1, 2), point(3, 4)],)
    values = [point(10, 20), point(30, 40)]
    assert batch_spec(columns, (True,)).unflatten(values) == values
