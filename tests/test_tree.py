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

from autoform import tree


@pytest.mark.parametrize(
    "container, shape, rows, expected",
    [
        pytest.param([0, 0], True, ["x", "y"], ["x", "y"], id="scalar"),
        pytest.param(
            (0, 0),
            (True, True),
            [("x", "a"), ("y", "b")],
            (("x", "y"), ("a", "b")),
            id="tuple",
        ),
        pytest.param(
            {"y": 0, "x": 0},
            {"b": True, "a": True},
            [{"b": "x", "a": "a"}, {"b": "y", "a": "b"}],
            {"b": {"y": "x", "x": "y"}, "a": {"y": "a", "x": "b"}},
            id="dict",
        ),
        pytest.param(
            [0, 0],
            (True,),
            [(["x", "y"],), (["a", "b"],)],
            ([["x", "y"], ["a", "b"]],),
            id="container-valued-field",
        ),
        pytest.param(
            [0, 0], (True, None), [("x", None), ("y", None)], (["x", "y"], None), id="none-node"
        ),
        pytest.param([0, 0], (), [(), ()], (), id="empty-result"),
        pytest.param([0, 0], False, ["x", "x"], "x", id="shared"),
    ],
)
def test_stack_unstack(container, shape, rows, expected):
    spec = tree.structure(container)
    actual = tree.stack(spec, shape, (row for row in rows))
    assert tree.equal(actual, expected)
    assert tree.unstack(spec, shape, actual) == rows


def test_stack_unstack_shared_field():
    shared = ["a", "b"]
    spec = tree.structure((0, 0))
    rows = tree.unstack(spec, (True, False), (("x", "y"), shared))
    assert rows == [("x", ["a", "b"]), ("y", ["a", "b"])]
    assert all(row[1] is shared for row in rows)
    actual = tree.stack(spec, (True, False), iter(rows))
    assert actual == (("x", "y"), shared)
    assert actual[1] is shared


def test_stack_unstack_custom_pytree():
    @tree.dataclass
    class Point:
        x: str
        y: str

    spec = tree.structure(Point(0, 0))
    rows = [Point("x", "a"), Point("y", "b")]
    shape = Point(True, True)
    actual = tree.stack(spec, shape, iter(rows))
    assert actual == Point(Point("x", "y"), Point("a", "b"))
    assert tree.unstack(spec, shape, actual) == rows


@pytest.mark.parametrize(
    "rows",
    [pytest.param(["x"], id="too-few"), pytest.param(["x", "y", "z"], id="too-many")],
)
def test_stack_rejects_wrong_item_count(rows):
    with pytest.raises(ValueError):
        tree.stack(tree.structure([0, 0]), True, iter(rows))


def test_stack_rejects_empty_container():
    with pytest.raises(ValueError, match="Cannot stack an empty container"):
        tree.stack(tree.structure([]), True, iter(()))


@pytest.mark.parametrize(
    "shape, rows",
    [
        pytest.param((True,), [("x",), ["y"]], id="container-type"),
        pytest.param((), [(), ("x",)], id="empty-result"),
    ],
)
def test_stack_rejects_mismatched_rows(shape, rows):
    with pytest.raises(ValueError):
        tree.stack(tree.structure([0, 0]), shape, iter(rows))


@pytest.mark.parametrize(
    "right, expected",
    [
        pytest.param(("a", ["b", "c"]), True, id="equal"),
        pytest.param(("a", ("b", "c")), False, id="structure"),
        pytest.param(("a", ["b", "d"]), False, id="leaves"),
    ],
)
def test_tree_equal(right, expected):
    assert tree.equal(("a", ["b", "c"]), right) is expected


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: tree.structure(x).unflatten(tree.leaves(x)), id="flatten"),
        pytest.param(lambda x: tree.map(lambda v: v, x), id="map"),
        pytest.param(lambda x: tree.broadcast_prefix(x, x), id="prefix"),
    ],
)
def test_order_and_none(operation):
    value = {"y": "x", "x": None, "z": "y"}
    result = operation(value)
    assert list(result.items()) == list(value.items())
    assert tree.leaves(result) == ["x", "y"]
    assert not tree.is_leaf(None)
    assert tree.map(str.upper, result) == {"y": "X", "x": None, "z": "Y"}


@pytest.mark.parametrize(
    "prefix, value, is_leaf, expected",
    [
        pytest.param(
            True, {"y": ("x", None), "x": []}, None, {"y": (True, None), "x": []}, id="nested"
        ),
        pytest.param((True, False), (["x", "y"], "Q"), None, ([True, True], False), id="mixed"),
        pytest.param((True,), (["x", "y"],), lambda x: isinstance(x, list), (True,), id="leaf"),
    ],
)
def test_broadcast_prefix(prefix, value, is_leaf, expected):
    assert tree.equal(tree.broadcast_prefix(prefix, value, is_leaf=is_leaf), expected)


@pytest.mark.parametrize(
    "decorate",
    [
        pytest.param(tree.dataclass, id="bare"),
        pytest.param(tree.dataclass(frozen=True), id="frozen"),
    ],
)
def test_dataclass_metadata(decorate):
    @decorate
    class State:
        value: str
        label: str = tree.field(pytree_node=False)

    state = State("x", "Q")
    assert tree.leaves(state) == ["x"]
    assert tree.map(str.upper, state) == State("X", "Q")
