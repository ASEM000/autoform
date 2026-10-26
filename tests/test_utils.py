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

import autoform.src.tree as tree
import autoform.src.utils as utils


@pytest.mark.parametrize(
    "values, expected",
    [pytest.param((1, 2), 6, id="numbers"), pytest.param(("a", "b"), "abab", id="text")],
)
def test_dispatch(values, expected):
    dispatch = utils.dispatch()
    dispatch.register((int, int), lambda values, *, scale: sum(values) * scale)
    dispatch.register((str, str), lambda values, *, scale: "".join(values) * scale)
    assert dispatch(values, scale=2) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param((1, 2), "exact", id="exact"),
        pytest.param(("a", "b"), "default", id="types"),
        pytest.param([1, 2], "default", id="structure"),
    ],
)
def test_dispatch_default(values, expected):
    @utils.dispatch
    def dispatch(values, *, scale):
        return "default", values, scale

    assert dispatch.__name__ == "dispatch"
    dispatch.register((int, int), lambda values, *, scale: ("exact", values, scale))
    selected, actual, scale = dispatch(values, scale=2)
    assert (selected, scale) == (expected, 2)
    assert actual is values


@pytest.mark.parametrize("error", [KeyError, TypeError], ids=["key", "type"])
@pytest.mark.parametrize("registered", [False, True], ids=["default", "exact"])
def test_dispatch_handler_errors(error, registered):
    failure = error("handler failed")

    def fail(values):
        raise failure

    dispatch = utils.dispatch((lambda values: values) if registered else fail)
    if registered:
        dispatch.register(int, fail)
    with pytest.raises(error) as raised:
        dispatch(1)
    assert raised.value is failure


@pytest.mark.parametrize(
    "types, values",
    [
        pytest.param((int, int), (True, 2), id="subclass"),
        pytest.param((int, float), (2.0, 1), id="order"),
        pytest.param((int, str), [1, "x"], id="structure"),
        pytest.param({"x": int, "y": str}, {"y": "x", "x": 1}, id="dict-order"),
        pytest.param(None, (), id="empty-structure"),
    ],
)
def test_dispatch_missing(types, values):
    dispatch = utils.dispatch()
    dispatch.register(types, lambda values: values)
    with pytest.raises(TypeError, match="No dispatch rule for"):
        dispatch(values)


@pytest.mark.parametrize(
    "types, values",
    [
        pytest.param(int, 1, id="scalar"),
        pytest.param((bool, int), (True, 2), id="explicit-subclass"),
        pytest.param([int, str], [1, "x"], id="list"),
        pytest.param({"y": (str, None), "x": [int]}, {"y": ("x", None), "x": [1]}, id="nested"),
        pytest.param(None, None, id="none"),
        pytest.param((), (), id="empty"),
    ],
)
def test_dispatch_pytrees(types, values):
    dispatch = utils.dispatch()
    dispatch.register(types, lambda values: values)
    assert dispatch(values) is values


@pytest.fixture
def point_type():
    @tree.dataclass
    class Point:
        x: int
        y: int

    return Point


def test_dispatch_custom_pytree(point_type):
    dispatch = utils.dispatch()
    dispatch.register(point_type(int, float), lambda values: values.x + values.y)
    assert dispatch(point_type(1, 2.0)) == 3.0
    with pytest.raises(TypeError, match="No dispatch rule for"):
        dispatch((1, 2.0))
