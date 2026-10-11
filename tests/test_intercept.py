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

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.control as control
import autoform.src.dead as dead
import autoform.src.intercept as intercept
import autoform.src.stage as stage
import autoform.src.string as string
from tests import aexecute, append_bang, execute


def checkpoint_chain(marks):
    def program(x):
        for key, collection in marks:
            x = intercept.checkpoint(x, key=key, collection=collection)
        return x

    return program


def expensive_checkpoint(x):
    expensive = string.concat("EXPENSIVE:", x)
    cached = intercept.checkpoint(expensive, key="result", collection="cache")
    return string.concat("Got: ", cached)


@pytest.mark.parametrize(
    "key, collection",
    [
        pytest.param("my_key", "my_col", id="strings"),
        pytest.param(100, 42, id="integers"),
        pytest.param(("b", 2), ("a", 1), id="tuples"),
    ],
)
def test_checkpoint_identity_and_collection(key, collection):
    program = checkpoint_chain([(key, collection)])
    assert program("hello") == "hello"
    ir = stage.trace(program)("test")
    with intercept.collect(collection=collection) as collected:
        assert ir.call("hello") == "hello"
    assert collected == {key: ["hello"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected, captured",
    [
        pytest.param(lambda ir: ir, ("hello",), "hello", ["hello"], id="call"),
        pytest.param(axis.batch, (["a", "b", "c"],), ["a", "b", "c"], ["a", "b", "c"], id="batch"),
        pytest.param(
            ad.pushforward,
            (("primal",), ("tangent",)),
            ("primal", "tangent"),
            ["primal", "tangent"],
            id="push",
        ),
        pytest.param(
            ad.pullback,
            (("primal",), "cotangent"),
            ("primal", ("cotangent",)),
            ["primal", "cotangent"],
            id="pull",
        ),
    ],
)
def test_collect_transforms(executor, transform, args, expected, captured):
    ir = transform(stage.trace(checkpoint_chain([("val", "debug")]))("test"))
    with intercept.collect(collection="debug") as collected:
        actual = executor(ir, *args)
        assert actual == expected
    assert collected == {"val": captured}


@pytest.mark.parametrize(
    "marks, collection, expected",
    [
        pytest.param(
            [("debug_val", "debug"), ("other_val", "other")],
            "debug",
            {"debug_val": ["hello"]},
            id="matching-collection",
        ),
        pytest.param(
            [("debug_val", "debug"), ("metrics_val", "metrics")],
            "metrics",
            {"metrics_val": ["hello"]},
            id="second-collection",
        ),
        pytest.param(
            [("a", "one"), ("b", "two")],
            ...,
            {"a": ["hello"], "b": ["hello"]},
            id="all-collections",
        ),
        pytest.param([("val", "other")], "debug", {}, id="no-match"),
    ],
)
def test_collect_filter(marks, collection, expected):
    ir = stage.trace(checkpoint_chain(marks))("test")
    with intercept.collect(collection=collection) as collected:
        assert ir.call("hello") == "hello"
    assert collected == expected


def test_collect_without_checkpoints():
    ir = stage.trace(append_bang)("test")
    with intercept.collect(collection="debug") as collected:
        assert ir.call("hello") == "hello!"
    assert collected == {}


@pytest.mark.parametrize(
    "prefix, suffix, keys, arg, expected, captured",
    [
        pytest.param(
            "",
            "!",
            ("first", "second"),
            "hi",
            "hi!",
            {"first": ["hi"], "second": ["hi!"]},
            id="chain",
        ),
        pytest.param(
            "Q: ",
            " A: 42",
            ("prompt", "response"),
            "What?",
            "Q: What? A: 42",
            {"prompt": ["Q: What?"], "response": ["Q: What? A: 42"]},
            id="response",
        ),
    ],
)
def test_collect_computed_values(prefix, suffix, keys, arg, expected, captured):
    def program(x):
        a = intercept.checkpoint(string.concat(prefix, x), key=keys[0], collection="debug")
        return intercept.checkpoint(string.concat(a, suffix), key=keys[1], collection="debug")

    ir = stage.trace(program)("test")
    with intercept.collect(collection="debug") as collected:
        assert ir.call(arg) == expected
    assert collected == captured


def test_nested_collectors():
    ir = stage.trace(checkpoint_chain([("debug", "debug"), ("cache", "cache")]))("test")
    with (
        intercept.collect(collection="debug") as debug,
        intercept.collect(collection="cache") as cache,
    ):
        assert ir.call("hello") == "hello"
    assert debug == {"debug": ["hello"]}
    assert cache == {"cache": ["hello"]}


def test_collect_switch_branch():
    branches = {
        key: stage.trace(
            lambda x: intercept.checkpoint(
                string.concat(key + ": ", x), key="result", collection="debug"
            ),
        )("x")
        for key in ("a", "b")
    }
    ir = stage.trace(lambda x: control.switch("a", branches, x))("input")
    with intercept.collect(collection="debug") as collected:
        assert ir.call("hello") == "a: hello"
    assert collected == {"result": ["a: hello"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_inject_replaces_value_and_restores_context(executor):
    ir = stage.trace(
        lambda x: intercept.checkpoint(
            string.concat("Hello, ", x), key="greeting", collection="cache"
        ),
    )("test")
    with intercept.inject(collection="cache", values={"greeting": ["CACHED"]}):
        actual = executor(ir, "World")
        assert actual == "CACHED"
    assert ir.call("World") == "Hello, World"


def test_inject_partial():
    def program(x):
        a = intercept.checkpoint(x, key="first", collection="cache")
        return intercept.checkpoint(string.concat(a, "!"), key="second", collection="cache")

    ir = stage.trace(program)("test")
    with intercept.inject(collection="cache", values={"first": ["INJECTED"]}):
        assert ir.call("ignored") == "INJECTED!"


@pytest.mark.parametrize(
    "marks, values, arg, expected",
    [
        pytest.param(
            [("val", "cache"), ("val", "other")],
            {"val": ["CACHED"]},
            "input",
            "CACHED",
            id="collection-filter",
        ),
        pytest.param([("val", "cache")], {}, "hello", "hello", id="empty"),
        pytest.param(
            [("val", "cache")],
            {"other": ["PLANTED"]},
            "hello",
            "hello",
            id="unmatched-key",
        ),
    ],
)
def test_inject_filter(marks, values, arg, expected):
    ir = stage.trace(checkpoint_chain(marks))("test")
    with intercept.inject(collection="cache", values=values):
        assert ir.call(arg) == expected


def test_inject_trace_specializes_and_dce_removes_dead_input():
    ir = stage.trace(expensive_checkpoint)("test")
    assert len(ir.eqns) == 3

    def wrapped(x):
        with intercept.inject(collection="cache", values={"result": ["CACHED"]}):
            return ir.call("ignored")

    specialized = stage.trace(wrapped)("example")
    assert [eqn.prim for eqn in specialized.eqns] == [string.concat_p] * 2
    optimized = dead.dce(specialized)
    assert len(optimized.eqns) == 1
    assert optimized.call("any_input") == "Got: CACHED"


def test_inject_dce_preserves_remaining_checkpoint():
    def program(x):
        first = intercept.checkpoint(string.concat("step1:", x), key="first", collection="cache")
        second = intercept.checkpoint(
            string.concat("step2:", first), key="second", collection="cache"
        )
        return string.concat("final:", second)

    ir = stage.trace(program)("test")
    assert len(ir.eqns) == 5

    def wrapped(x):
        with intercept.inject(collection="cache", values={"first": ["CACHED1"]}):
            return ir.call(x)

    optimized = dead.dce(stage.trace(wrapped)("example"))
    assert optimized.call("input") == "final:step2:CACHED1"


def test_inject_batch_encounter_order():
    ir = axis.batch(stage.trace(expensive_checkpoint)("test"))
    with intercept.inject(collection="cache", values={"result": ["A", "B"]}):
        assert ir.call(["x", "y"]) == ["Got: A", "Got: B"]
