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
from autoform.intercept import checkpoint
from tests import aexecute, append_bang, execute, trace_ir


def checkpoint_chain(marks):
    def program(x):
        for key, collection in marks:
            x = checkpoint(x, key=key, collection=collection)
        return x

    return program


def expensive_checkpoint(x):
    expensive = af.string.concat("EXPENSIVE:", x)
    cached = checkpoint(expensive, key="result", collection="cache")
    return af.string.concat("Got: ", cached)


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
    ir = trace_ir(program, "test")
    with af.collect(collection=collection) as collected:
        assert ir.call("hello") == "hello"
    assert collected == {key: ["hello"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected, captured",
    [
        pytest.param(lambda ir: ir, ("hello",), "hello", ["hello"], id="call"),
        pytest.param(af.batch, (["a", "b", "c"],), ["a", "b", "c"], ["a", "b", "c"], id="batch"),
        pytest.param(
            af.pushforward,
            (("primal",), ("tangent",)),
            ("primal", "tangent"),
            ["primal", "tangent"],
            id="push",
        ),
        pytest.param(
            af.pullback,
            (("primal",), "cotangent"),
            ("primal", ("cotangent",)),
            ["primal", "cotangent"],
            id="pull",
        ),
    ],
)
def test_collect_transforms(executor, transform, args, expected, captured):
    ir = transform(trace_ir(checkpoint_chain([("val", "debug")]), "test"))
    with af.collect(collection="debug") as collected:
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
    ir = trace_ir(checkpoint_chain(marks), "test")
    with af.collect(collection=collection) as collected:
        assert ir.call("hello") == "hello"
    assert collected == expected


def test_collect_without_checkpoints():
    ir = trace_ir(append_bang, "test")
    with af.collect(collection="debug") as collected:
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
        a = checkpoint(af.string.concat(prefix, x), key=keys[0], collection="debug")
        return checkpoint(af.string.concat(a, suffix), key=keys[1], collection="debug")

    ir = trace_ir(program, "test")
    with af.collect(collection="debug") as collected:
        assert ir.call(arg) == expected
    assert collected == captured


def test_nested_collectors():
    ir = trace_ir(checkpoint_chain([("debug", "debug"), ("cache", "cache")]), "test")
    with af.collect(collection="debug") as debug, af.collect(collection="cache") as cache:
        assert ir.call("hello") == "hello"
    assert debug == {"debug": ["hello"]}
    assert cache == {"cache": ["hello"]}


def test_collect_switch_branch():
    branches = {
        key: trace_ir(
            lambda x: checkpoint(af.string.concat(key + ": ", x), key="result", collection="debug"),
            "x",
        )
        for key in ("a", "b")
    }
    ir = trace_ir(lambda x: af.switch("a", branches, x), "input")
    with af.collect(collection="debug") as collected:
        assert ir.call("hello") == "a: hello"
    assert collected == {"result": ["a: hello"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_inject_replaces_value_and_restores_context(executor):
    ir = trace_ir(
        lambda x: checkpoint(af.string.concat("Hello, ", x), key="greeting", collection="cache"),
        "test",
    )
    with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
        actual = executor(ir, "World")
        assert actual == "CACHED"
    assert ir.call("World") == "Hello, World"


def test_inject_partial():
    def program(x):
        a = checkpoint(x, key="first", collection="cache")
        return checkpoint(af.string.concat(a, "!"), key="second", collection="cache")

    ir = trace_ir(program, "test")
    with af.inject(collection="cache", values={"first": ["INJECTED"]}):
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
    ir = trace_ir(checkpoint_chain(marks), "test")
    with af.inject(collection="cache", values=values):
        assert ir.call(arg) == expected


def test_inject_trace_specializes_and_dce_removes_dead_input():
    ir = trace_ir(expensive_checkpoint, "test")
    assert len(ir.eqns) == 3

    def wrapped(x):
        with af.inject(collection="cache", values={"result": ["CACHED"]}):
            return ir.call("ignored")

    specialized = trace_ir(wrapped, "example")
    assert [eqn.prim for eqn in specialized.eqns] == [af.string.concat_p] * 2
    optimized = af.dce(specialized)
    assert len(optimized.eqns) == 1
    assert optimized.call("any_input") == "Got: CACHED"


def test_inject_dce_preserves_remaining_checkpoint():
    def program(x):
        first = checkpoint(af.string.concat("step1:", x), key="first", collection="cache")
        second = checkpoint(af.string.concat("step2:", first), key="second", collection="cache")
        return af.string.concat("final:", second)

    ir = trace_ir(program, "test")
    assert len(ir.eqns) == 5

    def wrapped(x):
        with af.inject(collection="cache", values={"first": ["CACHED1"]}):
            return ir.call(x)

    optimized = af.dce(trace_ir(wrapped, "example"))
    assert optimized.call("input") == "final:step2:CACHED1"


def test_inject_batch_encounter_order():
    ir = af.batch(trace_ir(expensive_checkpoint, "test"))
    with af.inject(collection="cache", values={"result": ["A", "B"]}):
        assert ir.call(["x", "y"]) == ["Got: A", "Got: B"]
