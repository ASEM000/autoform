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
from autoform.core import using_interpreter
from autoform.intercept import checkpoint, checkpoint_p
from tests import CountingInterpreter, aexecute, append_bang, execute


@pytest.fixture
def checkpoint_switch():
    branch = af.trace(lambda x: checkpoint(x, key="save", collection="cache"))("x")
    return af.trace(lambda x: af.switch("a", {"a": branch}, x))("x")


def distinct_checkpoints(x):
    a = checkpoint(x, key="first", collection="debug")
    b = checkpoint(x, key="second", collection="debug")
    return af.string.concat(a, b)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_memoize_duplicate_primitives(executor):
    counter = CountingInterpreter()

    def program(x):
        a = af.string.concat(x, "!")
        b = af.string.concat(x, "!")
        return af.string.concat(a, b)

    ir = af.trace(program)("test")
    with using_interpreter(counter), af.memoize():
        actual = executor(ir, "hello")
        assert actual == "hello!hello!"
    assert counter.calls == 2


def test_memoize_scope():
    counter = CountingInterpreter()
    ir = af.trace(append_bang)("test")
    with using_interpreter(counter):
        for _ in range(2):
            with af.memoize():
                assert ir.call("hello") == "hello!"
    assert counter.calls == 2


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("transform", [lambda ir: ir, af.sched], ids=["switch", "scheduled-switch"])
def test_memoize_preserves_nested_checkpoints(executor, checkpoint_switch, transform):
    ir = transform(checkpoint_switch)
    with af.collect(collection="cache") as saved, af.memoize():
        for _ in range(4):
            assert executor(ir, "x") == "x"
    assert saved == {"save": ["x"] * 4}


def test_memoize_trace_preserves_nested_checkpoints(checkpoint_switch):
    def program(x):
        with af.memoize():
            return checkpoint_switch.call(x), checkpoint_switch.call(x)

    ir = af.trace(program)("x")
    with af.collect(collection="cache") as saved:
        assert ir.call("x") == ("x", "x")
    assert saved == {"save": ["x", "x"]}


def test_memoize_preserves_distinct_checkpoints():
    ir = af.trace(distinct_checkpoints)("test")
    assert [eqn.params["key"] for eqn in ir.eqns if eqn.prim is checkpoint_p] == ["first", "second"]
    with af.collect(collection="debug") as saved, af.memoize():
        assert ir.call("hi") == "hihi"
    assert saved == {"first": ["hi"], "second": ["hi"]}


def test_memoize_trace_preserves_distinct_checkpoints():
    with af.memoize():
        ir = af.trace(distinct_checkpoints)("test")
    assert [eqn.params["key"] for eqn in ir.eqns if eqn.prim is checkpoint_p] == ["first", "second"]
    with af.collect(collection="debug") as saved:
        assert ir.call("hi") == "hihi"
    assert saved == {"first": ["hi"], "second": ["hi"]}


def test_memoize_preserves_repeated_checkpoint():
    ir = af.trace(lambda x: checkpoint(x, key="val", collection="debug"))("test")
    with af.collect(collection="debug") as saved, af.memoize():
        assert ir.call("hello") == ir.call("hello") == "hello"
    assert saved == {"val": ["hello", "hello"]}


@pytest.mark.parametrize(
    "transform, first_args, second_args, expected, misses",
    [
        pytest.param(
            lambda ir: ir,
            ("hello",),
            ("hello",),
            ("hello!", "hello!"),
            1,
            id="same-input",
        ),
        pytest.param(
            lambda ir: ir,
            ("hello",),
            ("world",),
            ("hello!", "world!"),
            2,
            id="different-input",
        ),
        pytest.param(
            af.batch,
            (["a", "b"],),
            (["a", "b"],),
            (["a!", "b!"], ["a!", "b!"]),
            3,
            id="batch-cache-hit",
        ),
        pytest.param(
            af.pushforward,
            (("primal",), ("tangent",)),
            (("primal",), ("tangent",)),
            (("primal!", "tangent"), ("primal!", "tangent")),
            3,
            id="pushforward-cache-hit",
        ),
        pytest.param(
            af.pullback,
            (("primal",), "cotangent"),
            (("primal",), "cotangent"),
            (("primal!", ("cotangent",)), ("primal!", ("cotangent",))),
            2,
            id="pullback-cache-hit",
        ),
        pytest.param(
            af.batch,
            (["a", "b"],),
            (["c", "d"],),
            (["a!", "b!"], ["c!", "d!"]),
            6,
            id="batch-cache-miss",
        ),
    ],
)
def test_memoize_transformed_ir(transform, first_args, second_args, expected, misses):
    counter = CountingInterpreter()
    ir = transform(af.trace(append_bang)("test"))
    with using_interpreter(counter), af.memoize():
        results = ir.call(*first_args), ir.call(*second_args)
    assert results == expected
    assert counter.calls == misses
