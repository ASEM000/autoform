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

import asyncio
import json
from types import SimpleNamespace

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


@pytest.fixture
def waiting_client():
    class WaitingClient:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.error = None

        def responses(self, *, input, model, **kwargs):
            x = json.loads(input)["values"]["prompt"]
            return SimpleNamespace(output_text=json.dumps({"output": f"{model}|{x}"}))

        async def aresponses(self, **kwargs):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            if self.error is not None:
                raise self.error
            return self.responses(**kwargs)

    return WaitingClient()


def fill_text(x, *, model="m1"):
    return af.lm.fill({"prompt": x, "output": af.lm.Str()}, model=model)["output"]


@pytest.mark.parametrize(
    "transform, args, model, expected, misses",
    [
        pytest.param(af.sched, ("x", "x"), "m1", ("m1|x", "m1|x"), 1, id="scheduled-duplicates"),
        pytest.param(af.sched, ("x", "y"), "m1", ("m1|x", "m1|y"), 2, id="different-inputs"),
        pytest.param(af.sched, ("x", "x"), "m2", ("m1|x", "m2|x"), 2, id="different-models"),
        pytest.param(
            af.batch,
            (["x", "x"], ["x", "x"]),
            "m1",
            (["m1|x", "m1|x"], ["m1|x", "m1|x"]),
            1,
            id="batched-duplicates",
        ),
    ],
)
def test_memoize_concurrent_fill(transform, args, model, expected, misses, waiting_client):
    ir = transform(af.trace(lambda x, y: (fill_text(x), fill_text(y, model=model)))("x", "y"))

    async def run():
        with af.lm.client(waiting_client), af.memoize():
            task = asyncio.create_task(ir.acall(*args))
            await waiting_client.started.wait()
            await asyncio.sleep(0)
            assert waiting_client.calls == misses
            waiting_client.release.set()
            assert await task == expected
            assert await ir.acall(*args) == expected
            assert waiting_client.calls == misses

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_memoize_concurrent_failure_retries(waiting_client):
    ir = af.trace(fill_text)("x")

    async def run():
        with af.lm.client(waiting_client), af.memoize():
            first = asyncio.create_task(ir.acall("x"))
            await waiting_client.started.wait()
            second = asyncio.create_task(ir.acall("x"))
            await asyncio.sleep(0)
            waiting_client.error = RuntimeError("provider failed")
            waiting_client.release.set()
            errors = await asyncio.gather(first, second, return_exceptions=True)
            assert all(error is waiting_client.error for error in errors)
            waiting_client.error = None
            assert await ir.acall("x") == "m1|x"
            assert waiting_client.calls == 2

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("cancel_original", [True, False], ids=["original", "duplicate"])
def test_memoize_concurrent_cancellation(cancel_original, waiting_client):
    ir = af.trace(fill_text)("x")

    async def run():
        with af.lm.client(waiting_client), af.memoize():
            first = asyncio.create_task(ir.acall("x"))
            await waiting_client.started.wait()
            second = asyncio.create_task(ir.acall("x"))
            await asyncio.sleep(0)
            cancelled = first if cancel_original else second
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            if cancel_original:
                with pytest.raises(asyncio.CancelledError):
                    await second
            waiting_client.release.set()
            if not cancel_original:
                assert await first == "m1|x"
            assert await ir.acall("x") == "m1|x"
            assert waiting_client.calls == (2 if cancel_original else 1)

    asyncio.run(asyncio.wait_for(run(), timeout=5))
