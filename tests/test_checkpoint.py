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
from autoform.checkpoint import checkpoint
from autoform.core import Interpreter, active_interpreter, using_interpreter


class CountingInterpreter(Interpreter):
    def __init__(self):
        self.parent = active_interpreter.get()
        self.call_count = 0

    def interpret(self, prim, in_tree, /, **params):
        self.call_count += 1
        return self.parent.interpret(prim, in_tree, **params)

    async def ainterpret(self, prim, in_tree, /, **params):
        self.call_count += 1
        return await self.parent.ainterpret(prim, in_tree, **params)


class TestSow:
    def test_impl_is_identity(self):
        result = af.checkpoint("hello", key="test", collection="debug")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return af.checkpoint(x, key="my_name", collection="my_tag")

        ir = af.trace(func)("test")
        assert len(ir.ir_eqns) == 1
        assert ir.ir_eqns[0].prim.name == "checkpoint"
        assert ir.ir_eqns[0].params["collection"] == "my_tag"
        assert ir.ir_eqns[0].params["key"] == "my_name"

    def test_run_ir(self):
        def func(x):
            return af.checkpoint(x, key="value", collection="test")

        ir = af.trace(func)("test")
        result = ir.call("hello")
        assert result == "hello"

    def test_hashable_tags_and_names(self):
        assert af.checkpoint("x", key="str_name", collection="str_tag") == "x"
        assert af.checkpoint("x", key=100, collection=42) == "x"
        assert af.checkpoint("x", key=("b", 2), collection=("a", 1)) == "x"

    def test_pushforward_preserves_both(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="test")

        ir = af.trace(func)("a")
        pf_ir = af.pushforward(ir)
        primal_out, tangent_out = pf_ir.call(("primal",), ("tangent",))
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_preserves_cotangent(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="test")

        ir = af.trace(func)("a")
        pb_ir = af.pullback(ir)
        primal_out, cotangent_in = pb_ir.call(("primal",), "cotangent")
        assert primal_out == "primal"
        assert cotangent_in == ("cotangent",)

    def test_batch(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="test")

        ir = af.trace(func)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain(self):
        def func(x):
            sowed = af.checkpoint(x, key="input", collection="debug")
            return af.concat("[", sowed, "]")

        ir = af.trace(func)("a")
        result = ir.call("hello")
        assert result == "[hello]"


class TestRunAndReap:
    def test_reap_single_sow(self):
        def func(x):
            return af.checkpoint(x, key="captured", collection="debug")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = ir.call("hello")
        assert result == "hello"
        assert collected == {"captured": ["hello"]}

    def test_reap_multiple_sows_same_tag(self):
        def func(x):
            a = af.checkpoint(x, key="first", collection="debug")
            b = af.concat(a, "!")
            c = af.checkpoint(b, key="second", collection="debug")
            return c

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = ir.call("hi")
        assert result == "hi!"
        assert collected == {"first": ["hi"], "second": ["hi!"]}

    def test_reap_filters_by_tag(self):
        def func(x):
            a = af.checkpoint(x, key="debug_val", collection="debug")
            b = af.checkpoint(a, key="metrics_val", collection="metrics")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as debug_collected:
            ir.call("hello")
        assert debug_collected == {"debug_val": ["hello"]}

        with af.collect(collection="metrics") as metrics_collected:
            ir.call("hello")
        assert metrics_collected == {"metrics_val": ["hello"]}

    def test_reap_empty_when_no_match(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="other")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = ir.call("hello")
        assert result == "hello"
        assert collected == {}

    def test_reap_with_no_sows(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = ir.call("hello")
        assert result == "hello!"
        assert collected == {}

    def test_reap_preserves_execution(self):
        def func(x):
            a = af.checkpoint(af.format("Q: {}", x), key="prompt", collection="debug")
            response = af.concat(a, " A: 42")
            return af.checkpoint(response, key="response", collection="debug")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = ir.call("What?")
        assert result == "Q: What? A: 42"
        assert collected["prompt"] == ["Q: What?"]
        assert collected["response"] == ["Q: What? A: 42"]


class TestRunAndPlant:
    def test_plant_overrides_sow(self):
        def func(x):
            return af.checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.trace(func)("test")

        result = ir.call("World")
        assert result == "Hello, World"

        with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
            result = ir.call("World")
        assert result == "CACHED"

    def test_plant_partial(self):
        def func(x):
            a = af.checkpoint(x, key="first", collection="cache")
            b = af.checkpoint(af.concat(a, "!"), key="second", collection="cache")
            return b

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"first": ["PLANTED"]}):
            result = ir.call("ignored")
        assert result == "PLANTED!"

    def test_plant_filters_by_tag(self):
        def func(x):
            a = af.checkpoint(x, key="val", collection="cache")
            b = af.checkpoint(a, key="val", collection="other")
            return b

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"val": ["CACHED"]}):
            result = ir.call("input")
        assert result == "CACHED"

    def test_plant_empty_dict(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="cache")

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={}):
            result = ir.call("hello")
        assert result == "hello"

    def test_plant_unmatched_name(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="cache")

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"other": ["PLANTED"]}):
            result = ir.call("hello")
        assert result == "hello"


class TestTransformThenReap:
    def test_reap_captures_during_pushforward(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pf_ir = af.pushforward(ir)

        with af.collect(collection="debug") as collected:
            result = pf_ir.call(("primal",), ("tangent",))
        assert collected == {"val": ["primal", "tangent"]}

    def test_reap_captures_during_pullback(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pb_ir = af.pullback(ir)

        with af.collect(collection="debug") as collected:
            result = pb_ir.call(("primal",), "cotangent")
        assert collected == {"val": ["primal", "cotangent"]}

    def test_reap_captures_during_batch(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        with af.collect(collection="debug") as collected:
            result = batched.call(["a", "b", "c"])
        assert result == ["a", "b", "c"]
        assert collected == {"val": ["a", "b", "c"]}

    def test_reap_captures_in_switch_branches(self):
        def branch_a(x):
            return af.checkpoint(af.concat("a: ", x), key="result", collection="debug")

        def branch_b(x):
            return af.checkpoint(af.concat("b: ", x), key="result", collection="debug")

        ir_a = af.trace(branch_a)("x")
        ir_b = af.trace(branch_b)("x")

        def func(x):
            return af.switch("a", {"a": ir_a, "b": ir_b}, x)

        ir = af.trace(func)("input")
        with af.collect(collection="debug") as collected:
            result = ir.call("hello")
        assert result == "a: hello"
        assert collected == {"result": ["a: hello"]}


class TestInjectAndDCE:
    def test_inject_trace_creates_literal(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.checkpoint(expensive, key="result", collection="cache")
            return af.concat("Got: ", cached)

        ir = af.trace(program)("test")

        assert len(ir.ir_eqns) == 3

        def wrapped(x):
            with af.inject(collection="cache", values={"result": ["CACHED"]}):
                return ir.call("ignored")

        traced_ir = af.trace(wrapped)("example")
        assert len(traced_ir.ir_eqns) == 2

        last_eqn = traced_ir.ir_eqns[-1]
        assert last_eqn.prim.name == "concat"

    def test_dce_removes_dead_code_after_inject(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.checkpoint(expensive, key="result", collection="cache")
            return af.concat("Got: ", cached)

        ir = af.trace(program)("test")

        def wrapped(x):
            with af.inject(collection="cache", values={"result": ["CACHED"]}):
                return ir.call("ignored")

        traced_ir = af.trace(wrapped)("example")
        assert len(traced_ir.ir_eqns) == 2
        optimized_ir = af.dce(traced_ir)
        assert len(optimized_ir.ir_eqns) == 1

        result = optimized_ir.call("any_input")
        assert result == "Got: CACHED"

    def test_inject_dce_with_multiple_marks(self):
        def program(x):
            step1 = af.concat("step1:", x)
            saved1 = af.checkpoint(step1, key="first", collection="cache")
            step2 = af.concat("step2:", saved1)
            saved2 = af.checkpoint(step2, key="second", collection="cache")
            return af.concat("final:", saved2)

        ir = af.trace(program)("test")
        assert len(ir.ir_eqns) == 5

        def wrapped(x):
            with af.inject(collection="cache", values={"first": ["CACHED1"]}):
                return ir.call(x)

        traced_ir = af.trace(wrapped)("example")
        optimized_ir = af.dce(traced_ir)
        result = optimized_ir.call("input")
        assert result == "final:step2:CACHED1"

    def test_inject_works_with_nested_transforms(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.checkpoint(expensive, key="result", collection="cache")
            return af.concat("Got: ", cached)

        ir = af.trace(program)("test")
        batched_ir = af.batch(ir)

        with af.inject(collection="cache", values={"result": ["A", "B"]}):
            result = batched_ir.call(["x", "y"])

        assert result == ["Got: A", "Got: B"]


class TestSplit:
    def test_split_mark_at_end(self):
        def program(x):
            y = af.splitpoint(af.format("{}", x), key="s")
            return y

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="s")

        assert len(lhs.ir_eqns) == 1
        assert lhs.ir_eqns[0].prim.name == "format"

        assert len(rhs.ir_eqns) == 0

        lhs_result = lhs.call("Test")
        assert lhs_result == "Test"
        rhs_result = rhs.call(lhs_result)
        assert rhs_result == lhs_result

    def test_split_mark_in_middle(self):
        def program(x):
            y = af.format("Hello {}", x)
            z = af.splitpoint(y, key="mid")
            w = af.format("Result: {}", z)
            return w

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="mid")

        assert len(lhs.ir_eqns) == 1

        assert len(rhs.ir_eqns) == 1
        assert rhs.ir_eqns[0].prim.name == "format"

        lhs_result = lhs.call("World")
        assert lhs_result == "Hello World"

        rhs_result = rhs.call(lhs_result)
        assert rhs_result == "Result: Hello World"

        ir_full = af.trace(program)("x")
        full_result = ir_full.call("World")
        assert rhs_result == full_result

    def test_split_returns_marked_value_not_last_preceding_value(self):
        def program(x):
            marked = af.concat(x, "a")
            af.concat(x, "b")
            marked = af.splitpoint(marked, key="mid")
            return marked

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="mid")

        lhs_result = lhs.call("q")
        assert lhs_result == "qa"

        rhs_result = rhs.call(lhs_result)
        assert rhs_result == lhs_result
        assert rhs_result == ir.call("q")

    def test_split_rhs_with_extra_stuff_fails_on_execution(self):
        def program(x):
            y = af.concat(x, "a")
            z = af.concat(x, "b")
            y = af.splitpoint(y, key="mid")
            return af.concat(y, z)

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="mid")

        assert lhs.call("q") == "qa"
        with pytest.raises(KeyError):
            rhs.call("qa")

    def test_split_composition_equals_full(self):
        def program(x):
            a = af.format("Step1: {}", x)
            b = af.splitpoint(a, key="step1")
            c = af.format("Step2: {}", b)
            d = af.concat(c, "!")
            return d

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="step1")
        ir_full = af.trace(program)("x")

        for inp in ["a", "hello", "test123"]:
            lhs_result = lhs.call(inp)
            rhs_result = rhs.call(lhs_result)
            full_result = ir_full.call(inp)
            assert rhs_result == full_result

    def test_split_not_found_raises(self):
        def program(x):
            return af.format("{}", x)

        ir = af.trace(program)("...")
        with pytest.raises(AssertionError, match="could not find"):
            af.split(ir, key="nonexistent")

    def test_split_with_multiple_marks(self):
        def program(x):
            a = af.splitpoint(x, key="first")
            b = af.format("{}", a)
            c = af.splitpoint(b, key="second")
            d = af.concat(c, "!")
            return d

        ir = af.trace(program)("...")
        lhs1, rhs1 = af.split(ir, key="first")
        assert len(lhs1.ir_eqns) == 0
        assert len(rhs1.ir_eqns) == 3

        ir2 = af.trace(program)("...")
        lhs2, rhs2 = af.split(ir2, key="second")
        assert len(lhs2.ir_eqns) == 2
        assert len(rhs2.ir_eqns) == 1


class TestSplitpointPreservedThroughTransforms:
    def test_splitpoint_preserved_after_pushforward(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        assert len(pf_ir.ir_eqns) == 1
        assert pf_ir.ir_eqns[0].prim.name == "pushforward_call"
        nested_ir = pf_ir.ir_eqns[0].params["ir"]
        splitpoints = [eqn for eqn in nested_ir.ir_eqns if eqn.prim.name == "splitpoint"]
        assert len(splitpoints) == 1
        assert splitpoints[0].params["key"] == "mid"

    def test_splitpoint_preserved_after_pullback(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        assert len(pb_ir.ir_eqns) == 1
        assert pb_ir.ir_eqns[0].prim.name == "pullback_call"
        nested_ir = pb_ir.ir_eqns[0].params["ir"]
        splitpoints = [eqn for eqn in nested_ir.ir_eqns if eqn.prim.name == "splitpoint"]
        assert len(splitpoints) == 1
        assert splitpoints[0].params["key"] == "mid"

    def test_splitpoint_preserved_after_batch(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        batch_ir = af.batch(ir, in_axes=True)

        assert len(batch_ir.ir_eqns) == 1
        assert batch_ir.ir_eqns[0].prim.name == "batch_call"
        nested_ir = batch_ir.ir_eqns[0].params["ir"]
        splitpoints = [eqn for eqn in nested_ir.ir_eqns if eqn.prim.name == "splitpoint"]
        assert len(splitpoints) == 1
        assert splitpoints[0].params["key"] == "mid"


class TestSplitOnTransformedIR:
    def test_split_on_pushforward_ir(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        lhs, rhs = af.split(pf_ir, key="mid")

        assert len(lhs.ir_eqns) == 1
        assert lhs.ir_eqns[0].prim.name == "pushforward_call"
        nested_lhs = lhs.ir_eqns[0].params["ir"]

        assert nested_lhs.ir_eqns[-1].prim.name == "concat"

        assert len(rhs.ir_eqns) == 1
        assert rhs.ir_eqns[0].prim.name == "pushforward_call"
        nested_rhs = rhs.ir_eqns[0].params["ir"]

        assert len(nested_rhs.ir_eqns) == 1
        assert nested_rhs.ir_eqns[0].prim.name == "concat"

    def test_split_on_pullback_ir(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        lhs, rhs = af.split(pb_ir, key="mid")

        assert len(lhs.ir_eqns) == 1
        assert lhs.ir_eqns[0].prim.name == "pullback_call"
        assert len(rhs.ir_eqns) == 1
        assert rhs.ir_eqns[0].prim.name == "pullback_call"

    def test_split_on_batch_ir(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        batch_ir = af.batch(ir, in_axes=True)

        lhs, rhs = af.split(batch_ir, key="mid")

        assert len(lhs.ir_eqns) == 1
        assert lhs.ir_eqns[0].prim.name == "batch_call"
        assert len(rhs.ir_eqns) == 1
        assert rhs.ir_eqns[0].prim.name == "batch_call"

    def test_split_on_transformed_ir_execution(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")

        lhs, rhs = af.split(ir, key="mid")
        assert lhs.call("hello") == "hello!"
        assert rhs.call("hello!") == "hello!?"

    def test_split_on_double_pushforward(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pf_pf_ir = af.pushforward(af.pushforward(ir))

        lhs, rhs = af.split(pf_pf_ir, key="mid")

        assert len(lhs.ir_eqns) == 1
        assert lhs.ir_eqns[0].prim.name == "pushforward_call"
        assert len(rhs.ir_eqns) == 1
        assert rhs.ir_eqns[0].prim.name == "pushforward_call"

        inner_lhs = lhs.ir_eqns[0].params["ir"]
        inner_rhs = rhs.ir_eqns[0].params["ir"]
        assert inner_lhs.ir_eqns[0].prim.name == "pushforward_call"
        assert inner_rhs.ir_eqns[0].prim.name == "pushforward_call"

    def test_split_on_batch_pushforward(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        b_pf_ir = af.batch(af.pushforward(ir), in_axes=True)

        lhs, rhs = af.split(b_pf_ir, key="mid")

        assert len(lhs.ir_eqns) == 1
        assert lhs.ir_eqns[0].prim.name == "batch_call"
        assert len(rhs.ir_eqns) == 1
        assert rhs.ir_eqns[0].prim.name == "batch_call"

        inner_lhs = lhs.ir_eqns[0].params["ir"]
        inner_rhs = rhs.ir_eqns[0].params["ir"]
        assert inner_lhs.ir_eqns[0].prim.name == "pushforward_call"
        assert inner_rhs.ir_eqns[0].prim.name == "pushforward_call"

    def test_split_on_triple_nested(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        triple = af.pushforward(af.batch(af.pushforward(ir), in_axes=True))

        lhs, rhs = af.split(triple, key="mid")

        assert lhs.ir_eqns[0].prim.name == "pushforward_call"
        assert rhs.ir_eqns[0].prim.name == "pushforward_call"


class TestMemoizeBasic:
    def test_memoize_caches_duplicate_calls(self):
        def func(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(func)("test")

        counter = CountingInterpreter()
        with using_interpreter(counter):
            with af.memoize():
                result = ir.call("hello")

        assert result == "hello!hello!"
        assert counter.call_count == 2

    def test_memoize_returns_correct_result(self):
        def func(x):
            a = af.concat(x, "!")
            return a

        ir = af.trace(func)("test")

        with af.memoize():
            result = ir.call("hello")

        assert result == "hello!"

    def test_memoize_different_inputs_not_cached(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        counter = CountingInterpreter()
        with using_interpreter(counter):
            with af.memoize():
                r1 = ir.call("hello")
                r2 = ir.call("world")

        assert r1 == "hello!"
        assert r2 == "world!"
        assert counter.call_count == 2


class TestMemoizeWithCheckpoints:
    def test_memoize_with_checkpoint(self):
        def func(x):
            a = checkpoint(af.concat(x, "!"), key="val", collection="debug")
            return a

        ir = af.trace(func)("test")

        with af.memoize():
            with af.collect(collection="debug") as collected:
                result = ir.call("hello")

        assert result == "hello!"
        assert collected == {"val": ["hello!"]}

    def test_memoize_does_not_merge_different_checkpoints(self):
        def func(x):
            a = checkpoint(x, key="first", collection="debug")
            b = checkpoint(x, key="second", collection="debug")
            return af.concat(a, b)

        ir = af.trace(func)("test")

        with af.memoize():
            with af.collect(collection="debug") as collected:
                result = ir.call("hi")

        assert result == "hihi"
        assert "first" in collected
        assert "second" in collected

    def test_memoize_inside_trace_does_not_dedup_checkpoint_calls(self):
        def func(x):
            with af.memoize():
                a = checkpoint(x, key="first", collection="debug")
                b = checkpoint(x, key="second", collection="debug")
                return af.concat(a, b)

        ir = af.trace(func)("test")

        checkpoint_eqns = [eqn for eqn in ir.ir_eqns if eqn.prim.name == "checkpoint"]
        assert len(checkpoint_eqns) == 2
        assert checkpoint_eqns[0].params["key"] == "first"
        assert checkpoint_eqns[1].params["key"] == "second"

    def test_memoize_outside_collect_does_not_skip_checkpoints(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            with af.memoize():
                r1 = ir.call("hello")
                r2 = ir.call("hello")

        assert r1 == "hello"
        assert r2 == "hello"
        assert collected == {"val": ["hello", "hello"]}


class TestMemoizeMultipleCalls:
    def test_memoize_across_multiple_ir_calls(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        counter = CountingInterpreter()
        with using_interpreter(counter):
            with af.memoize():
                r1 = ir.call("hello")
                r2 = ir.call("hello")

        assert r1 == "hello!"
        assert r2 == "hello!"
        assert counter.call_count == 1

    def test_memoize_scope_is_context(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        counter = CountingInterpreter()
        with using_interpreter(counter):
            with af.memoize():
                ir.call("hello")

            with af.memoize():
                ir.call("hello")

        assert counter.call_count == 2


class TestMemoizeTransformedIRs:
    def count_misses(self, ir, *inputs):
        counter = CountingInterpreter()
        results = []
        with using_interpreter(counter):
            with af.memoize():
                for inp in inputs:
                    if isinstance(inp, tuple):
                        results.append(ir.call(*inp))
                    else:
                        results.append(ir.call(inp))
        return results, counter.call_count

    def test_memoize_batched_ir(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        (r1, r2), misses = self.count_misses(batched, ["a", "b"], ["a", "b"])

        assert r1 == ["a!", "b!"]
        assert r2 == ["a!", "b!"]

        assert misses == 3

    def test_memoize_pushforward_ir(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        pf_ir = af.pushforward(ir)

        (r1, r2), misses = self.count_misses(
            pf_ir, (("primal",), ("tangent",)), (("primal",), ("tangent",))
        )

        assert r1 == r2
        assert misses == 3

    def test_memoize_pullback_ir(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        pb_ir = af.pullback(ir)

        (r1, r2), misses = self.count_misses(
            pb_ir, (("primal",), "cotangent"), (("primal",), "cotangent")
        )

        assert r1 == r2
        assert misses == 2

    def test_memoize_batched_different_inputs(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        (r1, r2), misses = self.count_misses(batched, ["a", "b"], ["c", "d"])

        assert r1 == ["a!", "b!"]
        assert r2 == ["c!", "d!"]
        assert misses == 6
