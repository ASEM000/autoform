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
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "effect"
        assert ir.ireqns[0].effect.collection == "my_tag"
        assert ir.ireqns[0].effect.key == "my_name"

    def test_run_ir(self):
        def func(x):
            return af.checkpoint(x, key="value", collection="test")

        ir = af.trace(func)("test")
        result = af.call(ir)("hello")
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
        primal_out, tangent_out = af.call(pf_ir)(("primal", "tangent"))
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_preserves_cotangent(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="test")

        ir = af.trace(func)("a")
        pb_ir = af.pullback(ir)
        primal_out, cotangent_in = af.call(pb_ir)(("primal", "cotangent"))
        assert primal_out == "primal"
        assert cotangent_in == "cotangent"

    def test_batch(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="test")

        ir = af.trace(func)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain(self):
        def func(x):
            sowed = af.checkpoint(x, key="input", collection="debug")
            return af.concat("[", sowed, "]")

        ir = af.trace(func)("a")
        result = af.call(ir)("hello")
        assert result == "[hello]"


class TestRunAndReap:
    def test_reap_single_sow(self):
        def func(x):
            return af.checkpoint(x, key="captured", collection="debug")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = af.call(ir)("hello")
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
            result = af.call(ir)("hi")
        assert result == "hi!"
        assert collected == {"first": ["hi"], "second": ["hi!"]}

    def test_reap_filters_by_tag(self):
        def func(x):
            a = af.checkpoint(x, key="debug_val", collection="debug")
            b = af.checkpoint(a, key="metrics_val", collection="metrics")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as debug_collected:
            af.call(ir)("hello")
        assert debug_collected == {"debug_val": ["hello"]}

        with af.collect(collection="metrics") as metrics_collected:
            af.call(ir)("hello")
        assert metrics_collected == {"metrics_val": ["hello"]}

    def test_reap_empty_when_no_match(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="other")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = af.call(ir)("hello")
        assert result == "hello"
        assert collected == {}

    def test_reap_with_no_sows(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = af.call(ir)("hello")
        assert result == "hello!"
        assert collected == {}

    def test_reap_preserves_execution(self):
        def func(x):
            a = af.checkpoint(af.format("Q: {}", x), key="prompt", collection="debug")
            response = af.concat(a, " A: 42")
            return af.checkpoint(response, key="response", collection="debug")

        ir = af.trace(func)("test")
        with af.collect(collection="debug") as collected:
            result = af.call(ir)("What?")
        assert result == "Q: What? A: 42"
        assert collected["prompt"] == ["Q: What?"]
        assert collected["response"] == ["Q: What? A: 42"]


class TestRunAndPlant:
    def test_plant_overrides_sow(self):
        def func(x):
            return af.checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.trace(func)("test")

        result = af.call(ir)("World")
        assert result == "Hello, World"

        with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
            result = af.call(ir)("World")
        assert result == "CACHED"

    def test_plant_partial(self):
        def func(x):
            a = af.checkpoint(x, key="first", collection="cache")
            b = af.checkpoint(af.concat(a, "!"), key="second", collection="cache")
            return b

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"first": ["PLANTED"]}):
            result = af.call(ir)("ignored")
        assert result == "PLANTED!"

    def test_plant_filters_by_tag(self):
        def func(x):
            a = af.checkpoint(x, key="val", collection="cache")
            b = af.checkpoint(a, key="val", collection="other")
            return b

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"val": ["CACHED"]}):
            result = af.call(ir)("input")
        assert result == "CACHED"

    def test_plant_empty_dict(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="cache")

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={}):
            result = af.call(ir)("hello")
        assert result == "hello"

    def test_plant_unmatched_name(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="cache")

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"other": ["PLANTED"]}):
            result = af.call(ir)("hello")
        assert result == "hello"


class TestTransformThenReap:
    def test_reap_captures_during_pushforward(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pf_ir = af.pushforward(ir)

        with af.collect(collection="debug") as collected:
            result = af.call(pf_ir)(("primal", "tangent"))
        assert collected == {"val": ["primal", "tangent"]}

    def test_reap_captures_during_pullback(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pb_ir = af.pullback(ir)

        with af.collect(collection="debug") as collected:
            result = af.call(pb_ir)(("primal", "cotangent"))
        assert collected == {"val": ["primal", "cotangent"]}

    def test_reap_captures_during_batch(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        with af.collect(collection="debug") as collected:
            result = af.call(batched)(["a", "b", "c"])
        assert result == ["a", "b", "c"]
        assert collected == {"val": [["a", "b", "c"]]}

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
            result = af.call(ir)("hello")
        assert result == "a: hello"
        assert collected == {"result": ["a: hello"]}


class TestInjectAndDCE:
    def test_inject_trace_creates_literal(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.checkpoint(expensive, key="result", collection="cache")
            return af.concat("Got: ", cached)

        ir = af.trace(program)("test")

        assert len(ir.ireqns) == 3

        def wrapped(x):
            with af.inject(collection="cache", values={"result": ["CACHED"]}):
                return af.call(ir)("ignored")

        traced_ir = af.trace(wrapped)("example")
        assert len(traced_ir.ireqns) == 2

        last_eqn = traced_ir.ireqns[-1]
        assert last_eqn.prim.name == "concat"

    def test_dce_removes_dead_code_after_inject(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.checkpoint(expensive, key="result", collection="cache")
            return af.concat("Got: ", cached)

        ir = af.trace(program)("test")

        def wrapped(x):
            with af.inject(collection="cache", values={"result": ["CACHED"]}):
                return af.call(ir)("ignored")

        traced_ir = af.trace(wrapped)("example")
        assert len(traced_ir.ireqns) == 2
        optimized_ir = af.dce(traced_ir)
        assert len(optimized_ir.ireqns) == 1

        result = af.call(optimized_ir)("any_input")
        assert result == "Got: CACHED"

    def test_inject_dce_with_multiple_marks(self):
        def program(x):
            step1 = af.concat("step1:", x)
            saved1 = af.checkpoint(step1, key="first", collection="cache")
            step2 = af.concat("step2:", saved1)
            saved2 = af.checkpoint(step2, key="second", collection="cache")
            return af.concat("final:", saved2)

        ir = af.trace(program)("test")
        assert len(ir.ireqns) == 5

        def wrapped(x):
            with af.inject(collection="cache", values={"first": ["CACHED1"]}):
                return af.call(ir)(x)

        traced_ir = af.trace(wrapped)("example")
        optimized_ir = af.dce(traced_ir)
        result = af.call(optimized_ir)("input")
        assert result == "final:step2:CACHED1"

    def test_inject_works_with_nested_transforms(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.checkpoint(expensive, key="result", collection="cache")
            return af.concat("Got: ", cached)

        ir = af.trace(program)("test")
        batched_ir = af.batch(ir)

        with af.inject(collection="cache", values={"result": [["A", "B"]]}):
            result = af.call(batched_ir)(["x", "y"])

        assert result == ["Got: A", "Got: B"]


class TestSplit:
    def test_split_mark_at_end(self):
        def program(x):
            y = af.splitpoint(af.format("{}", x), key="s")
            return y

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="s")

        assert len(lhs.ireqns) == 1
        assert lhs.ireqns[0].prim.name == "format"

        assert len(rhs.ireqns) == 0

        lhs_result = af.call(lhs)("Test")
        assert lhs_result == "Test"
        rhs_result = af.call(rhs)(lhs_result)
        assert rhs_result == lhs_result

    def test_split_mark_in_middle(self):
        def program(x):
            y = af.format("Hello {}", x)
            z = af.splitpoint(y, key="mid")
            w = af.format("Result: {}", z)
            return w

        ir = af.trace(program)("...")
        lhs, rhs = af.split(ir, key="mid")

        assert len(lhs.ireqns) == 1

        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "format"

        lhs_result = af.call(lhs)("World")
        assert lhs_result == "Hello World"

        rhs_result = af.call(rhs)(lhs_result)
        assert rhs_result == "Result: Hello World"

        ir_full = af.trace(program)("x")
        full_result = af.call(ir_full)("World")
        assert rhs_result == full_result

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
            lhs_result = af.call(lhs)(inp)
            rhs_result = af.call(rhs)(lhs_result)
            full_result = af.call(ir_full)(inp)
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
        assert len(lhs1.ireqns) == 0
        assert len(rhs1.ireqns) == 3

        ir2 = af.trace(program)("...")
        lhs2, rhs2 = af.split(ir2, key="second")
        assert len(lhs2.ireqns) == 2
        assert len(rhs2.ireqns) == 1


class TestSplitpointPreservedThroughTransforms:
    def test_splitpoint_preserved_after_pushforward(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        assert len(pf_ir.ireqns) == 1
        assert pf_ir.ireqns[0].prim.name == "pushforward_call"
        nested_ir = pf_ir.ireqns[0].params["ir"]
        splitpoints = [eqn for eqn in nested_ir.ireqns if eqn.prim.name == "splitpoint"]
        assert len(splitpoints) == 1
        assert splitpoints[0].params["key"] == "mid"

    def test_splitpoint_preserved_after_pullback(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        assert len(pb_ir.ireqns) == 1
        assert pb_ir.ireqns[0].prim.name == "pullback_call"
        nested_ir = pb_ir.ireqns[0].params["ir"]
        splitpoints = [eqn for eqn in nested_ir.ireqns if eqn.prim.name == "splitpoint"]
        assert len(splitpoints) == 1
        assert splitpoints[0].params["key"] == "mid"

    def test_splitpoint_preserved_after_batch(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        batch_ir = af.batch(ir, in_axes=True)

        assert len(batch_ir.ireqns) == 1
        assert batch_ir.ireqns[0].prim.name == "batch_call"
        nested_ir = batch_ir.ireqns[0].params["ir"]
        splitpoints = [eqn for eqn in nested_ir.ireqns if eqn.prim.name == "splitpoint"]
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

        assert len(lhs.ireqns) == 1
        assert lhs.ireqns[0].prim.name == "pushforward_call"
        nested_lhs = lhs.ireqns[0].params["ir"]

        assert nested_lhs.ireqns[-1].prim.name == "concat"

        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "pushforward_call"
        nested_rhs = rhs.ireqns[0].params["ir"]

        assert len(nested_rhs.ireqns) == 1
        assert nested_rhs.ireqns[0].prim.name == "concat"

    def test_split_on_pullback_ir(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        lhs, rhs = af.split(pb_ir, key="mid")

        assert len(lhs.ireqns) == 1
        assert lhs.ireqns[0].prim.name == "pullback_call"
        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "pullback_call"

    def test_split_on_batch_ir(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        batch_ir = af.batch(ir, in_axes=True)

        lhs, rhs = af.split(batch_ir, key="mid")

        assert len(lhs.ireqns) == 1
        assert lhs.ireqns[0].prim.name == "batch_call"
        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "batch_call"

    def test_split_on_transformed_ir_execution(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")

        lhs, rhs = af.split(ir, key="mid")
        assert af.call(lhs)("hello") == "hello!"
        assert af.call(rhs)("hello!") == "hello!?"

    def test_split_on_double_pushforward(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        pf_pf_ir = af.pushforward(af.pushforward(ir))

        lhs, rhs = af.split(pf_pf_ir, key="mid")

        assert len(lhs.ireqns) == 1
        assert lhs.ireqns[0].prim.name == "pushforward_call"
        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "pushforward_call"

        inner_lhs = lhs.ireqns[0].params["ir"]
        inner_rhs = rhs.ireqns[0].params["ir"]
        assert inner_lhs.ireqns[0].prim.name == "pushforward_call"
        assert inner_rhs.ireqns[0].prim.name == "pushforward_call"

    def test_split_on_batch_pushforward(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        b_pf_ir = af.batch(af.pushforward(ir), in_axes=True)

        lhs, rhs = af.split(b_pf_ir, key="mid")

        assert len(lhs.ireqns) == 1
        assert lhs.ireqns[0].prim.name == "batch_call"
        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "batch_call"

        inner_lhs = lhs.ireqns[0].params["ir"]
        inner_rhs = rhs.ireqns[0].params["ir"]
        assert inner_lhs.ireqns[0].prim.name == "pushforward_call"
        assert inner_rhs.ireqns[0].prim.name == "pushforward_call"

    def test_split_on_triple_nested(self):
        def program(x):
            y = af.splitpoint(af.concat(x, "!"), key="mid")
            return af.concat(y, "?")

        ir = af.trace(program)("x")
        triple = af.pushforward(af.batch(af.pushforward(ir), in_axes=True))

        lhs, rhs = af.split(triple, key="mid")

        assert lhs.ireqns[0].prim.name == "pushforward_call"
        assert rhs.ireqns[0].prim.name == "pushforward_call"


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
                result = af.call(ir)("hello")

        assert result == "hello!hello!"
        assert counter.call_count == 2

    def test_memoize_returns_correct_result(self):
        def func(x):
            a = af.concat(x, "!")
            return a

        ir = af.trace(func)("test")

        with af.memoize():
            result = af.call(ir)("hello")

        assert result == "hello!"

    def test_memoize_different_inputs_not_cached(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        counter = CountingInterpreter()
        with using_interpreter(counter):
            with af.memoize():
                r1 = af.call(ir)("hello")
                r2 = af.call(ir)("world")

        assert r1 == "hello!"
        assert r2 == "world!"
        assert counter.call_count == 2


class TestMemoizeWithEffects:
    def test_memoize_with_checkpoint(self):
        def func(x):
            a = checkpoint(af.concat(x, "!"), key="val", collection="debug")
            return a

        ir = af.trace(func)("test")

        with af.memoize():
            with af.collect(collection="debug") as collected:
                result = af.call(ir)("hello")

        assert result == "hello!"
        assert collected == {"val": ["hello!"]}

    def test_memoize_does_not_merge_different_effects(self):
        def func(x):
            a = checkpoint(x, key="first", collection="debug")
            b = checkpoint(x, key="second", collection="debug")
            return af.concat(a, b)

        ir = af.trace(func)("test")

        with af.memoize():
            with af.collect(collection="debug") as collected:
                result = af.call(ir)("hi")

        assert result == "hihi"
        assert "first" in collected
        assert "second" in collected


class TestMemoizeMultipleCalls:
    def test_memoize_across_multiple_ir_calls(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        counter = CountingInterpreter()
        with using_interpreter(counter):
            with af.memoize():
                r1 = af.call(ir)("hello")
                r2 = af.call(ir)("hello")

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
                af.call(ir)("hello")

            with af.memoize():
                af.call(ir)("hello")

        assert counter.call_count == 2


class TestMemoizeTransformedIRs:
    def count_misses(self, ir, *inputs):
        counter = CountingInterpreter()
        results = []
        with using_interpreter(counter):
            with af.memoize():
                for inp in inputs:
                    results.append(af.call(ir)(inp))
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

        (r1, r2), misses = self.count_misses(pf_ir, ("primal", "tangent"), ("primal", "tangent"))

        assert r1 == r2
        assert misses == 3

    def test_memoize_pullback_ir(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")
        pb_ir = af.pullback(ir)

        (r1, r2), misses = self.count_misses(
            pb_ir, ("primal", "cotangent"), ("primal", "cotangent")
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
