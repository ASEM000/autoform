import pytest

import autoform as af


class TestSow:
    def test_impl_is_identity(self):
        result = af.mark("hello", collection="debug", name="test")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return af.mark(x, collection="my_tag", name="my_name")

        ir = af.build_ir(func)("test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "mark"
        assert ir.ireqns[0].params["collection"] == "my_tag"
        assert ir.ireqns[0].params["name"] == "my_name"

    def test_run_ir(self):
        def func(x):
            return af.mark(x, collection="test", name="value")

        ir = af.build_ir(func)("test")
        result = af.call(ir)("hello")
        assert result == "hello"

    def test_hashable_tags_and_names(self):
        assert af.mark("x", collection="str_tag", name="str_name") == "x"
        assert af.mark("x", collection=42, name=100) == "x"
        assert af.mark("x", collection=("a", 1), name=("b", 2)) == "x"

    def test_pushforward_preserves_both(self):
        def func(x):
            return af.mark(x, collection="test", name="val")

        ir = af.build_ir(func)("a")
        pf_ir = af.pushforward(ir)
        primal_out, tangent_out = af.call(pf_ir)(("primal", "tangent"))
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_preserves_cotangent(self):
        def func(x):
            return af.mark(x, collection="test", name="val")

        ir = af.build_ir(func)("a")
        pb_ir = af.pullback(ir)
        primal_out, cotangent_in = af.call(pb_ir)(("primal", "cotangent"))
        assert primal_out == "primal"
        assert cotangent_in == "cotangent"

    def test_batch(self):
        def func(x):
            return af.mark(x, collection="test", name="val")

        ir = af.build_ir(func)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain(self):
        def func(x):
            sowed = af.mark(x, collection="debug", name="input")
            return af.concat("[", sowed, "]")

        ir = af.build_ir(func)("a")
        result = af.call(ir)("hello")
        assert result == "[hello]"


class TestRunAndReap:
    def test_reap_single_sow(self):
        def func(x):
            return af.mark(x, collection="debug", name="captured")

        ir = af.build_ir(func)("test")
        result, collected = af.collect(ir, collection="debug")("hello")
        assert result == "hello"
        assert collected == {"captured": ["hello"]}

    def test_reap_multiple_sows_same_tag(self):
        def func(x):
            a = af.mark(x, collection="debug", name="first")
            b = af.concat(a, "!")
            c = af.mark(b, collection="debug", name="second")
            return c

        ir = af.build_ir(func)("test")
        result, collected = af.collect(ir, collection="debug")("hi")
        assert result == "hi!"
        assert collected == {"first": ["hi"], "second": ["hi!"]}

    def test_reap_filters_by_tag(self):
        def func(x):
            a = af.mark(x, collection="debug", name="debug_val")
            b = af.mark(a, collection="metrics", name="metrics_val")
            return b

        ir = af.build_ir(func)("test")

        _, debug_collected = af.collect(ir, collection="debug")("hello")
        assert debug_collected == {"debug_val": ["hello"]}

        _, metrics_collected = af.collect(ir, collection="metrics")("hello")
        assert metrics_collected == {"metrics_val": ["hello"]}

    def test_reap_empty_when_no_match(self):
        def func(x):
            return af.mark(x, collection="other", name="val")

        ir = af.build_ir(func)("test")
        result, collected = af.collect(ir, collection="debug")("hello")
        assert result == "hello"
        assert collected == {}

    def test_reap_with_no_sows(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.build_ir(func)("test")
        result, collected = af.collect(ir, collection="debug")("hello")
        assert result == "hello!"
        assert collected == {}

    def test_reap_preserves_execution(self):
        def func(x):
            a = af.mark(af.format("Q: {}", x), collection="debug", name="prompt")
            response = af.concat(a, " A: 42")
            return af.mark(response, collection="debug", name="response")

        ir = af.build_ir(func)("test")
        result, collected = af.collect(ir, collection="debug")("What?")
        assert result == "Q: What? A: 42"
        assert collected["prompt"] == ["Q: What?"]
        assert collected["response"] == ["Q: What? A: 42"]


class TestRunAndPlant:
    def test_plant_overrides_sow(self):
        def func(x):
            return af.mark(af.concat("Hello, ", x), collection="cache", name="greeting")

        ir = af.build_ir(func)("test")

        result = af.call(ir)("World")
        assert result == "Hello, World"

        result = af.inject(ir, collection="cache", values={"greeting": ["CACHED"]})("World")
        assert result == "CACHED"

    def test_plant_partial(self):
        def func(x):
            a = af.mark(x, collection="cache", name="first")
            b = af.mark(af.concat(a, "!"), collection="cache", name="second")
            return b

        ir = af.build_ir(func)("test")

        result = af.inject(ir, collection="cache", values={"first": ["PLANTED"]})("ignored")
        assert result == "PLANTED!"

    def test_plant_filters_by_tag(self):
        def func(x):
            a = af.mark(x, collection="cache", name="val")
            b = af.mark(a, collection="other", name="val")
            return b

        ir = af.build_ir(func)("test")

        result = af.inject(ir, collection="cache", values={"val": ["CACHED"]})("input")
        assert result == "CACHED"

    def test_plant_empty_dict(self):
        def func(x):
            return af.mark(x, collection="cache", name="val")

        ir = af.build_ir(func)("test")

        result = af.inject(ir, collection="cache", values={})("hello")
        assert result == "hello"

    def test_plant_unmatched_name(self):
        def func(x):
            return af.mark(x, collection="cache", name="val")

        ir = af.build_ir(func)("test")

        result = af.inject(ir, collection="cache", values={"other": ["PLANTED"]})("hello")
        assert result == "hello"


class TestTransformThenReap:
    def test_reap_captures_during_pushforward(self):
        def func(x):
            return af.mark(x, collection="debug", name="val")

        ir = af.build_ir(func)("test")
        pf_ir = af.pushforward(ir)

        result, primals = af.collect(pf_ir, collection=("debug", "primal"))(("primal", "tangent"))
        assert primals == {"val": ["primal"]}

        result, tangents = af.collect(pf_ir, collection=("debug", "tangent"))(("primal", "tangent"))
        assert tangents == {"val": ["tangent"]}

    def test_reap_captures_during_pullback(self):
        def func(x):
            return af.mark(x, collection="debug", name="val")

        ir = af.build_ir(func)("test")
        pb_ir = af.pullback(ir)

        result, primals = af.collect(pb_ir, collection=("debug", "primal"))(("primal", "cotangent"))
        assert primals == {"val": ["primal"]}

        result, grads = af.collect(pb_ir, collection=("debug", "cotangent"))((
            "primal",
            "cotangent",
        ))
        assert grads == {"val": ["cotangent"]}

    def test_reap_captures_during_batch(self):
        def func(x):
            return af.mark(x, collection="debug", name="val")

        ir = af.build_ir(func)("test")
        batched = af.batch(ir)
        result, collected = af.collect(batched, collection=("debug", "batch"))(["a", "b", "c"])
        assert result == ["a", "b", "c"]
        assert collected == {"val": [["a", "b", "c"]]}

    def test_reap_captures_in_switch_branches(self):
        def branch_a(x):
            return af.mark(af.concat("a: ", x), collection="debug", name="result")

        def branch_b(x):
            return af.mark(af.concat("b: ", x), collection="debug", name="result")

        ir_a = af.build_ir(branch_a)("x")
        ir_b = af.build_ir(branch_b)("x")

        def func(x):
            return af.switch("a", {"a": ir_a, "b": ir_b}, x)

        ir = af.build_ir(func)("input")
        result, collected = af.collect(ir, collection="debug")("hello")
        assert result == "a: hello"
        assert collected == {"result": ["a: hello"]}


class TestInjectAndDCE:
    def test_inject_trace_creates_literal(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.mark(expensive, collection="cache", name="result")
            return af.concat("Got: ", cached)

        ir = af.build_ir(program)("test")

        assert len(ir.ireqns) == 3

        def wrapped(x):
            return af.inject(ir, collection="cache", values={"result": ["CACHED"]})("ignored")

        traced_ir = af.build_ir(wrapped)("example")

        assert len(traced_ir.ireqns) == 2

        last_eqn = traced_ir.ireqns[-1]
        assert last_eqn.prim.name == "concat"

    def test_dce_removes_dead_code_after_inject(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.mark(expensive, collection="cache", name="result")
            return af.concat("Got: ", cached)

        ir = af.build_ir(program)("test")

        def wrapped(x):
            return af.inject(ir, collection="cache", values={"result": ["CACHED"]})("ignored")

        traced_ir = af.build_ir(wrapped)("example")

        assert len(traced_ir.ireqns) == 2

        optimized_ir = af.dce(traced_ir)
        assert len(optimized_ir.ireqns) == 0

        result = af.call(optimized_ir)("any_input")
        assert result == "Got: CACHED"

    def test_inject_dce_with_multiple_marks(self):
        def program(x):
            step1 = af.concat("step1:", x)
            saved1 = af.mark(step1, collection="cache", name="first")
            step2 = af.concat("step2:", saved1)
            saved2 = af.mark(step2, collection="cache", name="second")
            return af.concat("final:", saved2)

        ir = af.build_ir(program)("test")
        assert len(ir.ireqns) == 5

        def wrapped(x):
            return af.inject(ir, collection="cache", values={"first": ["CACHED1"]})(x)

        traced_ir = af.build_ir(wrapped)("example")
        optimized_ir = af.dce(traced_ir)
        result = af.call(optimized_ir)("input")
        assert result == "final:step2:CACHED1"

    def test_inject_works_with_nested_transforms(self):
        def program(x):
            expensive = af.concat("EXPENSIVE:", x)
            cached = af.mark(expensive, collection="cache", name="result")
            return af.concat("Got: ", cached)

        ir = af.build_ir(program)("test")
        batched_ir = af.batch(ir)

        result = af.inject(
            batched_ir, collection=("cache", "batch"), values={"result": [["A", "B"]]}
        )(["x", "y"])

        assert result == ["Got: A", "Got: B"]


class TestSplit:
    def test_split_mark_at_end(self):
        def program(x):
            y = af.mark(af.format("{}", x), collection="debug", name="s")
            return y

        lhs, rhs = af.split(program, name="s")("...")

        assert len(lhs.ireqns) == 2
        assert lhs.ireqns[0].prim.name == "format"
        assert lhs.ireqns[1].prim.name == "mark"

        assert len(rhs.ireqns) == 0

        lhs_result = af.call(lhs)("Test")
        assert lhs_result == "Test"
        rhs_result = af.call(rhs)(lhs_result)
        assert rhs_result == lhs_result

    def test_split_mark_in_middle(self):
        def program(x):
            y = af.format("Hello {}", x)
            z = af.mark(y, collection="cache", name="mid")
            w = af.format("Result: {}", z)
            return w

        lhs, rhs = af.split(program, name="mid")("...")

        assert len(lhs.ireqns) == 2

        assert len(rhs.ireqns) == 1
        assert rhs.ireqns[0].prim.name == "format"

        lhs_result = af.call(lhs)("World")
        assert lhs_result == "Hello World"

        rhs_result = af.call(rhs)(lhs_result)
        assert rhs_result == "Result: Hello World"

        ir_full = af.build_ir(program)("x")
        full_result = af.call(ir_full)("World")
        assert rhs_result == full_result

    def test_split_composition_equals_full(self):
        def program(x):
            a = af.format("Step1: {}", x)
            b = af.mark(a, collection="split", name="step1")
            c = af.format("Step2: {}", b)
            d = af.concat(c, "!")
            return d

        lhs, rhs = af.split(program, name="step1")("...")
        ir_full = af.build_ir(program)("x")

        for inp in ["a", "hello", "test123"]:
            lhs_result = af.call(lhs)(inp)
            rhs_result = af.call(rhs)(lhs_result)
            full_result = af.call(ir_full)(inp)
            assert rhs_result == full_result

    def test_split_not_found_raises(self):
        def program(x):
            return af.format("{}", x)

        with pytest.raises(AssertionError, match="could not find mark"):
            af.split(program, name="nonexistent")("...")

    def test_split_with_multiple_marks(self):
        def program(x):
            a = af.mark(x, collection="c", name="first")
            b = af.format("{}", a)
            c = af.mark(b, collection="c", name="second")
            d = af.concat(c, "!")
            return d

        lhs1, rhs1 = af.split(program, name="first")("...")
        assert len(lhs1.ireqns) == 1
        assert len(rhs1.ireqns) == 3

        lhs2, rhs2 = af.split(program, name="second")("...")
        assert len(lhs2.ireqns) == 3
        assert len(rhs2.ireqns) == 1
