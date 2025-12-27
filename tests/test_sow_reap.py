import autoform as af


class TestSow:
    def test_impl_is_identity(self):
        result = af.sow("hello", tag="debug", name="test")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return af.sow(x, tag="my_tag", name="my_name")

        ir = af.build_ir(func)("test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "sow"
        assert ir.ireqns[0].params["tag"] == "my_tag"
        assert ir.ireqns[0].params["name"] == "my_name"

    def test_run_ir(self):
        def func(x):
            return af.sow(x, tag="test", name="value")

        ir = af.build_ir(func)("test")
        result = af.run_ir(ir, "hello")
        assert result == "hello"

    def test_hashable_tags_and_names(self):
        assert af.sow("x", tag="str_tag", name="str_name") == "x"
        assert af.sow("x", tag=42, name=100) == "x"
        assert af.sow("x", tag=("a", 1), name=("b", 2)) == "x"

    def test_pushforward_preserves_both(self):
        def func(x):
            return af.sow(x, tag="test", name="val")

        ir = af.build_ir(func)("a")
        pf_ir = af.pushforward_ir(ir)
        primal_out, tangent_out = af.run_ir(pf_ir, ("primal", "tangent"))
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_preserves_cotangent(self):
        def func(x):
            return af.sow(x, tag="test", name="val")

        ir = af.build_ir(func)("a")
        pb_ir = af.pullback_ir(ir)
        primal_out, cotangent_in = af.run_ir(pb_ir, ("primal", "cotangent"))
        assert primal_out == "primal"
        assert cotangent_in == "cotangent"

    def test_batch(self):
        def func(x):
            return af.sow(x, tag="test", name="val")

        ir = af.build_ir(func)("a")
        batched_ir = af.batch_ir(ir)
        result = af.run_ir(batched_ir, ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain(self):
        def func(x):
            sowed = af.sow(x, tag="debug", name="input")
            return af.concat("[", sowed, "]")

        ir = af.build_ir(func)("a")
        result = af.run_ir(ir, "hello")
        assert result == "[hello]"


class TestRunAndReap:
    def test_reap_single_sow(self):
        def func(x):
            return af.sow(x, tag="debug", name="captured")

        ir = af.build_ir(func)("test")
        result, reaped = af.run_and_reap(ir, "hello", tag="debug")
        assert result == "hello"
        assert reaped == {"captured": "hello"}

    def test_reap_multiple_sows_same_tag(self):
        def func(x):
            a = af.sow(x, tag="debug", name="first")
            b = af.concat(a, "!")
            c = af.sow(b, tag="debug", name="second")
            return c

        ir = af.build_ir(func)("test")
        result, reaped = af.run_and_reap(ir, "hi", tag="debug")
        assert result == "hi!"
        assert reaped == {"first": "hi", "second": "hi!"}

    def test_reap_filters_by_tag(self):
        def func(x):
            a = af.sow(x, tag="debug", name="debug_val")
            b = af.sow(a, tag="metrics", name="metrics_val")
            return b

        ir = af.build_ir(func)("test")

        _, debug_reaped = af.run_and_reap(ir, "hello", tag="debug")
        assert debug_reaped == {"debug_val": "hello"}

        _, metrics_reaped = af.run_and_reap(ir, "hello", tag="metrics")
        assert metrics_reaped == {"metrics_val": "hello"}

    def test_reap_empty_when_no_match(self):
        def func(x):
            return af.sow(x, tag="other", name="val")

        ir = af.build_ir(func)("test")
        result, reaped = af.run_and_reap(ir, "hello", tag="debug")
        assert result == "hello"
        assert reaped == {}

    def test_reap_with_no_sows(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.build_ir(func)("test")
        result, reaped = af.run_and_reap(ir, "hello", tag="debug")
        assert result == "hello!"
        assert reaped == {}

    def test_reap_preserves_execution(self):
        def func(x):
            a = af.sow(af.format("Q: {}", x), tag="debug", name="prompt")
            response = af.concat(a, " A: 42")
            return af.sow(response, tag="debug", name="response")

        ir = af.build_ir(func)("test")
        result, reaped = af.run_and_reap(ir, "What?", tag="debug")
        assert result == "Q: What? A: 42"
        assert reaped["prompt"] == "Q: What?"
        assert reaped["response"] == "Q: What? A: 42"


class TestRunAndPlant:
    def test_plant_overrides_sow(self):
        def func(x):
            return af.sow(af.concat("Hello, ", x), tag="cache", name="greeting")

        ir = af.build_ir(func)("test")

        result = af.run_ir(ir, "World")
        assert result == "Hello, World"

        result = af.run_and_plant(ir, "World", {"greeting": "CACHED"}, tag="cache")
        assert result == "CACHED"

    def test_plant_partial(self):
        def func(x):
            a = af.sow(x, tag="cache", name="first")
            b = af.sow(af.concat(a, "!"), tag="cache", name="second")
            return b

        ir = af.build_ir(func)("test")

        result = af.run_and_plant(ir, "ignored", {"first": "PLANTED"}, tag="cache")
        assert result == "PLANTED!"

    def test_plant_filters_by_tag(self):
        def func(x):
            a = af.sow(x, tag="cache", name="val")
            b = af.sow(a, tag="other", name="val")
            return b

        ir = af.build_ir(func)("test")

        result = af.run_and_plant(ir, "input", {"val": "CACHED"}, tag="cache")
        assert result == "CACHED"

    def test_plant_empty_dict(self):
        def func(x):
            return af.sow(x, tag="cache", name="val")

        ir = af.build_ir(func)("test")

        result = af.run_and_plant(ir, "hello", {}, tag="cache")
        assert result == "hello"

    def test_plant_unmatched_name(self):
        def func(x):
            return af.sow(x, tag="cache", name="val")

        ir = af.build_ir(func)("test")

        result = af.run_and_plant(ir, "hello", {"other": "PLANTED"}, tag="cache")
        assert result == "hello"


class TestTransformThenReap:
    def test_reap_captures_during_pushforward(self):
        def func(x):
            return af.sow(x, tag="debug", name="val")

        ir = af.build_ir(func)("test")
        pf_ir = af.pushforward_ir(ir)

        result, primals = af.run_and_reap(pf_ir, ("primal", "tangent"), tag=("debug", "primal"))
        assert primals == {"val": "primal"}

        result, tangents = af.run_and_reap(pf_ir, ("primal", "tangent"), tag=("debug", "tangent"))
        assert tangents == {"val": "tangent"}

    def test_reap_captures_during_pullback(self):
        def func(x):
            return af.sow(x, tag="debug", name="val")

        ir = af.build_ir(func)("test")
        pb_ir = af.pullback_ir(ir)

        result, primals = af.run_and_reap(pb_ir, ("primal", "cotangent"), tag=("debug", "primal"))
        assert primals == {"val": "primal"}

        result, grads = af.run_and_reap(pb_ir, ("primal", "cotangent"), tag=("debug", "cotangent"))
        assert grads == {"val": "cotangent"}

    def test_reap_captures_during_batch(self):
        def func(x):
            return af.sow(x, tag="debug", name="val")

        ir = af.build_ir(func)("test")
        batched = af.batch_ir(ir)
        result, reaped = af.run_and_reap(batched, ["a", "b", "c"], tag=("debug", "batch"))
        assert result == ["a", "b", "c"]
        assert reaped == {"val": ["a", "b", "c"]}

    def test_reap_captures_in_switch_branches(self):
        def branch_a(x):
            return af.sow(af.concat("a: ", x), tag="debug", name="result")

        def branch_b(x):
            return af.sow(af.concat("b: ", x), tag="debug", name="result")

        ir_a = af.build_ir(branch_a)("x")
        ir_b = af.build_ir(branch_b)("x")

        def func(x):
            return af.switch("a", {"a": ir_a, "b": ir_b}, x)

        ir = af.build_ir(func)("input")
        result, reaped = af.run_and_reap(ir, "hello", tag="debug")
        assert result == "a: hello"
        assert reaped == {"result": "a: hello"}
