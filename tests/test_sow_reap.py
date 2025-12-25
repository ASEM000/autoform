import autoform.core as core


class TestSow:
    def test_impl_is_identity(self):
        result = core.sow("hello", tag="debug", name="test")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return core.sow(x, tag="my_tag", name="my_name")

        ir = core.build_ir(func, "test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "sow"
        assert ir.ireqns[0].params["tag"] == "my_tag"
        assert ir.ireqns[0].params["name"] == "my_name"

    def test_run_ir(self):
        def func(x):
            return core.sow(x, tag="test", name="value")

        ir = core.build_ir(func, "test")
        result = core.run_ir(ir, "hello")
        assert result == "hello"

    def test_hashable_tags_and_names(self):
        assert core.sow("x", tag="str_tag", name="str_name") == "x"
        assert core.sow("x", tag=42, name=100) == "x"
        assert core.sow("x", tag=("a", 1), name=("b", 2)) == "x"

    def test_pushforward_preserves_both(self):
        def func(x):
            return core.sow(x, tag="test", name="val")

        ir = core.build_ir(func, "a")
        pf_ir = core.pushforward_ir(ir)
        primal_out, tangent_out = core.run_ir(pf_ir, ("primal", "tangent"))
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_preserves_cotangent(self):
        def func(x):
            return core.sow(x, tag="test", name="val")

        ir = core.build_ir(func, "a")
        pb_ir = core.pullback_ir(ir)
        primal_out, cotangent_in = core.run_ir(pb_ir, ("primal", "cotangent"))
        assert primal_out == "primal"
        assert cotangent_in == "cotangent"

    def test_batch(self):
        def func(x):
            return core.sow(x, tag="test", name="val")

        ir = core.build_ir(func, "a")
        batched_ir = core.batch_ir(ir)
        result = core.run_ir(batched_ir, ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain(self):
        def func(x):
            sowed = core.sow(x, tag="debug", name="input")
            return core.concat("[", sowed, "]")

        ir = core.build_ir(func, "a")
        result = core.run_ir(ir, "hello")
        assert result == "[hello]"


class TestReapIr:
    def test_reap_single_sow(self):
        def func(x):
            return core.sow(x, tag="debug", name="captured")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        result, reaped = core.run_ir(reap, "hello")
        assert result == "hello"
        assert reaped == {"captured": "hello"}

    def test_reap_multiple_sows_same_tag(self):
        def func(x):
            a = core.sow(x, tag="debug", name="first")
            b = core.concat(a, "!")
            c = core.sow(b, tag="debug", name="second")
            return c

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        result, reaped = core.run_ir(reap, "hi")
        assert result == "hi!"
        assert reaped == {"first": "hi", "second": "hi!"}

    def test_reap_filters_by_tag(self):
        def func(x):
            a = core.sow(x, tag="debug", name="debug_val")
            b = core.sow(a, tag="metrics", name="metrics_val")
            return b

        ir = core.build_ir(func, "test")

        reap_debug = core.reap_ir(ir, tag="debug")
        _, debug_reaped = core.run_ir(reap_debug, "hello")
        assert debug_reaped == {"debug_val": "hello"}

        reap_metrics = core.reap_ir(ir, tag="metrics")
        _, metrics_reaped = core.run_ir(reap_metrics, "hello")
        assert metrics_reaped == {"metrics_val": "hello"}

    def test_reap_empty_when_no_match(self):
        def func(x):
            return core.sow(x, tag="other", name="val")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        result, reaped = core.run_ir(reap, "hello")
        assert result == "hello"
        assert reaped == {}

    def test_reap_with_no_sows(self):
        def func(x):
            return core.concat(x, "!")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        result, reaped = core.run_ir(reap, "hello")
        assert result == "hello!"
        assert reaped == {}

    def test_reap_preserves_execution(self):
        def func(x):
            a = core.sow(core.format("Q: {}", x), tag="debug", name="prompt")
            response = core.concat(a, " A: 42")
            return core.sow(response, tag="debug", name="response")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        result, reaped = core.run_ir(reap, "What?")
        assert result == "Q: What? A: 42"
        assert reaped["prompt"] == "Q: What?"
        assert reaped["response"] == "Q: What? A: 42"

    def test_reap_returns_ir(self):
        def func(x):
            return core.sow(x, tag="debug", name="val")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        assert isinstance(reap, core.IR)


class TestPlantIr:
    def test_plant_overrides_sow(self):
        def func(x):
            return core.sow(core.concat("Hello, ", x), tag="cache", name="greeting")

        ir = core.build_ir(func, "test")

        result = core.run_ir(ir, "World")
        assert result == "Hello, World"

        planted = core.plant_ir(ir, {"greeting": "CACHED"}, tag="cache")
        result = core.run_ir(planted, "World")
        assert result == "CACHED"

    def test_plant_partial(self):
        def func(x):
            a = core.sow(x, tag="cache", name="first")
            b = core.sow(core.concat(a, "!"), tag="cache", name="second")
            return b

        ir = core.build_ir(func, "test")

        planted = core.plant_ir(ir, {"first": "PLANTED"}, tag="cache")
        result = core.run_ir(planted, "ignored")
        assert result == "PLANTED!"

    def test_plant_filters_by_tag(self):
        def func(x):
            a = core.sow(x, tag="cache", name="val")
            b = core.sow(a, tag="other", name="val")
            return b

        ir = core.build_ir(func, "test")

        planted = core.plant_ir(ir, {"val": "CACHED"}, tag="cache")
        result = core.run_ir(planted, "input")
        assert result == "CACHED"

    def test_plant_empty_dict(self):
        def func(x):
            return core.sow(x, tag="cache", name="val")

        ir = core.build_ir(func, "test")

        planted = core.plant_ir(ir, {}, tag="cache")
        result = core.run_ir(planted, "hello")
        assert result == "hello"

    def test_plant_unmatched_name(self):
        def func(x):
            return core.sow(x, tag="cache", name="val")

        ir = core.build_ir(func, "test")

        planted = core.plant_ir(ir, {"other": "PLANTED"}, tag="cache")
        result = core.run_ir(planted, "hello")
        assert result == "hello"

    def test_plant_returns_ir(self):
        def func(x):
            return core.sow(x, tag="cache", name="val")

        ir = core.build_ir(func, "test")
        planted = core.plant_ir(ir, {"val": "X"}, tag="cache")
        assert isinstance(planted, core.IR)


class TestReapTransformComposition:
    def test_batch_of_reap(self):
        def func(x):
            return core.sow(core.concat("Hello, ", x), tag="debug", name="greeting")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        batched_reap = core.batch_ir(reap)

        result = core.run_ir(batched_reap, ["World", "Universe"])
        outputs, reaped = result
        assert outputs == ["Hello, World", "Hello, Universe"]

    def test_pushforward_of_reap(self):
        def func(x):
            return core.sow(x, tag="debug", name="val")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        pf_reap = core.pushforward_ir(reap)

        result = core.run_ir(pf_reap, ("primal", "tangent"))
        (primal_out, _), (tangent_out, _) = result
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_of_reap(self):
        def func(x):
            return core.sow(x, tag="debug", name="val")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        pb_reap = core.pullback_ir(reap)

        result = core.run_ir(pb_reap, ("primal", ("cotangent", {})))
        primal_out, cotangent_in = result
        assert primal_out[0] == "primal"


class TestPlantTransformComposition:
    def test_batch_of_plant(self):
        def func(x):
            return core.sow(core.concat("Hello, ", x), tag="cache", name="greeting")

        ir = core.build_ir(func, "test")
        planted = core.plant_ir(ir, {"greeting": "CACHED"}, tag="cache")
        batched_plant = core.batch_ir(planted)
        result = core.run_ir(batched_plant, ["World", "Universe"])
        assert result == ["Hello, World", "Hello, Universe"]

    def test_pushforward_of_plant(self):
        def func(x):
            return core.sow(x, tag="cache", name="val")

        ir = core.build_ir(func, "test")
        planted = core.plant_ir(ir, {"val": "CACHED"}, tag="cache")
        pf_plant = core.pushforward_ir(planted)

        result = core.run_ir(pf_plant, ("primal", "tangent"))
        primal_out, tangent_out = result
        assert primal_out == "primal"
        assert tangent_out == "tangent"

    def test_pullback_of_plant(self):
        def func(x):
            return core.sow(x, tag="cache", name="val")

        ir = core.build_ir(func, "test")
        planted = core.plant_ir(ir, {"val": "CACHED"}, tag="cache")
        pb_plant = core.pullback_ir(planted)

        result = core.run_ir(pb_plant, ("primal", "cotangent"))
        primal_out, cotangent_in = result
        assert primal_out == "CACHED"


class TestNestedReapPlantComposition:
    def test_reap_of_batch(self):
        def func(x):
            return core.sow(core.concat("Hello, ", x), tag="debug", name="greeting")

        ir = core.build_ir(func, "test")
        batched = core.batch_ir(ir)
        reap_batched = core.reap_ir(batched, tag="debug")

        result, reaped = core.run_ir(reap_batched, ["World", "Universe"])
        assert result == ["Hello, World", "Hello, Universe"]

    def test_reap_of_plant(self):
        def func(x):
            return core.sow(core.concat("Hello, ", x), tag="cache", name="greeting")

        ir = core.build_ir(func, "test")
        planted = core.plant_ir(ir, {"greeting": "OVERRIDE"}, tag="cache")
        reap_planted = core.reap_ir(planted, tag="cache")

        result, reaped = core.run_ir(reap_planted, "ignored")
        assert result == "OVERRIDE"

    def test_batch_of_batch_of_reap(self):
        def func(x):
            return core.sow(x, tag="debug", name="val")

        ir = core.build_ir(func, "test")
        reap = core.reap_ir(ir, tag="debug")
        batched = core.batch_ir(reap)
        double_batched = core.batch_ir(batched)
        result = core.run_ir(double_batched, [["a", "b"], ["c", "d"]])
        outputs, _ = result
        assert outputs == [["a", "b"], ["c", "d"]]
