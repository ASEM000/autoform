import autoform.core as core


class TestTraceRunIR:
    def test_trace_run_ir_inlines_operations(self):
        def inner_program(x):
            return core.format("Hello, {}!", x)

        inner_ir = core.build_ir(inner_program, "world")

        def program_with_run_ir(x):
            return core.run_ir(inner_ir, x)

        outer_ir = core.build_ir(program_with_run_ir, "test")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "format"
        result = core.run_ir(outer_ir, "Alice")
        assert result == "Hello, Alice!"

    def test_trace_run_ir_multiple_operations(self):
        def inner_program(x):
            y = core.format("[{}]", x)
            return core.concat(y, "!")

        inner_ir = core.build_ir(inner_program, "x")

        def program_with_run_ir(x):
            return core.run_ir(inner_ir, x)

        outer_ir = core.build_ir(program_with_run_ir, "test")
        assert len(outer_ir.ireqns) == 2
        result = core.run_ir(outer_ir, "hello")
        assert result == "[hello]!"


class TestTraceBatchIR:
    def test_trace_batch_ir_creates_batch_call(self):
        def inner_program(x):
            return core.format("Item: {}", x)

        inner_ir = core.build_ir(inner_program, "x")
        batched_inner_ir = core.batch_ir(inner_ir, in_axes=list)

        def program_with_batch(xs):
            return core.run_ir(batched_inner_ir, xs)

        outer_ir = core.build_ir(program_with_batch, ["a", "b", "c"])
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "batch_call"
        result = core.run_ir(outer_ir, ["x", "y", "z"])
        assert result == ["Item: x", "Item: y", "Item: z"]


class TestTracePushforwardIR:
    def test_trace_pushforward_ir_creates_pushforward_call(self):
        def inner_program(x):
            return core.format("[{}]", x)

        inner_ir = core.build_ir(inner_program, "x")
        pf_ir = core.pushforward_ir(inner_ir)

        def program_with_pushforward(primals, tangents):
            return core.run_ir(pf_ir, (primals, tangents))

        outer_ir = core.build_ir(program_with_pushforward, "p", "t")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "pushforward_call"
        result = core.run_ir(outer_ir, "primal", "tangent")
        assert result == ("[primal]", "[tangent]")


class TestTracePullbackIR:
    def test_trace_pullback_ir_creates_pullback_call(self):
        def inner_program(x):
            return core.format("<{}>", x)

        inner_ir = core.build_ir(inner_program, "x")
        pb_ir = core.pullback_ir(inner_ir)

        def program_with_pullback(primal, cotangent):
            return core.run_ir(pb_ir, (primal, cotangent))

        outer_ir = core.build_ir(program_with_pullback, "p", "c")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "pullback_call"
        result = core.run_ir(outer_ir, "primal", "cotan")
        assert result == ("<primal>", "cotan")


class TestMultiLevelTracing:
    def test_double_trace_flattens_operations(self):
        def base_program(x):
            return core.format("({})", x)

        base_ir = core.build_ir(base_program, "x")

        def level1(x):
            return core.run_ir(base_ir, x)

        level1_ir = core.build_ir(level1, "y")

        def level2(x):
            return core.run_ir(level1_ir, x)

        level2_ir = core.build_ir(level2, "z")
        assert len(level2_ir.ireqns) == 1
        assert level2_ir.ireqns[0].prim.name == "format"
        result = core.run_ir(level2_ir, "hello")
        assert result == "(hello)"

    def test_triple_trace_flattens_operations(self):
        def base_program(x):
            return core.concat(x, "!")

        base_ir = core.build_ir(base_program, "x")

        def level1(x):
            return core.run_ir(base_ir, x)

        level1_ir = core.build_ir(level1, "y")

        def level2(x):
            return core.run_ir(level1_ir, x)

        level2_ir = core.build_ir(level2, "z")

        def level3(x):
            return core.run_ir(level2_ir, x)

        level3_ir = core.build_ir(level3, "w")
        assert len(level3_ir.ireqns) == 1
        result = core.run_ir(level3_ir, "test")
        assert result == "test!"


class TestTransformOfTracedRunIR:
    def test_pushforward_of_traced_run_ir(self):
        def inner_program(x):
            return core.format("[{}]", x)

        inner_ir = core.build_ir(inner_program, "x")

        def program_with_run_ir(x):
            return core.run_ir(inner_ir, x)

        outer_ir = core.build_ir(program_with_run_ir, "test")
        pf_outer_ir = core.pushforward_ir(outer_ir)
        result = core.run_ir(pf_outer_ir, ("primal", "tangent"))
        assert result == ("[primal]", "[tangent]")

    def test_batch_of_traced_run_ir(self):
        def inner_program(x):
            return core.format("<{}>", x)

        inner_ir = core.build_ir(inner_program, "x")

        def program_with_run_ir(x):
            return core.run_ir(inner_ir, x)

        outer_ir = core.build_ir(program_with_run_ir, "test")
        batch_outer_ir = core.batch_ir(outer_ir, in_axes=list)
        result = core.run_ir(batch_outer_ir, ["a", "b", "c"])
        assert result == ["<a>", "<b>", "<c>"]

    def test_pullback_of_traced_run_ir(self):
        def inner_program(x):
            return core.concat(x, "!")

        inner_ir = core.build_ir(inner_program, "x")

        def program_with_run_ir(x):
            return core.run_ir(inner_ir, x)

        outer_ir = core.build_ir(program_with_run_ir, "test")
        pb_outer_ir = core.pullback_ir(outer_ir)
        result = core.run_ir(pb_outer_ir, ("primal", "cotan"))
        assert result == ("primal!", "cotan")
