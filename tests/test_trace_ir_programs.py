import autoform as af


class TestTraceRunIR:
    def test_trace_run_ir_inlines_operations(self):
        def inner_program(x):
            return af.format("Hello, {}!", x)

        inner_ir = af.build_ir(inner_program)("world")

        def program_with_run_ir(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "format"
        result = af.call(outer_ir)("Alice")
        assert result == "Hello, Alice!"

    def test_trace_run_ir_multiple_operations(self):
        def inner_program(x):
            y = af.format("[{}]", x)
            return af.concat(y, "!")

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        assert len(outer_ir.ireqns) == 2
        result = af.call(outer_ir)("hello")
        assert result == "[hello]!"


class TestTraceBatchIR:
    def test_trace_batch_creates_batch_call(self):
        def inner_program(x):
            return af.format("Item: {}", x)

        inner_ir = af.build_ir(inner_program)("x")
        batched_inner_ir = af.batch(inner_ir, in_axes=list)

        def program_with_batch(xs):
            return af.call(batched_inner_ir)(xs)

        outer_ir = af.build_ir(program_with_batch)(["a", "b", "c"])
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "batch_call"
        result = af.call(outer_ir)(["x", "y", "z"])
        assert result == ["Item: x", "Item: y", "Item: z"]


class TestTracePushforwardIR:
    def test_trace_pushforward_creates_pushforward_call(self):
        def inner_program(x):
            return af.format("[{}]", x)

        inner_ir = af.build_ir(inner_program)("x")
        pf_ir = af.pushforward(inner_ir)

        def program_with_pushforward(primals, tangents):
            return af.call(pf_ir)((primals, tangents))

        outer_ir = af.build_ir(program_with_pushforward)("p", "t")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "pushforward_call"
        result = af.call(outer_ir)("primal", "tangent")
        assert result == ("[primal]", "[tangent]")


class TestTracePullbackIR:
    def test_trace_pullback_creates_pullback_call(self):
        def inner_program(x):
            return af.format("<{}>", x)

        inner_ir = af.build_ir(inner_program)("x")
        pb_ir = af.pullback(inner_ir)

        def program_with_pullback(primal, cotangent):
            return af.call(pb_ir)((primal, cotangent))

        outer_ir = af.build_ir(program_with_pullback)("p", "c")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "pullback_call"
        result = af.call(outer_ir)("primal", "cotan")
        assert result == ("<primal>", "cotan")


class TestMultiLevelTracing:
    def test_double_trace_flattens_operations(self):
        def base_program(x):
            return af.format("({})", x)

        base_ir = af.build_ir(base_program)("x")

        def level1(x):
            return af.call(base_ir)(x)

        level1_ir = af.build_ir(level1)("y")

        def level2(x):
            return af.call(level1_ir)(x)

        level2_ir = af.build_ir(level2)("z")
        assert len(level2_ir.ireqns) == 1
        assert level2_ir.ireqns[0].prim.name == "format"
        result = af.call(level2_ir)("hello")
        assert result == "(hello)"

    def test_triple_trace_flattens_operations(self):
        def base_program(x):
            return af.concat(x, "!")

        base_ir = af.build_ir(base_program)("x")

        def level1(x):
            return af.call(base_ir)(x)

        level1_ir = af.build_ir(level1)("y")

        def level2(x):
            return af.call(level1_ir)(x)

        level2_ir = af.build_ir(level2)("z")

        def level3(x):
            return af.call(level2_ir)(x)

        level3_ir = af.build_ir(level3)("w")
        assert len(level3_ir.ireqns) == 1
        result = af.call(level3_ir)("test")
        assert result == "test!"


class TestTransformOfTracedRunIR:
    def test_pushforward_of_traced_run_ir(self):
        def inner_program(x):
            return af.format("[{}]", x)

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        pf_outer_ir = af.pushforward(outer_ir)
        result = af.call(pf_outer_ir)(("primal", "tangent"))
        assert result == ("[primal]", "[tangent]")

    def test_batch_of_traced_run_ir(self):
        def inner_program(x):
            return af.format("<{}>", x)

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        batch_outer_ir = af.batch(outer_ir, in_axes=list)
        result = af.call(batch_outer_ir)(["a", "b", "c"])
        assert result == ["<a>", "<b>", "<c>"]

    def test_pullback_of_traced_run_ir(self):
        def inner_program(x):
            return af.concat(x, "!")

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        pb_outer_ir = af.pullback(outer_ir)
        result = af.call(pb_outer_ir)(("primal", "cotan"))
        assert result == ("primal!", "cotan")
