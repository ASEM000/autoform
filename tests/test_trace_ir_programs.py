import autoform as af


class TestTraceRunIR:
    def test_trace_run_ir_inlines_operations(self):
        def inner_program(x):
            return af.format("Hello, {}!", x)

        inner_ir = af.build_ir(inner_program)("world")

        def program_with_run_ir(x):
            return inner_ir.call(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "format"
        result = outer_ir.call("Alice")
        assert result == "Hello, Alice!"

    def test_trace_run_ir_multiple_operations(self):
        def inner_program(x):
            y = af.format("[{}]", x)
            return af.concat(y, "!")

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return inner_ir.call(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        assert len(outer_ir.ireqns) == 2
        result = outer_ir.call("hello")
        assert result == "[hello]!"


class TestTraceBatchIR:
    def test_trace_batch_ir_creates_batch_call(self):
        def inner_program(x):
            return af.format("Item: {}", x)

        inner_ir = af.build_ir(inner_program)("x")
        batched_inner_ir = af.batch_ir(inner_ir, in_axes=list)

        def program_with_batch(xs):
            return batched_inner_ir.call(xs)

        outer_ir = af.build_ir(program_with_batch)(["a", "b", "c"])
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "batch_call"
        result = outer_ir.call(["x", "y", "z"])
        assert result == ["Item: x", "Item: y", "Item: z"]


class TestTracePushforwardIR:
    def test_trace_pushforward_ir_creates_pushforward_call(self):
        def inner_program(x):
            return af.format("[{}]", x)

        inner_ir = af.build_ir(inner_program)("x")
        pf_ir = af.pushforward_ir(inner_ir)

        def program_with_pushforward(primals, tangents):
            return pf_ir.call((primals, tangents))

        outer_ir = af.build_ir(program_with_pushforward)("p", "t")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "pushforward_call"
        result = outer_ir.call("primal", "tangent")
        assert result == ("[primal]", "[tangent]")


class TestTracePullbackIR:
    def test_trace_pullback_ir_creates_pullback_call(self):
        def inner_program(x):
            return af.format("<{}>", x)

        inner_ir = af.build_ir(inner_program)("x")
        pb_ir = af.pullback_ir(inner_ir)

        def program_with_pullback(primal, cotangent):
            return pb_ir.call((primal, cotangent))

        outer_ir = af.build_ir(program_with_pullback)("p", "c")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "pullback_call"
        result = outer_ir.call("primal", "cotan")
        assert result == ("<primal>", "cotan")


class TestMultiLevelTracing:
    def test_double_trace_flattens_operations(self):
        def base_program(x):
            return af.format("({})", x)

        base_ir = af.build_ir(base_program)("x")

        def level1(x):
            return base_ir.call(x)

        level1_ir = af.build_ir(level1)("y")

        def level2(x):
            return level1_ir.call(x)

        level2_ir = af.build_ir(level2)("z")
        assert len(level2_ir.ireqns) == 1
        assert level2_ir.ireqns[0].prim.name == "format"
        result = level2_ir.call("hello")
        assert result == "(hello)"

    def test_triple_trace_flattens_operations(self):
        def base_program(x):
            return af.concat(x, "!")

        base_ir = af.build_ir(base_program)("x")

        def level1(x):
            return base_ir.call(x)

        level1_ir = af.build_ir(level1)("y")

        def level2(x):
            return level1_ir.call(x)

        level2_ir = af.build_ir(level2)("z")

        def level3(x):
            return level2_ir.call(x)

        level3_ir = af.build_ir(level3)("w")
        assert len(level3_ir.ireqns) == 1
        result = level3_ir.call("test")
        assert result == "test!"


class TestTransformOfTracedRunIR:
    def test_pushforward_of_traced_run_ir(self):
        def inner_program(x):
            return af.format("[{}]", x)

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return inner_ir.call(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        pf_outer_ir = af.pushforward_ir(outer_ir)
        result = pf_outer_ir.call(("primal", "tangent"))
        assert result == ("[primal]", "[tangent]")

    def test_batch_of_traced_run_ir(self):
        def inner_program(x):
            return af.format("<{}>", x)

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return inner_ir.call(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        batch_outer_ir = af.batch_ir(outer_ir, in_axes=list)
        result = batch_outer_ir.call(["a", "b", "c"])
        assert result == ["<a>", "<b>", "<c>"]

    def test_pullback_of_traced_run_ir(self):
        def inner_program(x):
            return af.concat(x, "!")

        inner_ir = af.build_ir(inner_program)("x")

        def program_with_run_ir(x):
            return inner_ir.call(x)

        outer_ir = af.build_ir(program_with_run_ir)("test")
        pb_outer_ir = af.pullback_ir(outer_ir)
        result = pb_outer_ir.call(("primal", "cotan"))
        assert result == ("primal!", "cotan")
