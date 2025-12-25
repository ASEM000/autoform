import autoform.core as core
import functools as ft


class TestPrimitive:
    def test_creation(self):
        p = core.Primitive("test_prim")
        assert p.name == "test_prim"
        assert repr(p) == "test_prim"

    def test_def_impl_decorator(self):
        p = core.Primitive("test_impl")

        @ft.partial(core.impl_rules.def_rule, p)
        def impl(x):
            return x

        assert core.impl_rules[p] is impl

    def test_def_eval_decorator(self):
        p = core.Primitive("test_eval")

        @ft.partial(core.eval_rules.def_rule, p)
        def eval_rule(x):
            return core.Var()

        assert core.eval_rules[p] is eval_rule

    def test_def_batch_decorator(self):
        p = core.Primitive("test_batch")

        @ft.partial(core.batch_rules.def_rule, p)
        def batch_rule(batch_size, in_batched, in_tree):
            return in_tree, True

        assert core.batch_rules[p] is batch_rule

    def test_def_pushforward_decorator(self):
        p = core.Primitive("test_pushforward")

        @ft.partial(core.push_rules.def_rule, p)
        def pf_rule(primals, tangents):
            return primals, tangents

        assert core.push_rules[p] is pf_rule

    def test_def_pullback_forward_decorator(self):
        p = core.Primitive("test_pullback_fwd")

        @ft.partial(core.pull_fwd_rules.def_rule, p)
        def pb_fwd_rule(in_tree):
            return in_tree, in_tree

        assert core.pull_fwd_rules[p] is pb_fwd_rule

    def test_def_pullback_backward_decorator(self):
        p = core.Primitive("test_pullback_bwd")

        @ft.partial(core.pull_bwd_rules.def_rule, p)
        def pb_bwd_rule(residuals, cotangent_out):
            return cotangent_out

        assert core.pull_bwd_rules[p] is pb_bwd_rule

    def test_def_iter_decorator(self):
        p = core.Primitive("test_iter")

        @ft.partial(core.iter_rules.def_rule, p)
        def iter_rule(x):
            yield [x]

        assert core.iter_rules[p] is iter_rule

    def test_def_async_decorator(self):
        p = core.Primitive("test_async")

        @ft.partial(core.async_rules.def_rule, p)
        async def async_rule(x):
            return x

        assert core.async_rules[p] is async_rule


class TestFormatPrimitive:
    def test_basic_format(self):
        result = core.format("Hello, {}!", "World")
        assert result == "Hello, World!"

    def test_multiple_placeholders(self):
        result = core.format("{} + {} = {}", "1", "2", "3")
        assert result == "1 + 2 = 3"

    def test_format_ir(self):
        def func(x):
            return core.format("Value: {}", x)

        ir = core.build_ir(func, "test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "format"


class TestConcatPrimitive:
    def test_basic_concat(self):
        result = core.concat("Hello", " ", "World")
        assert result == "Hello World"

    def test_two_args(self):
        result = core.concat("A", "B")
        assert result == "AB"

    def test_concat_ir(self):
        def func(x, y):
            return core.concat(x, y)

        ir = core.build_ir(func, "a", "b")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "concat"


class TestBind:
    def test_bind_using(self):
        p = core.Primitive("custom_bind")

        @ft.partial(core.impl_rules.def_rule, p)
        def impl(in_tree, *, multiplier):
            return in_tree * multiplier

        @ft.partial(core.eval_rules.def_rule, p)
        def eval_rule(in_tree, *, multiplier):
            return core.Var()

        def func(x):
            return p.bind(x, multiplier=3)

        ir = core.build_ir(func, "A")
        assert ir.ireqns[0].params["multiplier"] == 3
        result = core.run_ir(ir, "B")
        assert result == "BBB"


class TestInterpreter:
    def test_eval_interpreter_is_default(self):
        result = core.concat("a", "b")
        assert result == "ab"

    def test_use_interpreter_context(self):
        tracer = core.TracingInterpreter()
        with core.using_interp(tracer) as t:
            assert t is tracer
            core.format("Hello, {}!", core.IRVar.fresh())
            assert len(tracer.ireqns) == 1
        result = core.concat("a", "b")
        assert result == "ab"

    def test_tracing_interpreter_creates_ireqns(self):
        tracer = core.TracingInterpreter()
        with core.using_interp(tracer):
            core.format("Hello, {}!", core.IRVar.fresh())
        assert len(tracer.ireqns) == 1


class TestStopGradient:
    def test_impl_is_identity(self):
        result = core.stop_gradient("hello")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, "test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "stop_gradient"

    def test_run_ir(self):
        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, "test")
        result = core.run_ir(ir, "hello")
        assert result == "hello"

    def test_pushforward_zeros_tangent(self):
        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, "a")
        pf_ir = core.pushforward_ir(ir)
        primal_out, tangent_out = core.run_ir(pf_ir, ("primal", "tangent"))
        assert primal_out == "primal"
        assert tangent_out == "" or (hasattr(tangent_out, "items") and len(tangent_out.items) == 0)

    def test_pullback_zeros_cotangent(self):
        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, "a")
        pb_ir = core.pullback_ir(ir)
        primal_out, cotangent_in = core.run_ir(pb_ir, ("primal", "cotangent"))
        assert primal_out == "primal"
        assert cotangent_in == "" or (
            hasattr(cotangent_in, "items") and len(cotangent_in.items) == 0
        )

    def test_batch(self):
        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, "a")
        batched_ir = core.batch_ir(ir)
        result = core.run_ir(batched_ir, ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain_stops_gradient(self):
        def is_zero_cotangent(val):
            return val == "" or (hasattr(val, "items") and len(val.items) == 0)

        def func(x, y):
            stopped = core.stop_gradient(x)
            return core.concat(stopped, y)

        ir = core.build_ir(func, "a", "b")
        pb_ir = core.pullback_ir(ir)
        _, (cotangent_x, cotangent_y) = core.run_ir(pb_ir, (("a", "b"), "grad"))
        assert is_zero_cotangent(cotangent_x)
        assert cotangent_y == "grad"

    def test_chained_with_format(self):
        def func(x):
            stopped = core.stop_gradient(x)
            return core.format("[{}]", stopped)

        ir = core.build_ir(func, "test")
        result = core.run_ir(ir, "hello")
        assert result == "[hello]"

    def test_tree_input(self):
        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, ("a", "b"))
        result = core.run_ir(ir, ("hello", "world"))
        assert result == ("hello", "world")

    def test_tree_pullback_zeros_all(self):
        def is_zero_cotangent(val):
            return val == "" or (hasattr(val, "items") and len(val.items) == 0)

        def func(x):
            return core.stop_gradient(x)

        ir = core.build_ir(func, ("a", "b"))
        pb_ir = core.pullback_ir(ir)
        _, cotangent_in = core.run_ir(pb_ir, (("p1", "p2"), ("c1", "c2")))
        assert is_zero_cotangent(cotangent_in[0])
        assert is_zero_cotangent(cotangent_in[1])


class TestirCall:
    def test_impl_executes_inner_ir(self):
        def inner(x):
            return core.format("[{}]", x)

        inner_ir = core.build_ir(inner, "X")
        result = core.ir_call(inner_ir, "hello")
        assert result == "[hello]"

    def test_impl_with_multiple_args(self):
        def inner(a, b):
            return core.concat(a, b)

        inner_ir = core.build_ir(inner, "A", "B")
        result = core.ir_call(inner_ir, "foo", "bar")
        assert result == "foobar"

    def test_ir_build_creates_ir_call_eqn(self):
        def inner(x):
            return core.format("[{}]", x)

        inner_ir = core.build_ir(inner, "X")

        def outer(prog, x):
            return core.ir_call(prog, x)

        outer_ir = core.build_ir(outer, inner_ir, "X")
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "ir_call"

    def test_run_ir_with_ir_call(self):
        def inner(x):
            return core.format("<{}>", x)

        inner_ir = core.build_ir(inner, "X")

        def outer(prog, x):
            return core.ir_call(prog, x)

        outer_ir = core.build_ir(outer, inner_ir, "X")
        result = core.run_ir(outer_ir, inner_ir, "test")
        assert result == "<test>"

    def test_pushforward(self):
        def inner(x):
            return core.concat(x, "!")

        inner_ir = core.build_ir(inner, "X")

        def outer(prog, x):
            return core.ir_call(prog, x)

        outer_ir = core.build_ir(outer, inner_ir, "X")
        pf_ir = core.pushforward_ir(outer_ir)
        (p_out, t_out) = core.run_ir(pf_ir, ((inner_ir, "primal"), (inner_ir, "tangent")))
        assert p_out == "primal!"
        assert t_out == "tangent!"

    def test_pullback_runs(self):
        def inner(x):
            return core.concat(x, "!")

        inner_ir = core.build_ir(inner, "X")

        def outer(prog, x):
            return core.ir_call(prog, x)

        outer_ir = core.build_ir(outer, inner_ir, "X")
        pb_ir = core.pullback_ir(outer_ir)
        _, cotangents = core.run_ir(pb_ir, ((inner_ir, "hello"), "grad"))
        assert len(cotangents) == 2

    def test_batch(self):
        def inner(x):
            return core.format("[{}]", x)

        inner_ir = core.build_ir(inner, "X")

        def outer(prog, x):
            return core.ir_call(prog, x)

        outer_ir = core.build_ir(outer, inner_ir, "X")
        batched_ir = core.batch_ir(outer_ir, in_axes=(None, list))
        result = core.run_ir(batched_ir, inner_ir, ["a", "b", "c"])
        assert result == ["[a]", "[b]", "[c]"]

    def test_nested_ir_call(self):
        def inner1(x):
            return core.concat(x, "1")

        def inner2(x):
            return core.concat(x, "2")

        ir1 = core.build_ir(inner1, "X")
        ir2 = core.build_ir(inner2, "X")

        def outer(p1, p2, x):
            r1 = core.ir_call(p1, x)
            return core.ir_call(p2, r1)

        outer_ir = core.build_ir(outer, ir1, ir2, "X")
        result = core.run_ir(outer_ir, ir1, ir2, "start")
        assert result == "start12"
