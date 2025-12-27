import autoform as af
import functools as ft


class TestPrimitive:
    def test_creation(self):
        p = af.core.Primitive("test_prim")
        assert p.name == "test_prim"
        assert repr(p) == "test_prim"

    def test_def_impl_decorator(self):
        p = af.core.Primitive("test_impl")

        @ft.partial(af.core.impl_rules.def_rule, p)
        def impl(x):
            return x

        assert af.core.impl_rules[p] is impl

    def test_def_eval_decorator(self):
        p = af.core.Primitive("test_eval")

        @ft.partial(af.core.eval_rules.def_rule, p)
        def eval_rule(x):
            return af.core.Var()

        assert af.core.eval_rules[p] is eval_rule

    def test_def_batch_decorator(self):
        p = af.core.Primitive("test_batch")

        @ft.partial(af.core.batch_rules.def_rule, p)
        def batch_rule(batch_size, in_batched, in_tree):
            return in_tree, True

        assert af.core.batch_rules[p] is batch_rule

    def test_def_pushforward_decorator(self):
        p = af.core.Primitive("test_pushforward")

        @ft.partial(af.core.push_rules.def_rule, p)
        def pf_rule(primals, tangents):
            return primals, tangents

        assert af.core.push_rules[p] is pf_rule

    def test_def_pullback_forward_decorator(self):
        p = af.core.Primitive("test_pullback_fwd")

        @ft.partial(af.core.pull_fwd_rules.def_rule, p)
        def pb_fwd_rule(in_tree):
            return in_tree, in_tree

        assert af.core.pull_fwd_rules[p] is pb_fwd_rule

    def test_def_pullback_backward_decorator(self):
        p = af.core.Primitive("test_pullback_bwd")

        @ft.partial(af.core.pull_bwd_rules.def_rule, p)
        def pb_bwd_rule(residuals, cotangent_out):
            return cotangent_out

        assert af.core.pull_bwd_rules[p] is pb_bwd_rule

    def test_def_iter_decorator(self):
        p = af.core.Primitive("test_iter")

        @ft.partial(af.core.iter_rules.def_rule, p)
        def iter_rule(x):
            yield [x]

        assert af.core.iter_rules[p] is iter_rule

    def test_def_async_decorator(self):
        p = af.core.Primitive("test_async")

        @ft.partial(af.core.async_rules.def_rule, p)
        async def async_rule(x):
            return x

        assert af.core.async_rules[p] is async_rule


class TestFormatPrimitive:
    def test_basic_format(self):
        result = af.format("Hello, {}!", "World")
        assert result == "Hello, World!"

    def test_multiple_placeholders(self):
        result = af.format("{} + {} = {}", "1", "2", "3")
        assert result == "1 + 2 = 3"

    def test_format_ir(self):
        def func(x):
            return af.format("Value: {}", x)

        ir = af.build_ir(func)("test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "format"


class TestConcatPrimitive:
    def test_basic_concat(self):
        result = af.concat("Hello", " ", "World")
        assert result == "Hello World"

    def test_two_args(self):
        result = af.concat("A", "B")
        assert result == "AB"

    def test_concat_ir(self):
        def func(x, y):
            return af.concat(x, y)

        ir = af.build_ir(func)("a", "b")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "concat"


class TestBind:
    def test_bind_using(self):
        p = af.core.Primitive("custom_bind")

        @ft.partial(af.core.impl_rules.def_rule, p)
        def impl(in_tree, *, multiplier):
            return in_tree * multiplier

        @ft.partial(af.core.eval_rules.def_rule, p)
        def eval_rule(in_tree, *, multiplier):
            return af.core.Var()

        def func(x):
            return p.bind(x, multiplier=3)

        ir = af.build_ir(func)("A")
        assert ir.ireqns[0].params["multiplier"] == 3
        result = af.call(ir)("B")
        assert result == "BBB"


class TestInterpreter:
    def test_eval_interpreter_is_default(self):
        result = af.concat("a", "b")
        assert result == "ab"

    def test_use_interpreter_context(self):
        tracer = af.core.TracingInterpreter()
        with af.core.using_interp(tracer) as t:
            assert t is tracer
            af.format("Hello, {}!", af.core.IRVar.fresh())
            assert len(tracer.ireqns) == 1
        result = af.concat("a", "b")
        assert result == "ab"

    def test_tracing_interpreter_creates_ireqns(self):
        tracer = af.core.TracingInterpreter()
        with af.core.using_interp(tracer):
            af.format("Hello, {}!", af.core.IRVar.fresh())
        assert len(tracer.ireqns) == 1


class TestStopGradient:
    def test_impl_is_identity(self):
        result = af.stop_gradient("hello")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)("test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "stop_gradient"

    def test_run_ir(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)("test")
        result = af.call(ir)("hello")
        assert result == "hello"

    def test_pushforward_zeros_tangent(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)("a")
        pf_ir = af.pushforward(ir)
        primal_out, tangent_out = af.call(pf_ir)(("primal", "tangent"))
        assert primal_out == "primal"
        assert tangent_out == "" or (hasattr(tangent_out, "items") and len(tangent_out.items) == 0)

    def test_pullback_zeros_cotangent(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)("a")
        pb_ir = af.pullback(ir)
        primal_out, cotangent_in = af.call(pb_ir)(("primal", "cotangent"))
        assert primal_out == "primal"
        assert cotangent_in == "" or (
            hasattr(cotangent_in, "items") and len(cotangent_in.items) == 0
        )

    def test_batch(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain_stops_gradient(self):
        def is_zero_cotangent(val):
            return val == "" or (hasattr(val, "items") and len(val.items) == 0)

        def func(x, y):
            stopped = af.stop_gradient(x)
            return af.concat(stopped, y)

        ir = af.build_ir(func)("a", "b")
        pb_ir = af.pullback(ir)
        _, (cotangent_x, cotangent_y) = af.call(pb_ir)((("a", "b"), "grad"))
        assert is_zero_cotangent(cotangent_x)
        assert cotangent_y == "grad"

    def test_chained_with_format(self):
        def func(x):
            stopped = af.stop_gradient(x)
            return af.format("[{}]", stopped)

        ir = af.build_ir(func)("test")
        result = af.call(ir)("hello")
        assert result == "[hello]"

    def test_tree_input(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)(("a", "b"))
        result = af.call(ir)(("hello", "world"))
        assert result == ("hello", "world")

    def test_tree_pullback_zeros_all(self):
        def is_zero_cotangent(val):
            return val == "" or (hasattr(val, "items") and len(val.items) == 0)

        def func(x):
            return af.stop_gradient(x)

        ir = af.build_ir(func)(("a", "b"))
        pb_ir = af.pullback(ir)
        _, cotangent_in = af.call(pb_ir)((("p1", "p2"), ("c1", "c2")))
        assert is_zero_cotangent(cotangent_in[0])
        assert is_zero_cotangent(cotangent_in[1])


class TestRunIRInline:
    """Tests for run_ir inlining behavior when called inside traced functions."""

    def test_run_ir_inlines_operations(self):
        """run_ir inside a traced function inlines the inner IR's operations."""
        inner_ir = af.build_ir(lambda x: af.format("[{}]", x))("X")

        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("X")
        # Operations are inlined, so we see format directly
        assert len(outer_ir.ireqns) == 1
        assert outer_ir.ireqns[0].prim.name == "format"

    def test_run_ir_inline_executes_correctly(self):
        """Inlined run_ir produces correct output."""
        inner_ir = af.build_ir(lambda x: af.format("<{}>", x))("X")

        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("X")
        result = af.call(outer_ir)("test")
        assert result == "<test>"

    def test_run_ir_inline_with_multiple_ops(self):
        """Multiple operations are all inlined."""

        def inner(x):
            a = af.concat(x, "!")
            b = af.format("[{}]", a)
            return b

        inner_ir = af.build_ir(inner)("X")

        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("X")
        assert len(outer_ir.ireqns) == 2
        result = af.call(outer_ir)("hello")
        assert result == "[hello!]"

    def test_nested_run_ir_inlines(self):
        """Nested run_ir calls all get inlined."""
        ir1 = af.build_ir(lambda x: af.concat(x, "1"))("X")
        ir2 = af.build_ir(lambda x: af.concat(x, "2"))("X")

        def outer(x):
            r1 = af.call(ir1)(x)
            return af.call(ir2)(r1)

        outer_ir = af.build_ir(outer)("X")
        assert len(outer_ir.ireqns) == 2
        result = af.call(outer_ir)("start")
        assert result == "start12"

    def test_pushforward_on_inlined_run_ir(self):
        """Pushforward works on inlined run_ir."""
        inner_ir = af.build_ir(lambda x: af.concat(x, "!"))("X")

        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("X")
        pf_ir = af.pushforward(outer_ir)
        (p_out, t_out) = af.call(pf_ir)(("primal", "tangent"))
        assert p_out == "primal!"
        assert t_out == "tangent!"

    def test_pullback_on_inlined_run_ir(self):
        """Pullback works on inlined run_ir."""
        inner_ir = af.build_ir(lambda x: af.concat(x, "!"))("X")

        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("X")
        pb_ir = af.pullback(outer_ir)
        _, cotangent = af.call(pb_ir)(("hello", "grad"))
        assert cotangent == "grad"

    def test_batch_on_inlined_run_ir(self):
        """Batch works on inlined run_ir."""
        inner_ir = af.build_ir(lambda x: af.format("[{}]", x))("X")

        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("X")
        batched_ir = af.batch(outer_ir, in_axes=list)
        result = af.call(batched_ir)(["a", "b", "c"])
        assert result == ["[a]", "[b]", "[c]"]
