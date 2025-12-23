import autoform.core as core


class TestIRCallEvalFix:
    def test_eval_returns_correct_structure(self):
        def inner(x):
            return core.concat("Hello, ", x)

        inner_ir = core.build_ir(inner, "world")

        def outer(y):
            return core.ir_call(inner_ir, y)

        outer_ir = core.build_ir(outer, "test")
        result = core.run_ir(outer_ir, "World")
        assert result == "Hello, World"

    def test_eval_with_structured_output(self):
        def inner(x):
            return core.concat(x, "!")

        inner_ir = core.build_ir(inner, "test")

        def outer(name):
            result = core.ir_call(inner_ir, name)
            return core.concat(result, "!")

        outer_ir = core.build_ir(outer, "Alice")
        result = core.run_ir(outer_ir, "Bob")
        assert result == "Bob!!"


class TestIRCallPullbackFix:
    def test_pullback_no_none_cotangents(self):
        def inner(x):
            return core.format("Input: {}", x)

        inner_ir = core.build_ir(inner, "test")

        def outer(name):
            processed = core.ir_call(inner_ir, name)
            return core.concat(processed, " [done]")

        outer_ir = core.build_ir(outer, "Alice")
        pb_ir = core.pullback_ir(outer_ir)
        output, cotangent_in = core.run_ir(pb_ir, ("Bob", "feedback"))

        assert output == "Input: Bob [done]"
        assert cotangent_in is not None
        assert isinstance(cotangent_in, str)

    def test_pullback_zero_cotangent_for_ir(self):
        def inner(x):
            return core.concat("Hello, ", x)

        inner_ir = core.build_ir(inner, "test")

        def outer(prog, name):
            return core.ir_call(prog, name)

        outer_ir = core.build_ir(outer, inner_ir, "test")
        pb_ir = core.pullback_ir(outer_ir)
        _, (cot_ir, cot_name) = core.run_ir(pb_ir, ((inner_ir, "Alice"), "grad"))

        assert cot_ir == ""
        assert cot_name == "grad"


class TestIRCallNested:
    def test_nested_ir_call_eval(self):
        def format_name(name):
            return core.format("Name: {}", name)

        format_ir = core.build_ir(format_name, "test")

        def add_greeting(name):
            formatted = core.ir_call(format_ir, name)
            return core.concat("Hello! ", formatted)

        greeting_ir = core.build_ir(add_greeting, "test")

        def final_message(name):
            greeting = core.ir_call(greeting_ir, name)
            return core.concat(greeting, " Welcome!")

        final_ir = core.build_ir(final_message, "Alice")
        result = core.run_ir(final_ir, "Bob")
        assert result == "Hello! Name: Bob Welcome!"

    def test_nested_ir_call_pullback(self):
        def inner1(x):
            return core.concat(x, "_1")

        ir1 = core.build_ir(inner1, "test")

        def inner2(x):
            result = core.ir_call(ir1, x)
            return core.concat(result, "_2")

        ir2 = core.build_ir(inner2, "test")

        def outer(x):
            return core.ir_call(ir2, x)

        outer_ir = core.build_ir(outer, "test")
        pb_ir = core.pullback_ir(outer_ir)
        output, cotangent = core.run_ir(pb_ir, ("start", "feedback"))

        assert output == "start_1_2"
        assert cotangent == "feedback"
