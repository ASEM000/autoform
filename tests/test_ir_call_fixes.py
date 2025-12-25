import autoform as af


class TestIRCallEvalFix:
    def test_eval_returns_correct_structure(self):
        def inner(x):
            return af.concat("Hello, ", x)

        inner_ir = af.build_ir(inner, "world")

        def outer(y):
            return af.ir_call(inner_ir, y)

        outer_ir = af.build_ir(outer, "test")
        result = af.run_ir(outer_ir, "World")
        assert result == "Hello, World"

    def test_eval_with_structured_output(self):
        def inner(x):
            return af.concat(x, "!")

        inner_ir = af.build_ir(inner, "test")

        def outer(name):
            result = af.ir_call(inner_ir, name)
            return af.concat(result, "!")

        outer_ir = af.build_ir(outer, "Alice")
        result = af.run_ir(outer_ir, "Bob")
        assert result == "Bob!!"


class TestIRCallPullbackFix:
    def test_pullback_no_none_cotangents(self):
        def inner(x):
            return af.format("Input: {}", x)

        inner_ir = af.build_ir(inner, "test")

        def outer(name):
            processed = af.ir_call(inner_ir, name)
            return af.concat(processed, " [done]")

        outer_ir = af.build_ir(outer, "Alice")
        pb_ir = af.pullback_ir(outer_ir)
        output, cotangent_in = af.run_ir(pb_ir, ("Bob", "feedback"))

        assert output == "Input: Bob [done]"
        assert cotangent_in is not None
        assert isinstance(cotangent_in, str)

    def test_pullback_zero_cotangent_for_ir(self):
        def inner(x):
            return af.concat("Hello, ", x)

        inner_ir = af.build_ir(inner, "test")

        def outer(prog, name):
            return af.ir_call(prog, name)

        outer_ir = af.build_ir(outer, inner_ir, "test")
        pb_ir = af.pullback_ir(outer_ir)
        _, (cot_ir, cot_name) = af.run_ir(pb_ir, ((inner_ir, "Alice"), "grad"))

        assert cot_ir == ""
        assert cot_name == "grad"


class TestIRCallNested:
    def test_nested_ir_call_eval(self):
        def format_name(name):
            return af.format("Name: {}", name)

        format_ir = af.build_ir(format_name, "test")

        def add_greeting(name):
            formatted = af.ir_call(format_ir, name)
            return af.concat("Hello! ", formatted)

        greeting_ir = af.build_ir(add_greeting, "test")

        def final_message(name):
            greeting = af.ir_call(greeting_ir, name)
            return af.concat(greeting, " Welcome!")

        final_ir = af.build_ir(final_message, "Alice")
        result = af.run_ir(final_ir, "Bob")
        assert result == "Hello! Name: Bob Welcome!"

    def test_nested_ir_call_pullback(self):
        def inner1(x):
            return af.concat(x, "_1")

        ir1 = af.build_ir(inner1, "test")

        def inner2(x):
            result = af.ir_call(ir1, x)
            return af.concat(result, "_2")

        ir2 = af.build_ir(inner2, "test")

        def outer(x):
            return af.ir_call(ir2, x)

        outer_ir = af.build_ir(outer, "test")
        pb_ir = af.pullback_ir(outer_ir)
        output, cotangent = af.run_ir(pb_ir, ("start", "feedback"))

        assert output == "start_1_2"
        assert cotangent == "feedback"
