import pytest

import autoform as af


class TestBuildIR:
    def test_trace_scalar_input_is_dynamic(self):
        def program(x):
            return af.format("{}", x)

        cases = [
            (1, 2, "2"),
            (1.5, 2.5, "2.5"),
            (True, False, "False"),
        ]

        for traced, runtime, expected in cases:
            ir = af.trace(program)(traced)
            assert isinstance(ir.in_irtree, af.core.IRVar)
            assert ir.in_irtree.type is type(traced)
            assert af.call(ir)(runtime) == expected

    def test_trace_dict_input_with_scalar_leaves(self):
        def program(payload):
            return af.format(
                "{} {} {} {}",
                payload["name"],
                payload["count"],
                payload["score"],
                payload["active"],
            )

        ir = af.trace(program)({"name": "cats", "count": 1, "score": 1.5, "active": True})
        result = af.call(ir)({"name": "dogs", "count": 2, "score": 2.5, "active": False})
        assert result == "dogs 2 2.5 False"

    def test_trace_unsupported_input_leaf_errors(self):
        class Opaque:
            pass

        def program(x):
            return x

        with pytest.raises(AssertionError, match="Unsupported input leaf type"):
            af.trace(program)(Opaque())

    def test_traces_literal_and_variable(self):
        def program(name):
            return af.concat("Hello, ", name)

        ir = af.trace(program)("Alice")
        assert len(ir.ireqns) == 1
        assert isinstance(ir.in_irtree, af.core.IRVar)
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 2
        lit_candidate = eqn.in_irtree[0]
        assert (
            isinstance(lit_candidate, af.core.IRLit) and lit_candidate.value == "Hello, "
        ) or lit_candidate == "Hello, "
        assert isinstance(eqn.in_irtree[1], af.core.IRVar)

    def test_trace_rejects_async_functions(self):
        async def program(name):
            return af.concat("Hello, ", name)

        with pytest.raises(AssertionError, match="only supports sync functions"):
            af.trace(program)

    def test_format_traces_template_and_args(self):
        def program(x):
            return af.format("Hello, {}!", x)

        ir = af.trace(program)("World")
        assert len(ir.ireqns) == 1
        eqn = ir.ireqns[0]
        args, kwargs_values = eqn.in_irtree
        assert len(args) == 1
        assert len(kwargs_values) == 0
        assert eqn.params["template"] == "Hello, {}!"
        assert isinstance(args[0], af.core.IRVar)
        assert af.call(ir)("Alice") == "Hello, Alice!"

    def test_multiple_operations(self):
        def program(x, y):
            a = af.concat(x, y)
            b = af.format("[{}]", a)
            return b

        ir = af.trace(program)("A", "B")
        assert len(ir.ireqns) == 2

    def test_single_input_tree_structure(self):
        def program(x):
            return af.concat(x, x)

        ir = af.trace(program)("test")
        assert isinstance(ir.in_irtree, af.core.IRVar)

    def test_tuple_input_tree_structure(self):
        def program(a, b):
            return af.concat(a, b)

        ir = af.trace(program)("A", "B")
        assert isinstance(ir.in_irtree, tuple)
        assert len(ir.in_irtree) == 2


class TestRunIR:
    def test_basic_execution(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("hello")
        result = af.call(ir)("world")
        assert result == "world!"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_basic_execution_async(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("hello")
        result = await af.acall(ir)("world")
        assert result == "world!"

    def test_chained_operations(self):
        def program(x):
            step1 = af.concat(x, x)
            step2 = af.format("[{}]", step1)
            return step2

        ir = af.trace(program)("A")
        result = af.call(ir)("B")
        assert result == "[BB]"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_chained_operations_async(self):
        def program(x):
            step1 = af.concat(x, x)
            step2 = af.format("[{}]", step1)
            return step2

        ir = af.trace(program)("A")
        result = await af.acall(ir)("B")
        assert result == "[BB]"

    def test_multiple_args(self):
        def program(a, b):
            return af.format("{} + {}", a, b)

        ir = af.trace(program)("x", "y")
        result = af.call(ir)("1", "2")
        assert result == "1 + 2"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_args_async(self):
        def program(a, b):
            return af.format("{} + {}", a, b)

        ir = af.trace(program)("x", "y")
        result = await af.acall(ir)("1", "2")
        assert result == "1 + 2"
