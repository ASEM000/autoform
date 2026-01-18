import pytest

import autoform as af


class TestBuildIR:
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

    @pytest.mark.asyncio(loop_scope="function")
    async def test_traces_literal_and_variable_async(self):
        async def program(name):
            return af.concat("Hello, ", name)

        ir = await af.atrace(program)("Alice")
        assert len(ir.ireqns) == 1
        assert isinstance(ir.in_irtree, af.core.IRVar)
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 2
        lit_candidate = eqn.in_irtree[0]
        assert (
            isinstance(lit_candidate, af.core.IRLit) and lit_candidate.value == "Hello, "
        ) or lit_candidate == "Hello, "
        assert isinstance(eqn.in_irtree[1], af.core.IRVar)

    def test_format_traces_template_and_args(self):
        def program(x):
            return af.format("Hello, {}!", x)

        ir = af.trace(program)("World")
        assert len(ir.ireqns) == 1
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 1
        assert eqn.params["template"] == "Hello, {}!"
        assert isinstance(eqn.in_irtree[0], af.core.IRVar)
        assert af.call(ir)("Alice") == "Hello, Alice!"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_format_traces_template_and_args_async(self):
        async def program(x):
            return af.format("Hello, {}!", x)

        ir = await af.atrace(program)("World")
        assert len(ir.ireqns) == 1
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 1
        assert eqn.params["template"] == "Hello, {}!"
        assert isinstance(eqn.in_irtree[0], af.core.IRVar)
        assert await af.acall(ir)("Alice") == "Hello, Alice!"

    def test_multiple_operations(self):
        def program(x, y):
            a = af.concat(x, y)
            b = af.format("[{}]", a)
            return b

        ir = af.trace(program)("A", "B")
        assert len(ir.ireqns) == 2

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_operations_async(self):
        async def program(x, y):
            a = af.concat(x, y)
            b = af.format("[{}]", a)
            return b

        ir = await af.atrace(program)("A", "B")
        assert len(ir.ireqns) == 2

    def test_single_input_tree_structure(self):
        def program(x):
            return af.concat(x, x)

        ir = af.trace(program)("test")
        assert isinstance(ir.in_irtree, af.core.IRVar)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_single_input_tree_structure_async(self):
        async def program(x):
            return af.concat(x, x)

        ir = await af.atrace(program)("test")
        assert isinstance(ir.in_irtree, af.core.IRVar)

    def test_tuple_input_tree_structure(self):
        def program(a, b):
            return af.concat(a, b)

        ir = af.trace(program)("A", "B")
        assert isinstance(ir.in_irtree, tuple)
        assert len(ir.in_irtree) == 2

    @pytest.mark.asyncio(loop_scope="function")
    async def test_tuple_input_tree_structure_async(self):
        async def program(a, b):
            return af.concat(a, b)

        ir = await af.atrace(program)("A", "B")
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

    @pytest.mark.asyncio(loop_scope="function")
    async def test_basic_execution_atrace(self):
        async def program(x):
            return af.concat(x, "!")

        ir = await af.atrace(program)("hello")
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

    @pytest.mark.asyncio(loop_scope="function")
    async def test_chained_operations_atrace(self):
        async def program(x):
            step1 = af.concat(x, x)
            step2 = af.format("[{}]", step1)
            return step2

        ir = await af.atrace(program)("A")
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

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_args_atrace(self):
        async def program(a, b):
            return af.format("{} + {}", a, b)

        ir = await af.atrace(program)("x", "y")
        result = await af.acall(ir)("1", "2")
        assert result == "1 + 2"


class TestABuildIRSpecific:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_atrace_with_await_inside(self):
        async def inner_async():
            return "async_value"

        async def program(x):
            val = await inner_async()
            return af.concat(x, val)

        ir = await af.atrace(program)("test")
        assert len(ir.ireqns) == 1
        result = await af.acall(ir)("prefix_")
        assert result == "prefix_async_value"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_atrace_produces_same_ir_structure(self):
        def sync_program(x):
            a = af.format("[{}]", x)
            b = af.concat(a, "!")
            return b

        async def async_program(x):
            a = af.format("[{}]", x)
            b = af.concat(a, "!")
            return b

        sync_ir = af.trace(sync_program)("test")
        async_ir = await af.atrace(async_program)("test")

        assert len(sync_ir.ireqns) == len(async_ir.ireqns)
        for s_eqn, a_eqn in zip(sync_ir.ireqns, async_ir.ireqns):
            assert s_eqn.prim.name == a_eqn.prim.name

    @pytest.mark.asyncio(loop_scope="function")
    async def test_atrace_with_transforms(self):
        async def program(x):
            return af.format("[{}]", x)

        ir = await af.atrace(program)("test")

        pf_ir = af.pushforward(ir)
        result = await af.acall(pf_ir)(("primal", "tangent"))
        assert result == ("[primal]", "[tangent]")

        pb_ir = af.pullback(ir)
        out, cot = await af.acall(pb_ir)(("primal", "grad"))
        assert out == "[primal]"
        assert cot == "grad"

        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["a", "b", "c"])
        assert result == ["[a]", "[b]", "[c]"]
