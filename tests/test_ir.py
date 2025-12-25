import pytest
import autoform.core as core
import functools as ft


class TestBuildIR:
    def test_traces_literal_and_variable(self):
        def program(name):
            return core.concat("Hello, ", name)

        ir = core.build_ir(program, "Alice")
        assert len(ir.ireqns) == 1
        assert core.is_irvar(ir.in_irtree)
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 2
        lit_candidate = eqn.in_irtree[0]
        assert (
            core.is_irlit(lit_candidate) and lit_candidate.value == "Hello, "
        ) or lit_candidate == "Hello, "
        assert core.is_irvar(eqn.in_irtree[1])

    def test_format_traces_template_and_args(self):
        def program(x):
            return core.format("Hello, {}!", x)

        ir = core.build_ir(program, "World")
        assert len(ir.ireqns) == 1
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 1
        assert eqn.params["template"] == "Hello, {}!"
        assert core.is_irvar(eqn.in_irtree[0])
        assert core.run_ir(ir, "Alice") == "Hello, Alice!"

    def test_multiple_operations(self):
        def program(x, y):
            a = core.concat(x, y)
            b = core.format("[{}]", a)
            return b

        ir = core.build_ir(program, "A", "B")
        assert len(ir.ireqns) == 2

    def test_single_input_tree_structure(self):
        def program(x):
            return core.concat(x, x)

        ir = core.build_ir(program, "test")
        assert core.is_irvar(ir.in_irtree)

    def test_tuple_input_tree_structure(self):
        def program(a, b):
            return core.concat(a, b)

        ir = core.build_ir(program, "A", "B")
        assert isinstance(ir.in_irtree, tuple)
        assert len(ir.in_irtree) == 2


class TestRunIR:
    def test_basic_execution(self):
        def program(x):
            return core.concat(x, "!")

        ir = core.build_ir(program, "hello")
        result = core.run_ir(ir, "world")
        assert result == "world!"

    def test_chained_operations(self):
        def program(x):
            step1 = core.concat(x, x)
            step2 = core.format("[{}]", step1)
            return step2

        ir = core.build_ir(program, "A")
        result = core.run_ir(ir, "B")
        assert result == "[BB]"

    def test_multiple_args(self):
        def program(a, b):
            return core.format("{} + {}", a, b)

        ir = core.build_ir(program, "x", "y")
        result = core.run_ir(ir, "1", "2")
        assert result == "1 + 2"


class TestIterIR:
    def test_streams_and_accumulates(self):
        stream_p = core.Primitive("stream")

        @ft.partial(core.eval_rules.def_rule, stream_p)
        def eval_stream(x):
            return core.Var()

        @ft.partial(core.iter_rules.def_rule, stream_p)
        def iter_stream(x):
            for ch in x:
                yield ch

        def stream(x):
            return stream_p.bind(x)

        ir = core.build_ir(stream, "AB")
        outputs = list(core.iter_ir(ir, "AB"))
        assert outputs[:-1] == ["A", "B"]
        assert outputs[-1] == "AB"

    def test_fallback_to_impl_rule(self):
        def program(x, y):
            return core.concat(x, y)

        ir = core.build_ir(program, "A", "B")
        outputs = list(core.iter_ir(ir, "A", "B"))
        assert outputs == ["AB"]

    def test_multiple_outputs(self):
        split_p = core.Primitive("split")

        @ft.partial(core.eval_rules.def_rule, split_p)
        def eval_split(x):
            return core.Var(), core.Var()

        @ft.partial(core.iter_rules.def_rule, split_p)
        def iter_split(x):
            for ch in x:
                yield (ch, ch)

        def split(x):
            return split_p.bind(x)

        ir = core.build_ir(split, "AB")
        iterator = core.iter_ir(ir, "AB")
        chunk1 = next(iterator)
        assert chunk1 == ("A", "A")
        chunk2 = next(iterator)
        assert chunk2 == ("B", "B")
        final_res = next(iterator)
        assert final_res == ("AB", "AB")

    def test_string_accumulation(self):
        p = core.Primitive("strs")

        @ft.partial(core.eval_rules.def_rule, p)
        def eval_rule(x):
            return core.Var()

        @ft.partial(core.iter_rules.def_rule, p)
        def iter_rule(x):
            yield "a"
            yield "b"
            yield "c"

        def func(x):
            return p.bind(x)

        ir = core.build_ir(func, "input")
        results = list(core.iter_ir(ir, "input"))
        assert results[-1] == "abc"

    def test_list_accumulation(self):
        p = core.Primitive("lists")

        @ft.partial(core.eval_rules.def_rule, p)
        def eval_rule(x):
            return core.Var()

        @ft.partial(core.iter_rules.def_rule, p)
        def iter_rule(x):
            yield [1, 2]
            yield [3, 4]

        def func(x):
            return p.bind(x)

        ir = core.build_ir(func, "input")
        results = list(core.iter_ir(ir, "input"))
        assert results[-1] == [1, 2, 3, 4]

    def test_program_call_streams_through(self):
        """Test that iter_ir streams through program_call via its iter_rule."""
        stream_p = core.Primitive("stream_tokens")

        @ft.partial(core.eval_rules.def_rule, stream_p)
        def eval_stream(x):
            return core.Var()

        @ft.partial(core.iter_rules.def_rule, stream_p)
        def iter_stream(in_tree):
            x = in_tree
            for ch in x:
                yield ch

        def stream_tokens(x):
            return stream_p.bind(x)

        inner_ir = core.build_ir(stream_tokens, "abc")

        def outer(x):
            return core.ir_call(inner_ir, x)

        outer_ir = core.build_ir(outer, "abc")
        outputs = list(core.iter_ir(outer_ir, "xyz"))
        assert outputs[:-1] == ["x", "y", "z"]
        assert outputs[-1] == "xyz"

    def test_nested_program_call_streams(self):
        """Test streaming through multiple levels of program_call."""
        stream_p = core.Primitive("deep_stream")

        @ft.partial(core.eval_rules.def_rule, stream_p)
        def eval_stream(x):
            return core.Var()

        @ft.partial(core.iter_rules.def_rule, stream_p)
        def iter_stream(in_tree):
            x = in_tree
            for i, ch in enumerate(x):
                yield f"{i}:{ch}"

        def deep_stream(x):
            return stream_p.bind(x)

        ir_level0 = core.build_ir(deep_stream, "ab")

        def level1(x):
            return core.ir_call(ir_level0, x)

        ir_level1 = core.build_ir(level1, "ab")

        def level2(x):
            return core.ir_call(ir_level1, x)

        ir_level2 = core.build_ir(level2, "ab")
        outputs = list(core.iter_ir(ir_level2, "XY"))
        assert outputs[:-1] == ["0:X", "1:Y"]
        assert outputs[-1] == "0:X1:Y"


class TestAsyncIR:
    @pytest.mark.asyncio
    async def test_basic_async(self):
        import asyncio

        p = core.Primitive("async_identity")

        @ft.partial(core.eval_rules.def_rule, p)
        def eval_rule(x):
            return core.Var()

        @ft.partial(core.async_rules.def_rule, p)
        async def async_rule(x):
            await asyncio.sleep(0.001)
            return x

        def func(x):
            return p.bind(x)

        ir = core.build_ir(func, "hello")
        result = await core.arun_ir(ir, "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_fallback_to_impl(self):
        p = core.Primitive("sync_only")

        @ft.partial(core.eval_rules.def_rule, p)
        def eval_rule(x):
            return core.Var()

        @ft.partial(core.impl_rules.def_rule, p)
        def impl_rule(x):
            return x + "!"

        def func(x):
            return p.bind(x)

        ir = core.build_ir(func, "hello")
        result = await core.arun_ir(ir, "hello")
        assert result == "hello!"


class TestIRAtoms:
    def test_irvar_creation(self):
        v = core.IRVar(42)
        assert v.id == 42
        assert core.is_irvar(v)
        assert core.is_iratom(v)

    def test_irlit_creation(self):
        lit = core.IRLit("hello")
        assert lit.value == "hello"
        assert core.is_irlit(lit)
        assert core.is_iratom(lit)

    def test_irzero_creation(self):
        lit = core.IRLit("test")
        zero = core.IRZero(lit.value)
        assert zero.value == "test"
        assert core.is_irlit(zero)

    def test_irpvar_is_irvar(self):
        pv = core.IRPVar(0)
        assert core.is_irvar(pv)
        assert isinstance(pv, core.IRVar)

    def test_irtvar_is_irvar(self):
        tv = core.IRTVar(0)
        assert core.is_irvar(tv)
        assert isinstance(tv, core.IRVar)

    def test_ircvar_is_irvar(self):
        cv = core.IRCVar(0)
        assert core.is_irvar(cv)
        assert isinstance(cv, core.IRVar)

    def test_irbvar_is_irvar(self):
        bv = core.IRBVar(0)
        assert core.is_irvar(bv)
        assert isinstance(bv, core.IRVar)


class TestGenerateTextCode:
    def test_basic_ir_repr(self):
        def program(x):
            return core.concat(x, x)

        ir = core.build_ir(program, "test")
        text = core.generate_text_code(ir)
        assert "func(" in text
        assert "concat" in text
        assert "}" in text

    def test_format_using(self):
        def program(x):
            return core.format("Hello, {}!", x)

        ir = core.build_ir(program, "test")
        text = core.generate_text_code(ir)
        assert "format" in text
        assert "template=" in text
