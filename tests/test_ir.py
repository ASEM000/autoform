import pytest
import autoform as af
import functools as ft


class TestBuildIR:
    def test_traces_literal_and_variable(self):
        def program(name):
            return af.concat("Hello, ", name)

        ir = af.build_ir(program)("Alice")
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

        ir = af.build_ir(program)("World")
        assert len(ir.ireqns) == 1
        eqn = ir.ireqns[0]
        assert len(eqn.in_irtree) == 1
        assert eqn.params["template"] == "Hello, {}!"
        assert isinstance(eqn.in_irtree[0], af.core.IRVar)
        assert af.call(ir)("Alice") == "Hello, Alice!"

    def test_multiple_operations(self):
        def program(x, y):
            a = af.concat(x, y)
            b = af.format("[{}]", a)
            return b

        ir = af.build_ir(program)("A", "B")
        assert len(ir.ireqns) == 2

    def test_single_input_tree_structure(self):
        def program(x):
            return af.concat(x, x)

        ir = af.build_ir(program)("test")
        assert isinstance(ir.in_irtree, af.core.IRVar)

    def test_tuple_input_tree_structure(self):
        def program(a, b):
            return af.concat(a, b)

        ir = af.build_ir(program)("A", "B")
        assert isinstance(ir.in_irtree, tuple)
        assert len(ir.in_irtree) == 2


class TestRunIR:
    def test_basic_execution(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program)("hello")
        result = af.call(ir)("world")
        assert result == "world!"

    def test_chained_operations(self):
        def program(x):
            step1 = af.concat(x, x)
            step2 = af.format("[{}]", step1)
            return step2

        ir = af.build_ir(program)("A")
        result = af.call(ir)("B")
        assert result == "[BB]"

    def test_multiple_args(self):
        def program(a, b):
            return af.format("{} + {}", a, b)

        ir = af.build_ir(program)("x", "y")
        result = af.call(ir)("1", "2")
        assert result == "1 + 2"


class TestIterIR:
    def test_streams_and_accumulates(self):
        stream_p = af.core.Primitive("stream")

        @ft.partial(af.core.eval_rules.def_rule, stream_p)
        def eval_stream(x):
            return af.core.Var()

        @ft.partial(af.core.iter_rules.def_rule, stream_p)
        def iter_stream(x):
            for ch in x:
                yield ch

        def stream(x):
            return stream_p.bind(x)

        ir = af.build_ir(stream)("AB")
        outputs = list(af.icall(ir)("AB"))
        assert outputs[:-1] == ["A", "B"]
        assert outputs[-1] == "AB"

    def test_fallback_to_impl_rule(self):
        def program(x, y):
            return af.concat(x, y)

        ir = af.build_ir(program)("A", "B")
        outputs = list(af.icall(ir)("A", "B"))
        assert outputs == ["AB"]

    def test_multiple_outputs(self):
        split_p = af.core.Primitive("split")

        @ft.partial(af.core.eval_rules.def_rule, split_p)
        def eval_split(x):
            return af.core.Var(), af.core.Var()

        @ft.partial(af.core.iter_rules.def_rule, split_p)
        def iter_split(x):
            for ch in x:
                yield (ch, ch)

        def split(x):
            return split_p.bind(x)

        ir = af.build_ir(split)("AB")
        iterator = af.icall(ir)("AB")
        chunk1 = next(iterator)
        assert chunk1 == ("A", "A")
        chunk2 = next(iterator)
        assert chunk2 == ("B", "B")
        final_res = next(iterator)
        assert final_res == ("AB", "AB")

    def test_string_accumulation(self):
        p = af.core.Primitive("strs")

        @ft.partial(af.core.eval_rules.def_rule, p)
        def eval_rule(x):
            return af.core.Var()

        @ft.partial(af.core.iter_rules.def_rule, p)
        def iter_rule(x):
            yield "a"
            yield "b"
            yield "c"

        def func(x):
            return p.bind(x)

        ir = af.build_ir(func)("input")
        results = list(af.icall(ir)("input"))
        assert results[-1] == "abc"

    def test_list_accumulation(self):
        p = af.core.Primitive("lists")

        @ft.partial(af.core.eval_rules.def_rule, p)
        def eval_rule(x):
            return af.core.Var()

        @ft.partial(af.core.iter_rules.def_rule, p)
        def iter_rule(x):
            yield [1, 2]
            yield [3, 4]

        def func(x):
            return p.bind(x)

        ir = af.build_ir(func)("input")
        results = list(af.icall(ir)("input"))
        assert results[-1] == [1, 2, 3, 4]

    def test_program_call_streams_through(self):
        stream_p = af.core.Primitive("stream_tokens")

        @ft.partial(af.core.eval_rules.def_rule, stream_p)
        def eval_stream(x):
            return af.core.Var()

        @ft.partial(af.core.iter_rules.def_rule, stream_p)
        def iter_stream(in_tree):
            x = in_tree
            for ch in x:
                yield ch

        def stream_tokens(x):
            return stream_p.bind(x)

        inner_ir = af.build_ir(stream_tokens)("abc")

        # run_ir inlines the streaming primitive directly
        def outer(x):
            return af.call(inner_ir)(x)

        outer_ir = af.build_ir(outer)("abc")
        outputs = list(af.icall(outer_ir)("xyz"))
        assert outputs[:-1] == ["x", "y", "z"]
        assert outputs[-1] == "xyz"

    def test_nested_program_call_streams(self):
        stream_p = af.core.Primitive("deep_stream")

        @ft.partial(af.core.eval_rules.def_rule, stream_p)
        def eval_stream(x):
            return af.core.Var()

        @ft.partial(af.core.iter_rules.def_rule, stream_p)
        def iter_stream(in_tree):
            x = in_tree
            for i, ch in enumerate(x):
                yield f"{i}:{ch}"

        def deep_stream(x):
            return stream_p.bind(x)

        ir_level0 = af.build_ir(deep_stream)("ab")

        def level1(x):
            return af.call(ir_level0)(x)

        ir_level1 = af.build_ir(level1)("ab")

        def level2(x):
            return af.call(ir_level1)(x)

        ir_level2 = af.build_ir(level2)("ab")
        outputs = list(af.icall(ir_level2)("XY"))
        assert outputs[:-1] == ["0:X", "1:Y"]
        assert outputs[-1] == "0:X1:Y"


class TestAsyncIR:
    @pytest.mark.asyncio
    async def test_basic_async(self):
        import asyncio

        p = af.core.Primitive("async_identity")

        @ft.partial(af.core.eval_rules.def_rule, p)
        def eval_rule(x):
            return af.core.Var()

        @ft.partial(af.core.async_rules.def_rule, p)
        async def async_rule(x):
            await asyncio.sleep(0.001)
            return x

        def func(x):
            return p.bind(x)

        ir = af.build_ir(func)("hello")
        result = await af.acall(ir)("hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_fallback_to_impl(self):
        p = af.core.Primitive("sync_only")

        @ft.partial(af.core.eval_rules.def_rule, p)
        def eval_rule(x):
            return af.core.Var()

        @ft.partial(af.core.impl_rules.def_rule, p)
        def impl_rule(x):
            return x + "!"

        def func(x):
            return p.bind(x)

        ir = af.build_ir(func)("hello")
        result = await af.acall(ir)("hello")
        assert result == "hello!"
