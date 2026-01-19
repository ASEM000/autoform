import functools as ft

import pytest

import autoform as af
from autoform.utils import rebatch


class TestBatchBasic:
    def test_single_arg(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["hello", "world"])
        assert result == ["hello!", "world!"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_single_arg_async(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["hello", "world"])
        assert result == ["hello!", "world!"]

    def test_two_args(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_two_args_async(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    def test_concat(self):
        def join(a, b):
            return af.concat(a, b)

        ir = af.trace(join)("Hello", " World")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["Hello", "Good"], [" World", " Day"])
        assert result == ["Hello World", "Good Day"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_concat_async(self):
        def join(a, b):
            return af.concat(a, b)

        ir = af.trace(join)("Hello", " World")
        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["Hello", "Good"], [" World", " Day"])
        assert result == ["Hello World", "Good Day"]

    def test_chained(self):
        def process(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        ir = af.trace(process)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b", "c"])
        assert result == ["[a]!", "[b]!", "[c]!"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_chained_async(self):
        def process(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        ir = af.trace(process)("a")
        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["a", "b", "c"])
        assert result == ["[a]!", "[b]!", "[c]!"]

    def test_nested_format(self):
        def template(name, value):
            inner = af.format("{} units", value)
            return af.format("{}: {}", name, inner)

        ir = af.trace(template)("temp", "25")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["temp", "pressure"], ["25", "101"])
        assert result == ["temp: 25 units", "pressure: 101 units"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_nested_format_async(self):
        def template(name, value):
            inner = af.format("{} units", value)
            return af.format("{}: {}", name, inner)

        ir = af.trace(template)("temp", "25")
        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["temp", "pressure"], ["25", "101"])
        assert result == ["temp: 25 units", "pressure: 101 units"]

    def test_empty_batch(self):
        def f(x):
            return af.format("{}!", x)

        ir = af.trace(f)("a")
        batched_ir = af.batch(ir)
        with pytest.raises(AssertionError):
            af.call(batched_ir)([])


class TestBatchIRStructure:
    def test_creates_single_eqn(self):
        def f(x):
            return af.concat(x, x)

        ir = af.trace(f)("hello")
        batched_ir = af.batch(ir)
        assert len(batched_ir.ireqns) == 1
        assert batched_ir.ireqns[0].prim.name == "batch_call"

    def test_has_in_axes_param(self):
        def f(x):
            return af.concat(x, x)

        ir = af.trace(f)("hello")
        batched_ir = af.batch(ir, in_axes=True)
        assert "in_axes" in batched_ir.ireqns[0].params

    def test_has_sub_ir_param(self):
        def f(x):
            return af.concat(x, x)

        ir = af.trace(f)("hello")
        batched_ir = af.batch(ir)
        assert "ir" in batched_ir.ireqns[0].params


class TestNestedBatch:
    def test_batch_of_batch(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        double_batched_ir = af.batch(batched_ir)
        result = af.call(double_batched_ir)([["a", "b"], ["c", "d", "e"]])
        assert result == [["a!", "b!"], ["c!", "d!", "e!"]]

    def test_batch_of_batch_two_args(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir)
        double_batched_ir = af.batch(batched_ir)
        result = af.call(double_batched_ir)(
            [["Asem", "Zeyad"], ["Zeyad"]],
            [["Hi", "Hello"], ["Hey"]],
        )
        assert result == [["Hi: Asem", "Hello: Zeyad"], ["Hey: Zeyad"]]


class TestBatchInAxes:
    def test_broadcast_second_arg(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = af.call(batched_ir)(["Asem", "Zeyad", "Zeyad"], "Hi")
        assert result == ["Hi: Asem", "Hi: Zeyad", "Hi: Zeyad"]

    def test_broadcast_first_arg(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(False, True))
        result = af.call(batched_ir)("Asem", ["Hi", "Hello", "Hey"])
        assert result == ["Hi: Asem", "Hello: Asem", "Hey: Asem"]

    def test_default_all_batched(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    def test_explicit_all_batched(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(True, True))
        result = af.call(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    def test_all_broadcast(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(False, False))
        with pytest.raises(AssertionError):
            af.call(batched_ir)("Asem", "Hi")


class TestBatchUtils:
    def test_basic_axes_tree(self):
        from autoform.batch import infer_batch_size

        col_tree = (["a", "b"], ["x", "y"])
        in_axes = True
        batch_size = infer_batch_size(col_tree, in_axes)
        assert batch_size == 2

    def test_broadcast_axes_tree(self):
        from autoform.batch import infer_batch_size

        col_tree = (["a", "b"], "single")
        in_axes = (True, False)
        batch_size = infer_batch_size(col_tree, in_axes)
        assert batch_size == 2

    def test_no_batched_returns_zero(self):
        from autoform.batch import infer_batch_size

        col_tree = ("a", "b")
        in_axes = (False, False)
        batch_size = infer_batch_size(col_tree, in_axes)
        assert batch_size == 0


class TestBatchRuleOutBatched:
    def test_format_out_batched_is_scalar(self):
        batch_size = 2
        in_batched = (True,)
        in_values = (["a", "b"],)
        out_vals, out_batched = af.core.batch_rules.get(af.string.format_p)(
            (batch_size, in_batched, in_values), template="{}"
        )
        assert out_batched
        assert out_vals == ["a", "b"]

    def test_concat_out_batched_is_scalar(self):
        batch_size = 2
        in_batched = (True, True)
        in_values = (["a", "b"], ["x", "y"])
        out_vals, out_batched = af.core.batch_rules.get(af.string.concat_p)((
            batch_size,
            in_batched,
            in_values,
        ))
        assert out_batched
        assert out_vals == ["ax", "by"]


class TestBatchMultipleOutputs:
    def test_batch_primitive_with_two_outputs(self):
        split_p = af.core.Primitive("split")

        @ft.partial(af.core.eval_rules.set, split_p)
        def eval_split(x):
            return af.core.Var(str), af.core.Var(str)

        @ft.partial(af.core.impl_rules.set, split_p)
        def impl_split(x):
            return x[0], x[1:]

        @ft.partial(af.core.batch_rules.set, split_p)
        def batch_split(in_tree):
            batch_size, in_batched, in_values = in_tree
            results = [impl_split(in_values[b]) for b in range(batch_size)]
            out1 = [r[0] for r in results]
            out2 = [r[1] for r in results]
            return (out1, out2), (True, True)

        def program(x):
            return split_p.bind(x)

        ir = af.trace(program)("abc")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["abc", "xyz", "123"])
        assert result == (["a", "x", "1"], ["bc", "yz", "23"])

    def test_batch_nested_tuple_output(self):
        nested_p = af.core.Primitive("nested")

        @ft.partial(af.core.eval_rules.set, nested_p)
        def eval_nested(x):
            return (af.core.Var(str), af.core.Var(str)), af.core.Var(str)

        @ft.partial(af.core.impl_rules.set, nested_p)
        def impl_nested(x):
            return (x + "1", x + "2"), x + "3"

        @ft.partial(af.core.batch_rules.set, nested_p)
        def batch_nested(in_tree):
            batch_size, in_batched, in_values = in_tree
            results = [impl_nested(in_values[b]) for b in range(batch_size)]
            out1 = ([r[0][0] for r in results], [r[0][1] for r in results])
            out2 = [r[1] for r in results]
            return (out1, out2), ((True, True), True)

        def program(x):
            return nested_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == ((["a1", "b1"], ["a2", "b2"]), ["a3", "b3"])


class TestBatchBroadcasting:
    def test_concat_mixed_batched(self):
        batch_size = 3
        in_batched = (True, False)
        in_values = (["a", "b", "c"], "!")
        out_vals, out_batched = af.core.batch_rules.get(af.string.concat_p)((
            batch_size,
            in_batched,
            in_values,
        ))
        assert out_vals == ["a!", "b!", "c!"]
        assert out_batched

    def test_format_mixed_batched(self):
        batch_size = 2
        in_batched = (True, False)
        in_values = (["Alice", "Bob"], "Hello")
        out_vals, out_batched = af.core.batch_rules.get(af.string.format_p)(
            (batch_size, in_batched, in_values), template="{1}, {0}!"
        )
        assert out_vals == ["Hello, Alice!", "Hello, Bob!"]
        assert out_batched

    def test_all_unbatched(self):
        batch_size = 0
        in_batched = (False, False)
        in_values = ("a", "b")
        out_vals, out_batched = af.core.batch_rules.get(af.string.concat_p)((
            batch_size,
            in_batched,
            in_values,
        ))
        assert out_vals == []
        assert out_batched


class TestBatchRuleOutBatchedValidation:
    def test_single_output_accepts_scalar_bool(self):
        single_p = af.core.Primitive("single_out")

        @ft.partial(af.core.impl_rules.set, single_p)
        def impl(x):
            return x

        @ft.partial(af.core.eval_rules.set, single_p)
        def eval_rule(x):
            return af.core.Var(str)

        @ft.partial(af.core.batch_rules.set, single_p)
        def batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            return [x[i] for i in range(batch_size)], True

        def program(x):
            return single_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == ["a", "b"]

    def test_tuple_output_requires_tuple_out_batched(self):
        tuple_p = af.core.Primitive("tuple_out")

        @ft.partial(af.core.impl_rules.set, tuple_p)
        def impl(x):
            return (x, x)

        @ft.partial(af.core.eval_rules.set, tuple_p)
        def eval_rule(x):
            return (af.core.Var(str), af.core.Var(str))

        @ft.partial(af.core.batch_rules.set, tuple_p)
        def bad_batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return (vals, vals), True

        def program(x):
            return tuple_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        with pytest.raises(ValueError):
            af.call(batched_ir)(["a", "b"])

    def test_tuple_output_with_correct_out_batched(self):
        tuple_p = af.core.Primitive("tuple_out_correct")

        @ft.partial(af.core.impl_rules.set, tuple_p)
        def impl(x):
            return (x, x)

        @ft.partial(af.core.eval_rules.set, tuple_p)
        def eval_rule(x):
            return (af.core.Var(str), af.core.Var(str))

        @ft.partial(af.core.batch_rules.set, tuple_p)
        def correct_batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return (vals, vals), (True, True)

        def program(x):
            return tuple_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == (["a", "b"], ["a", "b"])

    def test_nested_output_requires_nested_out_batched(self):
        nested_p = af.core.Primitive("nested_out")

        @ft.partial(af.core.impl_rules.set, nested_p)
        def impl(x):
            return {"first": x, "second": (x, x)}

        @ft.partial(af.core.eval_rules.set, nested_p)
        def eval_rule(x):
            return {"first": af.core.Var(str), "second": (af.core.Var(str), af.core.Var(str))}

        @ft.partial(af.core.batch_rules.set, nested_p)
        def bad_batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return {"first": vals, "second": (vals, vals)}, True

        def program(x):
            return nested_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        with pytest.raises(ValueError):
            af.call(batched_ir)(["a", "b"])

    def test_nested_output_with_correct_out_batched(self):
        nested_p = af.core.Primitive("nested_out_correct")

        @ft.partial(af.core.impl_rules.set, nested_p)
        def impl(x):
            return {"first": x, "second": (x, x)}

        @ft.partial(af.core.eval_rules.set, nested_p)
        def eval_rule(x):
            return {"first": af.core.Var(str), "second": (af.core.Var(str), af.core.Var(str))}

        @ft.partial(af.core.batch_rules.set, nested_p)
        def correct_batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return {"first": vals, "second": (vals, vals)}, {"first": True, "second": (True, True)}

        def program(x):
            return nested_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == {"first": ["a", "b"], "second": (["a", "b"], ["a", "b"])}

    def test_mixed_batched_output(self):
        mixed_p = af.core.Primitive("mixed_batch")

        @ft.partial(af.core.impl_rules.set, mixed_p)
        def impl(x):
            return (x, "constant")

        @ft.partial(af.core.eval_rules.set, mixed_p)
        def eval_rule(x):
            return (af.core.Var(str), af.core.Var(str))

        @ft.partial(af.core.batch_rules.set, mixed_p)
        def batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return (vals, ["constant"] * batch_size), (True, True)

        def program(x):
            return mixed_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == (["a", "b"], ["constant", "constant"])


class TestTransposeBatch:
    def test_list_structure(self):
        from autoform.utils import transpose_batch

        results = [["a", "x"], ["b", "y"], ["c", "z"]]
        out_batched = [True, True]
        out = transpose_batch(3, out_batched, results)
        assert out == [["a", "b", "c"], ["x", "y", "z"]]

    def test_tuple_structure(self):
        from autoform.utils import transpose_batch

        results = [("a", "x"), ("b", "y")]
        out_batched = (True, True)
        out = transpose_batch(2, out_batched, results)
        assert out == (["a", "b"], ["x", "y"])

    def test_dict_structure(self):
        from autoform.utils import transpose_batch

        results = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out_batched = {"a": True, "b": True}
        out = transpose_batch(2, out_batched, results)
        assert out == {"a": [1, 3], "b": [2, 4]}

    def test_struct_structure(self):
        from autoform.utils import transpose_batch

        class Point(af.Struct):
            x: int
            y: int

        results = [Point(x=1, y=2), Point(x=3, y=4)]
        out_batched = Point(x=True, y=True)
        out = transpose_batch(2, out_batched, results)
        assert out.x == [1, 3]
        assert out.y == [2, 4]


class TestRebatch:
    def test_list_to_list(self):
        in_tree = (["a", "b", "c"],)
        in_batched = (True,)
        results = ["x", "y", "z"]
        out = rebatch(in_tree, in_batched, results)
        assert out == ["x", "y", "z"]

    def test_tuple_to_tuple(self):
        in_tree = (("a", "b", "c"),)
        in_batched = (True,)
        results = ["x", "y", "z"]
        out = rebatch(in_tree, in_batched, results)
        assert out == ("x", "y", "z")

    def test_mixed_batched_uses_first(self):
        in_tree = (("a", "b"), "broadcast")
        in_batched = (True, False)
        results = ["x", "y"]
        out = rebatch(in_tree, in_batched, results)
        assert out == ("x", "y")

    def test_no_batched_returns_list(self):
        in_tree = ("a", "b")
        in_batched = (False, False)
        results = ["x", "y"]
        out = rebatch(in_tree, in_batched, results)
        assert out == ["x", "y"]

    def test_nested_tuple(self):
        in_tree = ((("a", "b", "c"),),)
        in_batched = ((True,),)
        results = ["x", "y", "z"]
        out = rebatch(in_tree, in_batched, results)
        assert out == ("x", "y", "z")

    def test_struct_container(self):
        class Point(af.Struct):
            x: int
            y: int

        in_tree = ([Point(x=1, y=2), Point(x=3, y=4)],)
        in_batched = (True,)
        results = [Point(x=10, y=20), Point(x=30, y=40)]
        out = rebatch(in_tree, in_batched, results)
        assert out == [Point(x=10, y=20), Point(x=30, y=40)]

    def test_struct_with_tuple_field(self):
        class Batch(af.Struct):
            codes: tuple

        in_tree = (Batch(codes=("a", "b", "c")),)
        in_batched = (True,)
        results = [("x", "y", "z")]
        out = rebatch(in_tree, in_batched, results)
        assert out == Batch(codes=("x", "y", "z"))
