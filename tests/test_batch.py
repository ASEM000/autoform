import pytest
import functools as ft
import autoform as af


class TestBatchBasic:
    def test_single_arg(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.build_ir(shout)("hello")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["hello", "world"])
        assert result == ["hello!", "world!"]

    def test_two_args(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    def test_concat(self):
        def join(a, b):
            return af.concat(a, b)

        ir = af.build_ir(join)("Hello", " World")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["Hello", "Good"], [" World", " Day"])
        assert result == ["Hello World", "Good Day"]

    def test_chained(self):
        def process(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        ir = af.build_ir(process)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b", "c"])
        assert result == ["[a]!", "[b]!", "[c]!"]

    def test_nested_format(self):
        def template(name, value):
            inner = af.format("{} units", value)
            return af.format("{}: {}", name, inner)

        ir = af.build_ir(template)("temp", "25")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["temp", "pressure"], ["25", "101"])
        assert result == ["temp: 25 units", "pressure: 101 units"]

    def test_empty_batch(self):
        def f(x):
            return af.format("{}!", x)

        ir = af.build_ir(f)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)([])
        assert result == []


class TestBatchIRStructure:
    def test_creates_single_eqn(self):
        def f(x):
            return af.concat(x, x)

        ir = af.build_ir(f)("hello")
        batched_ir = af.batch(ir)
        assert len(batched_ir.ireqns) == 1
        assert batched_ir.ireqns[0].prim.name == "batch_call"

    def test_has_in_axes_param(self):
        def f(x):
            return af.concat(x, x)

        ir = af.build_ir(f)("hello")
        batched_ir = af.batch(ir, in_axes=list)
        assert "in_axes" in batched_ir.ireqns[0].params

    def test_has_sub_ir_param(self):
        def f(x):
            return af.concat(x, x)

        ir = af.build_ir(f)("hello")
        batched_ir = af.batch(ir)
        assert "ir" in batched_ir.ireqns[0].params


class TestNestedBatch:
    def test_batch_of_batch(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.build_ir(shout)("hello")
        batched_ir = af.batch(ir)
        double_batched_ir = af.batch(batched_ir)
        result = af.call(double_batched_ir)([["a", "b"], ["c", "d", "e"]])
        assert result == [["a!", "b!"], ["c!", "d!", "e!"]]

    def test_batch_of_batch_two_args(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
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

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(list, None))
        result = af.call(batched_ir)(["Asem", "Zeyad", "Zeyad"], "Hi")
        assert result == ["Hi: Asem", "Hi: Zeyad", "Hi: Zeyad"]

    def test_broadcast_first_arg(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(None, list))
        result = af.call(batched_ir)("Asem", ["Hi", "Hello", "Hey"])
        assert result == ["Hi: Asem", "Hello: Asem", "Hey: Asem"]

    def test_default_all_batched(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    def test_explicit_all_batched(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(list, list))
        result = af.call(batched_ir)(["Asem", "Zeyad"], ["Hi", "Hello"])
        assert result == ["Hi: Asem", "Hello: Zeyad"]

    def test_all_broadcast(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(None, None))
        result = af.call(batched_ir)("Asem", "Hi")
        assert result == []


class TestBatchAsync:
    @pytest.mark.asyncio
    async def test_async_batch_basic(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.build_ir(shout)("hello")
        batched_ir = af.batch(ir)
        result = await af.acall(batched_ir)(["a", "b", "c"])
        assert result == ["a!", "b!", "c!"]

    @pytest.mark.asyncio
    async def test_async_batch_broadcast(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.build_ir(greet)("Asem", "Hi")
        batched_ir = af.batch(ir, in_axes=(list, None))
        result = await af.acall(batched_ir)(["A", "B"], "Hi")
        assert result == ["Hi: A", "Hi: B"]


class TestBatchUtils:
    def test_basic_axes_tree(self):
        from autoform.batch import infer_batch_size

        col_tree = (["a", "b"], ["x", "y"])
        in_axes = list
        batch_size = infer_batch_size(col_tree, in_axes)
        assert batch_size == 2

    def test_broadcast_axes_tree(self):
        from autoform.batch import infer_batch_size

        col_tree = (["a", "b"], "single")
        in_axes = (list, None)
        batch_size = infer_batch_size(col_tree, in_axes)
        assert batch_size == 2

    def test_no_batched_returns_zero(self):
        from autoform.batch import infer_batch_size

        col_tree = ("a", "b")
        in_axes = (None, None)
        batch_size = infer_batch_size(col_tree, in_axes)
        assert batch_size == 0

    def test_mixed_axes_tree(self):
        from autoform.batch import broadcast_in_axes_prefix

        in_axes = (list, None)
        tree = (["a", "b", "c"], {"x": 1, "y": 2})
        broadcasted_in_axes = (list, {"x": None, "y": None})
        assert broadcast_in_axes_prefix(in_axes, tree) == broadcasted_in_axes


class TestBatchRuleOutBatched:
    def test_format_out_batched_is_scalar(self):
        batch_size = 2
        in_batched = (True,)
        in_tree = (["a", "b"],)
        out_vals, out_batched = af.core.batch_rules[af.string.format_p](
            batch_size, in_batched, in_tree, template="{}"
        )
        assert out_batched
        assert out_vals == ["a", "b"]

    def test_concat_out_batched_is_scalar(self):
        batch_size = 2
        in_batched = (True, True)
        in_tree = (["a", "b"], ["x", "y"])
        out_vals, out_batched = af.core.batch_rules[af.string.concat_p](
            batch_size, in_batched, in_tree
        )
        assert out_batched
        assert out_vals == ["ax", "by"]


class TestBatchMultipleOutputs:
    def test_batch_primitive_with_two_outputs(self):
        split_p = af.core.Primitive("split")

        @ft.partial(af.core.eval_rules.def_rule, split_p)
        def eval_split(x):
            return af.core.Var(), af.core.Var()

        @ft.partial(af.core.impl_rules.def_rule, split_p)
        def impl_split(x):
            return x[0], x[1:]

        @ft.partial(af.core.batch_rules.def_rule, split_p)
        def batch_split(batch_size, in_batched, in_tree):
            results = [impl_split(in_tree[b]) for b in range(batch_size)]
            out1 = [r[0] for r in results]
            out2 = [r[1] for r in results]
            return (out1, out2), (True, True)

        def program(x):
            return split_p.bind(x)

        ir = af.build_ir(program)("abc")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["abc", "xyz", "123"])
        assert result == (["a", "x", "1"], ["bc", "yz", "23"])

    def test_batch_nested_tuple_output(self):
        nested_p = af.core.Primitive("nested")

        @ft.partial(af.core.eval_rules.def_rule, nested_p)
        def eval_nested(x):
            return (af.core.Var(), af.core.Var()), af.core.Var()

        @ft.partial(af.core.impl_rules.def_rule, nested_p)
        def impl_nested(x):
            return (x + "1", x + "2"), x + "3"

        @ft.partial(af.core.batch_rules.def_rule, nested_p)
        def batch_nested(batch_size, in_batched, in_tree):
            results = [impl_nested(in_tree[b]) for b in range(batch_size)]
            out1 = ([r[0][0] for r in results], [r[0][1] for r in results])
            out2 = [r[1] for r in results]
            return (out1, out2), ((True, True), True)

        def program(x):
            return nested_p.bind(x)

        ir = af.build_ir(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == ((["a1", "b1"], ["a2", "b2"]), ["a3", "b3"])


class TestBatchBroadcasting:
    def test_concat_mixed_batched(self):
        batch_size = 3
        in_batched = (True, False)
        in_tree = (["a", "b", "c"], "!")
        out_vals, out_batched = af.core.batch_rules[af.string.concat_p](
            batch_size, in_batched, in_tree
        )
        assert out_vals == ["a!", "b!", "c!"]
        assert out_batched

    def test_format_mixed_batched(self):
        batch_size = 2
        in_batched = (True, False)
        in_tree = (["Alice", "Bob"], "Hello")
        out_vals, out_batched = af.core.batch_rules[af.string.format_p](
            batch_size, in_batched, in_tree, template="{1}, {0}!"
        )
        assert out_vals == ["Hello, Alice!", "Hello, Bob!"]
        assert out_batched

    def test_all_unbatched(self):
        batch_size = 0
        in_batched = (False, False)
        in_tree = ("a", "b")
        out_vals, out_batched = af.core.batch_rules[af.string.concat_p](
            batch_size, in_batched, in_tree
        )
        assert out_vals == []
        assert out_batched


class TestBatchRuleOutBatchedValidation:
    def test_single_output_accepts_scalar_bool(self):
        single_p = af.core.Primitive("single_out")

        @ft.partial(af.core.impl_rules.def_rule, single_p)
        def impl(x):
            return x

        @ft.partial(af.core.eval_rules.def_rule, single_p)
        def eval_rule(x):
            return af.core.Var()

        @ft.partial(af.core.batch_rules.def_rule, single_p)
        def batch_rule(batch_size, in_batched, x):
            return [x[i] for i in range(batch_size)], True

        def program(x):
            return single_p.bind(x)

        ir = af.build_ir(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == ["a", "b"]

    def test_tuple_output_requires_tuple_out_batched(self):
        tuple_p = af.core.Primitive("tuple_out")

        @ft.partial(af.core.impl_rules.def_rule, tuple_p)
        def impl(x):
            return (x, x)

        @ft.partial(af.core.eval_rules.def_rule, tuple_p)
        def eval_rule(x):
            return (af.core.Var(), af.core.Var())

        @ft.partial(af.core.batch_rules.def_rule, tuple_p)
        def bad_batch_rule(batch_size, in_batched, x):
            vals = [x[i] for i in range(batch_size)]
            return (vals, vals), True

        def program(x):
            return tuple_p.bind(x)

        ir = af.build_ir(program)("a")
        batched_ir = af.batch(ir)
        with pytest.raises(ValueError, match="out_batched.*structure"):
            af.call(batched_ir)(["a", "b"])

    def test_tuple_output_with_correct_out_batched(self):
        tuple_p = af.core.Primitive("tuple_out_correct")

        @ft.partial(af.core.impl_rules.def_rule, tuple_p)
        def impl(x):
            return (x, x)

        @ft.partial(af.core.eval_rules.def_rule, tuple_p)
        def eval_rule(x):
            return (af.core.Var(), af.core.Var())

        @ft.partial(af.core.batch_rules.def_rule, tuple_p)
        def correct_batch_rule(batch_size, in_batched, x):
            vals = [x[i] for i in range(batch_size)]
            return (vals, vals), (True, True)

        def program(x):
            return tuple_p.bind(x)

        ir = af.build_ir(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == (["a", "b"], ["a", "b"])

    def test_nested_output_requires_nested_out_batched(self):
        nested_p = af.core.Primitive("nested_out")

        @ft.partial(af.core.impl_rules.def_rule, nested_p)
        def impl(x):
            return {"first": x, "second": (x, x)}

        @ft.partial(af.core.eval_rules.def_rule, nested_p)
        def eval_rule(x):
            return {"first": af.core.Var(), "second": (af.core.Var(), af.core.Var())}

        @ft.partial(af.core.batch_rules.def_rule, nested_p)
        def bad_batch_rule(batch_size, in_batched, x):
            vals = [x[i] for i in range(batch_size)]
            return {"first": vals, "second": (vals, vals)}, True

        def program(x):
            return nested_p.bind(x)

        ir = af.build_ir(program)("a")
        batched_ir = af.batch(ir)
        with pytest.raises(ValueError, match="out_batched.*structure"):
            af.call(batched_ir)(["a", "b"])

    def test_nested_output_with_correct_out_batched(self):
        nested_p = af.core.Primitive("nested_out_correct")

        @ft.partial(af.core.impl_rules.def_rule, nested_p)
        def impl(x):
            return {"first": x, "second": (x, x)}

        @ft.partial(af.core.eval_rules.def_rule, nested_p)
        def eval_rule(x):
            return {"first": af.core.Var(), "second": (af.core.Var(), af.core.Var())}

        @ft.partial(af.core.batch_rules.def_rule, nested_p)
        def correct_batch_rule(batch_size, in_batched, x):
            vals = [x[i] for i in range(batch_size)]
            return {"first": vals, "second": (vals, vals)}, {"first": True, "second": (True, True)}

        def program(x):
            return nested_p.bind(x)

        ir = af.build_ir(program)("a")
        batched_ir = af.batch(ir)
        result = af.call(batched_ir)(["a", "b"])
        assert result == {"first": ["a", "b"], "second": (["a", "b"], ["a", "b"])}

    def test_mixed_batched_output(self):
        mixed_p = af.core.Primitive("mixed_batch")

        @ft.partial(af.core.impl_rules.def_rule, mixed_p)
        def impl(x):
            return (x, "constant")

        @ft.partial(af.core.eval_rules.def_rule, mixed_p)
        def eval_rule(x):
            return (af.core.Var(), af.core.Var())

        @ft.partial(af.core.batch_rules.def_rule, mixed_p)
        def batch_rule(batch_size, in_batched, x):
            vals = [x[i] for i in range(batch_size)]
            return (vals, ["constant"] * batch_size), (True, True)

        def program(x):
            return mixed_p.bind(x)

        ir = af.build_ir(program)("a")
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
