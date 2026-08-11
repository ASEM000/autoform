# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft

import optree
import pytest

import autoform as af
from autoform.analysis import toposort_levels
from autoform.axes import BatchAVal, BatchBox, BatchInterpreter, fanout_p
from autoform.utils import batch_spec, batch_transpose

tree = optree.pytree.reexport(namespace=af.PYTREE_NAMESPACE)


class TestBatchBasic:
    def test_single_arg(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["hello", "world"])
        assert result == ["hello!", "world!"]

    def test_single_arg_tuple_batch_container(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        result = batched_ir.call(("hello", "world"))
        assert result == ("hello!", "world!")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_single_arg_async(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        result = await batched_ir.acall(["hello", "world"])
        assert result == ["hello!", "world!"]

    def test_two_args(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["x0", "x1"], ["Hi", "Hello"])
        assert result == ["Hi: x0", "Hello: x1"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_two_args_async(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir)
        result = await batched_ir.acall(["x0", "x1"], ["Hi", "Hello"])
        assert result == ["Hi: x0", "Hello: x1"]

    def test_concat(self):
        def join(a, b):
            return af.concat(a, b)

        ir = af.trace(join)("Hello", " World")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["Hello", "Good"], [" World", " Day"])
        assert result == ["Hello World", "Good Day"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_concat_async(self):
        def join(a, b):
            return af.concat(a, b)

        ir = af.trace(join)("Hello", " World")
        batched_ir = af.batch(ir)
        result = await batched_ir.acall(["Hello", "Good"], [" World", " Day"])
        assert result == ["Hello World", "Good Day"]

    def test_chained(self):
        def process(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        ir = af.trace(process)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b", "c"])
        assert result == ["[a]!", "[b]!", "[c]!"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_chained_async(self):
        def process(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        ir = af.trace(process)("a")
        batched_ir = af.batch(ir)
        result = await batched_ir.acall(["a", "b", "c"])
        assert result == ["[a]!", "[b]!", "[c]!"]

    def test_nested_format(self):
        def template(name, value):
            inner = af.format("{} units", value)
            return af.format("{}: {}", name, inner)

        ir = af.trace(template)("temp", "25")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["temp", "pressure"], ["25", "101"])
        assert result == ["temp: 25 units", "pressure: 101 units"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_nested_format_async(self):
        def template(name, value):
            inner = af.format("{} units", value)
            return af.format("{}: {}", name, inner)

        ir = af.trace(template)("temp", "25")
        batched_ir = af.batch(ir)
        result = await batched_ir.acall(["temp", "pressure"], ["25", "101"])
        assert result == ["temp: 25 units", "pressure: 101 units"]

    def test_empty_batch(self):
        def f(x):
            return af.format("{}!", x)

        ir = af.trace(f)("a")
        batched_ir = af.batch(ir)
        with pytest.raises(AssertionError):
            batched_ir.call([])

    def test_static_input_literal_is_not_boxed(self):
        def label(prefix, value):
            return af.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        batched_ir = af.batch(ir, in_axes=(False, True))

        assert batched_ir.call("Q", ["a", "b"]) == ["Qa", "Qb"]
        with pytest.raises(AssertionError, match="Static input mismatch"):
            batched_ir.call("R", ["a", "b"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_static_input_literal_is_not_boxed_async(self):
        def label(prefix, value):
            return af.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        batched_ir = af.batch(ir, in_axes=(False, True))

        assert await batched_ir.acall("Q", ["a", "b"]) == ["Qa", "Qb"]
        with pytest.raises(AssertionError, match="Static input mismatch"):
            await batched_ir.acall("R", ["a", "b"])


class TestBatchIRStructure:
    def test_creates_single_eqn(self):
        def f(x):
            return af.concat(x, x)

        ir = af.trace(f)("hello")
        batched_ir = af.batch(ir)
        assert len(batched_ir.eqns) == 1
        assert batched_ir.eqns[0].prim.name == "batch_call"

    def test_has_in_axes_param(self):
        def f(x):
            return af.concat(x, x)

        ir = af.trace(f)("hello")
        batched_ir = af.batch(ir, in_axes=True)
        assert "in_axes" in batched_ir.eqns[0].params

    def test_has_sub_ir_param(self):
        def f(x):
            return af.concat(x, x)

        ir = af.trace(f)("hello")
        batched_ir = af.batch(ir)
        assert "ir" in batched_ir.eqns[0].params

    def test_batch_wrapper_rewrites_batched_aval(self):
        class TaggedAVal(af.core.AVal):
            __slots__ = ["tag"]

            def __init__(self, tag):
                self.tag = tag

        aval = TaggedAVal("input")
        var = af.core.Var(aval=aval)
        ir = af.core.IR([], (var,), (var,))

        batched_ir = af.batch(ir)

        assert batched_ir.in_tree[0].aval == BatchAVal(aval)
        assert batched_ir.out_tree[0].aval == BatchAVal(aval)

    def test_batch_wrapper_preserves_broadcast_aval(self):
        class TaggedAVal(af.core.AVal):
            __slots__ = ["tag"]

            def __init__(self, tag):
                self.tag = tag

        aval = TaggedAVal("input")
        var = af.core.Var(aval=aval)
        ir = af.core.IR([], (var,), (var,))

        batched_ir = af.batch(ir, in_axes=False)

        assert batched_ir.in_tree[0].aval is aval
        assert batched_ir.out_tree[0].aval is aval

    def test_batch_constant_output_uses_mapped_wrapper_var(self):
        def f(x):
            return "c"

        ir = af.trace(f)("x")
        batched_ir = af.batch(ir)

        assert isinstance(batched_ir.out_tree, af.core.Var)
        assert batched_ir.out_tree.aval == BatchAVal(af.core.StrAVal())
        assert batched_ir.call(["a", "b"]) == ["c", "c"]

    def test_batch_broadcast_constant_output_stays_literal(self):
        def f(x):
            return "c"

        ir = af.trace(f)("x")
        batched_ir = af.batch(ir, in_axes=False)

        assert batched_ir.out_tree == "c"
        assert batched_ir.call("a") == "c"


class TestNestedBatch:
    def test_batch_of_batch(self):
        def shout(text):
            return af.format("{}!", text)

        ir = af.trace(shout)("hello")
        batched_ir = af.batch(ir)
        double_batched_ir = af.batch(batched_ir)
        assert double_batched_ir.in_tree[0].aval == BatchAVal(BatchAVal(af.core.StrAVal()))
        assert double_batched_ir.out_tree.aval == BatchAVal(BatchAVal(af.core.StrAVal()))
        result = double_batched_ir.call([["a", "b"], ["c", "d", "e"]])
        assert result == [["a!", "b!"], ["c!", "d!", "e!"]]

    def test_batch_of_batch_two_args(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir)
        double_batched_ir = af.batch(batched_ir)
        result = double_batched_ir.call(
            [["x0", "x1"], ["x1"]],
            [["Hi", "Hello"], ["Hey"]],
        )
        assert result == [["Hi: x0", "Hello: x1"], ["Hey: x1"]]


class TestBatchInAxes:
    def test_broadcast_second_arg(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["x0", "x1", "x1"], "Hi")
        assert result == ["Hi: x0", "Hi: x1", "Hi: x1"]

    def test_broadcast_first_arg(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir, in_axes=(False, True))
        result = batched_ir.call("x0", ["Hi", "Hello", "Hey"])
        assert result == ["Hi: x0", "Hello: x0", "Hey: x0"]

    def test_default_all_batched(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["x0", "x1"], ["Hi", "Hello"])
        assert result == ["Hi: x0", "Hello: x1"]

    def test_explicit_all_batched(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir, in_axes=(True, True))
        result = batched_ir.call(["x0", "x1"], ["Hi", "Hello"])
        assert result == ["Hi: x0", "Hello: x1"]

    def test_all_broadcast(self):
        def greet(name, greeting):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x0", "Hi")
        batched_ir = af.batch(ir, in_axes=(False, False))
        result = batched_ir.call("x0", "Hi")
        assert result == "Hi: x0"


class TestBatchUtils:
    def test_batch_box_treats_axis_spec_as_prefix(self):
        batcher = BatchInterpreter(batch_size=2, parent=af.core.active_interpreter.get())

        boxed = batcher.box((["a", "b"], True))

        assert isinstance(boxed, BatchBox)
        assert boxed.value == ["a", "b"]
        assert boxed.batched is True

    def test_basic_axes_tree(self):
        col_tree = (["a", "b"], ["x", "y"])
        in_axes = True
        batch_size = batch_spec(col_tree, in_axes).num_children
        assert batch_size == 2

    def test_broadcast_axes_tree(self):
        col_tree = (["a", "b"], "single")
        in_axes = (True, False)
        batch_size = batch_spec(col_tree, in_axes).num_children
        assert batch_size == 2

    def test_no_batched_returns_none(self):
        col_tree = ("a", "b")
        in_axes = (False, False)
        spec = batch_spec(col_tree, in_axes)
        assert spec is None

    def test_tree_equal_same_structure_and_leaves(self):
        assert af.utils.tree_equal(("a", ["b", "c"]), ("a", ["b", "c"]))

    def test_tree_equal_different_structure(self):
        assert not af.utils.tree_equal(("a", ["b", "c"]), ("a", ("b", "c")))

    def test_tree_equal_different_leaves(self):
        assert not af.utils.tree_equal(("a", ["b", "c"]), ("a", ["b", "d"]))


class TestBatchRuleOutBatched:
    def test_format_out_batched_is_scalar(self):
        batch_size = 2
        in_batched = ((True,), ())
        in_values = ((["a", "b"],), ())
        out_vals, out_batched = af.core.batch_rules.get(af.string.format_p)(
            (batch_size, in_batched, in_values), template="{}", keys=()
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
        split_p = af.core.Prim("split")

        @ft.partial(af.core.abstract_rules.set, split_p)
        def abstract_split(x):
            return af.core.StrAVal(), af.core.StrAVal()

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
        result = batched_ir.call(["abc", "xyz", "123"])
        assert result == (["a", "x", "1"], ["bc", "yz", "23"])

    def test_batch_nested_tuple_output(self):
        nested_p = af.core.Prim("nested")

        @ft.partial(af.core.abstract_rules.set, nested_p)
        def abstract_nested(x):
            return (af.core.StrAVal(), af.core.StrAVal()), af.core.StrAVal()

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
        result = batched_ir.call(["a", "b"])
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
        in_batched = ((True, False), ())
        in_values = ((["x0", "x1"], "Hello"), ())
        out_vals, out_batched = af.core.batch_rules.get(af.string.format_p)(
            (batch_size, in_batched, in_values), template="{1}, {0}!", keys=()
        )
        assert out_vals == ["Hello, x0!", "Hello, x1!"]
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
        assert out_vals == "ab"
        assert out_batched is False


class TestBatchRuleOutBatchedValidation:
    def test_single_output_accepts_scalar_bool(self):
        single_p = af.core.Prim("single_out")

        @ft.partial(af.core.impl_rules.set, single_p)
        def impl(x):
            return x

        @ft.partial(af.core.abstract_rules.set, single_p)
        def abstract_rule(x):
            return af.core.StrAVal()

        @ft.partial(af.core.batch_rules.set, single_p)
        def batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            return [x[i] for i in range(batch_size)], True

        def program(x):
            return single_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b"])
        assert result == ["a", "b"]

    def test_tuple_output_requires_tuple_out_batched(self):
        tuple_p = af.core.Prim("tuple_out")

        @ft.partial(af.core.impl_rules.set, tuple_p)
        def impl(x):
            return (x, x)

        @ft.partial(af.core.abstract_rules.set, tuple_p)
        def abstract_rule(x):
            return (af.core.StrAVal(), af.core.StrAVal())

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
            batched_ir.call(["a", "b"])

    def test_tuple_output_with_correct_out_batched(self):
        tuple_p = af.core.Prim("tuple_out_correct")

        @ft.partial(af.core.impl_rules.set, tuple_p)
        def impl(x):
            return (x, x)

        @ft.partial(af.core.abstract_rules.set, tuple_p)
        def abstract_rule(x):
            return (af.core.StrAVal(), af.core.StrAVal())

        @ft.partial(af.core.batch_rules.set, tuple_p)
        def correct_batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return (vals, vals), (True, True)

        def program(x):
            return tuple_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b"])
        assert result == (["a", "b"], ["a", "b"])

    def test_nested_output_requires_nested_out_batched(self):
        nested_p = af.core.Prim("nested_out")

        @ft.partial(af.core.impl_rules.set, nested_p)
        def impl(x):
            return {"first": x, "second": (x, x)}

        @ft.partial(af.core.abstract_rules.set, nested_p)
        def abstract_rule(x):
            return {
                "first": af.core.StrAVal(),
                "second": (af.core.StrAVal(), af.core.StrAVal()),
            }

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
            batched_ir.call(["a", "b"])

    def test_nested_output_with_correct_out_batched(self):
        nested_p = af.core.Prim("nested_out_correct")

        @ft.partial(af.core.impl_rules.set, nested_p)
        def impl(x):
            return {"first": x, "second": (x, x)}

        @ft.partial(af.core.abstract_rules.set, nested_p)
        def abstract_rule(x):
            return {
                "first": af.core.StrAVal(),
                "second": (af.core.StrAVal(), af.core.StrAVal()),
            }

        @ft.partial(af.core.batch_rules.set, nested_p)
        def correct_batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return {"first": vals, "second": (vals, vals)}, {"first": True, "second": (True, True)}

        def program(x):
            return nested_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b"])
        assert result == {"first": ["a", "b"], "second": (["a", "b"], ["a", "b"])}

    def test_mixed_batched_output(self):
        mixed_p = af.core.Prim("mixed_batch")

        @ft.partial(af.core.impl_rules.set, mixed_p)
        def impl(x):
            return (x, "constant")

        @ft.partial(af.core.abstract_rules.set, mixed_p)
        def abstract_rule(x):
            return (af.core.StrAVal(), af.core.StrAVal())

        @ft.partial(af.core.batch_rules.set, mixed_p)
        def batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return (vals, ["constant"] * batch_size), (True, True)

        def program(x):
            return mixed_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b"])
        assert result == (["a", "b"], ["constant", "constant"])

    def test_hop_broadcasts_scalar_output(self):
        mixed_p = af.core.Prim("mixed_batch_boundary")

        @ft.partial(af.core.impl_rules.set, mixed_p)
        def impl(x):
            return (x, "constant")

        @ft.partial(af.core.abstract_rules.set, mixed_p)
        def abstract_rule(x):
            return (af.core.StrAVal(), af.core.StrAVal())

        @ft.partial(af.core.batch_rules.set, mixed_p)
        def batch_rule(in_tree):
            batch_size, in_batched, x = in_tree
            vals = [x[i] for i in range(batch_size)]
            return (vals, "constant"), (True, False)

        def program(x):
            return mixed_p.bind(x)

        ir = af.trace(program)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b"])
        assert result == (["a", "b"], ["constant", "constant"])


class TestTransposeBatch:
    def test_list_structure(self):
        results = [["a", "x"], ["b", "y"], ["c", "z"]]
        out_batched = [True, True]
        out = batch_transpose(3, out_batched, results)
        assert out == [["a", "b", "c"], ["x", "y", "z"]]

    def test_tuple_structure(self):
        results = [("a", "x"), ("b", "y")]
        out_batched = (True, True)
        out = batch_transpose(2, out_batched, results)
        assert out == (["a", "b"], ["x", "y"])

    def test_dict_structure(self):
        results = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out_batched = {"a": True, "b": True}
        out = batch_transpose(2, out_batched, results)
        assert out == {"a": [1, 3], "b": [2, 4]}

    def test_struct_structure(self):
        @tree.dataclasses.dataclass
        class Point:
            x: int
            y: int

        results = [Point(x=1, y=2), Point(x=3, y=4)]
        out_batched = Point(x=True, y=True)
        out = batch_transpose(2, out_batched, results)
        assert out.x == [1, 3]
        assert out.y == [2, 4]


class TestBatchSpec:
    def test_list_to_list(self):
        in_tree = (["a", "b", "c"],)
        in_batched = (True,)
        results = ["x", "y", "z"]
        out = batch_spec(in_tree, in_batched).unflatten(results)
        assert out == ["x", "y", "z"]

    def test_tuple_to_tuple(self):
        in_tree = (("a", "b", "c"),)
        in_batched = (True,)
        results = ["x", "y", "z"]
        out = batch_spec(in_tree, in_batched).unflatten(results)
        assert out == ("x", "y", "z")

    def test_mixed_batched_uses_first(self):
        in_tree = (("a", "b"), "broadcast")
        in_batched = (True, False)
        results = ["x", "y"]
        out = batch_spec(in_tree, in_batched).unflatten(results)
        assert out == ("x", "y")

    def test_no_batched_returns_none(self):
        in_tree = ("a", "b")
        in_batched = (False, False)
        spec = batch_spec(in_tree, in_batched)
        assert spec is None

    def test_nested_tuple(self):
        in_tree = ((("a", "b", "c"),),)
        in_batched = ((True,),)
        results = ["x", "y", "z"]
        out = batch_spec(in_tree, in_batched).unflatten(results)
        assert out == ("x", "y", "z")

    def test_struct_container(self):
        @tree.dataclasses.dataclass
        class Point:
            x: int
            y: int

        in_tree = ([Point(x=1, y=2), Point(x=3, y=4)],)
        in_batched = (True,)
        results = [Point(x=10, y=20), Point(x=30, y=40)]
        out = batch_spec(in_tree, in_batched).unflatten(results)
        assert out == [Point(x=10, y=20), Point(x=30, y=40)]


class TestBatchRuleAllUnbatched:
    def test_format_all_unbatched(self):
        batch_size = 3
        in_batched = ((False, False), ())
        in_values = (("hello", "world"), ())
        out_vals, out_batched = af.core.batch_rules.get(af.string.format_p)(
            (batch_size, in_batched, in_values), template="{} {}", keys=()
        )
        assert out_vals == "hello world"
        assert out_batched is False

    def test_concat_all_unbatched(self):
        batch_size = 3
        in_batched = (False, False)
        in_values = ("hello", "world")
        out_vals, out_batched = af.core.batch_rules.get(af.string.concat_p)((
            batch_size,
            in_batched,
            in_values,
        ))
        assert out_vals == "helloworld"
        assert out_batched is False

    def test_match_all_unbatched(self):
        batch_size = 3
        in_batched = (False, False)
        in_values = ("hello", "hello")
        out_vals, out_batched = af.core.batch_rules.get(af.string.match_p)((
            batch_size,
            in_batched,
            in_values,
        ))
        assert out_vals is True
        assert out_batched is False

    def test_match_all_unbatched_not_equal(self):
        batch_size = 3
        in_batched = (False, False)
        in_values = ("hello", "world")
        out_vals, out_batched = af.core.batch_rules.get(af.string.match_p)((
            batch_size,
            in_batched,
            in_values,
        ))
        assert out_vals is False
        assert out_batched is False


class TestBatchWithMixedAxes:
    def test_two_outputs_mixed_batching(self):
        def program(x, y):
            return af.format("x={}", x), af.format("y={}", y)

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["a", "b", "c"], "constant")
        assert result == (["x=a", "x=b", "x=c"], ["y=constant", "y=constant", "y=constant"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_two_outputs_mixed_batching_async(self):
        def program(x, y):
            return af.format("x={}", x), af.format("y={}", y)

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = await batched_ir.acall(["a", "b", "c"], "constant")
        assert result == (["x=a", "x=b", "x=c"], ["y=constant", "y=constant", "y=constant"])

    def test_chained_with_broadcast(self):
        def program(x, prefix):
            prefixed = af.concat(prefix, x)
            return af.format("[{}]", prefixed)

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["a", "b", "c"], ">>")
        assert result == ["[>>a]", "[>>b]", "[>>c]"]

    def test_multiple_uses_of_broadcast_input(self):
        def program(x, sep):
            return af.concat(af.concat(x, sep), x)

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["a", "b"], "-")
        assert result == ["a-a", "b-b"]


def scheduled_parallel_formats():
    def program(x):
        a = af.format("[{}]", x)
        b = af.format("<{}>", x)
        return a, b

    return af.sched(af.trace(program)("a"))


class TestInternalFanoutViaSched:
    def test_two_independent_ops(self):
        scheduled = scheduled_parallel_formats()

        prim_names = [e.prim.name for e in scheduled.eqns]
        assert prim_names == ["fanout"]
        assert scheduled.call("A") == ("[A]", "<A>")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_two_independent_ops_async(self):
        scheduled = scheduled_parallel_formats()

        result = await scheduled.acall("A")
        assert result == ("[A]", "<A>")

    def test_exception_propagates(self):
        error_p = af.core.Prim("error")

        @ft.partial(af.core.abstract_rules.set, error_p)
        def abstract_error(x):
            return af.core.StrAVal()

        @ft.partial(af.core.impl_rules.set, error_p)
        def impl_error(x):
            raise ValueError("intentional error")

        def program(x):
            ok = af.format("[{}]", x)
            err = error_p.bind(x)
            return ok, err

        scheduled = af.sched(af.trace(program)("a"))

        with pytest.raises(ValueError, match="intentional error"):
            scheduled.call("A")


class TestFanoutWithTransforms:
    def test_pushforward(self):
        pf_ir = af.pushforward(scheduled_parallel_formats())
        (p_out, t_out) = pf_ir.call(("primal",), ("tangent",))
        assert p_out == ("[primal]", "<primal>")
        assert t_out == ("[tangent]", "<tangent>")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pushforward_async(self):
        pf_ir = af.pushforward(scheduled_parallel_formats())
        (p_out, t_out) = await pf_ir.acall(("primal",), ("tangent",))
        assert p_out == ("[primal]", "<primal>")
        assert t_out == ("[tangent]", "<tangent>")

    def test_pullback(self):
        pb_ir = af.pullback(scheduled_parallel_formats())
        out, cotangent = pb_ir.call(("primal",), ("grad1", "grad2"))
        assert out == ("[primal]", "<primal>")
        assert cotangent == ("grad1grad2",)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_async(self):
        pb_ir = af.pullback(scheduled_parallel_formats())
        out, cotangent = await pb_ir.acall(("primal",), ("grad1", "grad2"))
        assert out == ("[primal]", "<primal>")
        assert cotangent == ("grad1grad2",)


class TestFanoutWithBatch:
    def test_batch_fanout(self):
        batched_ir = af.batch(scheduled_parallel_formats())
        result = batched_ir.call(["A", "B", "C"])
        assert result == (["[A]", "[B]", "[C]"], ["<A>", "<B>", "<C>"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_fanout_async(self):
        batched_ir = af.batch(scheduled_parallel_formats())
        result = await batched_ir.acall(["A", "B", "C"])
        assert result == (["[A]", "[B]", "[C]"], ["<A>", "<B>", "<C>"])

    def test_batch_fanout_mixed_axes(self):
        def program(x, y):
            a = af.format("[{}]", x)
            b = af.format("<{}>", y)
            return a, b

        scheduled = af.sched(af.trace(program)("x", "y"))
        batched_ir = af.batch(scheduled, in_axes=(True, False))
        result = batched_ir.call(["A", "B", "C"], "STATIC")
        assert result == (["[A]", "[B]", "[C]"], ["<STATIC>", "<STATIC>", "<STATIC>"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_fanout_mixed_axes_async(self):
        def program(x, y):
            a = af.format("[{}]", x)
            b = af.format("<{}>", y)
            return a, b

        scheduled = af.sched(af.trace(program)("x", "y"))
        batched_ir = af.batch(scheduled, in_axes=(True, False))
        result = await batched_ir.acall(["X", "Y"], "STATIC")
        assert result == (["[X]", "[Y]"], ["<STATIC>", "<STATIC>"])


class TestFanoutWithDCE:
    def test_fanout_kept_when_used(self):
        dce_ir = af.dce(scheduled_parallel_formats())

        assert len(dce_ir.eqns) == 1
        assert dce_ir.eqns[0].prim.name == "fanout"

    def test_fanout_removed_when_unused(self):
        def program(x):
            _ = af.format("[{}]", x)
            _ = af.format("<{}>", x)
            return af.concat(x, "!")

        ir = af.trace(program)("a")
        scheduled = af.sched(ir, cond=lambda e: e.prim.name == "format")
        dce_ir = af.dce(scheduled)

        assert all(eqn.prim.name != "fanout" for eqn in dce_ir.eqns)

    def test_fanout_dce_propagates_to_branches(self):
        def program(x):
            live = af.format("[{}]", x)
            dead = af.format("<{}>", x)
            return live

        scheduled = af.sched(af.trace(program)("a"))
        dce_ir = af.dce(scheduled, out_used=True)

        fanout_eqn = dce_ir.eqns[0]
        dce_branch = fanout_eqn.params["irs"][1]

        assert len(dce_branch.eqns) == 0

    def test_fanout_partial_output_used(self):
        def program(x):
            a = af.format("[{}]", x)
            _ = af.format("<{}>", x)
            return a

        scheduled = af.sched(af.trace(program)("a"))
        dce_ir = af.dce(scheduled, out_used=True)
        fanout_eqns = [e for e in dce_ir.eqns if e.prim.name == "fanout"]
        assert len(fanout_eqns) == 1
        inner_irs = fanout_eqns[0].params["irs"]
        assert len(inner_irs[0].eqns) == 1
        assert len(inner_irs[1].eqns) == 0

    def test_fanout_unused_branch_structured_output_is_callable(self):
        def structured_with_dead(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            return (a, b)

        ir_live = af.trace(lambda x: af.concat(x, "!"))("x")
        ir_dead = af.trace(structured_with_dead)("x")

        def program(x):
            result = af.switch("live", {"live": ir_live}, x)
            _ = af.switch("dead", {"dead": ir_dead}, x)
            return result

        prog_ir = af.trace(program)("x")
        scheduled = af.sched(prog_ir)
        dce_ir = af.dce(scheduled)

        assert dce_ir.call("X") == "X!"

        fanout_eqn = [e for e in dce_ir.eqns if e.prim.name == "fanout"][0]
        inner_dead = fanout_eqn.params["irs"][1]
        assert len(inner_dead.eqns) == 0
        leaves = af.utils.tree.leaves(inner_dead.out_tree)
        assert all(x is None for x in leaves)


class TestFanoutContextPreservation:
    def test_preserves_collect(self):
        def program(a, b):
            x = af.checkpoint(a, key="val", collection="debug")
            y = af.checkpoint(b, key="val", collection="debug")
            return x, y

        scheduled = af.sched(af.trace(program)("a", "b"))

        with af.collect(collection="debug") as collected:
            results = scheduled.call("A", "B")

        assert results == ("A", "B")
        assert "val" in collected
        assert set(collected["val"]) == {"A", "B"}

    def test_preserves_inject(self):
        def program(a, b):
            x = af.checkpoint(af.format("[{}]", a), key="val", collection="cache")
            y = af.checkpoint(af.format("<{}>", b), key="val", collection="cache")
            return x, y

        scheduled = af.sched(af.trace(program)("a", "b"))

        with af.inject(collection="cache", values={"val": ["CACHED1", "CACHED2"]}):
            results = scheduled.call("A", "B")

        assert results == ("CACHED1", "CACHED2")

    def test_nested_fanout(self):
        def inner(x):
            return af.checkpoint(x, key="inner", collection="debug")

        branches = {
            "a": af.sched(af.trace(lambda x: (inner(x), inner(x)))("x")),
            "b": af.sched(af.trace(lambda x: (inner(x), inner(x)))("x")),
        }

        def outer(key, x):
            left, right = af.switch(key, branches, x)
            return af.concat(left, right)

        outer_ir = af.sched(af.trace(outer)("a", "x"))

        with af.collect(collection="debug") as collected:
            results = outer_ir.call("a", "A")

        assert results == "AA"
        assert "inner" in collected
        assert len(collected["inner"]) == 2


class TestSched:
    def test_parallel_equations_fused(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            c = af.concat(a, b)
            return c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        prim_names = [e.prim.name for e in scheduled.eqns]
        assert prim_names == ["fanout", "concat"]

        result = scheduled.call("test")
        assert result == "[test]<test>"

    def test_single_equation_not_wrapped(self):
        def program(x):
            return af.format("[{}]", x)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        prim_names = [e.prim.name for e in scheduled.eqns]
        assert prim_names == ["format"]

    def test_with_cond_filter(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.concat(x, "!")
            return a, b

        ir = af.trace(program)("x")

        scheduled = af.sched(ir, cond=lambda e: e.prim.name == "format")

        prim_names = {e.prim.name for e in scheduled.eqns}
        assert prim_names == {"format", "concat"}

    def test_checkpoints_can_be_parallelized(self):
        def program(a, b):
            x = af.checkpoint(af.format("{}", a), key="x")
            y = af.checkpoint(af.format("{}", b), key="y")
            return x, y

        ir = af.trace(program)("a", "b")
        scheduled = af.sched(ir)

        fanout_count = sum(1 for e in scheduled.eqns if e.prim.name == "fanout")
        assert fanout_count == 2

    def test_checkpoint_ordering_via_depends(self):
        def program(a, b):
            x = af.checkpoint(af.format("{}", a), key="x")
            y = af.checkpoint(af.format("{}", b), key="y")
            return af.depends(y, x)

        ir = af.trace(program)("a", "b")
        scheduled = af.sched(ir)

        result = scheduled.call("hello", "world")
        assert result == "world"

    def test_mixed_pure_and_checkpoints(self):
        def program(a, b, c):
            x = af.format("[{}]", a)
            y = af.format("<{}>", b)
            z = af.checkpoint(af.format("{{{}}}", c), key="z")
            return x, y, z

        ir = af.trace(program)("a", "b", "c")
        scheduled = af.sched(ir)

        fanout_count = sum(1 for e in scheduled.eqns if e.prim.name == "fanout")
        assert fanout_count == 1

        result = scheduled.call("a", "b", "c")
        assert result == ("[a]", "<b>", "{c}")


class TestSchedRecursive:
    def test_sched_switch_branches(self):
        branches = {
            "a": af.trace(lambda x: af.concat(af.format("[{}]", x), af.format("<{}>", x)))("x"),
            "b": af.trace(lambda x: af.format("({})", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")
        scheduled = af.sched(ir)

        switch_eqn = scheduled.eqns[0]
        branch_a = switch_eqn.params["branches"]["a"]
        assert any(e.prim.name == "fanout" for e in branch_a.eqns)

        assert scheduled.call("a", "hello") == "[hello]<hello>"
        assert scheduled.call("b", "hello") == "(hello)"

    def test_sched_nested_switch(self):
        inner_branches = {
            "x": af.trace(lambda a: af.concat(af.format("[{}]", a), af.format("<{}>", a)))("a"),
            "y": af.trace(lambda a: af.format("({})", a))("a"),
        }

        def inner_program(key, inp):
            return af.switch(key, inner_branches, inp)

        inner_ir = af.trace(inner_program)("x", "inp")

        outer_branches = {
            "A": inner_ir,
            "B": af.trace(lambda key, inp: af.format("{} {}", key, inp))("k", "i"),
        }

        def outer_program(outer_key, inner_key, x):
            return af.switch(outer_key, outer_branches, inner_key, x)

        ir = af.trace(outer_program)("A", "x", "test")
        scheduled = af.sched(ir)

        outer_switch = scheduled.eqns[0]
        inner_switch = outer_switch.params["branches"]["A"].eqns[0]
        inner_branch_x = inner_switch.params["branches"]["x"]
        assert any(e.prim.name == "fanout" for e in inner_branch_x.eqns)

        assert scheduled.call("A", "x", "hello") == "[hello]<hello>"
        assert scheduled.call("A", "y", "hello") == "(hello)"
        assert scheduled.call("B", "ignored", "world") == "ignored world"

    def test_sched_fanout_nested_irs(self):
        branches1 = {
            "a": af.trace(lambda x: af.concat(af.format("[{}]", x), af.format("<{}>", x)))("x")
        }
        branches2 = {
            "a": af.trace(lambda x: af.concat(af.format("({})", x), af.format("{{{}}}", x)))("x")
        }

        def program(key, a, b):
            left = af.switch(key, branches1, a)
            right = af.switch(key, branches2, b)
            return left, right

        ir = af.trace(program)("a", "hello", "world")
        scheduled = af.sched(ir)

        outer_fanout = scheduled.eqns[0]
        for inner_ir in outer_fanout.params["irs"]:
            switch_eqn = inner_ir.eqns[0]
            branch = switch_eqn.params["branches"]["a"]
            assert any(e.prim.name == "fanout" for e in branch.eqns)

        result = scheduled.call("a", "hello", "world")
        assert result == ("[hello]<hello>", "(world){world}")

    def test_sched_with_cond_propagates_to_nested(self):
        branches = {
            "a": af.trace(lambda x: (af.format("[{}]", x), af.concat(x, "!")))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")

        scheduled = af.sched(ir, cond=lambda e: e.prim.name == "format")

        switch_eqn = scheduled.eqns[0]
        branch_a = switch_eqn.params["branches"]["a"]
        assert not any(e.prim.name == "fanout" for e in branch_a.eqns)

        assert scheduled.call("a", "test") == ("[test]", "test!")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sched_recursive_async(self):
        branches = {
            "a": af.trace(lambda x: af.concat(af.format("[{}]", x), af.format("<{}>", x)))("x"),
            "b": af.trace(lambda x: af.format("({})", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")
        scheduled = af.sched(ir)

        result = await scheduled.acall("a", "hello")
        assert result == "[hello]<hello>"

    def test_sched_preserves_non_ir_params(self):
        branches = {
            "a": af.trace(lambda x: af.format("[{}]", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")
        scheduled = af.sched(ir)

        switch_eqn = scheduled.eqns[0]
        assert "branches" in switch_eqn.params
        assert "a" in switch_eqn.params["branches"]

        assert scheduled.call("a", "test") == "[test]"


class TestSchedComposition:
    def test_sched_then_pushforward(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        pf_ir = af.pushforward(scheduled)

        primals, tangents = pf_ir.call(("test",), ("tangent",))
        assert primals == "[test]<test>"
        assert tangents == "[tangent]<tangent>"

    def test_sched_then_pullback(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        pb_ir = af.pullback(scheduled)

        out, cotangent = pb_ir.call(("test",), "grad")
        assert out == "[test]<test>"
        assert isinstance(cotangent[0], str)

    def test_sched_then_batch(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        batched_ir = af.batch(scheduled)

        result = batched_ir.call(["A", "B", "C"])
        assert result == ["[A]<A>", "[B]<B>", "[C]<C>"]

    def test_sched_then_dce(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            c = af.concat(a, b)
            _ = af.format("dead: {}", c)
            return c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        dce_ir = af.dce(scheduled)

        result = dce_ir.call("test")
        assert result == "[test]<test>"

    def test_dce_then_sched(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            _ = af.format("dead", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        dce_ir = af.dce(ir)
        scheduled = af.sched(dce_ir)

        result = scheduled.call("test")
        assert result == "[test]<test>"


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncSched:
    async def test_basic_async_execution(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await scheduled.acall("test")
        assert result == "[test]<test>"

    async def test_parallel_independent_ops(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            c = af.format("{{{}}}", x)
            return a, b, c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await scheduled.acall("test")
        assert result == ("[test]", "<test>", "{test}")

    async def test_sequential_dependent_ops(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.concat(a, "!")
            return b

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await scheduled.acall("test")
        assert result == "[test]!"

    async def test_mixed_parallel_and_sequential(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)

            c = af.concat(a, b)
            return c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await scheduled.acall("test")
        assert result == "[test]<test>"


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncAcall:
    async def test_basic_acall(self):
        ir = af.trace(lambda x: af.format("[{}]", x))("a")
        result = await ir.acall("hello")
        assert result == "[hello]"

    async def test_acall_with_switch(self):
        branches = {
            "a": af.trace(lambda x: af.format("[{}]", x))("x"),
            "b": af.trace(lambda x: af.format("<{}>", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")

        result_a = await ir.acall("a", "test")
        assert result_a == "[test]"

        result_b = await ir.acall("b", "test")
        assert result_b == "<test>"


class TestDepends:
    def test_basic(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        result = ir.call("hello")
        assert result == "B: hello"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_basic_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        result = await ir.acall("hello")
        assert result == "B: hello"

    def test_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        result = ir.call("hello")
        assert result == "C: hello"

    def test_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            b_ordered = af.depends(b, a)
            c_ordered = af.depends(c, b_ordered)
            return c_ordered

        ir = af.trace(program)("x")
        result = ir.call("hello")
        assert result == "C: hello"

    def test_no_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            return af.depends(a)

        ir = af.trace(program)("x")
        result = ir.call("hello")
        assert result == "A: hello"

    def test_ir_structure(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        depends_eqns = [e for e in ir.eqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

        in_leaves = af.utils.tree.leaves(depends_eqns[0].in_tree)
        assert len(in_leaves) >= 2


class TestDependsWithDCE:
    def test_kept_when_used(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        dce_ir = af.dce(ir)

        depends_eqns = [e for e in dce_ir.eqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

    def test_removed_when_unused(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            _ = af.depends(b, a)
            return af.format("C: {}", x)

        ir = af.trace(program)("x")
        dce_ir = af.dce(ir)

        depends_eqns = [e for e in dce_ir.eqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 0


class TestDependsWithPushforward:
    def test_pushforward(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = pf_ir.call(("primal",), ("tangent",))
        assert primal == "B: primal"
        assert tangent == "B: tangent"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pushforward_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = await pf_ir.acall(("primal",), ("tangent",))
        assert primal == "B: primal"
        assert tangent == "B: tangent"

    def test_pushforward_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = pf_ir.call(("primal",), ("tangent",))
        assert primal == "C: primal"
        assert tangent == "C: tangent"

    def test_pushforward_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.depends(af.format("B: {}", x), a)
            c = af.depends(af.format("C: {}", x), b)
            return c

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = pf_ir.call(("primal",), ("tangent",))
        assert primal == "C: primal"
        assert tangent == "C: tangent"


class TestDependsWithPullback:
    def test_pullback(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = pb_ir.call(("primal",), "grad")
        assert out == "B: primal"
        assert cotangent == ("grad",)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = await pb_ir.acall(("primal",), "grad")
        assert out == "B: primal"
        assert cotangent == ("grad",)

    def test_pullback_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = pb_ir.call(("primal",), "grad")
        assert out == "C: primal"
        assert cotangent == ("grad",)

    def test_pullback_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.depends(af.format("B: {}", x), a)
            c = af.depends(af.format("C: {}", x), b)
            return c

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = pb_ir.call(("primal",), "grad")
        assert out == "C: primal"
        assert cotangent == ("grad",)


class TestDependsWithBatch:
    def test_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = batched_ir.call(["x", "y", "z"])
        assert result == ["B: x", "B: y", "B: z"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = await batched_ir.acall(["x", "y", "z"])
        assert result == ["B: x", "B: y", "B: z"]

    def test_batch_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = batched_ir.call(["x", "y"])
        assert result == ["C: x", "C: y"]

    def test_batch_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.depends(af.format("B: {}", x), a)
            c = af.depends(af.format("C: {}", x), b)
            return c

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = batched_ir.call(["x", "y"])
        assert result == ["C: x", "C: y"]


class TestDependsWithSched:
    def test_sched_basic(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = scheduled.call("hello")
        assert result == "B: hello"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sched_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await scheduled.acall("hello")
        assert result == "B: hello"

    def test_sched_preserves_depends(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            b_ordered = af.depends(b, a)
            return af.concat(a, b_ordered)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        depends_eqns = [e for e in scheduled.eqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

        result = scheduled.call("hello")
        assert result == "A: helloB: hello"

    def test_sched_data_dependency(self):
        def program(x):
            a = af.format("A: {}", x)
            a_barrier = af.depends(a)
            b = af.format("B: {}", a_barrier)
            return b

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = scheduled.call("hello")
        assert result == "B: A: hello"


class TestDependsNestedTransforms:
    def test_batch_of_pushforward(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        batch_pf_ir = af.batch(pf_ir, in_axes=(True, True))

        primals, tangents = batch_pf_ir.call((["a", "b"],), (["da", "db"],))
        assert primals == ["B: a", "B: b"]
        assert tangents == ["B: da", "B: db"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_of_pushforward_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        batch_pf_ir = af.batch(pf_ir, in_axes=(True, True))

        primals, tangents = await batch_pf_ir.acall((["a", "b"],), (["da", "db"],))
        assert primals == ["B: a", "B: b"]
        assert tangents == ["B: da", "B: db"]

    def test_batch_of_pullback(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        batch_pb_ir = af.batch(pb_ir, in_axes=(True, True))

        outs, cotangents = batch_pb_ir.call((["a", "b"],), ["g1", "g2"])
        assert outs == ["B: a", "B: b"]
        assert cotangents == (["g1", "g2"],)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_of_pullback_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        batch_pb_ir = af.batch(pb_ir, in_axes=(True, True))

        outs, cotangents = await batch_pb_ir.acall((["a", "b"],), ["g1", "g2"])
        assert outs == ["B: a", "B: b"]
        assert cotangents == (["g1", "g2"],)

    def test_pushforward_of_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        pf_batched_ir = af.pushforward(batched_ir)

        primals, tangents = pf_batched_ir.call((["a", "b"],), (["da", "db"],))
        assert primals == ["B: a", "B: b"]
        assert tangents == ["B: da", "B: db"]

    def test_pullback_of_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        pb_batched_ir = af.pullback(batched_ir)

        outs, cotangents = pb_batched_ir.call((["a", "b"],), ["g1", "g2"])
        assert outs == ["B: a", "B: b"]
        assert cotangents == (["g1", "g2"],)

    def test_sched_of_pushforward(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        sched_pf_ir = af.sched(pf_ir)

        primal, tangent = sched_pf_ir.call(("primal",), ("tangent",))
        assert primal == "B: primal"
        assert tangent == "B: tangent"

    def test_sched_of_pullback(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        sched_pb_ir = af.sched(pb_ir)

        out, cotangent = sched_pb_ir.call(("primal",), "grad")
        assert out == "B: primal"
        assert cotangent == ("grad",)

    def test_sched_of_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        sched_batched_ir = af.sched(batched_ir)

        result = sched_batched_ir.call(["x", "y", "z"])
        assert result == ["B: x", "B: y", "B: z"]


class TestToposortLevels:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = af.trace(program)("input")
        levels = toposort_levels(ir)

        assert levels == []

    def test_single_equation(self):
        def program(x):
            return af.format("{}", x)

        ir = af.trace(program)("input")
        levels = toposort_levels(ir)

        assert len(levels) == 1
        assert len(levels[0]) == 1

    def test_independent_equations(self):
        def program(a, b):
            x = af.format("hello {}", a)
            y = af.format("world {}", b)
            return x, y

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_dependent_equations(self):
        def program(a, b):
            x = af.format("hello {}", a)
            y = af.format("world {}", b)
            z = af.concat(x, y)
            return z

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        assert len(levels) == 2
        assert len(levels[0]) == 2
        assert len(levels[1]) == 1

    def test_chain_of_equations(self):
        def program(x):
            a = af.format("{}", x)
            b = af.concat(a, "!")
            c = af.concat(b, "?")
            return c

        ir = af.trace(program)("input")
        levels = toposort_levels(ir)

        assert len(levels) == 3
        assert len(levels[0]) == 1
        assert len(levels[1]) == 1
        assert len(levels[2]) == 1


class TestToposortLevelsWithCheckpoints:
    def test_checkpoint_equations_can_parallelize(self):
        def program(a, b):
            x = af.checkpoint(af.format("hello {}", a), key="x")
            y = af.checkpoint(af.format("world {}", b), key="y")
            return x, y

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        checkpoint_eqns = [e for lvl in levels for e in lvl if e.prim.name == "checkpoint"]
        assert len(checkpoint_eqns) == 2

        checkpoint_levels = []
        for i, lvl in enumerate(levels):
            for e in lvl:
                if e.prim.name == "checkpoint":
                    checkpoint_levels.append(i)

        assert checkpoint_levels[0] == checkpoint_levels[1]

    def test_checkpoint_ordering_via_depends(self):
        def program(a, b):
            x = af.checkpoint(af.format("hello {}", a), key="x")
            y = af.checkpoint(af.format("world {}", b), key="y")
            return af.depends(y, x)

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        checkpoint_levels = []
        for i, lvl in enumerate(levels):
            for e in lvl:
                if e.prim.name == "checkpoint":
                    checkpoint_levels.append(i)

        assert checkpoint_levels[0] == checkpoint_levels[1]

        depends_level = None
        for i, lvl in enumerate(levels):
            for e in lvl:
                if e.prim.name == "depends":
                    depends_level = i

        assert depends_level > checkpoint_levels[0]

    def test_pure_equations_parallelize_around_checkpoints(self):
        def program(a, b, c):
            x = af.format("{}", a)
            y = af.checkpoint(af.format("{}", b), key="cp")
            z = af.format("{}", c)
            return x, y, z

        ir = af.trace(program)("a", "b", "c")
        levels = toposort_levels(ir)

        has_parallel = any(len(lvl) > 1 for lvl in levels)

        assert len(levels) == 2
        assert has_parallel


class TestFanoutBatchAllUnbatched:
    def test_fanout_batch_all_unbatched(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")
        irs = [ir1, ir2]

        batch_size = 3
        in_batched = [False, False]
        in_values = [("hello",), ("world",)]

        out_vals, out_batched = af.core.batch_rules.get(fanout_p)(
            (batch_size, in_batched, in_values), irs=irs
        )
        assert out_vals == ["[hello]", "<world>"]
        assert out_batched == [False, False]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_fanout_batch_all_unbatched_async(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")
        irs = [ir1, ir2]

        batch_size = 3
        in_batched = [False, False]
        in_values = [("hello",), ("world",)]

        out_vals, out_batched = await af.core.batch_rules.aget(fanout_p)(
            (batch_size, in_batched, in_values), irs=irs
        )
        assert out_vals == ["[hello]", "<world>"]
        assert out_batched == [False, False]

    def test_fanout_batch_single_ir_unbatched(self):
        ir = af.trace(lambda x: af.format("[{}]", x))("a")
        irs = [ir]

        batch_size = 3
        in_batched = [False]
        in_values = [("hello",)]

        out_vals, out_batched = af.core.batch_rules.get(fanout_p)(
            (batch_size, in_batched, in_values), irs=irs
        )
        assert out_vals == ["[hello]"]
        assert out_batched == [False]

    def test_fanout_integration_mixed_batched(self):
        def program(x, y):
            a = af.format("[{}]", x)
            b = af.format("<{}>", y)
            return a, b

        scheduled = af.sched(af.trace(program)("a", "b"))
        batched_ir = af.batch(scheduled, in_axes=(True, False))
        result = batched_ir.call(["a", "b"], "constant")
        assert result == (["[a]", "[b]"], ["<constant>", "<constant>"])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_fanout_integration_mixed_batched_async(self):
        def program(x, y):
            a = af.format("[{}]", x)
            b = af.format("<{}>", y)
            return a, b

        scheduled = af.sched(af.trace(program)("a", "b"))
        batched_ir = af.batch(scheduled, in_axes=(True, False))
        result = await batched_ir.acall(["a", "b"], "constant")
        assert result == (["[a]", "[b]"], ["<constant>", "<constant>"])
