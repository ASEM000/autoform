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

import pytest

import autoform as af
from autoform.axis import BatchAVal
from tests import aexecute, angle_text, append_bang, bracket_text, execute


def greet(name, greeting):
    return af.string.format("{greeting}: {name}", greeting=greeting, name=name)


class TaggedAVal(af.core.AVal):
    def __init__(self, tag):
        self.tag = tag


class TestBatchBasic:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
    def test_single_arg(self, executor, container):
        ir = af.batch(af.trace(append_bang)("hello"))
        values = container(("hello", "world"))
        result = executor(ir, values)
        assert result == container(("hello!", "world!"))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "program, traced, args, expected",
        [
            pytest.param(
                af.string.concat,
                ("Hello", " World"),
                (["Hello", "Good"], [" World", " Day"]),
                ["Hello World", "Good Day"],
                id="two-inputs",
            ),
            pytest.param(
                lambda x: af.string.concat(af.string.format("[{x}]", x=x), "!"),
                ("a",),
                (["a", "b", "c"],),
                ["[a]!", "[b]!", "[c]!"],
                id="chain",
            ),
            pytest.param(
                lambda name, value: af.string.format(
                    "{name}: {inner}",
                    name=name,
                    inner=af.string.format("{value} units", value=value),
                ),
                ("temp", "25"),
                (["temp", "pressure"], ["25", "101"]),
                ["temp: 25 units", "pressure: 101 units"],
                id="nested-format",
            ),
        ],
    )
    def test_dataflow(self, executor, program, traced, args, expected):
        ir = af.batch(af.trace(program)(*traced))
        actual = executor(ir, *args)
        assert actual == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_static_input_literal_is_not_boxed(self, executor):

        def label(prefix, value):
            return af.string.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        batched_ir = af.batch(ir, in_axes=(False, True))
        actual = executor(batched_ir, "Q", ["a", "b"])
        assert actual == ["Qa", "Qb"]
        with pytest.raises(AssertionError, match="Static input mismatch"):
            executor(batched_ir, "R", ["a", "b"])


class TestBatchIRStructure:
    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(af.core.tangent_s, id="tangent"),
            pytest.param(af.core.cotangent_s, id="cotangent"),
        ],
    )
    def test_batch_aval_ad_space(self, space):
        aval = BatchAVal(af.core.StrAVal())
        assert space.avalof(aval) == BatchAVal(af.core.StrAVal())

    def test_mapped_wrapper_aval(self):
        aval = TaggedAVal("input")
        var = af.core.Var(aval=aval)
        ir = af.batch(af.core.IR([], (var,), (var,)), in_axes=True)
        for wrapped in (ir.in_tree[0].aval, ir.out_tree[0].aval):
            assert wrapped == BatchAVal(aval)

    def test_broadcast_wrapper_aval(self):
        aval = TaggedAVal("input")
        var = af.core.Var(aval=aval)
        ir = af.batch(af.core.IR([], (var,), (var,)), in_axes=False)
        for wrapped in (ir.in_tree[0].aval, ir.out_tree[0].aval):
            assert wrapped is aval

    def test_mapped_constant_output(self):
        ir = af.batch(af.trace(lambda x: "c")("x"), in_axes=True)
        assert isinstance(ir.out_tree, af.core.Var)
        assert ir.out_tree.aval == BatchAVal(af.core.StrAVal())
        assert ir.call(["a", "b"]) == ["c", "c"]

    def test_broadcast_constant_output(self):
        ir = af.batch(af.trace(lambda x: "c")("x"), in_axes=False)
        assert ir.out_tree == "c"
        assert ir.call("a") == "c"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "program, depth, values, expected",
    [
        pytest.param(
            append_bang,
            2,
            [["a", "b"], ["c", "d", "e"]],
            [["a!", "b!"], ["c!", "d!", "e!"]],
            id="double-ragged",
        ),
        pytest.param(
            append_bang,
            3,
            [[["a", "b"], ["c"]], [["d", "e", "f"]]],
            [[["a!", "b!"], ["c!"]], [["d!", "e!", "f!"]]],
            id="triple-ragged",
        ),
        pytest.param(bracket_text, 4, [[[["a"]]]], [[[["[a]"]]]], id="quadruple-single"),
    ],
)
def test_nested_batch(executor, program, depth, values, expected):
    ir = af.trace(program)("hello")
    aval = af.core.StrAVal()
    for _ in range(depth):
        ir = af.batch(ir)
        aval = BatchAVal(aval)
    assert ir.in_tree[0].aval == aval
    assert ir.out_tree.aval == aval
    actual = executor(ir, values)
    assert actual == expected


@pytest.mark.parametrize(
    "program, depth, values",
    [
        pytest.param(append_bang, 1, [], id="empty"),
        pytest.param(append_bang, 3, [[[], []], []], id="deepest-empty"),
        pytest.param(angle_text, 2, [["a", "b"], [], ["c"]], id="mixed-empty"),
    ],
)
def test_empty_nested_batch(program, depth, values):
    ir = af.trace(program)("x")
    for _ in range(depth):
        ir = af.batch(ir)
    with pytest.raises(AssertionError):
        ir.call(values)


@pytest.mark.parametrize(
    "program, traced, args, expected",
    [
        pytest.param(
            lambda name, greeting: af.string.format(
                "{greeting}: {name}",
                greeting=greeting,
                name=name,
            ),
            ("x0", "Hi"),
            ([["x0", "x1"], ["x1"]], [["Hi", "Hello"], ["Hey"]]),
            [["Hi: x0", "Hello: x1"], ["Hey: x1"]],
            id="format",
        ),
        pytest.param(
            af.string.concat,
            ("a", "b"),
            ([["a1", "a2"], ["a3"]], [["b1", "b2"], ["b3"]]),
            [["a1b1", "a2b2"], ["a3b3"]],
            id="concat",
        ),
    ],
)
def test_nested_batch_two_inputs(program, traced, args, expected):
    ir = af.batch(af.batch(af.trace(program)(*traced)))
    assert ir.call(*args) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "in_axes, args, expected",
    [
        pytest.param(True, (["x0", "x1"], ["Hi", "Hello"]), ["Hi: x0", "Hello: x1"], id="default"),
        pytest.param(
            (True, True),
            (["x0", "x1"], ["Hi", "Hello"]),
            ["Hi: x0", "Hello: x1"],
            id="both-mapped",
        ),
        pytest.param(
            (True, False),
            (["x0", "x1", "x1"], "Hi"),
            ["Hi: x0", "Hi: x1", "Hi: x1"],
            id="broadcast-second",
        ),
        pytest.param(
            (False, True),
            ("x0", ["Hi", "Hello", "Hey"]),
            ["Hi: x0", "Hello: x0", "Hey: x0"],
            id="broadcast-first",
        ),
        pytest.param((False, False), ("x0", "Hi"), "Hi: x0", id="both-broadcast"),
    ],
)
def test_batch_axes(executor, in_axes, args, expected):
    ir = af.batch(af.trace(greet)("x0", "Hi"), in_axes=in_axes)
    actual = executor(ir, *args)
    assert actual == expected


def test_numeric_program_with_broadcast_input():
    ir = af.trace(lambda x, scale: x * x + x * scale)(1.0, 1.0)
    assert af.batch(ir, in_axes=(True, False)).call([1.0, 2.0, 3.0], 2.0) == [3.0, 8.0, 15.0]


def batch_primitive(sample_output, batch_rule, traced="a"):
    primitive = af.core.Prim("batch_output")
    af.core.abstract_rules.set(
        primitive,
        lambda _: af.utils.tree.map(af.core.primal_s.avalof, sample_output),
    )
    af.core.batch_rules.set(primitive, batch_rule)
    return af.batch(af.trace(primitive.bind)(traced))


@pytest.mark.parametrize(
    "sample, traced, inputs, output, flags",
    [
        pytest.param(
            ("a", "bc"),
            "abc",
            ["abc", "xyz", "123"],
            (["a", "x", "1"], ["bc", "yz", "23"]),
            (True, True),
            id="split",
        ),
        pytest.param(
            (("a1", "a2"), "a3"),
            "a",
            ["a", "b"],
            ((["a1", "b1"], ["a2", "b2"]), ["a3", "b3"]),
            ((True, True), True),
            id="nested-tuple",
        ),
    ],
)
def test_multiple_outputs(sample, traced, inputs, output, flags):
    ir = batch_primitive(sample, lambda _: (output, flags), traced)
    assert ir.call(inputs) == output


@pytest.mark.parametrize(
    "shape, flags, expected",
    [
        pytest.param(lambda x: x, True, ["a", "b"], id="scalar"),
        pytest.param(lambda x: (x, x), (True, True), (["a", "b"], ["a", "b"]), id="tuple"),
        pytest.param(
            lambda x: {"first": x, "second": (x, x)},
            {"first": True, "second": (True, True)},
            {"first": ["a", "b"], "second": (["a", "b"], ["a", "b"])},
            id="nested",
        ),
    ],
)
def test_batch_rule_output_shape(shape, flags, expected):
    ir = batch_primitive(shape("a"), lambda inputs: (shape(inputs[2]), flags))
    assert ir.call(["a", "b"]) == expected


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(lambda x: (x, x), id="tuple"),
        pytest.param(lambda x: {"first": x, "second": (x, x)}, id="nested"),
    ],
)
def test_batch_rule_rejects_mismatched_output_flags(shape):
    ir = batch_primitive(shape("a"), lambda inputs: (shape(inputs[2]), True))
    with pytest.raises(ValueError, match="out_batched must match the structure"):
        ir.call(["a", "b"])


@pytest.mark.parametrize(
    "mapped_constant, constant",
    [
        pytest.param(True, ["constant", "constant"], id="mapped"),
        pytest.param(False, "constant", id="broadcast"),
    ],
)
def test_batch_rule_constant_output(mapped_constant, constant):
    ir = batch_primitive(
        ("a", "constant"),
        lambda inputs: ((inputs[2], constant), (True, mapped_constant)),
    )
    assert ir.call(["a", "b"]) == (["a", "b"], ["constant", "constant"])


class TestBatchWithMixedAxes:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_two_outputs_mixed_batching(self, executor):

        def program(x, y):
            return (af.string.format("x={x}", x=x), af.string.format("y={y}", y=y))

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = executor(batched_ir, ["a", "b", "c"], "constant")
        assert result == (["x=a", "x=b", "x=c"], ["y=constant", "y=constant", "y=constant"])

    def test_chained_with_broadcast(self):
        def program(x, prefix):
            prefixed = af.string.concat(prefix, x)
            return af.string.format("[{prefixed}]", prefixed=prefixed)

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["a", "b", "c"], ">>")
        assert result == ["[>>a]", "[>>b]", "[>>c]"]

    def test_multiple_uses_of_broadcast_input(self):
        def program(x, sep):
            return af.string.concat(af.string.concat(x, sep), x)

        ir = af.trace(program)("...", "...")
        batched_ir = af.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["a", "b"], "-")
        assert result == ["a-a", "b-b"]


def test_batch_box_treats_axis_spec_as_prefix():
    batcher = af.axis.BatchInterpreter(batch_size=2, parent=af.core.active_interpreter.get())
    boxed = batcher.box((["a", "b"], True))
    assert isinstance(boxed, af.axis.BatchBox)
    assert boxed.value == ["a", "b"]
    assert boxed.batched is True
