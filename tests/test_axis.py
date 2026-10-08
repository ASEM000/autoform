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

from collections import namedtuple

import pytest

import autoform as af
from autoform.axis import BatchAVal
from tests import BlobAVal, aexecute, angle_text, append_bang, bracket_text, execute


def greet(name, greeting):
    return af.string.format("{greeting}: {name}", greeting=greeting, name=name)


class TaggedAVal(af.core.AVal):
    def __init__(self, tag):
        self.tag = tag


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(BatchAVal(af.string.StrAVal()), id="aval"),
        pytest.param(af.core.Zero(BatchAVal(af.string.StrAVal())), id="zero"),
    ],
)
def test_check_batch_aval(value):
    BatchAVal(af.string.StrAVal()).check(value)
    with pytest.raises(TypeError, match="Expected FloatAVal"):
        BatchAVal(af.numeric.FloatAVal()).check(value)


class TestBatchBasic:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_valued_items(self, executor):
        x = af.stage.Var(aval=BatchAVal(af.string.StrAVal()))
        ir = af.batch(af.stage.IR([], (x,), x))
        values = [[], ["a", "b"]]
        assert executor(ir, values) == values
        with pytest.raises(TypeError, match="Expected BatchAVal"):
            executor(ir, ["a", "b"])

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
            pytest.param(af.core.primal_s, id="primal"),
            pytest.param(af.core.tangent_s, id="tangent"),
            pytest.param(af.core.cotangent_s, id="cotangent"),
        ],
    )
    def test_batch_aval_ad_space(self, space):
        aval = BatchAVal(af.string.StrAVal())
        assert space.map(aval) == BatchAVal(af.string.StrAVal())

    def test_mapped_wrapper_aval(self):
        aval = TaggedAVal("input")
        var = af.stage.Var(aval=aval)
        ir = af.batch(af.stage.IR([], (var,), (var,)), in_axes=True)
        for wrapped in (ir.in_tree[0].aval, ir.out_tree[0].aval):
            assert wrapped == BatchAVal(aval)

    def test_broadcast_wrapper_aval(self):
        aval = TaggedAVal("input")
        var = af.stage.Var(aval=aval)
        ir = af.batch(af.stage.IR([], (var,), (var,)), in_axes=False)
        for wrapped in (ir.in_tree[0].aval, ir.out_tree[0].aval):
            assert wrapped is aval

    def test_mapped_constant_output(self):
        ir = af.batch(af.trace(lambda x: "c")("x"), in_axes=True)
        assert isinstance(ir.out_tree, af.stage.Var)
        assert ir.out_tree.aval == BatchAVal(af.string.StrAVal())
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
    aval = af.string.StrAVal()
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
    prim = af.core.Prim("batch_output")
    af.extend.register_abstract(prim, lambda _: af.utils.tree.map(af.core.avalof, sample_output))
    af.extend.register_batch(prim, batch_rule)
    return af.batch(af.trace(prim.bind)(traced))


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


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "container",
    [
        pytest.param(list, id="list"),
        pytest.param(tuple, id="tuple"),
        pytest.param(lambda x: {"x": x[0], "y": x[1]}, id="dict"),
    ],
)
@pytest.mark.parametrize(
    "cotangents, shared",
    [
        pytest.param(["g1", "g2"], "g1g2", id="strings"),
        pytest.param([af.core.Zero(af.string.StrAVal()), "g2"], "g2", id="mixed-zero"),
        pytest.param(
            [af.core.Zero(af.string.StrAVal()), af.core.Zero(af.string.StrAVal())],
            af.core.Zero(af.string.StrAVal()),
            id="all-zero",
        ),
    ],
)
def test_pullback_of_batch_accumulates_shared_input(executor, container, cotangents, shared):
    ir = af.trace(af.string.concat)("x", "y")
    primals = ("x", container(["a", "b"]))
    feedback = container(cotangents)
    outputs = container(["xa", "xb"])
    pb = af.pullback(af.batch(ir, in_axes=(False, True)))
    actual = executor(pb, primals, feedback)
    assert actual == (outputs, (shared, feedback))
    assert af.core.avalof(actual[1][0]) == pb.out_tree[1][0].aval
    lane_pb = af.batch(af.pullback(ir), in_axes=((False, True), True))
    assert executor(lane_pb, primals, feedback) == (outputs, (feedback, feedback))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_of_batch_accumulates_shared_pytree(executor):
    def program(x, y):
        return x["scale"] * y + x["bias"]

    ir = af.trace(program)({"scale": 2.0, "bias": 1.0}, 3.0)
    pb = af.pullback(af.batch(ir, in_axes=(False, True)))
    actual = executor(pb, ({"scale": 2.0, "bias": 1.0}, [3.0, 4.0]), [1.0, 1.0])
    assert actual == ([7.0, 9.0], ({"scale": 7.0, "bias": 2.0}, [2.0, 2.0]))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_of_batch_without_mapped_inputs(executor):
    ir = af.trace(af.string.concat)("x", "y")
    pb = af.pullback(af.batch(ir, in_axes=False))
    assert executor(pb, ("a", "b"), "g") == ("ab", ("g", "g"))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_of_nested_batch_accumulates_shared_batch(executor):
    ir = af.trace(af.string.concat)("x", "y")
    inner = af.batch(ir, in_axes=(True, False))
    pb = af.pullback(af.batch(inner, in_axes=(False, True)))
    actual = executor(pb, (["a", "b"], ["c", "d"]), [["g1", "g2"], ["g3", "g4"]])
    assert actual == ([["ac", "bc"], ["ad", "bd"]], (["g1g3", "g2g4"], ["g1g2", "g3g4"]))


def test_batch_box_treats_axis_spec_as_prefix():
    batcher = af.axis.BatchInterpreter(batch_size=2, parent=af.core.active_interpreter.get())
    boxed = batcher.box((["a", "b"], True))
    assert isinstance(boxed, af.axis.BatchBox)
    assert boxed.value == ["a", "b"]
    assert boxed.batched is True
    assert af.core.avalof(boxed) == af.string.StrAVal()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(("x", "y"), id="tuple"),
        pytest.param({"x": "x", "y": "y"}, id="dict"),
        pytest.param(namedtuple("Pair", "x y")("x", "y"), id="namedtuple"),
    ],
)
def test_batch_box_avalof_container(values):
    batcher = af.axis.BatchInterpreter(batch_size=2, parent=af.core.active_interpreter.get())
    box = batcher.box((values, True))

    assert af.core.avalof(box) == af.string.StrAVal()
    assert box.value is values


def test_broadcast_box_avalof_zero_metadata():
    aval = BlobAVal(3)
    x = af.core.Zero(aval)
    box = af.axis.BatchBox(object(), x, False)

    assert af.core.avalof(box) is aval


def test_batch_box_avalof_zero_metadata():
    aval = BlobAVal(3)
    x = af.core.Zero(aval)
    box = af.axis.BatchBox(object(), [x, x], True)
    actual = af.core.avalof(box)

    assert actual is aval


def test_avalof_nested_batch_ad_and_trace_boxes():
    aval = TaggedAVal("input")
    x = af.stage.TraceBox(owner=af.stage.TraceInterpreter(), var=af.stage.Var(aval=aval))
    x = af.ad.PushforwardBox(object(), x, object())
    x = af.axis.BatchBox(object(), (x, x), True)
    x = af.axis.BatchBox(object(), {"x": x, "y": x}, True)
    x = af.ad.PullbackBwdBox(object(), x)

    actual = af.core.avalof(x)

    assert actual is aval


@pytest.mark.parametrize(
    "values, message",
    [
        pytest.param([], "empty batch", id="empty-list"),
        pytest.param((), "empty batch", id="empty-tuple"),
        pytest.param("x", "Expected a batch container", id="scalar"),
        pytest.param(["x", 1], "different avals", id="mixed-types"),
        pytest.param(
            [af.core.Zero(BlobAVal(3)), af.core.Zero(BlobAVal(4))],
            "different avals",
            id="mixed-metadata",
        ),
    ],
)
def test_batch_box_avalof_rejects_invalid_batch(values, message):
    box = af.axis.BatchBox(object(), values, True)

    with pytest.raises(TypeError, match=message):
        af.core.avalof(box)
