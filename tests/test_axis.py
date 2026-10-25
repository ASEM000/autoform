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

import asyncio
from collections import namedtuple

import pytest

import autoform.extend as extend
import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.check as check
import autoform.src.core as core
import autoform.src.numeric as numeric
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.tree as tree
from tests import AValBlob, aexecute, angle_text, append_bang, bracket_text, execute


def greet(name, greeting):
    return string.format("{greeting}: {name}", greeting=greeting, name=name)


class AValTagged(core.AVal):
    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def __init__(self, tag):
        self.tag = tag


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(axis.ABatch(string.AStr()), id="aval"),
        pytest.param(core.Zero(axis.ABatch(string.AStr())), id="zero"),
    ],
)
def test_check_batch_aval(value):
    assert check.typecheck(value, axis.ABatch(string.AStr())) is value
    with pytest.raises(TypeError, match="Cannot join AFloat"):
        check.typecheck(value, axis.ABatch(numeric.AFloat()))


def test_join_custom_container():
    @tree.dataclass
    class Pair:
        x: object
        y: object

    aval = axis.ABatch(string.AStr())
    value = Pair("x", core.Zero(string.AStr()))
    assert check.typecheck(value, aval) is value
    with pytest.raises(TypeError, match="Cannot join AStr"):
        check.typecheck(Pair("x", 1.0), aval)


@pytest.mark.parametrize("abstract", [False, True], ids=["concrete", "abstract"])
def test_inherited_batch_join(abstract):
    class AValCustom(axis.ABatch): ...

    aval = AValCustom(string.AStr())
    value = axis.ABatch(aval.base) if abstract else ["x"]
    result = aval.join(value)
    assert type(result) is AValCustom
    assert result == aval
    if abstract:
        assert aval.join(aval) == aval
        result = value.join(aval)
        assert type(result) is axis.ABatch
        assert result == value


class TestBatchBasic:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_broadcast_symbolic_zero(self, executor):
        ir = axis.batch(stage.trace(lambda x, y: ad.cot_accum([x, y]))(1.0, 2.0))
        zero = core.Zero(axis.ABatch(numeric.AFloat()))

        assert executor(ir, zero, [2.0, 3.0]) == [2.0, 3.0]
        with pytest.raises(TypeError, match="Cannot infer batch layout from symbolic zeros alone"):
            executor(ir, zero, zero)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_valued_items(self, executor):
        x = stage.Var(aval=axis.ABatch(string.AStr()))
        ir = axis.batch(stage.IR([], (x,), x))
        values = [[], ["a", "b"]]
        assert executor(ir, values) == values
        with pytest.raises(TypeError, match="Cannot join ABatch"):
            executor(ir, ["a", "b"])

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
    def test_single_arg(self, executor, container):
        ir = axis.batch(stage.trace(append_bang)("hello"))
        values = container(("hello", "world"))
        result = executor(ir, values)
        assert result == container(("hello!", "world!"))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "program, traced, args, expected",
        [
            pytest.param(
                string.concat,
                ("Hello", " World"),
                (["Hello", "Good"], [" World", " Day"]),
                ["Hello World", "Good Day"],
                id="two-inputs",
            ),
            pytest.param(
                lambda x: string.concat(string.format("[{x}]", x=x), "!"),
                ("a",),
                (["a", "b", "c"],),
                ["[a]!", "[b]!", "[c]!"],
                id="chain",
            ),
            pytest.param(
                lambda name, value: string.format(
                    "{name}: {inner}",
                    name=name,
                    inner=string.format("{value} units", value=value),
                ),
                ("temp", "25"),
                (["temp", "pressure"], ["25", "101"]),
                ["temp: 25 units", "pressure: 101 units"],
                id="nested-format",
            ),
        ],
    )
    def test_dataflow(self, executor, program, traced, args, expected):
        ir = axis.batch(stage.trace(program)(*traced))
        actual = executor(ir, *args)
        assert actual == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_input_literal_is_not_boxed(self, executor):

        def label(prefix, value):
            return string.concat(prefix, value)

        traced = stage.trace(lambda value: label("Q", value))("x")
        ir = stage.IR(list(traced.eqns), ("Q", *traced.in_tree), traced.out_tree)
        batched_ir = axis.batch(ir, in_axes=(False, True))
        actual = executor(batched_ir, "Q", ["a", "b"])
        assert actual == ["Qa", "Qb"]
        with pytest.raises(ValueError, match="Static input mismatch"):
            executor(batched_ir, "R", ["a", "b"])

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_default_batch_with_captured_prefix(self, executor):
        prefix = "Q"
        ir = stage.trace(lambda value: string.concat(prefix, value))("x")
        assert executor(axis.batch(ir), ["a", "b"]) == ["Qa", "Qb"]


class TestBatchIRStructure:
    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(core.s_primal, id="primal"),
            pytest.param(core.s_tangent, id="tangent"),
            pytest.param(core.s_cotangent, id="cotangent"),
        ],
    )
    def test_batch_aval_ad_space(self, space):
        aval = axis.ABatch(string.AStr())
        assert space.map(aval) == axis.ABatch(string.AStr())

    def test_mapped_wrapper_aval(self):
        aval = AValTagged("input")
        var = stage.Var(aval=aval)
        ir = axis.batch(stage.IR([], (var,), (var,)), in_axes=True)
        for wrapped in (ir.in_tree[0].aval, ir.out_tree[0].aval):
            assert wrapped == axis.ABatch(aval)

    def test_broadcast_wrapper_aval(self):
        aval = AValTagged("input")
        var = stage.Var(aval=aval)
        ir = axis.batch(stage.IR([], (var,), (var,)), in_axes=False)
        for wrapped in (ir.in_tree[0].aval, ir.out_tree[0].aval):
            assert wrapped is aval

    def test_mapped_constant_output(self):
        ir = axis.batch(stage.trace(lambda x: "c")("x"), in_axes=True)
        assert isinstance(ir.out_tree, stage.Var)
        assert ir.out_tree.aval == axis.ABatch(string.AStr())
        assert ir.call(["a", "b"]) == ["c", "c"]

    def test_broadcast_constant_output(self):
        ir = axis.batch(stage.trace(lambda x: "c")("x"), in_axes=False)
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
    ir = stage.trace(program)("hello")
    aval = string.AStr()
    for _ in range(depth):
        ir = axis.batch(ir)
        aval = axis.ABatch(aval)
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
    ir = stage.trace(program)("x")
    for _ in range(depth):
        ir = axis.batch(ir)
    with pytest.raises(AssertionError):
        ir.call(values)


@pytest.mark.parametrize(
    "program, traced, args, expected",
    [
        pytest.param(
            lambda name, greeting: string.format(
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
            string.concat,
            ("a", "b"),
            ([["a1", "a2"], ["a3"]], [["b1", "b2"], ["b3"]]),
            [["a1b1", "a2b2"], ["a3b3"]],
            id="concat",
        ),
    ],
)
def test_nested_batch_two_inputs(program, traced, args, expected):
    ir = axis.batch(axis.batch(stage.trace(program)(*traced)))
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
    ir = axis.batch(stage.trace(greet)("x0", "Hi"), in_axes=in_axes)
    actual = executor(ir, *args)
    assert actual == expected


def test_numeric_program_with_broadcast_input():
    ir = stage.trace(lambda x, scale: x * x + x * scale)(1.0, 1.0)
    assert axis.batch(ir, in_axes=(True, False)).call([1.0, 2.0, 3.0], 2.0) == [3.0, 8.0, 15.0]


def batch_primitive(sample_output, batch_rule, traced="a"):
    p_batch_output = core.Prim("batch_output")
    extend.r_abstract.set(p_batch_output, lambda _: tree.map(core.avalof, sample_output))
    extend.r_batch.set(p_batch_output, batch_rule)
    return axis.batch(stage.trace(p_batch_output.bind)(traced))


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
            return (string.format("x={x}", x=x), string.format("y={y}", y=y))

        ir = stage.trace(program)("...", "...")
        batched_ir = axis.batch(ir, in_axes=(True, False))
        result = executor(batched_ir, ["a", "b", "c"], "constant")
        assert result == (["x=a", "x=b", "x=c"], ["y=constant", "y=constant", "y=constant"])

    def test_chained_with_broadcast(self):
        def program(x, prefix):
            prefixed = string.concat(prefix, x)
            return string.format("[{prefixed}]", prefixed=prefixed)

        ir = stage.trace(program)("...", "...")
        batched_ir = axis.batch(ir, in_axes=(True, False))
        result = batched_ir.call(["a", "b", "c"], ">>")
        assert result == ["[>>a]", "[>>b]", "[>>c]"]

    def test_multiple_uses_of_broadcast_input(self):
        def program(x, sep):
            return string.concat(string.concat(x, sep), x)

        ir = stage.trace(program)("...", "...")
        batched_ir = axis.batch(ir, in_axes=(True, False))
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
    "c, shared",
    [
        pytest.param(["g1", "g2"], "g1g2", id="strings"),
        pytest.param([core.Zero(string.AStr()), "g2"], "g2", id="mixed-zero"),
        pytest.param(
            [core.Zero(string.AStr()), core.Zero(string.AStr())],
            core.Zero(string.AStr()),
            id="all-zero",
        ),
    ],
)
def test_pullback_of_batch_accumulates_shared_input(executor, container, c, shared):
    ir = stage.trace(string.concat)("x", "y")
    p = ("x", container(["a", "b"]))
    feedback = container(c)
    outputs = container(["xa", "xb"])
    pb = ad.pullback(axis.batch(ir, in_axes=(False, True)))
    actual = executor(pb, p, feedback)
    assert actual == (outputs, (shared, feedback))
    assert core.avalof(actual[1][0]) == pb.out_tree[1][0].aval
    lane_pb = axis.batch(ad.pullback(ir), in_axes=((False, True), True))
    assert executor(lane_pb, p, feedback) == (outputs, (feedback, feedback))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_of_batch_accumulates_shared_pytree(executor):
    def program(x, y):
        return x["scale"] * y + x["bias"]

    ir = stage.trace(program)({"scale": 2.0, "bias": 1.0}, 3.0)
    pb = ad.pullback(axis.batch(ir, in_axes=(False, True)))
    actual = executor(pb, ({"scale": 2.0, "bias": 1.0}, [3.0, 4.0]), [1.0, 1.0])
    assert actual == ([7.0, 9.0], ({"scale": 7.0, "bias": 2.0}, [2.0, 2.0]))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_of_batch_without_mapped_inputs(executor):
    ir = stage.trace(string.concat)("x", "y")
    pb = ad.pullback(axis.batch(ir, in_axes=False))
    assert executor(pb, ("a", "b"), "g") == ("ab", ("g", "g"))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_of_nested_batch_accumulates_shared_batch(executor):
    ir = stage.trace(string.concat)("x", "y")
    inner = axis.batch(ir, in_axes=(True, False))
    pb = ad.pullback(axis.batch(inner, in_axes=(False, True)))
    actual = executor(pb, (["a", "b"], ["c", "d"]), [["g1", "g2"], ["g3", "g4"]])
    assert actual == ([["ac", "bc"], ["ad", "bd"]], (["g1g3", "g2g4"], ["g1g2", "g3g4"]))


def test_batch_box_treats_axis_spec_as_prefix():
    batcher = axis.BatchInterp(batch_size=2, parent=core.active_interp.get())
    boxed = batcher.box((["a", "b"], True))
    assert isinstance(boxed, axis.BatchBox)
    assert boxed.value == ["a", "b"]
    assert boxed.batched is True
    assert core.avalof(boxed) == string.AStr()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(("x", "y"), id="tuple"),
        pytest.param({"x": "x", "y": "y"}, id="dict"),
        pytest.param(namedtuple("Pair", "x y")("x", "y"), id="namedtuple"),
    ],
)
def test_batch_box_avalof_container(values):
    batcher = axis.BatchInterp(batch_size=2, parent=core.active_interp.get())
    box = batcher.box((values, True))

    assert core.avalof(box) == string.AStr()
    assert box.value is values


def test_broadcast_box_avalof_zero_metadata():
    aval = AValBlob(3)
    x = core.Zero(aval)
    box = axis.BatchBox(object(), x, False)

    assert core.avalof(box) is aval


def test_batch_box_avalof_zero_metadata():
    aval = AValBlob(3)
    x = core.Zero(aval)
    box = axis.BatchBox(object(), [x, x], True)
    actual = core.avalof(box)

    assert actual is aval


def test_avalof_nested_batch_ad_and_trace_boxes():
    aval = AValTagged("input")
    x = stage.TraceBox(owner=stage.TraceInterp(), var=stage.Var(aval=aval))
    x = ad.PushforwardBox(object(), x, object())
    x = axis.BatchBox(object(), (x, x), True)
    x = axis.BatchBox(object(), {"x": x, "y": x}, True)
    x = ad.PullbackBwdBox(object(), x)

    actual = core.avalof(x)

    assert actual is aval


@pytest.mark.parametrize(
    "values, message",
    [
        pytest.param([], "empty batch", id="empty-list"),
        pytest.param((), "empty batch", id="empty-tuple"),
        pytest.param("x", "Expected a batch container", id="scalar"),
        pytest.param(["x", 1], "different avals", id="mixed-types"),
        pytest.param(
            [core.Zero(AValBlob(3)), core.Zero(AValBlob(4))],
            "different avals",
            id="mixed-metadata",
        ),
    ],
)
def test_batch_box_avalof_rejects_invalid_batch(values, message):
    box = axis.BatchBox(object(), values, True)

    with pytest.raises(TypeError, match=message):
        core.avalof(box)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_shared_primitive_without_batch_rule(executor):
    p_shared = core.Prim("shared")
    calls = []

    def impl(x):
        calls.append(x)
        return {"x": x, "y": (x + "!",)}

    core.r_abstract.set(p_shared, lambda x: {"x": x, "y": (x,)})
    core.r_impl.set(p_shared, impl)
    core.r_aimpl.set(p_shared, utils.asyncify(impl))
    ir = stage.trace(lambda x, y: (x, p_shared.bind(y)))("x", "Q")
    actual = executor(axis.batch(ir, in_axes=(True, False)), ("x", "y"), "Q")
    assert actual == (("x", "y"), {"x": ("Q", "Q"), "y": (("Q!", "Q!"),)})
    assert calls == ["Q"]


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(core.Prim.bind, id="sync"),
        pytest.param(lambda p, x: asyncio.run(p.abind(x)), id="async"),
    ],
)
@pytest.mark.parametrize(
    "value, batched, expected",
    [
        pytest.param("Q", False, ["impl"], id="shared"),
        pytest.param(["x", "y"], True, ["batch"], id="outer-batched"),
    ],
)
def test_shared_batch_interp_delegates_to_parent(executor, value, batched, expected):
    p_shared = core.Prim("shared")
    calls = []

    def impl(x):
        calls.append("impl")
        return x

    def batch_rule(in_tree):
        size, batched, x = in_tree
        assert size == 2
        assert batched is True
        calls.append("batch")
        return x, True

    core.r_impl.set(p_shared, impl)
    core.r_aimpl.set(p_shared, utils.asyncify(impl))
    core.r_batch.set(p_shared, batch_rule)
    core.r_abatch.set(p_shared, utils.asyncify(batch_rule))
    parent = axis.BatchInterp(batch_size=2, parent=core.active_interp.get())
    inner = axis.BatchInterp(batch_size=3, parent=parent)
    x = inner.box((parent.box((value, batched)), False))
    with core.using_interp(inner):
        actual = executor(p_shared, x)
    assert parent.unbox(actual) == (value, batched)
    assert calls == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_nested_batch_shared_inputs_execute_once(executor):
    p_mark = core.Prim("mark")
    calls = []

    def impl(x):
        calls.append(x)
        return x + "!"

    def batch_rule(in_tree):
        _, batched, values = in_tree
        assert batched is True
        return [p_mark.bind(x) for x in values], True

    async def abatch_rule(in_tree):
        _, batched, values = in_tree
        assert batched is True
        return [await p_mark.abind(x) for x in values], True

    core.r_abstract.set(p_mark, lambda x: x)
    core.r_impl.set(p_mark, impl)
    core.r_aimpl.set(p_mark, utils.asyncify(impl))
    core.r_batch.set(p_mark, batch_rule)
    core.r_abatch.set(p_mark, abatch_rule)
    inner = axis.batch(stage.trace(p_mark.bind)("a"))
    ir = stage.trace(lambda x, y: (x, inner.call(y)))("x", ["a", "b"])
    actual = executor(axis.batch(ir, in_axes=(True, False)), ["x", "y", "z"], ["a", "b"])
    assert actual == (["x", "y", "z"], [["a!", "b!"]] * 3)
    assert calls == ["a", "b"]
