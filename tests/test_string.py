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

from typing import NamedTuple

import pytest

import autoform as af
from autoform.core import BoolAVal, StrAVal
from autoform.string import abstract_match
from tests import aexecute, execute


def match_yes(x):
    return af.string.match(x, "yes")


@pytest.mark.parametrize(
    "left, right, expected",
    [
        pytest.param("yes", "yes", True, id="equal"),
        pytest.param("yes", "no", False, id="unequal"),
        pytest.param("", "", True, id="empty"),
        pytest.param("", "x", False, id="empty-nonempty"),
        pytest.param("yes", 1, False, id="nonstring"),
    ],
)
def test_match(left, right, expected):
    assert af.string.match(left, right) is expected


@pytest.mark.parametrize(
    "program, traced, equal, unequal",
    [
        pytest.param(
            lambda x: af.string.match(x, "yes"),
            ("dummy",),
            ("yes",),
            ("no",),
            id="match",
        ),
        pytest.param(lambda x: x == "yes", ("dummy",), ("yes",), ("no",), id="eq"),
        pytest.param(lambda x: "yes" == x, ("dummy",), ("yes",), ("no",), id="reverse-eq"),
        pytest.param(
            lambda a, b: a == b,
            ("a", "b"),
            ("same", "same"),
            ("same", "other"),
            id="eq-two-inputs",
        ),
        pytest.param(
            af.string.match,
            ("a", "b"),
            ("hello", "hello"),
            ("hello", "world"),
            id="match-two-inputs",
        ),
        pytest.param(
            lambda x: af.string.match(x, "target"),
            ("dummy",),
            ("target",),
            ("other",),
            id="literal",
        ),
    ],
)
@pytest.mark.parametrize("expected", [True, False], ids=["equal", "unequal"])
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_match_lowering(program, traced, equal, unequal, expected, executor):
    ir = af.trace(program)(*traced)
    assert [eqn.prim for eqn in ir.eqns] == [af.string.match_p]
    args = equal if expected else unequal
    actual = executor(ir, *args)
    assert actual is expected


@pytest.mark.parametrize(
    "program, traced, args, expected",
    [
        pytest.param(lambda x: x + "!", ("a",), ("hello",), "hello!", id="add"),
        pytest.param(lambda x: "Hello, " + x, ("a",), ("world",), "Hello, world", id="reverse-add"),
        pytest.param(
            lambda a, b: a + b,
            ("a", "b"),
            ("left", "right"),
            "leftright",
            id="add-two-inputs",
        ),
        pytest.param(af.string.concat, ("a", "b"), ("hello", " world"), "hello world", id="concat"),
        pytest.param(
            lambda x: af.string.format("Value: {x}", x=x),
            ("test",),
            ("hello",),
            "Value: hello",
            id="format",
        ),
    ],
)
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_concat_lowering(program, traced, args, expected, executor):
    ir = af.trace(program)(*traced)
    assert [eqn.prim for eqn in ir.eqns] == [af.string.concat_p]
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize(
    "program, args, error, message",
    [
        pytest.param(af.string.match, ("yes", 1), AssertionError, "Expected strings", id="match"),
        pytest.param(
            lambda x: x == 1,
            ("yes",),
            AssertionError,
            "Expected strings",
            id="eq-string",
        ),
        pytest.param(
            lambda x: x == 1,
            (1,),
            TypeError,
            r"No trace rule for eq on values of type IntAVal\(\)",
            id="eq-integer",
        ),
        pytest.param(lambda x: x + 1, ("a",), AssertionError, "Expected strings", id="add"),
        pytest.param(lambda x: 1 + x, ("a",), AssertionError, "Expected strings", id="reverse-add"),
        pytest.param(
            lambda x: x + 1,
            (1,),
            TypeError,
            r"No trace rule for add on values of type IntAVal\(\)",
            id="add-integer",
        ),
        pytest.param(
            af.string.concat,
            ("a", "b", 1),
            AssertionError,
            "Expected strings",
            id="concat",
        ),
    ],
)
def test_invalid_traced_operands(program, args, error, message):
    with pytest.raises(error, match=message):
        af.trace(program)(*args)


@pytest.mark.parametrize(
    "transform, args, expected",
    [
        pytest.param(af.batch, (["yes", "yes", "yes"],), [True, True, True], id="batch-equal"),
        pytest.param(af.batch, (["yes", "no", "yes"],), [True, False, True], id="batch-mixed"),
        pytest.param(
            af.pushforward,
            (("yes",), ("tangent_input",)),
            (True, af.ad.Zero(BoolAVal())),
            id="pushforward-true",
        ),
        pytest.param(
            af.pushforward,
            (("no",), ("tangent_input",)),
            (False, af.ad.Zero(BoolAVal())),
            id="pushforward-false",
        ),
        pytest.param(
            af.pullback,
            (("yes",), "feedback"),
            (True, (af.ad.Zero(StrAVal()),)),
            id="pullback-true",
        ),
        pytest.param(
            af.pullback,
            (("no",), "feedback"),
            (False, (af.ad.Zero(StrAVal()),)),
            id="pullback-false",
        ),
    ],
)
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_match_transforms(transform, args, expected, executor):
    ir = transform(af.trace(match_yes)("dummy"))
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize(
    "axes, args",
    [
        pytest.param((True, True), (["a", "b", "c"], ["a", "x", "c"]), id="mapped"),
        pytest.param((True, False), (["target", "other", "target"], "target"), id="broadcast"),
    ],
)
def test_match_batch_axes(axes, args):
    ir = af.batch(af.trace(af.string.match)("a", "b"), in_axes=axes)
    assert ir.call(*args) == [True, False, True]


@pytest.mark.parametrize(
    "operands",
    [
        pytest.param(("yes", "yes"), id="equal"),
        pytest.param(("yes", "no"), id="unequal"),
        pytest.param((StrAVal(), "yes"), id="left-abstract"),
        pytest.param(("yes", StrAVal()), id="right-abstract"),
        pytest.param((StrAVal(), StrAVal()), id="both-abstract"),
    ],
)
def test_match_abstract(operands):
    assert abstract_match(operands) == BoolAVal()


def test_match_abstract_rejects_nonstring():
    with pytest.raises(AssertionError, match="Expected strings"):
        abstract_match(("yes", 1))


@pytest.mark.parametrize("status, expected", [("active", True), ("inactive", False)])
def test_match_in_larger_program(status, expected):
    def process(status, text):
        return af.string.match(status, "active"), af.string.format(
            "Status check: {text}",
            text=text,
        )

    assert af.trace(process)("status", "text").call(status, "hello") == (
        expected,
        "Status check: hello",
    )


def test_batch_match_with_format():
    def process(status):
        return af.string.match(status, "yes"), af.string.format(
            "Input was: {status}",
            status=status,
        )

    ir = af.batch(af.trace(process)("status"))
    assert ir.call(["yes", "no", "yes"]) == (
        [True, False, True],
        ["Input was: yes", "Input was: no", "Input was: yes"],
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(("Hello", " ", "World"), "Hello World", id="three"),
        pytest.param(("A", "B"), "AB", id="two"),
    ],
)
def test_concat(args, expected):
    assert af.string.concat(*args) == expected


def test_concat_rejects_nonstring():
    with pytest.raises(TypeError):
        af.string.concat("A", 1)


@pytest.mark.parametrize(
    "template, values, expected",
    [
        pytest.param("Hello, {value}!", {"value": "World"}, "Hello, World!", id="single"),
        pytest.param(
            "{value_1} + {value_2} = {value_3}",
            {"value_1": "1", "value_2": "2", "value_3": "3"},
            "1 + 2 = 3",
            id="multiple",
        ),
    ],
)
def test_format(template, values, expected):
    assert af.string.format(template, **values) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_format_batch_broadcast(executor):

    def greet(greeting, name):
        return af.string.format("{greeting}: {name}", greeting=greeting, name=name)

    ir = af.trace(greet)("x", "y")
    batched = af.batch(ir, in_axes=(False, True))
    expected = ["Hello: x0", "Hello: x1"]
    args = ("Hello", ["x0", "x1"])
    actual = executor(batched, *args)
    assert actual == expected


@pytest.mark.parametrize(
    "template, values, expected, feedback",
    [
        ("Hello {x}/{x}", {"x": "A"}, "AA", {"x": "gg"}),
        (
            "{right}/{left}/{right}",
            {"left": "A", "right": "B"},
            "BAB",
            {"left": "g", "right": "gg"},
        ),
        ("{a}{b}", {"a": "A", "b": "B"}, "AB", {"a": "g", "b": "g"}),
        ("{{{name}}}", {"name": "A"}, "A", {"name": "g"}),
        ("constant", {}, "", {}),
        ("", {}, "", {}),
    ],
)
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_format_lowers_to_concat(template, values, expected, feedback, executor):

    def program(values):
        return af.string.format(template, **values)

    ir = af.trace(program)(values)
    assert len(ir.eqns) == 1
    assert ir.eqns[0].prim is af.string.concat_p
    primal = template.format(**values)
    assert program(values) == primal
    actual = executor(ir, values)
    assert actual == primal
    pf = af.pushforward(ir)
    result = executor(pf, (values,), (values,))
    assert af.ad.materialize(result) == (primal, expected)
    pb = af.pullback(ir)
    result = executor(pb, (values,), "g")
    assert af.ad.materialize(result) == (primal, (feedback,))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_format_explicit_leaf_access_routes_feedback_to_selected_leaves(executor):
    class Record(NamedTuple):
        name: str
        items: tuple[str, str]

    def program(row):
        return af.string.format("{name}/{item}/{name}", name=row.name, item=row.items[0])

    row = Record("A", ("B", "C"))
    ir = af.trace(program)(row)
    assert ir.eqns[0].prim is af.string.concat_p
    pf = af.pushforward(ir)
    direction = Record("da", ("db", "dc"))
    assert executor(pf, (row,), (direction,)) == ("A/B/A", "dadbda")
    pb = af.pullback(ir)
    expected = ("A/B/A", (Record("gg", ("g", "")),))
    assert af.ad.materialize(executor(pb, (row,), "g")) == expected


@pytest.mark.parametrize(
    "template, unsupported",
    [
        ("{x!r}", "conversions"),
        ("{x!s}", "conversions"),
        ("{x!a}", "conversions"),
        ("{x:>8}", "format specifications"),
    ],
)
def test_format_rejects_unsupported_fields(template, unsupported):
    def program(x):
        return af.string.format(template, x=x)

    with pytest.raises(AssertionError, match="`format` does not support " + unsupported):
        program("A")
    with pytest.raises(AssertionError, match="`format` does not support " + unsupported):
        af.trace(program)("A")


@pytest.mark.parametrize("field", ["", "0", "0.real", "0[0]", "x.name", "x[0]"])
def test_format_does_not_resolve_field_paths(field):
    def program(x):
        return af.string.format("{" + field + "}", x=x)

    with pytest.raises(AssertionError, match="Template field name is not found"):
        program("A")
    with pytest.raises(AssertionError, match="Template field name is not found"):
        af.trace(program)("A")


@pytest.mark.parametrize(
    "template, values",
    [
        ("Hello {x}", {"x": "A", "y": "B"}),
        ("{x}{x}", {"x": "A", "y": "B"}),
        ("constant", {"x": "A"}),
    ],
)
def test_format_rejects_unused_arguments(template, values):
    def program(values):
        return af.string.format(template, **values)

    with pytest.raises(AssertionError, match="Unused format arguments"):
        program(values)
    with pytest.raises(AssertionError, match="Unused format arguments"):
        af.trace(program)(values)


def test_format_requires_keyword_arguments():
    with pytest.raises(TypeError):
        af.string.format("{x}", "A")


def test_format_requires_string_values():
    with pytest.raises(TypeError):
        af.string.format("{value}", value=1)
    with pytest.raises(AssertionError, match="Expected strings"):
        af.trace(lambda x: af.string.format("{x}", x=x))(1)


@pytest.mark.parametrize(
    "primitive, size, axes, values, expected, out_axis",
    [
        pytest.param(
            af.string.concat_p,
            2,
            (True,),
            (["a", "b"],),
            ["a", "b"],
            True,
            id="single-mapped",
        ),
        pytest.param(
            af.string.concat_p,
            2,
            (True, True),
            (["a", "b"], ["x", "y"]),
            ["ax", "by"],
            True,
            id="both-mapped",
        ),
        pytest.param(
            af.string.concat_p,
            3,
            (True, False),
            (["a", "b", "c"], "!"),
            ["a!", "b!", "c!"],
            True,
            id="broadcast",
        ),
        pytest.param(
            af.string.concat_p,
            2,
            (False, False, True, False),
            ("Hello", ", ", ["x0", "x1"], "!"),
            ["Hello, x0!", "Hello, x1!"],
            True,
            id="format-literals",
        ),
        pytest.param(
            af.string.concat_p,
            0,
            (False, False),
            ("a", "b"),
            "ab",
            False,
            id="no-mapped-zero-size",
        ),
        pytest.param(
            af.string.concat_p,
            3,
            (False, False, False),
            ("hello", " ", "world"),
            "hello world",
            False,
            id="format-unmapped",
        ),
        pytest.param(
            af.string.concat_p,
            3,
            (False, False),
            ("hello", "world"),
            "helloworld",
            False,
            id="concat-unmapped",
        ),
        pytest.param(
            af.string.match_p,
            3,
            (False, False),
            ("hello", "hello"),
            True,
            False,
            id="match-unmapped-true",
        ),
        pytest.param(
            af.string.match_p,
            3,
            (False, False),
            ("hello", "world"),
            False,
            False,
            id="match-unmapped-false",
        ),
    ],
)
def test_batch_rule(primitive, size, axes, values, expected, out_axis):
    assert af.core.batch_rules.get(primitive)((size, axes, values)) == (expected, out_axis)
