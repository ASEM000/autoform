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

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.core as core
import autoform.src.numeric as numeric
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import aexecute, execute


def match_yes(x):
    return string.match(x, "yes")


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
    assert string.match(left, right) is expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "program, traced, equal, unequal",
    [
        pytest.param(
            lambda x: string.match(x, "yes"),
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
            string.match,
            ("a", "b"),
            ("hello", "hello"),
            ("hello", "world"),
            id="match-two-inputs",
        ),
        pytest.param(
            lambda x: string.match(x, "target"),
            ("dummy",),
            ("target",),
            ("other",),
            id="literal",
        ),
    ],
)
@pytest.mark.parametrize("expected", [True, False], ids=["equal", "unequal"])
def test_match_lowering(executor, program, traced, equal, unequal, expected):
    ir = stage.trace(program)(*traced)
    assert [eqn.prim for eqn in ir.eqns] == [string.p_match]
    args = equal if expected else unequal
    actual = executor(ir, *args)
    assert actual is expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
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
        pytest.param(string.concat, ("a", "b"), ("hello", " world"), "hello world", id="concat"),
        pytest.param(
            lambda x: string.format("Value: {x}", x=x),
            ("test",),
            ("hello",),
            "Value: hello",
            id="format",
        ),
    ],
)
def test_concat_lowering(executor, program, traced, args, expected):
    ir = stage.trace(program)(*traced)
    assert [eqn.prim for eqn in ir.eqns] == [string.p_concat]
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize(
    "program, args, error, message",
    [
        pytest.param(string.match, ("yes", 1), AssertionError, "Expected strings", id="match"),
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
            r"No trace rule for eq on values of type AValInt\(\)",
            id="eq-integer",
        ),
        pytest.param(lambda x: x + 1, ("a",), AssertionError, "Expected strings", id="add"),
        pytest.param(lambda x: 1 + x, ("a",), AssertionError, "Expected strings", id="reverse-add"),
        pytest.param(
            lambda x: x + 1,
            (1,),
            TypeError,
            r"No trace rule for add on values of type AValInt\(\)",
            id="add-integer",
        ),
        pytest.param(
            string.concat,
            ("a", "b", 1),
            AssertionError,
            "Expected strings",
            id="concat",
        ),
    ],
)
def test_invalid_traced_operands(program, args, error, message):
    with pytest.raises(error, match=message):
        stage.trace(program)(*args)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected",
    [
        pytest.param(axis.batch, (["yes", "yes", "yes"],), [True, True, True], id="batch-equal"),
        pytest.param(axis.batch, (["yes", "no", "yes"],), [True, False, True], id="batch-mixed"),
        pytest.param(
            ad.pushforward,
            (("yes",), ("tangent_input",)),
            (True, core.Zero(numeric.AValBool())),
            id="pushforward-true",
        ),
        pytest.param(
            ad.pushforward,
            (("no",), ("tangent_input",)),
            (False, core.Zero(numeric.AValBool())),
            id="pushforward-false",
        ),
        pytest.param(
            ad.pullback,
            (("yes",), core.Zero(numeric.AValBool())),
            (True, (core.Zero(string.AValStr()),)),
            id="pullback-true",
        ),
        pytest.param(
            ad.pullback,
            (("no",), core.Zero(numeric.AValBool())),
            (False, (core.Zero(string.AValStr()),)),
            id="pullback-false",
        ),
    ],
)
def test_match_transforms(executor, transform, args, expected):
    ir = transform(stage.trace(match_yes)("dummy"))
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
    ir = axis.batch(stage.trace(string.match)("a", "b"), in_axes=axes)
    assert ir.call(*args) == [True, False, True]


@pytest.mark.parametrize(
    "operands",
    [
        pytest.param(("yes", "yes"), id="equal"),
        pytest.param(("yes", "no"), id="unequal"),
        pytest.param((string.AValStr(), "yes"), id="left-abstract"),
        pytest.param(("yes", string.AValStr()), id="right-abstract"),
        pytest.param((string.AValStr(), string.AValStr()), id="both-abstract"),
    ],
)
def test_match_abstract(operands):
    assert string.abstract_match(operands) == numeric.AValBool()


def test_match_abstract_rejects_nonstring():
    with pytest.raises(AssertionError, match="Expected strings"):
        string.abstract_match(("yes", 1))


@pytest.mark.parametrize(
    "status, expected",
    [pytest.param("active", True, id="active"), pytest.param("inactive", False, id="inactive")],
)
def test_match_in_larger_program(status, expected):
    def process(status, text):
        return string.match(status, "active"), string.format(
            "Status check: {text}",
            text=text,
        )

    assert stage.trace(process)("status", "text").call(status, "hello") == (
        expected,
        "Status check: hello",
    )


def test_batch_match_with_format():
    def process(status):
        return string.match(status, "yes"), string.format(
            "Input was: {status}",
            status=status,
        )

    ir = axis.batch(stage.trace(process)("status"))
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
    assert string.concat(*args) == expected


def test_concat_rejects_nonstring():
    with pytest.raises(TypeError):
        string.concat("A", 1)


def test_format_lowers_template_and_args_to_concat():
    def program(x):
        return string.format("Hello, {x}!", x=x)

    ir = stage.trace(program)("World")
    assert len(ir.eqns) == 1
    eqn = ir.eqns[0]
    assert eqn.prim is string.p_concat
    prefix, value, suffix = eqn.in_tree
    assert prefix == "Hello, "
    assert suffix == "!"
    assert isinstance(value, stage.Var)
    assert ir.call("x0") == "Hello, x0!"


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
    assert string.format(template, **values) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_format_batch_broadcast(executor):

    def greet(greeting, name):
        return string.format("{greeting}: {name}", greeting=greeting, name=name)

    ir = stage.trace(greet)("x", "y")
    batched = axis.batch(ir, in_axes=(False, True))
    expected = ["Hello: x0", "Hello: x1"]
    args = ("Hello", ["x0", "x1"])
    actual = executor(batched, *args)
    assert actual == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "template, values, expected, feedback",
    [
        pytest.param("Hello {x}/{x}", {"x": "A"}, "AA", {"x": "gg"}, id="repeated-field"),
        pytest.param(
            "{right}/{left}/{right}",
            {"left": "A", "right": "B"},
            "BAB",
            {"left": "g", "right": "gg"},
            id="reordered-repeated-fields",
        ),
        pytest.param(
            "{a}{b}",
            {"a": "A", "b": "B"},
            "AB",
            {"a": "g", "b": "g"},
            id="adjacent-fields",
        ),
        pytest.param("{{{name}}}", {"name": "A"}, "A", {"name": "g"}, id="escaped-braces"),
        pytest.param("constant", {}, "", {}, id="constant"),
        pytest.param("", {}, "", {}, id="empty"),
    ],
)
def test_format_lowers_to_concat(executor, template, values, expected, feedback):

    def program(values):
        return string.format(template, **values)

    ir = stage.trace(program)(values)
    assert len(ir.eqns) == 1
    assert ir.eqns[0].prim is string.p_concat
    p = template.format(**values)
    assert program(values) == p
    actual = executor(ir, values)
    assert actual == p
    pf = ad.pushforward(ir)
    result = executor(pf, (values,), (values,))
    zero = lambda x: x.aval.zero() if isinstance(x, core.Zero) else x
    assert utils.tree.map(zero, result) == (p, expected)
    pb = ad.pullback(ir)
    result = executor(pb, (values,), "g")
    assert utils.tree.map(zero, result) == (p, (feedback,))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_format_explicit_leaf_access_routes_feedback_to_selected_leaves(executor):
    class Record(NamedTuple):
        name: str
        items: tuple[str, str]

    def program(row):
        return string.format("{name}/{item}/{name}", name=row.name, item=row.items[0])

    row = Record("A", ("B", "C"))
    ir = stage.trace(program)(row)
    assert ir.eqns[0].prim is string.p_concat
    pf = ad.pushforward(ir)
    direction = Record("da", ("db", "dc"))
    assert executor(pf, (row,), (direction,)) == ("A/B/A", "dadbda")
    pb = ad.pullback(ir)
    expected = ("A/B/A", (Record("gg", ("g", "")),))
    zero = lambda x: x.aval.zero() if isinstance(x, core.Zero) else x
    assert utils.tree.map(zero, executor(pb, (row,), "g")) == expected


@pytest.mark.parametrize(
    "template, unsupported",
    [
        pytest.param("{x!r}", "conversions", id="repr-conversion"),
        pytest.param("{x!s}", "conversions", id="str-conversion"),
        pytest.param("{x!a}", "conversions", id="ascii-conversion"),
        pytest.param("{x:>8}", "format specifications", id="format-specification"),
    ],
)
def test_format_rejects_unsupported_fields(template, unsupported):
    def program(x):
        return string.format(template, x=x)

    with pytest.raises(AssertionError, match="`format` does not support " + unsupported):
        program("A")
    with pytest.raises(AssertionError, match="`format` does not support " + unsupported):
        stage.trace(program)("A")


@pytest.mark.parametrize("field", ["", "0", "0.real", "0[0]", "x.name", "x[0]"])
def test_format_does_not_resolve_field_paths(field):
    def program(x):
        return string.format("{" + field + "}", x=x)

    with pytest.raises(AssertionError, match="Template field name is not found"):
        program("A")
    with pytest.raises(AssertionError, match="Template field name is not found"):
        stage.trace(program)("A")


@pytest.mark.parametrize(
    "template, values",
    [
        pytest.param("Hello {x}", {"x": "A", "y": "B"}, id="extra-field"),
        pytest.param("{x}{x}", {"x": "A", "y": "B"}, id="extra-field-with-repetition"),
        pytest.param("constant", {"x": "A"}, id="constant-with-argument"),
    ],
)
def test_format_rejects_unused_arguments(template, values):
    def program(values):
        return string.format(template, **values)

    with pytest.raises(AssertionError, match="Unused format arguments"):
        program(values)
    with pytest.raises(AssertionError, match="Unused format arguments"):
        stage.trace(program)(values)


def test_format_requires_keyword_arguments():
    with pytest.raises(TypeError):
        string.format("{x}", "A")


def test_format_requires_string_values():
    with pytest.raises(TypeError):
        string.format("{value}", value=1)
    with pytest.raises(AssertionError, match="Expected strings"):
        stage.trace(lambda x: string.format("{x}", x=x))(1)
