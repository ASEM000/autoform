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


@pytest.mark.asyncio
async def test_format_batch_broadcast():
    def greet(greeting, name):
        return af.string.format("{greeting}: {name}", greeting=greeting, name=name)

    ir = af.trace(greet)("x", "y")
    batched = af.batch(ir, in_axes=(False, True))
    expected = ["Hello: x0", "Hello: x1"]
    assert batched.call("Hello", ["x0", "x1"]) == expected
    assert await batched.acall("Hello", ["x0", "x1"]) == expected


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
@pytest.mark.asyncio
async def test_format_lowers_to_concat(template, values, expected, feedback):
    def program(values):
        return af.string.format(template, **values)

    ir = af.trace(program)(values)
    assert len(ir.eqns) == 1
    assert ir.eqns[0].prim is af.string.concat_p
    assert ir.eqns[0].params == {}
    primal = template.format(**values)
    assert program(values) == primal
    assert ir.call(values) == primal
    assert await ir.acall(values) == primal

    pf = af.pushforward(ir)
    assert af.ad.materialize(pf.call((values,), (values,))) == (primal, expected)
    assert af.ad.materialize(await pf.acall((values,), (values,))) == (primal, expected)
    pb = af.pullback(ir)
    assert af.ad.materialize(pb.call((values,), "g")) == (primal, (feedback,))
    assert af.ad.materialize(await pb.acall((values,), "g")) == (primal, (feedback,))


@pytest.mark.asyncio
async def test_format_explicit_leaf_access_routes_feedback_to_selected_leaves():
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
    assert pf.call((row,), (direction,)) == ("A/B/A", "dadbda")
    assert await pf.acall((row,), (direction,)) == ("A/B/A", "dadbda")
    pb = af.pullback(ir)
    expected = ("A/B/A", (Record("gg", ("g", "")),))
    assert af.ad.materialize(pb.call((row,), "g")) == expected
    assert af.ad.materialize(await pb.acall((row,), "g")) == expected


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
