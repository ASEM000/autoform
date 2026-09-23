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
from autoform.order import depends_p, fanout_p
from tests import (
    aexecute,
    dependent_formats,
    execute,
    switch_program,
)


def parallel_formats(*, joined=False):
    def program(x):
        a = af.string.format("[{x}]", x=x)
        b = af.string.format("<{x}>", x=x)
        return af.string.concat(a, b) if joined else (a, b)

    return program


def format_switch_ir():
    branches = {
        "x": af.trace(parallel_formats(joined=True))("x"),
        "y": af.trace(lambda x: af.string.format("({x})", x=x))("x"),
    }
    return af.trace(switch_program(branches))("x", "inp")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "joined, transform, args, expected",
    [
        pytest.param(False, lambda ir: ir, ("A",), ("[A]", "<A>"), id="fanout"),
        pytest.param(True, lambda ir: ir, ("test",), "[test]<test>", id="joined"),
        pytest.param(
            False,
            af.pushforward,
            (("primal",), ("tangent",)),
            (("[primal]", "<primal>"), ("tangent", "tangent")),
            id="push",
        ),
        pytest.param(
            False,
            af.pullback,
            (("primal",), ("grad1", "grad2")),
            (("[primal]", "<primal>"), ("grad1grad2",)),
            id="pull",
        ),
        pytest.param(
            False,
            af.batch,
            (["A", "B", "C"],),
            (["[A]", "[B]", "[C]"], ["<A>", "<B>", "<C>"]),
            id="batch",
        ),
        pytest.param(
            True,
            af.pushforward,
            (("test",), ("tangent",)),
            ("[test]<test>", "tangenttangent"),
            id="joined-push",
        ),
        pytest.param(
            True,
            af.pullback,
            (("test",), "grad"),
            ("[test]<test>", ("gradgrad",)),
            id="joined-pull",
        ),
        pytest.param(
            True,
            af.batch,
            (["A", "B", "C"],),
            ["[A]<A>", "[B]<B>", "[C]<C>"],
            id="joined-batch",
        ),
    ],
)
def test_fanout_transforms(joined, transform, args, expected, executor):
    scheduled = af.sched(af.trace(parallel_formats(joined=joined))("a"))
    assert [e.prim for e in scheduled.eqns] == [fanout_p] + ([af.string.concat_p] if joined else [])
    ir = transform(scheduled)
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("c_out", [(1.0, 0.0), (2.0, 3.0)], ids=["one-live", "both-live"])
def test_fanout_pullback_repeated_operand(c_out, executor):
    ir = af.trace(lambda x: (x * x, x + 1.0))(3.0)
    scheduled = af.sched(ir)
    assert [e.prim for e in scheduled.eqns] == [fanout_p]
    expected = ((9.0, 4.0), (6.0 * c_out[0] + c_out[1],))
    assert af.pullback(ir).call((3.0,), c_out) == expected
    pb = af.pullback(scheduled)
    assert executor(pb, (3.0,), c_out) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_fanout_exception(executor):

    def impl_error(x):
        raise ValueError("intentional error")

    error_p = af.core.Prim("error")
    af.core.impl_rules.set(error_p, impl_error)
    af.core.abstract_rules.set(error_p, lambda x: af.core.StrAVal())
    af.core.impl_rules.aset(error_p, af.utils.asyncify(impl_error))
    ir = af.sched(af.trace(lambda x: (af.string.format("[{x}]", x=x), error_p.bind(x)))("a"))
    with pytest.raises(ValueError, match="intentional error"):
        executor(ir, "A")


@pytest.mark.parametrize(
    "executor, values, expected",
    [
        pytest.param(execute, ["A", "B", "C"], ["[A]", "[B]", "[C]"], id="sync"),
        pytest.param(aexecute, ["X", "Y"], ["[X]", "[Y]"], id="async"),
    ],
)
def test_fanout_mixed_axes(executor, values, expected):
    ir = af.trace(lambda x, y: (af.string.format("[{x}]", x=x), af.string.format("<{y}>", y=y)))(
        "x",
        "y",
    )
    ir = af.batch(af.sched(ir), in_axes=(True, False))
    result = executor(ir, values, "STATIC")
    assert result == (expected, ["<STATIC>"] * len(values))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "program, prims, expected",
    [
        pytest.param(
            lambda x: af.string.format("[{x}]", x=x),
            [af.string.concat_p],
            "[test]",
            id="single",
        ),
        pytest.param(
            lambda x: af.string.concat(af.string.format("[{x}]", x=x), "!"),
            [af.string.concat_p] * 2,
            "[test]!",
            id="sequential",
        ),
        pytest.param(
            lambda x: tuple((af.string.format(f, x=x) for f in ("[{x}]", "<{x}>", "{{{x}}}"))),
            [fanout_p],
            ("[test]", "<test>", "{test}"),
            id="three-independent",
        ),
    ],
)
def test_sched_levels(program, prims, expected, executor):
    ir = af.sched(af.trace(program)("x"))
    assert [e.prim for e in ir.eqns] == prims
    actual = executor(ir, "test")
    assert actual == expected


def test_sched_filter():
    ir = af.trace(lambda x: (af.string.format("[{x}]", x=x), af.string.concat(x, "!")))("x")
    scheduled = af.sched(ir, cond=lambda e: e.prim is af.string.concat_p and len(e.in_tree) == 3)
    assert [e.prim for e in scheduled.eqns] == [af.string.concat_p] * 2
    assert scheduled.call("test") == ("[test]", "test!")


def test_sched_filter_propagates_into_switch():
    inner = af.trace(lambda x: (af.string.format("[{x}]", x=x), af.string.concat(x, "!")))("x")
    ir = af.trace(lambda x: af.switch("a", {"a": inner}, x))("x")
    scheduled = af.sched(ir, cond=lambda e: e.prim is af.string.concat_p and len(e.in_tree) == 3)
    checked = scheduled.eqns[0].params["branches"]["a"]
    assert [e.prim for e in checked.eqns] == [af.string.concat_p] * 2
    assert scheduled.call("test") == ("[test]", "test!")


def test_sched_parallel_checkpoints():
    def program(a, b):
        x = af.checkpoint(af.string.format("{a}", a=a), key="x")
        y = af.checkpoint(af.string.format("{b}", b=b), key="y")
        return x, y

    ir = af.sched(af.trace(program)("a", "b"))
    assert sum(e.prim is fanout_p for e in ir.eqns) == 2
    assert ir.call("hello", "world") == ("hello", "world")


def test_sched_dependent_checkpoints():
    def program(a, b):
        x = af.checkpoint(af.string.format("{a}", a=a), key="x")
        y = af.checkpoint(af.string.format("{b}", b=b), key="y")
        return af.depends(y, x)

    ir = af.sched(af.trace(program)("a", "b"))
    assert sum(e.prim is fanout_p for e in ir.eqns) == 2
    assert ir.call("hello", "world") == "world"


def test_sched_mixed_checkpoint():
    def program(a, b, c):
        x = af.string.format("[{a}]", a=a)
        y = af.string.format("<{b}>", b=b)
        z = af.checkpoint(af.string.format("{{{c}}}", c=c), key="z")
        return x, y, z

    ir = af.sched(af.trace(program)("a", "b", "c"))
    assert sum(e.prim is fanout_p for e in ir.eqns) == 1
    assert ir.call("a", "b", "c") == ("[a]", "<b>", "{c}")


def test_fanout_collect():
    def program(a, b):
        return (
            af.checkpoint(a, key="val", collection="debug"),
            af.checkpoint(b, key="val", collection="debug"),
        )

    ir = af.sched(af.trace(program)("a", "b"))
    with af.collect(collection="debug") as collected:
        assert ir.call("A", "B") == ("A", "B")
    assert set(collected["val"]) == {"A", "B"}


def test_fanout_inject():
    def program(a, b):
        x = af.checkpoint(af.string.format("[{a}]", a=a), key="val", collection="cache")
        y = af.checkpoint(af.string.format("<{b}>", b=b), key="val", collection="cache")
        return x, y

    ir = af.sched(af.trace(program)("a", "b"))
    with af.inject(collection="cache", values={"val": ["CACHED1", "CACHED2"]}):
        assert ir.call("A", "B") == ("CACHED1", "CACHED2")


def test_nested_fanout_collect():
    def inner(x):
        return af.checkpoint(x, key="inner", collection="debug")

    branches = {key: af.sched(af.trace(lambda x: (inner(x), inner(x)))("x")) for key in ("a", "b")}

    def outer(key, x):
        left, right = af.switch(key, branches, x)
        return af.string.concat(left, right)

    ir = af.sched(af.trace(outer)("a", "x"))
    with af.collect(collection="debug") as collected:
        assert ir.call("a", "A") == "AA"
    assert collected == {"inner": ["A", "A"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(("x", "hello"), "[hello]<hello>", id="parallel-branch"),
        pytest.param(("y", "hello"), "(hello)", id="single-branch"),
    ],
)
def test_sched_switch(executor, args, expected):
    scheduled = af.sched(format_switch_ir())
    assert scheduled.eqns[0].params["branches"]["x"].eqns[0].prim is fanout_p
    actual = executor(scheduled, *args)
    assert actual == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param(("A", "x", "hello"), "[hello]<hello>", id="nested-parallel-branch"),
        pytest.param(("A", "y", "hello"), "(hello)", id="nested-single-branch"),
        pytest.param(("B", "ignored", "world"), "ignored world", id="outer-branch"),
    ],
)
def test_sched_nested_switch(executor, args, expected):
    branches = {
        "A": format_switch_ir(),
        "B": af.trace(lambda key, inp: af.string.format("{key} {inp}", key=key, inp=inp))("k", "i"),
    }
    ir = af.trace(lambda key, inner_key, x: af.switch(key, branches, inner_key, x))(
        "A",
        "x",
        "test",
    )
    scheduled = af.sched(ir)
    inner = scheduled.eqns[0].params["branches"]["A"]
    assert inner.eqns[0].params["branches"]["x"].eqns[0].prim is fanout_p
    actual = executor(scheduled, *args)
    assert actual == expected


def test_sched_nested_fanout():
    branches1 = {"a": af.trace(parallel_formats(joined=True))("x")}
    branches2 = {
        "a": af.trace(
            lambda x: af.string.concat(
                af.string.format("({x})", x=x),
                af.string.format("{{{x}}}", x=x),
            ),
        )("x")
    }
    ir = af.trace(lambda key, a, b: (af.switch(key, branches1, a), af.switch(key, branches2, b)))(
        "a",
        "hello",
        "world",
    )
    scheduled = af.sched(ir)
    assert scheduled.eqns[0].prim is fanout_p
    for inner in scheduled.eqns[0].params["irs"]:
        assert inner.eqns[0].params["branches"]["a"].eqns[0].prim is fanout_p
    assert scheduled.call("a", "hello", "world") == ("[hello]<hello>", "(world){world}")


def no_dependencies(x):
    return af.depends(af.string.format("A: {x}", x=x))


def chained_dependencies(x):
    a = af.string.format("A: {x}", x=x)
    b = af.string.format("B: {x}", x=x)
    c = af.string.format("C: {x}", x=x)
    return af.depends(c, af.depends(b, a))


def multiple_dependencies(x):
    a = af.string.format("A: {x}", x=x)
    b = af.string.format("B: {x}", x=x)
    c = af.string.format("C: {x}", x=x)
    return af.depends(c, a, b)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected",
    [
        pytest.param(lambda ir: ir, ("hello",), "B: hello", id="call"),
        pytest.param(af.sched, ("hello",), "B: hello", id="sched"),
        pytest.param(
            af.pushforward,
            (("primal",), ("tangent",)),
            ("B: primal", "tangent"),
            id="push",
        ),
        pytest.param(af.pullback, (("primal",), "grad"), ("B: primal", ("grad",)), id="pull"),
        pytest.param(af.batch, (["x", "y", "z"],), ["B: x", "B: y", "B: z"], id="batch"),
        pytest.param(
            lambda ir: af.batch(af.pushforward(ir), in_axes=(True, True)),
            ((["a", "b"],), (["da", "db"],)),
            (["B: a", "B: b"], ["da", "db"]),
            id="batch-push",
        ),
        pytest.param(
            lambda ir: af.batch(af.pullback(ir), in_axes=(True, True)),
            ((["a", "b"],), ["g1", "g2"]),
            (["B: a", "B: b"], (["g1", "g2"],)),
            id="batch-pull",
        ),
        pytest.param(
            lambda ir: af.pushforward(af.batch(ir)),
            ((["a", "b"],), (["da", "db"],)),
            (["B: a", "B: b"], ["da", "db"]),
            id="push-batch",
        ),
        pytest.param(
            lambda ir: af.pullback(af.batch(ir)),
            ((["a", "b"],), ["g1", "g2"]),
            (["B: a", "B: b"], (["g1", "g2"],)),
            id="pull-batch",
        ),
        pytest.param(
            lambda ir: af.sched(af.pushforward(ir)),
            (("primal",), ("tangent",)),
            ("B: primal", "tangent"),
            id="sched-push",
        ),
        pytest.param(
            lambda ir: af.sched(af.pullback(ir)),
            (("primal",), "grad"),
            ("B: primal", ("grad",)),
            id="sched-pull",
        ),
        pytest.param(
            lambda ir: af.sched(af.batch(ir)),
            (["x", "y", "z"],),
            ["B: x", "B: y", "B: z"],
            id="sched-batch",
        ),
    ],
)
def test_depends_transforms(transform, args, expected, executor):
    ir = af.trace(dependent_formats)("x")
    [barrier] = [e for e in ir.eqns if e.prim is depends_p]
    assert len(af.utils.tree.leaves(barrier.in_tree)) == 2
    ir = transform(ir)
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize(
    "program, transform, args, expected",
    [
        pytest.param(no_dependencies, lambda ir: ir, ("hello",), "A: hello", id="no-deps"),
        pytest.param(multiple_dependencies, lambda ir: ir, ("hello",), "C: hello", id="multiple"),
        pytest.param(chained_dependencies, lambda ir: ir, ("hello",), "C: hello", id="chain"),
        pytest.param(
            multiple_dependencies,
            af.pushforward,
            (("primal",), ("tangent",)),
            ("C: primal", "tangent"),
            id="multiple-push",
        ),
        pytest.param(
            chained_dependencies,
            af.pushforward,
            (("primal",), ("tangent",)),
            ("C: primal", "tangent"),
            id="chain-push",
        ),
        pytest.param(
            multiple_dependencies,
            af.pullback,
            (("primal",), "grad"),
            ("C: primal", ("grad",)),
            id="multiple-pull",
        ),
        pytest.param(
            chained_dependencies,
            af.pullback,
            (("primal",), "grad"),
            ("C: primal", ("grad",)),
            id="chain-pull",
        ),
        pytest.param(
            multiple_dependencies,
            af.batch,
            (["x", "y"],),
            ["C: x", "C: y"],
            id="multiple-batch",
        ),
        pytest.param(
            chained_dependencies,
            af.batch,
            (["x", "y"],),
            ["C: x", "C: y"],
            id="chain-batch",
        ),
    ],
)
def test_depends_shapes(program, transform, args, expected):
    assert transform(af.trace(program)("x")).call(*args) == expected


def test_sched_data_dependency():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        return af.string.format("B: {a_barrier}", a_barrier=af.depends(a))

    ir = af.sched(af.trace(program)("x"))
    assert sum(e.prim is depends_p for e in ir.eqns) == 1
    assert ir.call("hello") == "B: A: hello"


def test_sched_shared_dependency():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.string.format("B: {x}", x=x)
        return af.string.concat(a, af.depends(b, a))

    ir = af.sched(af.trace(program)("x"))
    assert sum(e.prim is depends_p for e in ir.eqns) == 1
    assert ir.call("hello") == "A: helloB: hello"
