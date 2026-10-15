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

import autoform.extend as extend
import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.control as control
import autoform.src.core as core
import autoform.src.intercept as intercept
import autoform.src.order as order
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import aexecute, dependent_formats, execute, switch_program


def parallel_formats(*, joined=False):
    def program(x):
        a = string.format("[{x}]", x=x)
        b = string.format("<{x}>", x=x)
        return string.concat(a, b) if joined else (a, b)

    return program


def format_switch_ir():
    branches = {
        "x": stage.trace(parallel_formats(joined=True))("x"),
        "y": stage.trace(lambda x: string.format("({x})", x=x))("x"),
    }
    return stage.trace(switch_program(branches))("x", "inp")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "joined, transform, args, expected",
    [
        pytest.param(False, lambda ir: ir, ("A",), ("[A]", "<A>"), id="fanout"),
        pytest.param(True, lambda ir: ir, ("test",), "[test]<test>", id="joined"),
        pytest.param(
            False,
            ad.pushforward,
            (("primal",), ("tangent",)),
            (("[primal]", "<primal>"), ("tangent", "tangent")),
            id="push",
        ),
        pytest.param(
            False,
            ad.pullback,
            (("primal",), ("grad1", "grad2")),
            (("[primal]", "<primal>"), ("grad1grad2",)),
            id="pull",
        ),
        pytest.param(
            False,
            axis.batch,
            (["A", "B", "C"],),
            (["[A]", "[B]", "[C]"], ["<A>", "<B>", "<C>"]),
            id="batch",
        ),
        pytest.param(
            True,
            ad.pushforward,
            (("test",), ("tangent",)),
            ("[test]<test>", "tangenttangent"),
            id="joined-push",
        ),
        pytest.param(
            True,
            ad.pullback,
            (("test",), "grad"),
            ("[test]<test>", ("gradgrad",)),
            id="joined-pull",
        ),
        pytest.param(
            True,
            axis.batch,
            (["A", "B", "C"],),
            ["[A]<A>", "[B]<B>", "[C]<C>"],
            id="joined-batch",
        ),
    ],
)
def test_fanout_transforms(executor, joined, transform, args, expected):
    scheduled = order.sched(stage.trace(parallel_formats(joined=joined))("a"))
    assert [e.prim for e in scheduled.eqns] == [order.p_fanout] + (
        [string.p_concat] if joined else []
    )
    ir = transform(scheduled)
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("out_c", [(1.0, 0.0), (2.0, 3.0)], ids=["one-live", "both-live"])
def test_fanout_pullback_repeated_operand(executor, out_c):
    ir = stage.trace(lambda x: (x * x, x + 1.0))(3.0)
    scheduled = order.sched(ir)
    assert [e.prim for e in scheduled.eqns] == [order.p_fanout]
    expected = ((9.0, 4.0), (6.0 * out_c[0] + out_c[1],))
    assert ad.pullback(ir).call((3.0,), out_c) == expected
    pb = ad.pullback(scheduled)
    assert executor(pb, (3.0,), out_c) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_fanout_exception(executor):

    def impl_error(x):
        raise ValueError("intentional error")

    p_error = core.Prim("error")
    extend.r_impl.set(p_error, impl_error)
    extend.r_abstract.set(p_error, lambda x: string.AValStr())
    extend.r_aimpl.set(p_error, utils.asyncify(impl_error))
    ir = order.sched(stage.trace(lambda x: (string.format("[{x}]", x=x), p_error.bind(x)))("a"))
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
    ir = stage.trace(lambda x, y: (string.format("[{x}]", x=x), string.format("<{y}>", y=y)))(
        "x",
        "y",
    )
    ir = axis.batch(order.sched(ir), in_axes=(True, False))
    result = executor(ir, values, "STATIC")
    assert result == (expected, ["<STATIC>"] * len(values))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "program, prims, expected",
    [
        pytest.param(
            lambda x: string.format("[{x}]", x=x),
            [string.p_concat],
            "[test]",
            id="single",
        ),
        pytest.param(
            lambda x: string.concat(string.format("[{x}]", x=x), "!"),
            [string.p_concat] * 2,
            "[test]!",
            id="sequential",
        ),
        pytest.param(
            lambda x: tuple((string.format(f, x=x) for f in ("[{x}]", "<{x}>", "{{{x}}}"))),
            [order.p_fanout],
            ("[test]", "<test>", "{test}"),
            id="three-independent",
        ),
    ],
)
def test_sched_levels(executor, program, prims, expected):
    ir = order.sched(stage.trace(program)("x"))
    assert [e.prim for e in ir.eqns] == prims
    actual = executor(ir, "test")
    assert actual == expected


def test_sched_filter():
    ir = stage.trace(lambda x: (string.format("[{x}]", x=x), string.concat(x, "!")))("x")
    scheduled = order.sched(ir, cond=lambda e: e.prim is string.p_concat and len(e.in_tree) == 3)
    assert [e.prim for e in scheduled.eqns] == [string.p_concat] * 2
    assert scheduled.call("test") == ("[test]", "test!")


def test_sched_filter_propagates_into_switch():
    inner = stage.trace(lambda x: (string.format("[{x}]", x=x), string.concat(x, "!")))("x")
    ir = stage.trace(lambda x: control.switch("a", {"a": inner}, x))("x")
    scheduled = order.sched(ir, cond=lambda e: e.prim is string.p_concat and len(e.in_tree) == 3)
    checked = scheduled.eqns[0].params["branches"]["a"]
    assert [e.prim for e in checked.eqns] == [string.p_concat] * 2
    assert scheduled.call("test") == ("[test]", "test!")


def test_sched_parallel_checkpoints():
    def program(a, b):
        x = intercept.checkpoint(string.format("{a}", a=a), key="x")
        y = intercept.checkpoint(string.format("{b}", b=b), key="y")
        return x, y

    ir = order.sched(stage.trace(program)("a", "b"))
    assert sum(e.prim is order.p_fanout for e in ir.eqns) == 2
    assert ir.call("hello", "world") == ("hello", "world")


def test_sched_dependent_checkpoints():
    def program(a, b):
        x = intercept.checkpoint(string.format("{a}", a=a), key="x")
        y = intercept.checkpoint(string.format("{b}", b=b), key="y")
        return order.depends(y, x)

    ir = order.sched(stage.trace(program)("a", "b"))
    assert sum(e.prim is order.p_fanout for e in ir.eqns) == 2
    assert ir.call("hello", "world") == "world"


def test_sched_mixed_checkpoint():
    def program(a, b, c):
        x = string.format("[{a}]", a=a)
        y = string.format("<{b}>", b=b)
        z = intercept.checkpoint(string.format("{{{c}}}", c=c), key="z")
        return x, y, z

    ir = order.sched(stage.trace(program)("a", "b", "c"))
    assert sum(e.prim is order.p_fanout for e in ir.eqns) == 1
    assert ir.call("a", "b", "c") == ("[a]", "<b>", "{c}")


def test_fanout_collect():
    def program(a, b):
        return (
            intercept.checkpoint(a, key="val", collection="debug"),
            intercept.checkpoint(b, key="val", collection="debug"),
        )

    ir = order.sched(stage.trace(program)("a", "b"))
    with intercept.collect(collection="debug") as collected:
        assert ir.call("A", "B") == ("A", "B")
    assert set(collected["val"]) == {"A", "B"}


def test_fanout_inject():
    def program(a, b):
        x = intercept.checkpoint(string.format("[{a}]", a=a), key="val", collection="cache")
        y = intercept.checkpoint(string.format("<{b}>", b=b), key="val", collection="cache")
        return x, y

    ir = order.sched(stage.trace(program)("a", "b"))
    with intercept.inject(collection="cache", values={"val": ["CACHED1", "CACHED2"]}):
        assert ir.call("A", "B") == ("CACHED1", "CACHED2")


def test_nested_fanout_collect():
    def inner(x):
        return intercept.checkpoint(x, key="inner", collection="debug")

    branches = {
        key: order.sched(stage.trace(lambda x: (inner(x), inner(x)))("x")) for key in ("a", "b")
    }

    def outer(key, x):
        left, right = control.switch(key, branches, x)
        return string.concat(left, right)

    ir = order.sched(stage.trace(outer)("a", "x"))
    with intercept.collect(collection="debug") as collected:
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
    scheduled = order.sched(format_switch_ir())
    assert scheduled.eqns[0].params["branches"]["x"].eqns[0].prim is order.p_fanout
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
        "B": stage.trace(lambda key, inp: string.format("{key} {inp}", key=key, inp=inp))("k", "i"),
    }
    ir = stage.trace(lambda key, inner_key, x: control.switch(key, branches, inner_key, x))(
        "A",
        "x",
        "test",
    )
    scheduled = order.sched(ir)
    inner = scheduled.eqns[0].params["branches"]["A"]
    assert inner.eqns[0].params["branches"]["x"].eqns[0].prim is order.p_fanout
    actual = executor(scheduled, *args)
    assert actual == expected


def test_sched_nested_fanout():
    branches1 = {"a": stage.trace(parallel_formats(joined=True))("x")}
    branches2 = {
        "a": stage.trace(
            lambda x: string.concat(
                string.format("({x})", x=x),
                string.format("{{{x}}}", x=x),
            ),
        )("x")
    }
    ir = stage.trace(
        lambda key, a, b: (control.switch(key, branches1, a), control.switch(key, branches2, b))
    )(
        "a",
        "hello",
        "world",
    )
    scheduled = order.sched(ir)
    assert scheduled.eqns[0].prim is order.p_fanout
    for inner in scheduled.eqns[0].params["irs"]:
        assert inner.eqns[0].params["branches"]["a"].eqns[0].prim is order.p_fanout
    assert scheduled.call("a", "hello", "world") == ("[hello]<hello>", "(world){world}")


def no_dependencies(x):
    return order.depends(string.format("A: {x}", x=x))


def chained_dependencies(x):
    a = string.format("A: {x}", x=x)
    b = string.format("B: {x}", x=x)
    c = string.format("C: {x}", x=x)
    return order.depends(c, order.depends(b, a))


def multiple_dependencies(x):
    a = string.format("A: {x}", x=x)
    b = string.format("B: {x}", x=x)
    c = string.format("C: {x}", x=x)
    return order.depends(c, a, b)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected",
    [
        pytest.param(lambda ir: ir, ("hello",), "B: hello", id="call"),
        pytest.param(order.sched, ("hello",), "B: hello", id="sched"),
        pytest.param(
            ad.pushforward,
            (("primal",), ("tangent",)),
            ("B: primal", "tangent"),
            id="push",
        ),
        pytest.param(ad.pullback, (("primal",), "grad"), ("B: primal", ("grad",)), id="pull"),
        pytest.param(axis.batch, (["x", "y", "z"],), ["B: x", "B: y", "B: z"], id="batch"),
        pytest.param(
            lambda ir: axis.batch(ad.pushforward(ir), in_axes=(True, True)),
            ((["a", "b"],), (["da", "db"],)),
            (["B: a", "B: b"], ["da", "db"]),
            id="batch-push",
        ),
        pytest.param(
            lambda ir: axis.batch(ad.pullback(ir), in_axes=(True, True)),
            ((["a", "b"],), ["g1", "g2"]),
            (["B: a", "B: b"], (["g1", "g2"],)),
            id="batch-pull",
        ),
        pytest.param(
            lambda ir: ad.pushforward(axis.batch(ir)),
            ((["a", "b"],), (["da", "db"],)),
            (["B: a", "B: b"], ["da", "db"]),
            id="push-batch",
        ),
        pytest.param(
            lambda ir: ad.pullback(axis.batch(ir)),
            ((["a", "b"],), ["g1", "g2"]),
            (["B: a", "B: b"], (["g1", "g2"],)),
            id="pull-batch",
        ),
        pytest.param(
            lambda ir: order.sched(ad.pushforward(ir)),
            (("primal",), ("tangent",)),
            ("B: primal", "tangent"),
            id="sched-push",
        ),
        pytest.param(
            lambda ir: order.sched(ad.pullback(ir)),
            (("primal",), "grad"),
            ("B: primal", ("grad",)),
            id="sched-pull",
        ),
        pytest.param(
            lambda ir: order.sched(axis.batch(ir)),
            (["x", "y", "z"],),
            ["B: x", "B: y", "B: z"],
            id="sched-batch",
        ),
    ],
)
def test_depends_transforms(executor, transform, args, expected):
    ir = stage.trace(dependent_formats)("x")
    [barrier] = [e for e in ir.eqns if e.prim is order.p_depends]
    assert len(utils.tree.leaves(barrier.in_tree)) == 2
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
            ad.pushforward,
            (("primal",), ("tangent",)),
            ("C: primal", "tangent"),
            id="multiple-push",
        ),
        pytest.param(
            chained_dependencies,
            ad.pushforward,
            (("primal",), ("tangent",)),
            ("C: primal", "tangent"),
            id="chain-push",
        ),
        pytest.param(
            multiple_dependencies,
            ad.pullback,
            (("primal",), "grad"),
            ("C: primal", ("grad",)),
            id="multiple-pull",
        ),
        pytest.param(
            chained_dependencies,
            ad.pullback,
            (("primal",), "grad"),
            ("C: primal", ("grad",)),
            id="chain-pull",
        ),
        pytest.param(
            multiple_dependencies,
            axis.batch,
            (["x", "y"],),
            ["C: x", "C: y"],
            id="multiple-batch",
        ),
        pytest.param(
            chained_dependencies,
            axis.batch,
            (["x", "y"],),
            ["C: x", "C: y"],
            id="chain-batch",
        ),
    ],
)
def test_depends_shapes(program, transform, args, expected):
    assert transform(stage.trace(program)("x")).call(*args) == expected


def test_sched_data_dependency():
    def program(x):
        a = string.format("A: {x}", x=x)
        return string.format("B: {a_barrier}", a_barrier=order.depends(a))

    ir = order.sched(stage.trace(program)("x"))
    assert sum(e.prim is order.p_depends for e in ir.eqns) == 1
    assert ir.call("hello") == "B: A: hello"


def test_sched_shared_dependency():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = string.format("B: {x}", x=x)
        return string.concat(a, order.depends(b, a))

    ir = order.sched(stage.trace(program)("x"))
    assert sum(e.prim is order.p_depends for e in ir.eqns) == 1
    assert ir.call("hello") == "A: helloB: hello"
