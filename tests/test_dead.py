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
from autoform.core import IR, Var
from autoform.intercept import checkpoint_p
from autoform.order import depends_p, fanout_p
from tests import aexecute, execute, fixpoint_program, switch_program, while_program


def live_and_dead(x):
    af.string.concat(x, " DEAD")
    return af.string.concat(x, " LIVE")


def paired_outputs(x):
    return af.string.concat(x, "a"), af.string.concat(x, "b")


def pair_with_dead(x):
    result = paired_outputs(x)
    af.string.concat(x, "dead")
    return result


def carry_state(state, theta):
    visible, hidden = state
    return af.string.concat(visible, hidden), af.string.concat(hidden, theta)


def checkpoint_then_bang(x):
    x = af.checkpoint(x, key="save", collection="cache")
    return af.string.concat(x, "!")


def unused_checkpoint_switch_ir():
    def save(x):
        af.string.concat(x, "dead")
        return af.checkpoint(x, key="save", collection="cache")

    branch = af.trace(save)("x")

    def program(x):
        af.switch("a", {"a": branch}, af.string.concat(x, "!"))
        return x

    return af.trace(program)("x")


def test_removes_dead_code():
    ir = af.trace(live_and_dead)("test")
    dced = af.dce(ir)
    assert len(ir.eqns) == 2
    assert [e.prim for e in dced.eqns] == [af.string.concat_p]
    assert dced.call("x") == "x LIVE"


def test_removes_inlined_dead_code():
    inner = af.trace(live_and_dead)("test")
    ir = af.trace(lambda x: inner.call(x))("input")
    dced = af.dce(ir)
    assert len(ir.eqns) == 2
    assert [e.prim for e in dced.eqns] == [af.string.concat_p]
    assert dced.call("x") == "x LIVE"


@pytest.mark.parametrize(
    "program, before, after, expected",
    [
        pytest.param(lambda x: x, 0, 0, "x", id="empty"),
        pytest.param(
            lambda x: (af.string.concat(x, "dead1"), af.string.concat(x, "dead2"), x)[-1],
            2,
            0,
            "x",
            id="all-dead",
        ),
        pytest.param(
            lambda x: af.string.concat(af.stop_gradient(x), "!"),
            2,
            2,
            "x!",
            id="stop-gradient",
        ),
    ],
)
def test_dce_edges(program, before, after, expected):
    ir = af.trace(program)("x")
    dced = af.dce(ir)
    assert len(ir.eqns) == before
    assert len(dced.eqns) == after
    assert dced.call("x") == expected


def test_preserves_dependency_order():
    def program(x):
        y = af.string.concat(x, "a")
        z = af.string.concat(y, "b")
        return af.string.concat(z, "c")

    dced = af.dce(af.trace(program)("x"))
    assert len(dced.eqns) == 3
    for current, following in zip(dced.eqns, dced.eqns[1:]):
        assert current.out_tree in af.utils.tree.leaves(following.in_tree)
    assert dced.call("x") == "xabc"


def test_diamond_dependency():
    def program(x):
        a = af.string.concat(x, "a")
        return af.string.concat(af.string.concat(a, "b"), af.string.concat(a, "c"))

    dced = af.dce(af.trace(program)("x"))
    assert len(dced.eqns) == 4
    assert dced.call("x") == "xabxac"


@pytest.mark.parametrize(
    "used, count, expected",
    [pytest.param(True, 1, "", id="used"), pytest.param(False, 0, None, id="unused")],
)
def test_no_input_equation(used, count, expected):
    dced = af.dce(af.trace(lambda: af.string.concat())(), out_used=used)
    assert len(dced.eqns) == count
    assert dced.call() == expected


def test_dangling_used_output():
    bad_ir = IR([], in_tree=(), out_tree=Var.fresh(aval=af.string.StrAVal()))
    with pytest.raises(AssertionError):
        af.dce(bad_ir, out_used=True)


def test_preserves_used_switch():
    branches = {
        key: af.trace(lambda x: af.string.concat(x, suffix))("x")
        for key, suffix in (("a", " A"), ("b", " B"))
    }
    ir = af.trace(switch_program(branches))("a", "input")
    dced = af.dce(ir)
    assert [e.prim for e in dced.eqns] == [af.control.switch_p]
    assert dced.call("a", "input") == "input A"


def test_removes_unused_switch():
    branches = {
        key: af.trace(lambda x: af.string.concat(x, suffix))("x")
        for key, suffix in (("a", " A"), ("b", " B"))
    }

    def program(key, x):
        af.switch(key, branches, x)
        return af.string.concat(x, "live")

    ir = af.trace(program)("a", "input")
    dced = af.dce(ir)
    assert [e.prim for e in dced.eqns] == [af.string.concat_p]
    assert dced.call("a", "input") == "inputlive"


@pytest.mark.parametrize(
    "transform, mask, count, args, expected",
    [
        pytest.param(af.batch, None, 1, (["x"],), ["x LIVE"], id="batch"),
        pytest.param(af.pushforward, None, 1, (("x",), ("t",)), ("x LIVE", "t"), id="push"),
        pytest.param(
            af.pullback,
            (True, (False,)),
            1,
            (("x",), "cot"),
            ("x LIVE", ("cot",)),
            id="pull-primal",
        ),
        pytest.param(
            af.pullback,
            (False, (True,)),
            2,
            (("x",), "cot"),
            ("x LIVE", ("cot",)),
            id="pull-cotangent",
        ),
    ],
)
def test_dce_inside_transform(transform, mask, count, args, expected):
    inner = af.trace(live_and_dead)("test")
    dced = af.dce(transform(inner), out_used=mask)
    assert len(dced.eqns[0].params["ir"].eqns) == count
    assert len(inner.eqns) == 2
    assert dced.call(*args) == expected


def test_nested_switch():
    branch = af.trace(live_and_dead)("test")
    branches = {"a": branch, "b": af.trace(lambda x: af.string.concat(x, " B"))("test")}
    ir = af.trace(switch_program(branches))("a", "input")
    dced = af.dce(ir)
    assert len(dced.eqns[0].params["branches"]["a"].eqns) == 1
    assert dced.call("a", "hello") == "hello LIVE"


def test_batched_nested_switch():
    branch = af.trace(live_and_dead)("test")
    branches = {"a": branch, "b": af.trace(lambda x: af.string.concat(x, " B"))("test")}
    ir = af.trace(switch_program(branches))("a", "input")
    dced = af.dce(af.batch(ir, in_axes=(False, True)))
    inner = dced.eqns[0].params["ir"]
    assert len(inner.eqns[0].params["branches"]["a"].eqns) == 1
    assert dced.call("a", ["hello"]) == ["hello LIVE"]


def test_dce_inside_while_condition():
    def cond(state):
        af.string.concat(state, " DEAD")
        return af.string.match(state, "go")

    cond_ir = af.trace(cond)("go")
    body_ir = af.trace(lambda state: af.string.concat(state, "!"))("go")
    ir = af.trace(while_program(cond_ir, body_ir, max_iters=1))("go")
    dced = af.dce(ir)
    assert len(ir.eqns[0].params["cond_ir"].eqns) == 2
    assert len(dced.eqns[0].params["cond_ir"].eqns) == 1
    assert dced.call("go") == "go!"


def test_dce_inside_while_body():
    def body(state):
        af.string.concat(state, " DEAD")
        return af.string.concat(state, "!")

    cond_ir = af.trace(lambda state: af.string.match(state, "go"))("go")
    body_ir = af.trace(body)("go")
    ir = af.trace(while_program(cond_ir, body_ir, max_iters=1))("go")
    dced = af.dce(ir)
    assert len(ir.eqns[0].params["body_ir"].eqns) == 2
    assert len(dced.eqns[0].params["body_ir"].eqns) == 1
    assert dced.call("go") == "go!"


def test_dce_inside_fixpoint_step():
    def step(state, theta):
        af.string.concat(theta, " DEAD")
        return af.string.concat(state, theta)

    program = fixpoint_program(af.trace(step)("x", "!"), max_iters=1)
    ir = af.trace(program)("x", "!")
    dced = af.dce(ir)
    assert len(ir.eqns[0].params["step_ir"].eqns) == 2
    assert len(dced.eqns[0].params["step_ir"].eqns) == 1
    assert dced.call("x", "!") == "x!"


def test_dce_inside_fixpoint_equivalence():
    def stable(prev, new):
        af.string.concat(prev, " DEAD")
        return af.string.match(new, "x!")

    program = fixpoint_program(
        af.trace(af.string.concat)("x", "!"),
        max_iters=2,
        equiv_ir=af.trace(stable)("x", "x!"),
    )
    ir = af.trace(program)("x", "!")
    dced = af.dce(ir)
    assert len(ir.eqns[0].params["equiv_ir"].eqns) == 2
    assert len(dced.eqns[0].params["equiv_ir"].eqns) == 1
    assert dced.call("x", "!") == "x!"


def test_while_carried_state():
    cond = af.trace(lambda state: af.string.match(state[0], "v"))(("v", "h"))
    body = af.trace(lambda state: carry_state(state, "!"))(("v", "h"))
    program = while_program(cond, body, max_iters=1)
    dced = af.dce(af.trace(program)(("v", "h")), out_used=(True, False))
    assert dced.eqns[0].params["body_ir"].out_tree[1] is not None
    assert dced.call(("v", "h")) == ("vh", "h!")


def test_fixpoint_carried_state():
    program = fixpoint_program(af.trace(carry_state)(("v", "h"), "!"), max_iters=2)
    dced = af.dce(af.trace(program)(("v", "h"), "!"), out_used=(True, False))
    assert dced.eqns[0].params["step_ir"].out_tree[1] is not None
    assert dced.call(("v", "h"), "!") == ("vhh!", "h!!")


@pytest.mark.parametrize(
    "mask, expected, count",
    [
        pytest.param((True, True), ("Xa", "Xb"), 2, id="both"),
        pytest.param((True, False), ("Xa", None), 1, id="first"),
        pytest.param((False, True), (None, "Xb"), 1, id="second"),
        pytest.param((False, False), (None, None), 0, id="neither"),
    ],
)
def test_output_masks(mask, expected, count):
    ir = af.trace(paired_outputs)("x")
    dced = af.dce(ir, out_used=mask)
    assert len(dced.eqns) == count
    assert dced.call("X") == expected


@pytest.mark.parametrize(
    "mask, expected",
    [
        pytest.param((True, True), ("Xa", "Xb"), id="both"),
        pytest.param((True, False), ("Xa", None), id="first"),
        pytest.param((False, True), (None, "Xb"), id="second"),
    ],
)
def test_fanout_output_masks(mask, expected):
    ir = af.sched(af.trace(paired_outputs)("x"))
    dced = af.dce(ir, out_used=mask)
    assert [e.prim for e in dced.eqns] == [fanout_p]
    assert [len(branch.eqns) for branch in dced.eqns[0].params["irs"]] == list(map(int, mask))
    assert dced.call("X") == expected


def test_removes_fanout_with_unused_outputs():
    ir = af.sched(af.trace(paired_outputs)("x"))
    dced = af.dce(ir, out_used=(False, False))
    assert len(dced.eqns) == 0
    assert dced.call("X") == (None, None)


def test_mask_shared_dependency():
    def program(x):
        shared = af.string.concat(x, "shared")
        return af.string.concat(shared, "a"), af.string.concat(shared, "b")

    dced = af.dce(af.trace(program)("x"), out_used=(True, False))
    assert len(dced.eqns) == 2
    assert dced.call("x") == ("xshareda", None)


def test_batched_output_masks():
    inner = af.trace(pair_with_dead)("x")
    ir = af.batch(inner)
    dced = af.dce(ir, out_used=(True, True))
    nested = dced.eqns[0].params["ir"]
    assert len(inner.eqns) == 3
    assert len(nested.eqns) == 2
    assert dced.call(["x"]) == (["xa"], ["xb"])


def test_switch_output_masks():
    inner = af.trace(pair_with_dead)("x")
    ir = af.trace(lambda x: af.switch("a", {"a": inner, "b": inner}, x))("x")
    dced = af.dce(ir, out_used=(True, True))
    nested = dced.eqns[0].params["branches"]["a"]
    assert len(inner.eqns) == 3
    assert len(nested.eqns) == 2
    assert dced.call("x") == ("xa", "xb")


def test_unused_fanout_structured_branch():
    live = af.trace(lambda x: af.string.concat(x, "!"))("x")
    dead = af.trace(lambda x: (af.string.concat(x, "a"), af.string.concat(x, "b")))("x")

    def program(x):
        result = af.switch("live", {"live": live}, x)
        af.switch("dead", {"dead": dead}, x)
        return result

    dced = af.dce(af.sched(af.trace(program)("x")))
    assert dced.call("X") == "X!"
    inner = dced.eqns[0].params["irs"][1]
    assert not inner.eqns
    assert all(leaf is None for leaf in af.utils.tree.leaves(inner.out_tree))


def test_schedule_after_dce():
    def program(x):
        a = af.string.format("[{x}]", x=x)
        b = af.string.format("<{x}>", x=x)
        c = af.string.concat(a, b)
        af.string.format("dead: {x}", x=x)
        return c

    ir = af.trace(program)("x")
    result = af.sched(af.dce(ir))
    assert result.call("test") == "[test]<test>"
    assert [e.prim for e in result.eqns] == [fanout_p, af.string.concat_p]


def test_dce_after_schedule():
    def program(x):
        a = af.string.format("[{x}]", x=x)
        b = af.string.format("<{x}>", x=x)
        c = af.string.concat(a, b)
        af.string.format("dead: {x}", x=c)
        return c

    ir = af.trace(program)("x")
    result = af.dce(af.sched(ir))
    assert result.call("test") == "[test]<test>"
    assert [e.prim for e in result.eqns] == [fanout_p, af.string.concat_p]


def test_preserves_checkpoint():
    def program(x):
        af.checkpoint(x, key="save", collection="cache")
        return x

    dced = af.dce(af.trace(program)("test"))
    assert len(dced.eqns) == 1
    assert dced.eqns[-1].prim is checkpoint_p
    assert dced.eqns[-1].params["key"] == "save"
    with af.collect(collection="cache") as saved:
        assert dced.call("test") == "test"
    assert saved == {"save": ["test"]}


def test_preserves_checkpoint_input():
    def program(x):
        value = af.string.concat(x, "!")
        af.checkpoint(value, key="save", collection="cache")
        return x

    dced = af.dce(af.trace(program)("test"))
    assert len(dced.eqns) == 2
    assert dced.eqns[-1].prim is checkpoint_p
    assert dced.eqns[-1].params["key"] == "save"
    with af.collect(collection="cache") as saved:
        assert dced.call("test") == "test"
    assert saved == {"save": ["test!"]}


def test_removes_dead_code_around_checkpoint():
    def program(x):
        af.string.concat(x, "dead")
        af.checkpoint(x, key="save", collection="cache")
        return x

    dced = af.dce(af.trace(program)("test"))
    assert len(dced.eqns) == 1
    assert dced.eqns[-1].prim is checkpoint_p
    assert dced.eqns[-1].params["key"] == "save"
    with af.collect(collection="cache") as saved:
        assert dced.call("test") == "test"
    assert saved == {"save": ["test"]}


def test_preserves_used_dependency():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.string.format("B: {x}", x=x)
        return af.depends(b, a)

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 2
    assert sum(e.prim is depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "B: x"


def test_removes_unused_dependency():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.string.format("B: {x}", x=x)
        af.depends(b, a)
        return af.string.format("C: {x}", x=x)

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 1
    assert sum(e.prim is depends_p for e in dced.eqns) == 0
    assert dced.call("x") == "C: x"


def test_removes_all_unused_dependencies():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.string.format("B: {x}", x=x)
        af.depends(b, a)
        return x

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 0
    assert sum(e.prim is depends_p for e in dced.eqns) == 0
    assert dced.call("x") == "x"


def test_preserves_dependency_chain():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.depends(af.string.format("B: {x}", x=x), a)
        return af.depends(af.string.format("C: {x}", x=x), b)

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 3
    assert sum(e.prim is depends_p for e in dced.eqns) == 2
    assert dced.call("x") == "C: x"


def test_prunes_dependency_chain():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.depends(af.string.format("B: {x}", x=x), a)
        af.depends(af.string.format("C: {x}", x=x), b)
        return b

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 2
    assert sum(e.prim is depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "B: x"


def test_preserves_multiple_dependencies():
    def program(x):
        a = af.string.format("A: {x}", x=x)
        b = af.string.format("B: {x}", x=x)
        return af.depends(af.string.format("C: {x}", x=x), a, b)

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 3
    assert sum(e.prim is depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "C: x"


def test_preserves_barrier_without_dependencies():
    def program(x):
        return af.depends(af.string.format("A: {x}", x=x))

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 1
    assert sum(e.prim is depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "A: x"


def test_preserves_checkpoint_dependency():
    def program(x):
        a = af.checkpoint(x, key="a")
        return af.depends(af.string.format("B: {x}", x=x), a)

    dced = af.dce(af.trace(program)("x"))
    assert sum(e.prim is af.string.concat_p for e in dced.eqns) == 1
    assert sum(e.prim is depends_p for e in dced.eqns) == 1
    assert sum(e.prim is checkpoint_p for e in dced.eqns) == 1
    assert dced.call("x") == "B: x"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_pullback_preserves_cotangent_structure(executor):
    ir = af.trace(lambda x: (af.string.concat(x, "!"), af.string.concat(x, "?")))("x")
    pb = af.pullback(ir)
    dced = af.dce(pb, out_used=((True, False), (False,)))
    args = (("x",), ("g", "h"))
    expected = pb.call(*args)

    assert executor(dced, *args) == expected
    assert executor(af.dce(dced), *args) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_fanout_composes_with_dce_and_batch(executor):
    ir = af.sched(af.trace(lambda x: (af.string.concat(x, "!"), af.string.concat(x, "?")))("x"))
    dced = af.dce(ir, out_used=(True, False))

    assert executor(dced, "x") == ("x!", None)
    assert executor(af.dce(dced), "x") == ("x!", None)
    batched = af.batch(dced)
    assert executor(batched, ["x", "y"]) == (["x!", "y!"], None)
    assert ir.call("x") == ("x!", "x?")


def test_partial_fanout_masks_nested_outputs():
    inner = af.trace(lambda x: (x, af.string.concat(x, "!")))("x")
    ir = af.trace(lambda x: af.order.fanout_p.bind([(x,)], irs=[inner]))("x")
    dced = af.dce(ir, out_used=[(False, True)])

    assert af.dce(dced).call("x") == [(None, "x!")]


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_switch_keeps_branch_outputs_consistent(executor):
    branches = {
        "a": af.trace(lambda x: (af.string.concat(x, "!"), x))("x"),
        "b": af.trace(lambda x: (af.string.concat(x, "?"), af.string.concat(x, ".")))("x"),
    }
    ir = af.trace(lambda key, x: af.switch(key, branches, x))("a", "x")
    dced = af.dce(ir, out_used=(True, False))

    assert executor(dced, "a", "x") == ("x!", None)
    assert executor(af.dce(dced), "b", "x") == ("x?", None)
    batched = af.batch(dced)
    assert executor(batched, ["a", "b"], ["x", "y"]) == (["x!", "y?"], None)
    assert ir.call("b", "x") == ("x?", "x.")


def test_partial_switch_keeps_shared_dependencies():
    def branch(x):
        shared = af.string.concat(x, "!")
        return af.string.concat(shared, "?"), shared

    branch_ir = af.trace(branch)("x")
    ir = af.trace(lambda x: af.switch("a", {"a": branch_ir}, x))("x")
    dced = af.dce(ir, out_used=(True, False))

    assert dced.call("x") == ("x!?", None)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args",
    [
        pytest.param(af.batch, (["x"],), id="batch"),
        pytest.param(af.pushforward, (("x",), ("t",)), id="pushforward"),
        pytest.param(af.weight, ("x",), id="weight"),
    ],
)
def test_unused_wrapper_keeps_checkpoint(executor, transform, args):
    wrapped = transform(af.trace(checkpoint_then_bang)("x"))

    def program(*xs):
        wrapped.call(*xs)
        return "done"

    ir = af.trace(program)(*args)
    with af.collect(collection="cache") as expected:
        executor(ir, *args)
    assert expected
    dced = af.dce(ir)
    assert len(dced.eqns[0].params["ir"].eqns) == 1
    assert len(wrapped.eqns[0].params["ir"].eqns) == 2
    for candidate in (dced, af.dce(dced)):
        with af.collect(collection="cache") as saved:
            assert executor(candidate, *args) == "done"
        assert saved == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_unused_pullback_keeps_checkpoint(executor):
    wrapped = af.pullback(af.trace(checkpoint_then_bang)("x"))

    def program(*xs):
        wrapped.call(*xs)
        return "done"

    args = (("x",), "g")
    ir = af.trace(program)(*args)
    with af.collect(collection="cache") as expected:
        executor(ir, *args)
    assert expected
    dced = af.dce(ir)
    assert len(wrapped.eqns[0].params["ir"].eqns) == 2
    for candidate in (dced, af.dce(dced)):
        with af.collect(collection="cache") as saved:
            assert executor(candidate, *args) == "done"
        assert saved == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_batch_keeps_checkpoint_and_live_output(executor):
    def save(x):
        af.checkpoint(x, key="save", collection="cache")
        return af.string.concat(x, "!"), af.string.concat(x, "?")

    ir = af.batch(af.trace(save)("x"))
    dced = af.dce(ir, out_used=(True, False))
    assert dced.eqns[0].out_tree[0] is ir.eqns[0].out_tree[0]
    assert len(dced.eqns[0].params["ir"].eqns) == 2
    for candidate in (dced, af.dce(dced)):
        with af.collect(collection="cache") as saved:
            assert executor(candidate, ["x"]) == (["x!"], None)
            assert executor(candidate, ["y"]) == (["y!"], None)
        assert saved == {"save": ["x", "y"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_unused_switch_keeps_checkpoint(executor):
    dced = af.dce(unused_checkpoint_switch_ir())
    assert len(dced.eqns[-1].params["branches"]["a"].eqns) == 1
    for candidate in (dced, af.dce(dced)):
        with af.collect(collection="cache") as saved:
            assert executor(candidate, "x") == "x"
            assert executor(candidate, "y") == "y"
        assert saved == {"save": ["x!", "y!"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_unused_scheduled_switch_keeps_checkpoint(executor):
    dced = af.dce(af.sched(unused_checkpoint_switch_ir()))
    for candidate in (dced, af.dce(dced)):
        with af.collect(collection="cache") as saved:
            assert executor(candidate, "x") == "x"
            assert executor(candidate, "y") == "y"
        assert saved == {"save": ["x!", "y!"]}
