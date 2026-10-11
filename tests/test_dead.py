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

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.control as control
import autoform.src.dead as dead
import autoform.src.intercept as intercept
import autoform.src.order as order
import autoform.src.path as path
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import aexecute, execute, fixpoint_program, switch_program, while_program


def live_and_dead(x):
    string.concat(x, " DEAD")
    return string.concat(x, " LIVE")


def paired_outputs(x):
    return string.concat(x, "a"), string.concat(x, "b")


def pair_with_dead(x):
    result = paired_outputs(x)
    string.concat(x, "dead")
    return result


def carry_state(state, theta):
    visible, hidden = state
    return string.concat(visible, hidden), string.concat(hidden, theta)


def checkpoint_then_bang(x):
    x = intercept.checkpoint(x, key="save", collection="cache")
    return string.concat(x, "!")


def unused_checkpoint_switch_ir():
    def save(x):
        string.concat(x, "dead")
        return intercept.checkpoint(x, key="save", collection="cache")

    branch = stage.trace(save)("x")

    def program(x):
        control.switch("a", {"a": branch}, string.concat(x, "!"))
        return x

    return stage.trace(program)("x")


def test_removes_dead_code():
    ir = stage.trace(live_and_dead)("test")
    dced = dead.dce(ir)
    assert len(ir.eqns) == 2
    assert [e.prim for e in dced.eqns] == [string.concat_p]
    assert dced.call("x") == "x LIVE"


def test_removes_inlined_dead_code():
    inner = stage.trace(live_and_dead)("test")
    ir = stage.trace(lambda x: inner.call(x))("input")
    dced = dead.dce(ir)
    assert len(ir.eqns) == 2
    assert [e.prim for e in dced.eqns] == [string.concat_p]
    assert dced.call("x") == "x LIVE"


@pytest.mark.parametrize(
    "program, before, after, expected",
    [
        pytest.param(lambda x: x, 0, 0, "x", id="empty"),
        pytest.param(
            lambda x: (string.concat(x, "dead1"), string.concat(x, "dead2"), x)[-1],
            2,
            0,
            "x",
            id="all-dead",
        ),
        pytest.param(
            lambda x: string.concat(control.stop_gradient(x), "!"),
            2,
            2,
            "x!",
            id="stop-gradient",
        ),
    ],
)
def test_dce_edges(program, before, after, expected):
    ir = stage.trace(program)("x")
    dced = dead.dce(ir)
    assert len(ir.eqns) == before
    assert len(dced.eqns) == after
    assert dced.call("x") == expected


def test_preserves_dependency_order():
    def program(x):
        y = string.concat(x, "a")
        z = string.concat(y, "b")
        return string.concat(z, "c")

    dced = dead.dce(stage.trace(program)("x"))
    assert len(dced.eqns) == 3
    for current, following in zip(dced.eqns, dced.eqns[1:]):
        assert current.out_tree in utils.tree.leaves(following.in_tree)
    assert dced.call("x") == "xabc"


def test_diamond_dependency():
    def program(x):
        a = string.concat(x, "a")
        return string.concat(string.concat(a, "b"), string.concat(a, "c"))

    dced = dead.dce(stage.trace(program)("x"))
    assert len(dced.eqns) == 4
    assert dced.call("x") == "xabxac"


@pytest.mark.parametrize(
    "used, count, expected",
    [pytest.param(True, 1, "", id="used"), pytest.param(False, 0, None, id="unused")],
)
def test_no_input_equation(used, count, expected):
    dced = dead.dce(stage.trace(lambda: string.concat())(), out_used=used)
    assert len(dced.eqns) == count
    assert dced.call() == expected


def test_dangling_used_output():
    bad_ir = stage.IR([], in_tree=(), out_tree=stage.Var.fresh(aval=string.StrAVal()))
    with pytest.raises(AssertionError):
        dead.dce(bad_ir, out_used=True)


def test_preserves_used_switch():
    branches = {
        key: stage.trace(lambda x: string.concat(x, suffix))("x")
        for key, suffix in (("a", " A"), ("b", " B"))
    }
    ir = stage.trace(switch_program(branches))("a", "input")
    dced = dead.dce(ir)
    assert [e.prim for e in dced.eqns] == [control.switch_p]
    assert dced.call("a", "input") == "input A"


def test_removes_unused_switch():
    branches = {
        key: stage.trace(lambda x: string.concat(x, suffix))("x")
        for key, suffix in (("a", " A"), ("b", " B"))
    }

    def program(key, x):
        control.switch(key, branches, x)
        return string.concat(x, "live")

    ir = stage.trace(program)("a", "input")
    dced = dead.dce(ir)
    assert [e.prim for e in dced.eqns] == [string.concat_p]
    assert dced.call("a", "input") == "inputlive"


@pytest.mark.parametrize(
    "transform, mask, count, args, expected",
    [
        pytest.param(axis.batch, None, 1, (["x"],), ["x LIVE"], id="batch"),
        pytest.param(ad.pushforward, None, 1, (("x",), ("t",)), ("x LIVE", "t"), id="push"),
        pytest.param(
            ad.pullback,
            (True, (False,)),
            1,
            (("x",), "cot"),
            ("x LIVE", ("cot",)),
            id="pull-primal",
        ),
        pytest.param(
            ad.pullback,
            (False, (True,)),
            2,
            (("x",), "cot"),
            ("x LIVE", ("cot",)),
            id="pull-cotangent",
        ),
    ],
)
def test_dce_inside_transform(transform, mask, count, args, expected):
    inner = stage.trace(live_and_dead)("test")
    dced = dead.dce(transform(inner), out_used=mask)
    assert len(dced.eqns[0].params["ir"].eqns) == count
    assert len(inner.eqns) == 2
    assert dced.call(*args) == expected


def test_nested_switch():
    branch = stage.trace(live_and_dead)("test")
    branches = {"a": branch, "b": stage.trace(lambda x: string.concat(x, " B"))("test")}
    ir = stage.trace(switch_program(branches))("a", "input")
    dced = dead.dce(ir)
    assert len(dced.eqns[0].params["branches"]["a"].eqns) == 1
    assert dced.call("a", "hello") == "hello LIVE"


def test_batched_nested_switch():
    branch = stage.trace(live_and_dead)("test")
    branches = {"a": branch, "b": stage.trace(lambda x: string.concat(x, " B"))("test")}
    ir = stage.trace(switch_program(branches))("a", "input")
    dced = dead.dce(axis.batch(ir, in_axes=(False, True)))
    inner = dced.eqns[0].params["ir"]
    assert len(inner.eqns[0].params["branches"]["a"].eqns) == 1
    assert dced.call("a", ["hello"]) == ["hello LIVE"]


def test_dce_inside_while_condition():
    def cond(state):
        string.concat(state, " DEAD")
        return string.match(state, "go")

    cond_ir = stage.trace(cond)("go")
    body_ir = stage.trace(lambda state: string.concat(state, "!"))("go")
    ir = stage.trace(while_program(cond_ir, body_ir, max_iters=1))("go")
    dced = dead.dce(ir)
    assert len(ir.eqns[0].params["cond_ir"].eqns) == 2
    assert len(dced.eqns[0].params["cond_ir"].eqns) == 1
    assert dced.call("go") == "go!"


def test_dce_inside_while_body():
    def body(state):
        string.concat(state, " DEAD")
        return string.concat(state, "!")

    cond_ir = stage.trace(lambda state: string.match(state, "go"))("go")
    body_ir = stage.trace(body)("go")
    ir = stage.trace(while_program(cond_ir, body_ir, max_iters=1))("go")
    dced = dead.dce(ir)
    assert len(ir.eqns[0].params["body_ir"].eqns) == 2
    assert len(dced.eqns[0].params["body_ir"].eqns) == 1
    assert dced.call("go") == "go!"


def test_dce_inside_fixpoint_step():
    def step(state, theta):
        string.concat(theta, " DEAD")
        return string.concat(state, theta)

    program = fixpoint_program(stage.trace(step)("x", "!"), max_iters=1)
    ir = stage.trace(program)("x", "!")
    dced = dead.dce(ir)
    assert len(ir.eqns[0].params["step_ir"].eqns) == 2
    assert len(dced.eqns[0].params["step_ir"].eqns) == 1
    assert dced.call("x", "!") == "x!"


def test_dce_inside_fixpoint_equivalence():
    def stable(prev, new):
        string.concat(prev, " DEAD")
        return string.match(new, "x!")

    program = fixpoint_program(
        stage.trace(string.concat)("x", "!"),
        max_iters=2,
        equiv_ir=stage.trace(stable)("x", "x!"),
    )
    ir = stage.trace(program)("x", "!")
    dced = dead.dce(ir)
    assert len(ir.eqns[0].params["equiv_ir"].eqns) == 2
    assert len(dced.eqns[0].params["equiv_ir"].eqns) == 1
    assert dced.call("x", "!") == "x!"


def test_while_carried_state():
    cond = stage.trace(lambda state: string.match(state[0], "v"))(("v", "h"))
    body = stage.trace(lambda state: carry_state(state, "!"))(("v", "h"))
    program = while_program(cond, body, max_iters=1)
    dced = dead.dce(stage.trace(program)(("v", "h")), out_used=(True, False))
    assert dced.eqns[0].params["body_ir"].out_tree[1] is not None
    assert dced.call(("v", "h")) == ("vh", "h!")


def test_fixpoint_carried_state():
    program = fixpoint_program(stage.trace(carry_state)(("v", "h"), "!"), max_iters=2)
    dced = dead.dce(stage.trace(program)(("v", "h"), "!"), out_used=(True, False))
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
    ir = stage.trace(paired_outputs)("x")
    dced = dead.dce(ir, out_used=mask)
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
    ir = order.sched(stage.trace(paired_outputs)("x"))
    dced = dead.dce(ir, out_used=mask)
    assert [e.prim for e in dced.eqns] == [order.fanout_p]
    assert [len(branch.eqns) for branch in dced.eqns[0].params["irs"]] == list(map(int, mask))
    assert dced.call("X") == expected


def test_removes_fanout_with_unused_outputs():
    ir = order.sched(stage.trace(paired_outputs)("x"))
    dced = dead.dce(ir, out_used=(False, False))
    assert len(dced.eqns) == 0
    assert dced.call("X") == (None, None)


def test_mask_shared_dependency():
    def program(x):
        shared = string.concat(x, "shared")
        return string.concat(shared, "a"), string.concat(shared, "b")

    dced = dead.dce(stage.trace(program)("x"), out_used=(True, False))
    assert len(dced.eqns) == 2
    assert dced.call("x") == ("xshareda", None)


def test_batched_output_masks():
    inner = stage.trace(pair_with_dead)("x")
    ir = axis.batch(inner)
    dced = dead.dce(ir, out_used=(True, True))
    nested = dced.eqns[0].params["ir"]
    assert len(inner.eqns) == 3
    assert len(nested.eqns) == 2
    assert dced.call(["x"]) == (["xa"], ["xb"])


def test_switch_output_masks():
    inner = stage.trace(pair_with_dead)("x")
    ir = stage.trace(lambda x: control.switch("a", {"a": inner, "b": inner}, x))("x")
    dced = dead.dce(ir, out_used=(True, True))
    nested = dced.eqns[0].params["branches"]["a"]
    assert len(inner.eqns) == 3
    assert len(nested.eqns) == 2
    assert dced.call("x") == ("xa", "xb")


def test_unused_fanout_structured_branch():
    live = stage.trace(lambda x: string.concat(x, "!"))("x")
    dead_ir = stage.trace(lambda x: (string.concat(x, "a"), string.concat(x, "b")))("x")

    def program(x):
        result = control.switch("live", {"live": live}, x)
        control.switch("dead", {"dead": dead_ir}, x)
        return result

    dced = dead.dce(order.sched(stage.trace(program)("x")))
    assert dced.call("X") == "X!"
    inner = dced.eqns[0].params["irs"][1]
    assert not inner.eqns
    assert all(leaf is None for leaf in utils.tree.leaves(inner.out_tree))


def test_schedule_after_dce():
    def program(x):
        a = string.format("[{x}]", x=x)
        b = string.format("<{x}>", x=x)
        c = string.concat(a, b)
        string.format("dead: {x}", x=x)
        return c

    ir = stage.trace(program)("x")
    result = order.sched(dead.dce(ir))
    assert result.call("test") == "[test]<test>"
    assert [e.prim for e in result.eqns] == [order.fanout_p, string.concat_p]


def test_dce_after_schedule():
    def program(x):
        a = string.format("[{x}]", x=x)
        b = string.format("<{x}>", x=x)
        c = string.concat(a, b)
        string.format("dead: {x}", x=c)
        return c

    ir = stage.trace(program)("x")
    result = dead.dce(order.sched(ir))
    assert result.call("test") == "[test]<test>"
    assert [e.prim for e in result.eqns] == [order.fanout_p, string.concat_p]


def test_preserves_checkpoint():
    def program(x):
        intercept.checkpoint(x, key="save", collection="cache")
        return x

    dced = dead.dce(stage.trace(program)("test"))
    assert len(dced.eqns) == 1
    assert dced.eqns[-1].prim is intercept.checkpoint_p
    assert dced.eqns[-1].params["key"] == "save"
    with intercept.collect(collection="cache") as saved:
        assert dced.call("test") == "test"
    assert saved == {"save": ["test"]}


def test_preserves_checkpoint_input():
    def program(x):
        value = string.concat(x, "!")
        intercept.checkpoint(value, key="save", collection="cache")
        return x

    dced = dead.dce(stage.trace(program)("test"))
    assert len(dced.eqns) == 2
    assert dced.eqns[-1].prim is intercept.checkpoint_p
    assert dced.eqns[-1].params["key"] == "save"
    with intercept.collect(collection="cache") as saved:
        assert dced.call("test") == "test"
    assert saved == {"save": ["test!"]}


def test_removes_dead_code_around_checkpoint():
    def program(x):
        string.concat(x, "dead")
        intercept.checkpoint(x, key="save", collection="cache")
        return x

    dced = dead.dce(stage.trace(program)("test"))
    assert len(dced.eqns) == 1
    assert dced.eqns[-1].prim is intercept.checkpoint_p
    assert dced.eqns[-1].params["key"] == "save"
    with intercept.collect(collection="cache") as saved:
        assert dced.call("test") == "test"
    assert saved == {"save": ["test"]}


def test_preserves_used_dependency():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = string.format("B: {x}", x=x)
        return order.depends(b, a)

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 2
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "B: x"


def test_removes_unused_dependency():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = string.format("B: {x}", x=x)
        order.depends(b, a)
        return string.format("C: {x}", x=x)

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 1
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 0
    assert dced.call("x") == "C: x"


def test_removes_all_unused_dependencies():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = string.format("B: {x}", x=x)
        order.depends(b, a)
        return x

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 0
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 0
    assert dced.call("x") == "x"


def test_preserves_dependency_chain():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = order.depends(string.format("B: {x}", x=x), a)
        return order.depends(string.format("C: {x}", x=x), b)

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 3
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 2
    assert dced.call("x") == "C: x"


def test_prunes_dependency_chain():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = order.depends(string.format("B: {x}", x=x), a)
        order.depends(string.format("C: {x}", x=x), b)
        return b

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 2
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "B: x"


def test_preserves_multiple_dependencies():
    def program(x):
        a = string.format("A: {x}", x=x)
        b = string.format("B: {x}", x=x)
        return order.depends(string.format("C: {x}", x=x), a, b)

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 3
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "C: x"


def test_preserves_barrier_without_dependencies():
    def program(x):
        return order.depends(string.format("A: {x}", x=x))

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 1
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 1
    assert dced.call("x") == "A: x"


def test_preserves_checkpoint_dependency():
    def program(x):
        a = intercept.checkpoint(x, key="a")
        return order.depends(string.format("B: {x}", x=x), a)

    dced = dead.dce(stage.trace(program)("x"))
    assert sum(e.prim is string.concat_p for e in dced.eqns) == 1
    assert sum(e.prim is order.depends_p for e in dced.eqns) == 1
    assert sum(e.prim is intercept.checkpoint_p for e in dced.eqns) == 1
    assert dced.call("x") == "B: x"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_pullback_preserves_cotangent_structure(executor):
    ir = stage.trace(lambda x: (string.concat(x, "!"), string.concat(x, "?")))("x")
    pb = ad.pullback(ir)
    dced = dead.dce(pb, out_used=((True, False), (False,)))
    args = (("x",), ("g", "h"))
    expected = pb.call(*args)

    assert executor(dced, *args) == expected
    assert executor(dead.dce(dced), *args) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_fanout_composes_with_dce_and_batch(executor):
    ir = order.sched(stage.trace(lambda x: (string.concat(x, "!"), string.concat(x, "?")))("x"))
    dced = dead.dce(ir, out_used=(True, False))

    assert executor(dced, "x") == ("x!", None)
    assert executor(dead.dce(dced), "x") == ("x!", None)
    batched = axis.batch(dced)
    assert executor(batched, ["x", "y"]) == (["x!", "y!"], None)
    assert ir.call("x") == ("x!", "x?")


def test_partial_fanout_masks_nested_outputs():
    inner = stage.trace(lambda x: (x, string.concat(x, "!")))("x")
    ir = stage.trace(lambda x: order.fanout_p.bind([(x,)], irs=[inner]))("x")
    dced = dead.dce(ir, out_used=[(False, True)])

    assert dead.dce(dced).call("x") == [(None, "x!")]


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_switch_keeps_branch_outputs_consistent(executor):
    branches = {
        "a": stage.trace(lambda x: (string.concat(x, "!"), x))("x"),
        "b": stage.trace(lambda x: (string.concat(x, "?"), string.concat(x, ".")))("x"),
    }
    ir = stage.trace(lambda key, x: control.switch(key, branches, x))("a", "x")
    dced = dead.dce(ir, out_used=(True, False))

    assert executor(dced, "a", "x") == ("x!", None)
    assert executor(dead.dce(dced), "b", "x") == ("x?", None)
    batched = axis.batch(dced)
    assert executor(batched, ["a", "b"], ["x", "y"]) == (["x!", "y?"], None)
    assert ir.call("b", "x") == ("x?", "x.")


def test_partial_switch_keeps_shared_dependencies():
    def branch(x):
        shared = string.concat(x, "!")
        return string.concat(shared, "?"), shared

    branch_ir = stage.trace(branch)("x")
    ir = stage.trace(lambda x: control.switch("a", {"a": branch_ir}, x))("x")
    dced = dead.dce(ir, out_used=(True, False))

    assert dced.call("x") == ("x!?", None)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args",
    [
        pytest.param(axis.batch, (["x"],), id="batch"),
        pytest.param(ad.pushforward, (("x",), ("t",)), id="pushforward"),
        pytest.param(path.weight, ("x",), id="weight"),
    ],
)
def test_unused_wrapper_keeps_checkpoint(executor, transform, args):
    wrapped = transform(stage.trace(checkpoint_then_bang)("x"))

    def program(*xs):
        wrapped.call(*xs)
        return "done"

    ir = stage.trace(program)(*args)
    with intercept.collect(collection="cache") as expected:
        executor(ir, *args)
    assert expected
    dced = dead.dce(ir)
    assert len(dced.eqns[0].params["ir"].eqns) == 1
    assert len(wrapped.eqns[0].params["ir"].eqns) == 2
    for candidate in (dced, dead.dce(dced)):
        with intercept.collect(collection="cache") as saved:
            assert executor(candidate, *args) == "done"
        assert saved == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_unused_pullback_keeps_checkpoint(executor):
    wrapped = ad.pullback(stage.trace(checkpoint_then_bang)("x"))

    def program(*xs):
        wrapped.call(*xs)
        return "done"

    args = (("x",), "g")
    ir = stage.trace(program)(*args)
    with intercept.collect(collection="cache") as expected:
        executor(ir, *args)
    assert expected
    dced = dead.dce(ir)
    assert len(wrapped.eqns[0].params["ir"].eqns) == 2
    for candidate in (dced, dead.dce(dced)):
        with intercept.collect(collection="cache") as saved:
            assert executor(candidate, *args) == "done"
        assert saved == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_partial_batch_keeps_checkpoint_and_live_output(executor):
    def save(x):
        intercept.checkpoint(x, key="save", collection="cache")
        return string.concat(x, "!"), string.concat(x, "?")

    ir = axis.batch(stage.trace(save)("x"))
    dced = dead.dce(ir, out_used=(True, False))
    assert dced.eqns[0].out_tree[0] is ir.eqns[0].out_tree[0]
    assert len(dced.eqns[0].params["ir"].eqns) == 2
    for candidate in (dced, dead.dce(dced)):
        with intercept.collect(collection="cache") as saved:
            assert executor(candidate, ["x"]) == (["x!"], None)
            assert executor(candidate, ["y"]) == (["y!"], None)
        assert saved == {"save": ["x", "y"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_unused_switch_keeps_checkpoint(executor):
    dced = dead.dce(unused_checkpoint_switch_ir())
    assert len(dced.eqns[-1].params["branches"]["a"].eqns) == 1
    for candidate in (dced, dead.dce(dced)):
        with intercept.collect(collection="cache") as saved:
            assert executor(candidate, "x") == "x"
            assert executor(candidate, "y") == "y"
        assert saved == {"save": ["x!", "y!"]}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_unused_scheduled_switch_keeps_checkpoint(executor):
    dced = dead.dce(order.sched(unused_checkpoint_switch_ir()))
    for candidate in (dced, dead.dce(dced)):
        with intercept.collect(collection="cache") as saved:
            assert executor(candidate, "x") == "x"
            assert executor(candidate, "y") == "y"
        assert saved == {"save": ["x!", "y!"]}
