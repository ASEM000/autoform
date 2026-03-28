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
from autoform.core import IR, IRVar


class TestDCE:
    def test_removes_unused_equation(self):
        def program(x):
            dead = af.concat(x, "dead")
            live = af.concat(x, "live")
            return live

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 2
        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].prim.name == "concat"

    def test_keeps_chained_dependencies(self):
        def program(x):
            y = af.concat(x, "a")
            z = af.concat(y, "b")
            return z

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 2
        assert len(dce.ir_eqns) == 2

    def test_preserves_equation_order(self):
        def program(x):
            y = af.concat(x, "a")
            z = af.concat(y, "b")
            w = af.concat(z, "c")
            return w

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(dce.ir_eqns) == 3

        for i in range(len(dce.ir_eqns) - 1):
            curr_out = dce.ir_eqns[i].out_ir_tree
            next_in_leaves = af.utils.treelib.leaves(dce.ir_eqns[i + 1].in_ir_tree)
            assert curr_out in next_in_leaves

    def test_errors_on_dangling_used_output_irvar(self):
        dangling = IRVar.fresh(type=str)
        bad_ir = IR([], in_ir_tree=(), out_ir_tree=dangling)

        with pytest.raises(AssertionError):
            af.dce(bad_ir, out_used=True)

    def test_keeps_all_if_all_used(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == len(dce.ir_eqns)

    def test_multiple_outputs_partial_use(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            c = af.concat(a, "c")
            return c

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 3
        assert len(dce.ir_eqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ir_eqns]
        assert prim_names == ["concat", "concat"]


class TestDCEWithHigherOrderPrimitives:
    def test_run_ir_inlines_for_dce(self):
        inner_ir = af.trace(lambda x: af.concat(x, "!"))("x")

        def program(x):
            return inner_ir.call(x)

        ir = af.trace(program)("input")
        dce = af.dce(ir)

        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].prim.name == "concat"

    def test_inlined_dead_code_removed(self):
        def inner(x):
            dead = af.concat(x, "dead")
            live = af.concat(x, "live")
            return live

        inner_ir = af.trace(inner)("x")

        def program(x):
            return inner_ir.call(x)

        ir = af.trace(program)("input")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 2
        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].prim.name == "concat"

    def test_switch_kept_when_used(self):
        branches = {
            "a": af.trace(lambda x: af.concat(x, " A"))("x"),
            "b": af.trace(lambda x: af.concat(x, " B"))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "input")
        dce = af.dce(ir)

        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].prim.name == "switch"

    def test_switch_removed_when_unused(self):
        branches = {
            "a": af.trace(lambda x: af.concat(x, " A"))("x"),
            "b": af.trace(lambda x: af.concat(x, " B"))("x"),
        }

        def program(key, x):
            dead = af.switch(key, branches, x)
            live = af.concat(x, "live")
            return live

        ir = af.trace(program)("a", "input")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 2
        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].prim.name == "concat"


class TestDCEWithTransformedIR:
    def test_dce_on_pushforward(self):
        def program(x):
            y = af.concat(x, "a")
            dead = af.concat(x, "dead")
            return y

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        dce = af.dce(pf_ir)

        assert len(dce.ir_eqns) <= len(pf_ir.ir_eqns)

    def test_dce_on_pullback(self):
        def program(x):
            y = af.concat(x, "a")
            dead = af.concat(x, "dead")
            return y

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        dce = af.dce(pb_ir)

        assert len(dce.ir_eqns) <= len(pb_ir.ir_eqns)

    def test_dce_on_batch(self):
        def program(x):
            y = af.concat(x, "a")
            dead = af.concat(x, "dead")
            return y

        ir = af.trace(program)("x")
        batch = af.batch(ir, in_axes=True)
        dce = af.dce(batch)

        assert len(dce.ir_eqns) <= len(batch.ir_eqns)


class TestDCEEdgeCases:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 0
        assert len(dce.ir_eqns) == 0

    def test_all_dead(self):
        def program(x):
            dead1 = af.concat(x, "dead1")
            dead2 = af.concat(x, "dead2")
            return x

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 2
        assert len(dce.ir_eqns) == 0

    def test_diamond_dependency(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(a, "b")
            c = af.concat(a, "c")
            d = af.concat(b, c)
            return d

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(ir.ir_eqns) == 4
        assert len(dce.ir_eqns) == 4

    def test_stop_gradient_kept(self):
        def program(x):
            y = af.stop_gradient(x)
            z = af.concat(y, "!")
            return z

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(dce.ir_eqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ir_eqns]
        assert prim_names == ["stop_gradient", "concat"]


class TestNestedDCE:
    def test_switch_dces_inner_branches(self):
        def branch_a_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        branch_a = af.trace(branch_a_fn)("test")
        branch_b = af.trace(lambda x: af.concat(x, " B"))("test")

        assert len(branch_a.ir_eqns) == 2

        def program(key, x):
            return af.switch(key, {"a": branch_a, "b": branch_b}, x)

        ir = af.trace(program)("a", "input")
        dce = af.dce(ir)

        dced_branch_a = dce.ir_eqns[0].params["branches"]["a"]
        assert len(dced_branch_a.ir_eqns) == 1

        result = dce.call("a", "hello")
        assert result == "hello LIVE"

    def test_batch_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.trace(inner_fn)("test")
        assert len(inner_ir.ir_eqns) == 2

        batch = af.batch(inner_ir, in_axes=True)
        dce = af.dce(batch)

        dced_inner = dce.ir_eqns[0].params["ir"]
        assert len(dced_inner.ir_eqns) == 1

    def test_pushforward_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.trace(inner_fn)("test")
        assert len(inner_ir.ir_eqns) == 2

        pf_ir = af.pushforward(inner_ir)
        dce = af.dce(pf_ir)

        dced_inner = dce.ir_eqns[0].params["ir"]
        assert len(dced_inner.ir_eqns) == 1

    def test_pullback_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.trace(inner_fn)("test")
        assert len(inner_ir.ir_eqns) == 2

        pb_ir = af.pullback(inner_ir)
        dce = af.dce(pb_ir, out_used=(True, (False,)))

        dced_inner = dce.ir_eqns[0].params["ir"]
        assert len(dced_inner.ir_eqns) == 1

    def test_pullback_call_keeps_inner_ir_when_cotangent_is_used(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.trace(inner_fn)("test")
        assert len(inner_ir.ir_eqns) == 2

        pb_ir = af.pullback(inner_ir)
        dce = af.dce(pb_ir, out_used=(False, (True,)))

        dced_inner = dce.ir_eqns[0].params["ir"]
        assert len(dced_inner.ir_eqns) == 2
        assert dce.call(("x",), "cot") == ("x LIVE", ("cot",))

    def test_deeply_nested_dce(self):
        def branch_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        branch = af.trace(branch_fn)("test")
        assert len(branch.ir_eqns) == 2

        def outer_fn(key, x):
            return af.switch(key, {"a": branch}, x)

        outer_ir = af.trace(outer_fn)("a", "test")
        batch = af.batch(outer_ir, in_axes=(False, True))
        dce = af.dce(batch)

        batch_inner = dce.ir_eqns[0].params["ir"]
        switch_eqn = batch_inner.ir_eqns[0]
        nested_branch = switch_eqn.params["branches"]["a"]

        assert len(nested_branch.ir_eqns) == 1


class TestDCEWithOutUsed:
    def test_out_used_single_output_true_keeps_deps(self):
        def program(x):
            y = af.concat(x, "a")
            return y

        ir = af.trace(program)("x")
        dce = af.dce(ir, out_used=True)

        assert len(dce.ir_eqns) == 1

    def test_out_used_single_output_false_removes_all(self):
        def program(x):
            y = af.concat(x, "a")
            return y

        ir = af.trace(program)("x")
        dce = af.dce(ir, out_used=False)

        assert len(dce.ir_eqns) == 0

    def test_out_used_tuple_partial(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            return (a, b)

        ir = af.trace(program)("x")

        dce_both = af.dce(ir, out_used=(True, True))
        assert len(dce_both.ir_eqns) == 2

        dce_first = af.dce(ir, out_used=(True, False))
        assert len(dce_first.ir_eqns) == 1

        dce_second = af.dce(ir, out_used=(False, True))
        assert len(dce_second.ir_eqns) == 1

        dce_none = af.dce(ir, out_used=(False, False))
        assert len(dce_none.ir_eqns) == 0

    def test_out_used_partial_is_callable_and_fills_none(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            return (a, b)

        ir = af.trace(program)("x")
        dced = af.dce(ir, out_used=(True, False))
        assert dced.call("X") == ("Xa", None)

    def test_out_used_with_shared_dependency(self):
        def program(x):
            shared = af.concat(x, "shared")
            a = af.concat(shared, "a")
            b = af.concat(shared, "b")
            return (a, b)

        ir = af.trace(program)("x")

        dce = af.dce(ir, out_used=(True, False))
        assert len(dce.ir_eqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ir_eqns]
        assert prim_names == ["concat", "concat"]

    def test_out_used_propagates_to_batch(self):
        def inner(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            dead = af.concat(x, "dead")
            return (a, b)

        inner_ir = af.trace(inner)("x")
        assert len(inner_ir.ir_eqns) == 3

        batch_ir = af.batch(inner_ir, in_axes=True)

        dce = af.dce(batch_ir, out_used=(True, True))
        assert len(dce.ir_eqns) == 1
        dced_inner = dce.ir_eqns[0].params["ir"]
        assert len(dced_inner.ir_eqns) == 2

    def test_out_used_propagates_to_switch(self):
        def branch_fn(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            dead = af.concat(x, "dead")
            return (a, b)

        branch = af.trace(branch_fn)("x")
        assert len(branch.ir_eqns) == 3

        branches = {"a": branch, "b": branch}

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")

        dce = af.dce(ir, out_used=(True, True))
        assert len(dce.ir_eqns) == 1
        dced_branch = dce.ir_eqns[0].params["branches"]["a"]
        assert len(dced_branch.ir_eqns) == 2


class TestDCEWithIntercepts:
    def test_intercepted_equation_not_removed(self):
        def program(x):
            saved = af.checkpoint(x, key="save", collection="cache")
            return x

        ir = af.trace(program)("test")
        assert len(ir.ir_eqns) == 1

        dce = af.dce(ir)
        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].intercept is not None

    def test_intercepted_inputs_remain_active(self):
        def program(x):
            computed = af.concat(x, "!")
            saved = af.checkpoint(computed, key="save", collection="cache")
            return x

        ir = af.trace(program)("test")
        assert len(ir.ir_eqns) == 2

        dce = af.dce(ir)
        assert len(dce.ir_eqns) == 2

    def test_mixed_intercepted_and_dead(self):
        def program(x):
            dead = af.concat(x, "dead")
            saved = af.checkpoint(x, key="save", collection="cache")
            return x

        ir = af.trace(program)("test")
        assert len(ir.ir_eqns) == 2

        dce = af.dce(ir, keep_intercepts=False)
        assert len(dce.ir_eqns) == 0

        dce = af.dce(ir, keep_intercepts=True)
        assert len(dce.ir_eqns) == 1
        assert dce.ir_eqns[0].intercept is not None


class TestDCEWithDepends:
    def test_depends_kept_when_output_used(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        depends_eqns = [e for e in dce.ir_eqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

    def test_depends_removed_when_output_unused(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            _ = af.depends(b, a)
            return af.format("C: {}", x)

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        depends_eqns = [e for e in dce.ir_eqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 0

    def test_depends_keeps_its_deps_alive(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        format_eqns = [e for e in dce.ir_eqns if e.prim.name == "format"]
        assert len(format_eqns) == 2

    def test_depends_deps_removed_when_depends_unused(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            _ = af.depends(b, a)
            return x

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        assert len(dce.ir_eqns) == 0

    def test_depends_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            b_ord = af.depends(b, a)
            c_ord = af.depends(c, b_ord)
            return c_ord

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        format_eqns = [e for e in dce.ir_eqns if e.prim.name == "format"]
        depends_eqns = [e for e in dce.ir_eqns if e.prim.name == "depends"]
        assert len(format_eqns) == 3
        assert len(depends_eqns) == 2

    def test_depends_partial_chain_kept(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            b_ord = af.depends(b, a)
            _ = af.depends(c, b_ord)
            return b_ord

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        format_eqns = [e for e in dce.ir_eqns if e.prim.name == "format"]
        depends_eqns = [e for e in dce.ir_eqns if e.prim.name == "depends"]
        assert len(format_eqns) == 2
        assert len(depends_eqns) == 1

    def test_depends_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        format_eqns = [e for e in dce.ir_eqns if e.prim.name == "format"]
        assert len(format_eqns) == 3

    def test_depends_no_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            return af.depends(a)

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        format_eqns = [e for e in dce.ir_eqns if e.prim.name == "format"]
        depends_eqns = [e for e in dce.ir_eqns if e.prim.name == "depends"]
        assert len(format_eqns) == 1
        assert len(depends_eqns) == 1

    def test_depends_with_intercepted(self):
        def program(x):
            a = af.checkpoint(x, key="a")
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        dce = af.dce(ir)

        intercept_eqns = [e for e in dce.ir_eqns if e.intercept is not None]
        depends_eqns = [e for e in dce.ir_eqns if e.prim.name == "depends"]
        assert len(intercept_eqns) == 1
        assert len(depends_eqns) == 1
