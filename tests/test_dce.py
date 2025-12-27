import functools as ft

import autoform as af


class TestDCE:
    def test_removes_unused_equation(self):
        def program(x):
            dead = af.concat(x, "dead")
            live = af.concat(x, "live")
            return live

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"

    def test_keeps_chained_dependencies(self):
        def program(x):
            y = af.concat(x, "a")
            z = af.concat(y, "b")
            return z

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 2

    def test_preserves_equation_order(self):
        def program(x):
            y = af.concat(x, "a")
            z = af.concat(y, "b")
            w = af.concat(z, "c")
            return w

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(dce.ireqns) == 3
        for i in range(len(dce.ireqns) - 1):
            curr_out = dce.ireqns[i].out_irtree
            next_in_leaves = af.utils.treelib.leaves(dce.ireqns[i + 1].in_irtree)
            assert curr_out in next_in_leaves

    def test_removes_constant_folded_equation(self):
        const_p = af.core.Primitive("test_const_fold")

        @ft.partial(af.core.impl_rules.def_rule, const_p)
        def impl(x):
            return "constant"

        @ft.partial(af.core.eval_rules.def_rule, const_p)
        def eval_const(x):
            return "constant"

        def program(x):
            y = const_p.bind(x)
            z = af.concat(y, "!")
            return z

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"

    def test_keeps_all_if_all_used(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == len(dce.ireqns)

    def test_multiple_outputs_partial_use(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            c = af.concat(a, "c")
            return c

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 3
        assert len(dce.ireqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ireqns]
        assert prim_names == ["concat", "concat"]


class TestDCEWithHigherOrderPrimitives:
    def test_run_ir_inlines_for_dce(self):
        """run_ir inlines operations, so DCE sees the inner ops directly."""
        inner_ir = af.build_ir(lambda x: af.concat(x, "!"))("x")

        def program(x):
            return af.call(inner_ir)(x)

        ir = af.build_ir(program)("input")
        dce = af.dce(ir)

        # Operations are inlined, so we see concat directly
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"

    def test_inlined_dead_code_removed(self):
        """DCE removes dead code from inlined run_ir."""

        def inner(x):
            dead = af.concat(x, "dead")  # This is dead
            live = af.concat(x, "live")  # This is returned
            return live

        inner_ir = af.build_ir(inner)("x")

        def program(x):
            return af.call(inner_ir)(x)

        ir = af.build_ir(program)("input")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 2  # Both ops are inlined
        assert len(dce.ireqns) == 1  # DCE removes dead one
        assert dce.ireqns[0].prim.name == "concat"

    def test_switch_kept_when_used(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat(x, " A"))("x"),
            "b": af.build_ir(lambda x: af.concat(x, " B"))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("a", "input")
        dce = af.dce(ir)

        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "switch"

    def test_switch_removed_when_unused(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat(x, " A"))("x"),
            "b": af.build_ir(lambda x: af.concat(x, " B"))("x"),
        }

        def program(key, x):
            dead = af.switch(key, branches, x)
            live = af.concat(x, "live")
            return live

        ir = af.build_ir(program)("a", "input")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"


class TestDCEWithTransformedIR:
    def test_dce_on_pushforward(self):
        def program(x):
            y = af.concat(x, "a")
            dead = af.concat(x, "dead")
            return y

        ir = af.build_ir(program)("x")
        pf_ir = af.pushforward(ir)
        dce = af.dce(pf_ir)

        assert len(dce.ireqns) <= len(pf_ir.ireqns)

    def test_dce_on_pullback(self):
        def program(x):
            y = af.concat(x, "a")
            dead = af.concat(x, "dead")
            return y

        ir = af.build_ir(program)("x")
        pb_ir = af.pullback(ir)
        dce = af.dce(pb_ir)

        assert len(dce.ireqns) <= len(pb_ir.ireqns)

    def test_dce_on_batch(self):
        def program(x):
            y = af.concat(x, "a")
            dead = af.concat(x, "dead")
            return y

        ir = af.build_ir(program)("x")
        batch = af.batch(ir, in_axes=list)
        dce = af.dce(batch)

        assert len(dce.ireqns) <= len(batch.ireqns)


class TestDCEEdgeCases:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 0
        assert len(dce.ireqns) == 0

    def test_all_dead(self):
        def program(x):
            dead1 = af.concat(x, "dead1")
            dead2 = af.concat(x, "dead2")
            return x

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 0

    def test_diamond_dependency(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(a, "b")
            c = af.concat(a, "c")
            d = af.concat(b, c)
            return d

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(ir.ireqns) == 4
        assert len(dce.ireqns) == 4

    def test_stop_gradient_kept(self):
        def program(x):
            y = af.stop_gradient(x)
            z = af.concat(y, "!")
            return z

        ir = af.build_ir(program)("x")
        dce = af.dce(ir)

        assert len(dce.ireqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ireqns]
        assert prim_names == ["stop_gradient", "concat"]


class TestNestedDCE:
    def test_switch_dces_inner_branches(self):
        def branch_a_fn(x):
            dead = af.concat(x, " DEAD")  # unused
            live = af.concat(x, " LIVE")  # returned
            return live

        branch_a = af.build_ir(branch_a_fn)("test")
        branch_b = af.build_ir(lambda x: af.concat(x, " B"))("test")

        assert len(branch_a.ireqns) == 2

        def program(key, x):
            return af.switch(key, {"a": branch_a, "b": branch_b}, x)

        ir = af.build_ir(program)("a", "input")
        dce = af.dce(ir)

        dced_branch_a = dce.ireqns[0].params["branches"]["a"]
        assert len(dced_branch_a.ireqns) == 1

        result = af.call(dce)("a", "hello")
        assert result == "hello LIVE"

    def test_batch_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.build_ir(inner_fn)("test")
        assert len(inner_ir.ireqns) == 2

        batch = af.batch(inner_ir, in_axes=list)
        dce = af.dce(batch)

        dced_inner = dce.ireqns[0].params["ir"]
        assert len(dced_inner.ireqns) == 1

    def test_pushforward_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.build_ir(inner_fn)("test")
        assert len(inner_ir.ireqns) == 2

        pf_ir = af.pushforward(inner_ir)
        dce = af.dce(pf_ir)

        dced_inner = dce.ireqns[0].params["ir"]
        assert len(dced_inner.ireqns) == 1

    def test_pullback_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        inner_ir = af.build_ir(inner_fn)("test")
        assert len(inner_ir.ireqns) == 2

        pb_ir = af.pullback(inner_ir)
        dce = af.dce(pb_ir)

        dced_inner = dce.ireqns[0].params["ir"]
        assert len(dced_inner.ireqns) == 1

    def test_deeply_nested_dce(self):
        def branch_fn(x):
            dead = af.concat(x, " DEAD")
            live = af.concat(x, " LIVE")
            return live

        branch = af.build_ir(branch_fn)("test")
        assert len(branch.ireqns) == 2

        def outer_fn(key, x):
            return af.switch(key, {"a": branch}, x)

        outer_ir = af.build_ir(outer_fn)("a", "test")
        batch = af.batch(outer_ir, in_axes=(None, list))
        dce = af.dce(batch)

        batch_inner = dce.ireqns[0].params["ir"]
        switch_eqn = batch_inner.ireqns[0]
        nested_branch = switch_eqn.params["branches"]["a"]

        assert len(nested_branch.ireqns) == 1
