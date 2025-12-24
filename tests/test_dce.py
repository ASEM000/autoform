import functools as ft

import autoform.core as core


class TestDCE:
    def test_removes_unused_equation(self):
        def program(x):
            dead = core.concat(x, "dead")
            live = core.concat(x, "live")
            return live

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"

    def test_keeps_chained_dependencies(self):
        def program(x):
            y = core.concat(x, "a")
            z = core.concat(y, "b")
            return z

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 2

    def test_preserves_equation_order(self):
        def program(x):
            y = core.concat(x, "a")
            z = core.concat(y, "b")
            w = core.concat(z, "c")
            return w

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(dce.ireqns) == 3
        for i in range(len(dce.ireqns) - 1):
            curr_out = dce.ireqns[i].out_irtree
            next_in_leaves = core.treelib.leaves(dce.ireqns[i + 1].in_irtree)
            assert curr_out in next_in_leaves

    def test_removes_constant_folded_equation(self):
        const_p = core.Primitive("test_const_fold")

        @ft.partial(core.impl_rules.set, const_p)
        def impl(x):
            return "constant"

        @ft.partial(core.eval_rules.set, const_p)
        def eval_const(x):
            return "constant"

        def program(x):
            y = const_p.bind(x)
            z = core.concat(y, "!")
            return z

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"

    def test_keeps_all_if_all_used(self):
        def program(x):
            return core.concat(x, "!")

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == len(dce.ireqns)

    def test_multiple_outputs_partial_use(self):
        def program(x):
            a = core.concat(x, "a")
            b = core.concat(x, "b")
            c = core.concat(a, "c")
            return c

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 3
        assert len(dce.ireqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ireqns]
        assert prim_names == ["concat", "concat"]


class TestDCEWithHigherOrderPrimitives:
    def test_ir_call_kept_when_used(self):
        inner_ir = core.build_ir(lambda x: core.concat(x, "!"), "x")

        def program(x):
            return core.ir_call(inner_ir, x)

        ir = core.build_ir(program, "input")
        dce = core.dce_ir(ir)

        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "ir_call"

    def test_ir_call_removed_when_unused(self):
        inner_ir = core.build_ir(lambda x: core.concat(x, "!"), "x")

        def program(x):
            dead = core.ir_call(inner_ir, x)
            live = core.concat(x, "live")
            return live

        ir = core.build_ir(program, "input")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"

    def test_switch_kept_when_used(self):
        branches = {
            "a": core.build_ir(lambda x: core.concat(x, " A"), "x"),
            "b": core.build_ir(lambda x: core.concat(x, " B"), "x"),
        }

        def program(key, x):
            return core.switch(key, branches, x)

        ir = core.build_ir(program, "a", "input")
        dce = core.dce_ir(ir)

        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "switch"

    def test_switch_removed_when_unused(self):
        branches = {
            "a": core.build_ir(lambda x: core.concat(x, " A"), "x"),
            "b": core.build_ir(lambda x: core.concat(x, " B"), "x"),
        }

        def program(key, x):
            dead = core.switch(key, branches, x)
            live = core.concat(x, "live")
            return live

        ir = core.build_ir(program, "a", "input")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 1
        assert dce.ireqns[0].prim.name == "concat"


class TestDCEWithTransformedIR:
    def test_dce_on_pushforward_ir(self):
        def program(x):
            y = core.concat(x, "a")
            dead = core.concat(x, "dead")
            return y

        ir = core.build_ir(program, "x")
        pf_ir = core.pushforward_ir(ir)
        dce = core.dce_ir(pf_ir)

        assert len(dce.ireqns) <= len(pf_ir.ireqns)

    def test_dce_on_pullback_ir(self):
        def program(x):
            y = core.concat(x, "a")
            dead = core.concat(x, "dead")
            return y

        ir = core.build_ir(program, "x")
        pb_ir = core.pullback_ir(ir)
        dce = core.dce_ir(pb_ir)

        assert len(dce.ireqns) <= len(pb_ir.ireqns)

    def test_dce_on_batch_ir(self):
        def program(x):
            y = core.concat(x, "a")
            dead = core.concat(x, "dead")
            return y

        ir = core.build_ir(program, "x")
        batch_ir = core.batch_ir(ir, in_axes=list)
        dce = core.dce_ir(batch_ir)

        assert len(dce.ireqns) <= len(batch_ir.ireqns)


class TestDCEEdgeCases:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 0
        assert len(dce.ireqns) == 0

    def test_all_dead(self):
        def program(x):
            dead1 = core.concat(x, "dead1")
            dead2 = core.concat(x, "dead2")
            return x

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 2
        assert len(dce.ireqns) == 0

    def test_diamond_dependency(self):
        def program(x):
            a = core.concat(x, "a")
            b = core.concat(a, "b")
            c = core.concat(a, "c")
            d = core.concat(b, c)
            return d

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(ir.ireqns) == 4
        assert len(dce.ireqns) == 4

    def test_stop_gradient_kept(self):
        def program(x):
            y = core.stop_gradient(x)
            z = core.concat(y, "!")
            return z

        ir = core.build_ir(program, "x")
        dce = core.dce_ir(ir)

        assert len(dce.ireqns) == 2
        prim_names = [eqn.prim.name for eqn in dce.ireqns]
        assert prim_names == ["stop_gradient", "concat"]


class TestNestedDCE:
    def test_switch_dces_inner_branches(self):
        def branch_a_fn(x):
            dead = core.concat(x, " DEAD")  # unused
            live = core.concat(x, " LIVE")  # returned
            return live

        branch_a = core.build_ir(branch_a_fn, "test")
        branch_b = core.build_ir(lambda x: core.concat(x, " B"), "test")

        assert len(branch_a.ireqns) == 2

        def program(key, x):
            return core.switch(key, {"a": branch_a, "b": branch_b}, x)

        ir = core.build_ir(program, "a", "input")
        dce = core.dce_ir(ir)

        dced_branch_a = dce.ireqns[0].params["branches"]["a"]
        assert len(dced_branch_a.ireqns) == 1

        result = core.run_ir(dce, "a", "hello")
        assert result == "hello LIVE"

    def test_batch_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = core.concat(x, " DEAD")
            live = core.concat(x, " LIVE")
            return live

        inner_ir = core.build_ir(inner_fn, "test")
        assert len(inner_ir.ireqns) == 2

        batch_ir = core.batch_ir(inner_ir, in_axes=list)
        dce = core.dce_ir(batch_ir)

        dced_inner = dce.ireqns[0].params["ir"]
        assert len(dced_inner.ireqns) == 1

    def test_pushforward_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = core.concat(x, " DEAD")
            live = core.concat(x, " LIVE")
            return live

        inner_ir = core.build_ir(inner_fn, "test")
        assert len(inner_ir.ireqns) == 2

        pf_ir = core.pushforward_ir(inner_ir)
        dce = core.dce_ir(pf_ir)

        dced_inner = dce.ireqns[0].params["ir"]
        assert len(dced_inner.ireqns) == 1

    def test_pullback_call_dces_inner_ir(self):
        def inner_fn(x):
            dead = core.concat(x, " DEAD")
            live = core.concat(x, " LIVE")
            return live

        inner_ir = core.build_ir(inner_fn, "test")
        assert len(inner_ir.ireqns) == 2

        pb_ir = core.pullback_ir(inner_ir)
        dce = core.dce_ir(pb_ir)

        dced_inner = dce.ireqns[0].params["ir"]
        assert len(dced_inner.ireqns) == 1

    def test_deeply_nested_dce(self):
        def branch_fn(x):
            dead = core.concat(x, " DEAD")
            live = core.concat(x, " LIVE")
            return live

        branch = core.build_ir(branch_fn, "test")
        assert len(branch.ireqns) == 2

        def outer_fn(key, x):
            return core.switch(key, {"a": branch}, x)

        outer_ir = core.build_ir(outer_fn, "a", "test")
        batch_ir = core.batch_ir(outer_ir, in_axes=(None, list))
        dce = core.dce_ir(batch_ir)

        batch_inner = dce.ireqns[0].params["ir"]
        switch_eqn = batch_inner.ireqns[0]
        nested_branch = switch_eqn.params["branches"]["a"]

        assert len(nested_branch.ireqns) == 1
