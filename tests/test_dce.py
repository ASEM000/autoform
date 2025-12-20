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
            curr_out = dce.ireqns[i].out_ir_tree
            next_in_leaves = core.treelib.leaves(dce.ireqns[i + 1].in_ir_tree)
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
            y = core.bind(const_p, x)
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
