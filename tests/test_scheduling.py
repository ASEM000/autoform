import functools as ft

import pytest

import autoform as af
from autoform.scheduling import toposort_levels


class TestGatherBasic:
    def test_single_ir(self):
        ir = af.trace(lambda x: af.format("[{}]", x))("a")
        result = af.gather([(ir, "A")])
        assert result == ["[A]"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_single_ir_async(self):
        ir = af.trace(lambda x: af.format("[{}]", x))("a")

        def program(x):
            return af.gather([(ir, x)])

        prog_ir = af.trace(program)("a")
        result = await af.acall(prog_ir)("A")
        assert result == ["[A]"]

    def test_two_irs(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")
        result = af.gather([(ir1, "A"), (ir2, "B")])
        assert result == ["[A]", "<B>"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_two_irs_async(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(a, b):
            return af.gather([(ir1, a), (ir2, b)])

        prog_ir = af.trace(program)("a", "b")
        result = await af.acall(prog_ir)("A", "B")
        assert result == ["[A]", "<B>"]

    def test_three_irs(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")
        ir3 = af.trace(lambda x: af.format("{{{}}}", x))("a")
        result = af.gather([(ir1, "X"), (ir2, "Y"), (ir3, "Z")])
        assert result == ["[X]", "<Y>", "{Z}"]

    def test_chained_operations(self):
        def chain(x):
            a = af.concat(x, "!")
            return af.format("[{}]", a)

        ir = af.trace(chain)("a")
        result = af.gather([(ir, "hello"), (ir, "world")])
        assert result == ["[hello!]", "[world!]"]


class TestGatherValidation:
    def test_empty_raises(self):
        with pytest.raises(AssertionError):
            af.gather([])

    def test_invalid_pair_raises(self):
        with pytest.raises(TypeError, match="Expected \\(ir, inputs\\) tuple"):
            af.gather(["not_a_tuple"])

    def test_non_ir_raises(self):
        with pytest.raises(TypeError, match="Expected \\(ir, inputs\\) tuple"):
            af.gather([("not_an_ir", "input")])

    def test_exception_propagates(self):
        error_p = af.core.Primitive("error")

        @ft.partial(af.core.eval_rules.set, error_p)
        def eval_error(x):
            return af.core.Var(str)

        @ft.partial(af.core.impl_rules.set, error_p)
        def impl_error(x):
            raise ValueError("intentional error")

        ir_ok = af.trace(lambda x: af.format("[{}]", x))("a")
        ir_error = af.trace(lambda x: error_p.bind(x))("a")

        with pytest.raises(ValueError, match="intentional error"):
            af.gather([(ir_ok, "A"), (ir_error, "B")])


class TestGatherWithTransforms:
    def test_pushforward(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir1, x), (ir2, x)])

        prog_ir = af.trace(program)("a")
        pf_ir = af.pushforward(prog_ir)
        (p_out, t_out) = af.call(pf_ir)(("primal", "tangent"))
        assert p_out == ["[primal]", "<primal>"]
        assert t_out == ["[tangent]", "<tangent>"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pushforward_async(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir1, x), (ir2, x)])

        prog_ir = af.trace(program)("a")
        pf_ir = af.pushforward(prog_ir)
        (p_out, t_out) = await af.acall(pf_ir)(("primal", "tangent"))
        assert p_out == ["[primal]", "<primal>"]
        assert t_out == ["[tangent]", "<tangent>"]

    def test_pullback(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir1, x), (ir2, x)])

        prog_ir = af.trace(program)("a")
        pb_ir = af.pullback(prog_ir)
        out, cotangent = af.call(pb_ir)(("primal", ["grad1", "grad2"]))
        assert out == ["[primal]", "<primal>"]
        assert isinstance(cotangent, str)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_async(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir1, x), (ir2, x)])

        prog_ir = af.trace(program)("a")
        pb_ir = af.pullback(prog_ir)
        out, cotangent = await af.acall(pb_ir)(("primal", ["grad1", "grad2"]))
        assert out == ["[primal]", "<primal>"]
        assert isinstance(cotangent, str)


class TestGatherWithBatch:
    def test_batch_gather(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir1, x), (ir2, x)])

        prog_ir = af.trace(program)("a")
        batched_ir = af.batch(prog_ir)
        result = af.call(batched_ir)(["A", "B", "C"])
        assert result == [["[A]", "[B]", "[C]"], ["<A>", "<B>", "<C>"]]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_gather_async(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir1, x), (ir2, x)])

        prog_ir = af.trace(program)("a")
        batched_ir = af.batch(prog_ir)
        result = await af.acall(batched_ir)(["A", "B", "C"])
        assert result == [["[A]", "[B]", "[C]"], ["<A>", "<B>", "<C>"]]


class TestGatherWithDCE:
    def test_gather_kept_when_used(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            results = af.gather([(ir1, x), (ir2, x)])
            return results

        prog_ir = af.trace(program)("a")
        dce_ir = af.dce(prog_ir)

        assert len(dce_ir.ireqns) == 1
        assert dce_ir.ireqns[0].prim.name == "gather"

    def test_gather_removed_when_unused(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            _ = af.gather([(ir1, x), (ir2, x)])
            return af.format("constant: {}", x)

        prog_ir = af.trace(program)("a")
        dce_ir = af.dce(prog_ir)

        assert all(eqn.prim.name != "gather" for eqn in dce_ir.ireqns)

    def test_gather_dce_propagates_to_branches(self):
        def with_dead_code(x):
            _ = af.format("dead: {}", x)
            return af.format("[{}]", x)

        ir_with_dead = af.trace(with_dead_code)("a")
        ir_simple = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            return af.gather([(ir_with_dead, x), (ir_simple, x)])

        prog_ir = af.trace(program)("a")
        dce_ir = af.dce(prog_ir)

        gather_eqn = dce_ir.ireqns[0]
        dce_branch = gather_eqn.params["irs"][0]

        assert len(dce_branch.ireqns) == 1

    def test_gather_partial_output_used(self):
        ir1 = af.trace(lambda x: af.format("[{}]", x))("a")
        ir2 = af.trace(lambda x: af.format("<{}>", x))("a")

        def program(x):
            results = af.gather([(ir1, x), (ir2, x)])
            return results[0]

        prog_ir = af.trace(program)("a")
        dce_ir = af.dce(prog_ir, out_used=True)
        gather_eqns = [e for e in dce_ir.ireqns if e.prim.name == "gather"]
        assert len(gather_eqns) == 1
        inner_irs = gather_eqns[0].params["irs"]
        assert len(inner_irs[0].ireqns) == 1
        assert len(inner_irs[1].ireqns) == 0

    def test_gather_unused_branch_structured_output_is_callable(self):
        def structured_with_dead(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            return (a, b)

        ir_live = af.trace(lambda x: af.concat(x, "!"))("x")
        ir_dead = af.trace(structured_with_dead)("x")

        def program(x):
            result, _ = af.gather([(ir_live, x), (ir_dead, x)])
            return result

        prog_ir = af.trace(program)("x")
        dce_ir = af.dce(prog_ir)

        assert af.call(dce_ir)("X") == "X!"

        gather_eqn = [e for e in dce_ir.ireqns if e.prim.name == "gather"][0]
        inner_dead = gather_eqn.params["irs"][1]
        assert len(inner_dead.ireqns) == 0
        leaves = af.utils.treelib.leaves(inner_dead.out_irtree)
        assert all(af.core.is_irlit(x) and x.value is None for x in leaves)


class TestGatherContextPreservation:
    def test_preserves_collect(self):
        def func(x):
            return af.checkpoint(x, key="val", collection="debug")

        ir1 = af.trace(func)("a")
        ir2 = af.trace(func)("b")

        with af.collect(collection="debug") as collected:
            results = af.gather([(ir1, "A"), (ir2, "B")])

        assert results == ["A", "B"]
        assert "val" in collected
        assert set(collected["val"]) == {"A", "B"}

    def test_preserves_inject(self):
        def func(x):
            return af.checkpoint(af.format("[{}]", x), key="val", collection="cache")

        ir1 = af.trace(func)("a")
        ir2 = af.trace(func)("b")

        with af.inject(collection="cache", values={"val": ["CACHED1", "CACHED2"]}):
            results = af.gather([(ir1, "A"), (ir2, "B")])

        assert results == ["CACHED1", "CACHED2"]

    def test_nested_gather(self):
        def inner(x):
            return af.checkpoint(x, key="inner", collection="debug")

        inner_ir = af.trace(inner)("x")

        def outer(x):
            results = af.gather([(inner_ir, x), (inner_ir, x)])
            return af.concat(results[0], results[1])

        outer_ir = af.trace(outer)("x")

        with af.collect(collection="debug") as collected:
            results = af.gather([(outer_ir, "A"), (outer_ir, "B")])

        assert "inner" in collected
        assert len(collected["inner"]) == 4


class TestSched:
    def test_parallel_equations_fused(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            c = af.concat(a, b)
            return c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        prim_names = [e.prim.name for e in scheduled.ireqns]
        assert prim_names == ["gather", "concat"]

        result = af.call(scheduled)("test")
        assert result == "[test]<test>"

    def test_single_equation_not_wrapped(self):
        def program(x):
            return af.format("[{}]", x)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        prim_names = [e.prim.name for e in scheduled.ireqns]
        assert prim_names == ["format"]

    def test_with_cond_filter(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.concat(x, "!")
            return a, b

        ir = af.trace(program)("x")

        scheduled = af.sched(ir, cond=lambda e: e.prim.name == "format")

        prim_names = {e.prim.name for e in scheduled.ireqns}
        assert prim_names == {"format", "concat"}

    def test_effectful_can_be_parallelized(self):
        def program(a, b):
            x = af.checkpoint(af.format("{}", a), key="x")
            y = af.checkpoint(af.format("{}", b), key="y")
            return x, y

        ir = af.trace(program)("a", "b")
        scheduled = af.sched(ir)

        gather_count = sum(1 for e in scheduled.ireqns if e.prim.name == "gather")
        assert gather_count == 2

    def test_effectful_ordering_via_depends(self):
        def program(a, b):
            x = af.checkpoint(af.format("{}", a), key="x")
            y = af.checkpoint(af.format("{}", b), key="y")
            return af.depends(y, x)

        ir = af.trace(program)("a", "b")
        scheduled = af.sched(ir)

        result = af.call(scheduled)("hello", "world")
        assert result == "world"

    def test_mixed_pure_and_effectful(self):
        def program(a, b, c):
            x = af.format("[{}]", a)
            y = af.format("<{}>", b)
            z = af.checkpoint(af.format("{{{}}}", c), key="z")
            return x, y, z

        ir = af.trace(program)("a", "b", "c")
        scheduled = af.sched(ir)

        gather_count = sum(1 for e in scheduled.ireqns if e.prim.name == "gather")
        assert gather_count == 1

        result = af.call(scheduled)("a", "b", "c")
        assert result == ("[a]", "<b>", "{c}")


class TestSchedRecursive:
    def test_sched_switch_branches(self):
        branches = {
            "a": af.trace(lambda x: af.concat(af.format("[{}]", x), af.format("<{}>", x)))("x"),
            "b": af.trace(lambda x: af.format("({})", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")
        scheduled = af.sched(ir)

        switch_eqn = scheduled.ireqns[0]
        branch_a = switch_eqn.params["branches"]["a"]
        assert any(e.prim.name == "gather" for e in branch_a.ireqns)

        assert af.call(scheduled)("a", "hello") == "[hello]<hello>"
        assert af.call(scheduled)("b", "hello") == "(hello)"

    def test_sched_nested_switch(self):
        inner_branches = {
            "x": af.trace(lambda a: af.concat(af.format("[{}]", a), af.format("<{}>", a)))("a"),
            "y": af.trace(lambda a: af.format("({})", a))("a"),
        }

        def inner_program(key, inp):
            return af.switch(key, inner_branches, inp)

        inner_ir = af.trace(inner_program)("x", "inp")

        outer_branches = {
            "A": inner_ir,
            "B": af.trace(lambda key, inp: af.format("{} {}", key, inp))("k", "i"),
        }

        def outer_program(outer_key, inner_key, x):
            return af.switch(outer_key, outer_branches, inner_key, x)

        ir = af.trace(outer_program)("A", "x", "test")
        scheduled = af.sched(ir)

        outer_switch = scheduled.ireqns[0]
        inner_switch = outer_switch.params["branches"]["A"].ireqns[0]
        inner_branch_x = inner_switch.params["branches"]["x"]
        assert any(e.prim.name == "gather" for e in inner_branch_x.ireqns)

        assert af.call(scheduled)("A", "x", "hello") == "[hello]<hello>"
        assert af.call(scheduled)("A", "y", "hello") == "(hello)"
        assert af.call(scheduled)("B", "ignored", "world") == "ignored world"

    def test_sched_gather_nested_irs(self):
        ir1 = af.trace(lambda x: af.concat(af.format("[{}]", x), af.format("<{}>", x)))("x")
        ir2 = af.trace(lambda x: af.concat(af.format("({})", x), af.format("{{{}}}", x)))("x")

        def program(a, b):
            results = af.gather([(ir1, a), (ir2, b)])
            return results

        ir = af.trace(program)("a", "b")
        scheduled = af.sched(ir)

        outer_gather = scheduled.ireqns[0]
        for inner_ir in outer_gather.params["irs"]:
            assert any(e.prim.name == "gather" for e in inner_ir.ireqns)

        result = af.call(scheduled)("hello", "world")
        assert result == ["[hello]<hello>", "(world){world}"]

    def test_sched_with_cond_propagates_to_nested(self):
        branches = {
            "a": af.trace(lambda x: (af.format("[{}]", x), af.concat(x, "!")))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")

        scheduled = af.sched(ir, cond=lambda e: e.prim.name == "format")

        switch_eqn = scheduled.ireqns[0]
        branch_a = switch_eqn.params["branches"]["a"]
        assert not any(e.prim.name == "gather" for e in branch_a.ireqns)

        assert af.call(scheduled)("a", "test") == ("[test]", "test!")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sched_recursive_async(self):
        branches = {
            "a": af.trace(lambda x: af.concat(af.format("[{}]", x), af.format("<{}>", x)))("x"),
            "b": af.trace(lambda x: af.format("({})", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")
        scheduled = af.sched(ir)

        result = await af.acall(scheduled)("a", "hello")
        assert result == "[hello]<hello>"

    def test_sched_preserves_non_ir_params(self):
        branches = {
            "a": af.trace(lambda x: af.format("[{}]", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")
        scheduled = af.sched(ir)

        switch_eqn = scheduled.ireqns[0]
        assert "branches" in switch_eqn.params
        assert "a" in switch_eqn.params["branches"]

        assert af.call(scheduled)("a", "test") == "[test]"


class TestSchedComposition:
    def test_sched_then_pushforward(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        pf_ir = af.pushforward(scheduled)

        primals, tangents = af.call(pf_ir)(("test", "tangent"))
        assert primals == "[test]<test>"
        assert tangents == "[tangent]<tangent>"

    def test_sched_then_pullback(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        pb_ir = af.pullback(scheduled)

        out, cotangent = af.call(pb_ir)(("test", "grad"))
        assert out == "[test]<test>"
        assert isinstance(cotangent, str)

    def test_sched_then_batch(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        batched_ir = af.batch(scheduled)

        result = af.call(batched_ir)(["A", "B", "C"])
        assert result == ["[A]<A>", "[B]<B>", "[C]<C>"]

    def test_sched_then_dce(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            c = af.concat(a, b)
            _ = af.format("dead: {}", c)
            return c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        dce_ir = af.dce(scheduled)

        result = af.call(dce_ir)("test")
        assert result == "[test]<test>"

    def test_dce_then_sched(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            _ = af.format("dead", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        dce_ir = af.dce(ir)
        scheduled = af.sched(dce_ir)

        result = af.call(scheduled)("test")
        assert result == "[test]<test>"


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncSched:
    async def test_basic_async_execution(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            return af.concat(a, b)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await af.acall(scheduled)("test")
        assert result == "[test]<test>"

    async def test_parallel_independent_ops(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)
            c = af.format("{{{}}}", x)
            return a, b, c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await af.acall(scheduled)("test")
        assert result == ("[test]", "<test>", "{test}")

    async def test_sequential_dependent_ops(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.concat(a, "!")
            return b

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await af.acall(scheduled)("test")
        assert result == "[test]!"

    async def test_mixed_parallel_and_sequential(self):
        def program(x):
            a = af.format("[{}]", x)
            b = af.format("<{}>", x)

            c = af.concat(a, b)
            return c

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await af.acall(scheduled)("test")
        assert result == "[test]<test>"


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncAcall:
    async def test_basic_acall(self):
        ir = af.trace(lambda x: af.format("[{}]", x))("a")
        result = await af.acall(ir)("hello")
        assert result == "[hello]"

    async def test_acall_with_switch(self):
        branches = {
            "a": af.trace(lambda x: af.format("[{}]", x))("x"),
            "b": af.trace(lambda x: af.format("<{}>", x))("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("a", "x")

        result_a = await af.acall(ir)("a", "test")
        assert result_a == "[test]"

        result_b = await af.acall(ir)("b", "test")
        assert result_b == "<test>"


class TestDepends:
    def test_basic(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        result = af.call(ir)("hello")
        assert result == "B: hello"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_basic_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        result = await af.acall(ir)("hello")
        assert result == "B: hello"

    def test_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        result = af.call(ir)("hello")
        assert result == "C: hello"

    def test_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            b_ordered = af.depends(b, a)
            c_ordered = af.depends(c, b_ordered)
            return c_ordered

        ir = af.trace(program)("x")
        result = af.call(ir)("hello")
        assert result == "C: hello"

    def test_no_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            return af.depends(a)

        ir = af.trace(program)("x")
        result = af.call(ir)("hello")
        assert result == "A: hello"

    def test_ir_structure(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        depends_eqns = [e for e in ir.ireqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

        in_leaves = af.utils.treelib.leaves(depends_eqns[0].in_irtree)
        assert len(in_leaves) >= 2


class TestDependsWithDCE:
    def test_kept_when_used(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        dce_ir = af.dce(ir)

        depends_eqns = [e for e in dce_ir.ireqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

    def test_removed_when_unused(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            _ = af.depends(b, a)
            return af.format("C: {}", x)

        ir = af.trace(program)("x")
        dce_ir = af.dce(ir)

        depends_eqns = [e for e in dce_ir.ireqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 0


class TestDependsWithPushforward:
    def test_pushforward(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = af.call(pf_ir)(("primal", "tangent"))
        assert primal == "B: primal"
        assert tangent == "B: tangent"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pushforward_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = await af.acall(pf_ir)(("primal", "tangent"))
        assert primal == "B: primal"
        assert tangent == "B: tangent"

    def test_pushforward_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = af.call(pf_ir)(("primal", "tangent"))
        assert primal == "C: primal"
        assert tangent == "C: tangent"

    def test_pushforward_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.depends(af.format("B: {}", x), a)
            c = af.depends(af.format("C: {}", x), b)
            return c

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        primal, tangent = af.call(pf_ir)(("primal", "tangent"))
        assert primal == "C: primal"
        assert tangent == "C: tangent"


class TestDependsWithPullback:
    def test_pullback(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = af.call(pb_ir)(("primal", "grad"))
        assert out == "B: primal"
        assert cotangent == "grad"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = await af.acall(pb_ir)(("primal", "grad"))
        assert out == "B: primal"
        assert cotangent == "grad"

    def test_pullback_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = af.call(pb_ir)(("primal", "grad"))
        assert out == "C: primal"
        assert cotangent == "grad"

    def test_pullback_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.depends(af.format("B: {}", x), a)
            c = af.depends(af.format("C: {}", x), b)
            return c

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        out, cotangent = af.call(pb_ir)(("primal", "grad"))
        assert out == "C: primal"
        assert cotangent == "grad"


class TestDependsWithBatch:
    def test_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = af.call(batched_ir)(["x", "y", "z"])
        assert result == ["B: x", "B: y", "B: z"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = await af.acall(batched_ir)(["x", "y", "z"])
        assert result == ["B: x", "B: y", "B: z"]

    def test_batch_multiple_deps(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            c = af.format("C: {}", x)
            return af.depends(c, a, b)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = af.call(batched_ir)(["x", "y"])
        assert result == ["C: x", "C: y"]

    def test_batch_chained(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.depends(af.format("B: {}", x), a)
            c = af.depends(af.format("C: {}", x), b)
            return c

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)

        result = af.call(batched_ir)(["x", "y"])
        assert result == ["C: x", "C: y"]


class TestDependsWithSched:
    def test_sched_basic(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = af.call(scheduled)("hello")
        assert result == "B: hello"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_sched_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = await af.acall(scheduled)("hello")
        assert result == "B: hello"

    def test_sched_preserves_depends(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            b_ordered = af.depends(b, a)
            return af.concat(a, b_ordered)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        depends_eqns = [e for e in scheduled.ireqns if e.prim.name == "depends"]
        assert len(depends_eqns) == 1

        result = af.call(scheduled)("hello")
        assert result == "A: helloB: hello"

    def test_sched_data_dependency(self):
        def program(x):
            a = af.format("A: {}", x)
            a_barrier = af.depends(a)
            b = af.format("B: {}", a_barrier)
            return b

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)

        result = af.call(scheduled)("hello")
        assert result == "B: A: hello"


class TestDependsNestedTransforms:
    def test_batch_of_pushforward(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        batch_pf_ir = af.batch(pf_ir, in_axes=(True, True))

        primals, tangents = af.call(batch_pf_ir)(["a", "b"], ["da", "db"])
        assert primals == ["B: a", "B: b"]
        assert tangents == ["B: da", "B: db"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_of_pushforward_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        batch_pf_ir = af.batch(pf_ir, in_axes=(True, True))

        primals, tangents = await af.acall(batch_pf_ir)(["a", "b"], ["da", "db"])
        assert primals == ["B: a", "B: b"]
        assert tangents == ["B: da", "B: db"]

    def test_batch_of_pullback(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        batch_pb_ir = af.batch(pb_ir, in_axes=(True, True))

        outs, cotangents = af.call(batch_pb_ir)(["a", "b"], ["g1", "g2"])
        assert outs == ["B: a", "B: b"]
        assert cotangents == ["g1", "g2"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_of_pullback_async(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        batch_pb_ir = af.batch(pb_ir, in_axes=(True, True))

        outs, cotangents = await af.acall(batch_pb_ir)(["a", "b"], ["g1", "g2"])
        assert outs == ["B: a", "B: b"]
        assert cotangents == ["g1", "g2"]

    def test_pushforward_of_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        pf_batched_ir = af.pushforward(batched_ir)

        primals, tangents = af.call(pf_batched_ir)((["a", "b"], ["da", "db"]))
        assert primals == ["B: a", "B: b"]
        assert tangents == ["B: da", "B: db"]

    def test_pullback_of_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        pb_batched_ir = af.pullback(batched_ir)

        outs, cotangents = af.call(pb_batched_ir)((["a", "b"], ["g1", "g2"]))
        assert outs == ["B: a", "B: b"]
        assert cotangents == ["g1", "g2"]

    def test_sched_of_pushforward(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)
        sched_pf_ir = af.sched(pf_ir)

        primal, tangent = af.call(sched_pf_ir)(("primal", "tangent"))
        assert primal == "B: primal"
        assert tangent == "B: tangent"

    def test_sched_of_pullback(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)
        sched_pb_ir = af.sched(pb_ir)

        out, cotangent = af.call(sched_pb_ir)(("primal", "grad"))
        assert out == "B: primal"
        assert cotangent == "grad"

    def test_sched_of_batch(self):
        def program(x):
            a = af.format("A: {}", x)
            b = af.format("B: {}", x)
            return af.depends(b, a)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        sched_batched_ir = af.sched(batched_ir)

        result = af.call(sched_batched_ir)(["x", "y", "z"])
        assert result == ["B: x", "B: y", "B: z"]


class TestToposortLevels:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = af.trace(program)("input")
        levels = toposort_levels(ir)

        assert levels == []

    def test_single_equation(self):
        def program(x):
            return af.format("{}", x)

        ir = af.trace(program)("input")
        levels = toposort_levels(ir)

        assert len(levels) == 1
        assert len(levels[0]) == 1

    def test_independent_equations(self):
        def program(a, b):
            x = af.format("hello {}", a)
            y = af.format("world {}", b)
            return x, y

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_dependent_equations(self):
        def program(a, b):
            x = af.format("hello {}", a)
            y = af.format("world {}", b)
            z = af.concat(x, y)
            return z

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        assert len(levels) == 2
        assert len(levels[0]) == 2
        assert len(levels[1]) == 1

    def test_chain_of_equations(self):
        def program(x):
            a = af.format("{}", x)
            b = af.concat(a, "!")
            c = af.concat(b, "?")
            return c

        ir = af.trace(program)("input")
        levels = toposort_levels(ir)

        assert len(levels) == 3
        assert len(levels[0]) == 1
        assert len(levels[1]) == 1
        assert len(levels[2]) == 1


class TestToposortLevelsWithEffects:
    def test_effectful_equations_can_parallelize(self):
        def program(a, b):
            x = af.checkpoint(af.format("hello {}", a), key="x")
            y = af.checkpoint(af.format("world {}", b), key="y")
            return x, y

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        effect_eqns = [e for lvl in levels for e in lvl if e.effect is not None]
        assert len(effect_eqns) == 2

        effect_levels = []
        for i, lvl in enumerate(levels):
            for e in lvl:
                if e.effect is not None:
                    effect_levels.append(i)

        assert effect_levels[0] == effect_levels[1]

    def test_effectful_ordering_via_depends(self):
        def program(a, b):
            x = af.checkpoint(af.format("hello {}", a), key="x")
            y = af.checkpoint(af.format("world {}", b), key="y")
            return af.depends(y, x)

        ir = af.trace(program)("a", "b")
        levels = toposort_levels(ir)

        effect_levels = []
        for i, lvl in enumerate(levels):
            for e in lvl:
                if e.effect is not None:
                    effect_levels.append(i)

        assert effect_levels[0] == effect_levels[1]

        depends_level = None
        for i, lvl in enumerate(levels):
            for e in lvl:
                if e.prim.name == "depends":
                    depends_level = i

        assert depends_level > effect_levels[0]

    def test_pure_equations_parallelize_around_effects(self):
        def program(a, b, c):
            x = af.format("{}", a)
            y = af.checkpoint(af.format("{}", b), key="cp")
            z = af.format("{}", c)
            return x, y, z

        ir = af.trace(program)("a", "b", "c")
        levels = toposort_levels(ir)

        has_parallel = any(len(lvl) > 1 for lvl in levels)

        assert len(levels) == 2
        assert has_parallel
