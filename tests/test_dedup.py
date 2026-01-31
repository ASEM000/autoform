import autoform as af


class TestDedup:
    def test_removes_duplicate_equation(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 2

    def test_keeps_distinct_equations(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 3

    def test_transitive_deduplication(self):
        def program(x):
            v1 = af.format("hello {}", x)
            v2 = af.concat(v1, "!")
            v3 = af.format("hello {}", x)
            v4 = af.concat(v3, "!")
            return af.concat(v2, v4)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 5
        assert len(mem.ireqns) == 3

    def test_output_substitution(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return b

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(mem.ireqns) == 1
        out_leaves = af.utils.treelib.leaves(mem.out_irtree)
        kept_out = af.utils.treelib.leaves(mem.ireqns[0].out_irtree)
        assert out_leaves[0] is kept_out[0]

    def test_preserves_execution_semantics(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        original = af.call(ir)("hello")
        memoized = af.call(mem)("hello")

        assert original == memoized == "hello!hello!"

    def test_transitive_preserves_execution(self):
        def program(x):
            v1 = af.format("hello {}", x)
            v2 = af.concat(v1, "!")
            v3 = af.format("hello {}", x)
            v4 = af.concat(v3, "!")
            return af.concat(v2, v4)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        original = af.call(ir)("world")
        memoized = af.call(mem)("world")

        assert original == memoized == "hello world!hello world!"


class TestDedupEdgeCases:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 0
        assert len(mem.ireqns) == 0

    def test_single_equation(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 1
        assert len(mem.ireqns) == 1

    def test_no_duplicates(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(a, "b")
            c = af.concat(b, "c")
            return c

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == len(mem.ireqns)

    def test_all_duplicates(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            c = af.concat(x, "!")
            return (a, b, c)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 1

        result = af.call(mem)("hi")
        assert result == ("hi!", "hi!", "hi!")

    def test_diamond_dependency(self):
        def program(x):
            shared = af.concat(x, "!")
            a = af.concat(shared, "a")
            b = af.concat(shared, "b")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 4
        assert len(mem.ireqns) == 4

    def test_duplicate_with_different_params(self):
        def program(x):
            a = af.format("hello {}", x)
            b = af.format("goodbye {}", x)
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 3

    def test_multiple_inputs(self):
        def program(x, y):
            a = af.concat(x, y)
            b = af.concat(x, y)
            return af.concat(a, b)

        ir = af.trace(program)("a", "b")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 2

        result = af.call(mem)("hello", " world")
        assert result == "hello worldhello world"


class TestDedupWithEffects:
    def test_does_not_deduplicate_different_effects(self):
        def program(x):
            a = af.checkpoint(x, key="first", collection="cache")
            b = af.checkpoint(x, key="second", collection="cache")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 3

    def test_does_not_deduplicate_identical_effects(self):
        def program(x):
            a = af.checkpoint(x, key="same", collection="cache")
            b = af.checkpoint(x, key="same", collection="cache")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)

        assert len(ir.ireqns) == 3
        assert len(mem.ireqns) == 3


class TestDedupWithTransformedIR:
    def test_dedup_on_pushforward(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        pf_ir = af.pushforward(ir)
        mem = af.dedup(pf_ir)

        assert len(mem.ireqns) <= len(pf_ir.ireqns)

    def test_dedup_on_pullback(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        pb_ir = af.pullback(ir)
        mem = af.dedup(pb_ir)

        assert len(mem.ireqns) <= len(pb_ir.ireqns)

    def test_dedup_on_batch(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        batch_ir = af.batch(ir, in_axes=True)
        mem = af.dedup(batch_ir)

        assert len(mem.ireqns) <= len(batch_ir.ireqns)


class TestNestedDedup:
    def test_pushforward_dedups_inner_ir(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        pf_ir = af.pushforward(ir)
        mem = af.dedup(pf_ir)

        inner_ir = mem.ireqns[0].params["ir"]
        assert len(inner_ir.ireqns) == 2

    def test_pullback_dedups_inner_ir(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        pb_ir = af.pullback(ir)
        mem = af.dedup(pb_ir)

        inner_ir = mem.ireqns[0].params["ir"]
        assert len(inner_ir.ireqns) == 2

    def test_batch_dedups_inner_ir(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        batch_ir = af.batch(ir, in_axes=True)
        mem = af.dedup(batch_ir)

        inner_ir = mem.ireqns[0].params["ir"]
        assert len(inner_ir.ireqns) == 2

    def test_deeply_nested_dedup(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        double = af.pushforward(af.pushforward(ir))
        mem = af.dedup(double)

        inner1 = mem.ireqns[0].params["ir"]
        inner2 = inner1.ireqns[0].params["ir"]
        assert len(inner2.ireqns) == 2


class TestDedupComposition:
    def test_dedup_then_dce(self):
        def program(x):
            a = af.concat(x, "!")
            b = af.concat(x, "!")
            dead = af.concat(x, "dead")
            return af.concat(a, b)

        ir = af.trace(program)("test")
        mem = af.dedup(ir)
        dced = af.dce(mem)

        assert len(ir.ireqns) == 4
        assert len(mem.ireqns) == 3
        assert len(dced.ireqns) == 2

        result = af.call(dced)("hi")
        assert result == "hi!hi!"

    def test_fold_then_dedup(self):
        def program(x):
            a = af.concat("hello", " world")
            b = af.concat("hello", " world")
            return af.concat(af.concat(x, a), b)

        ir = af.trace(program)("test")
        folded = af.fold(ir)
        mem = af.dedup(folded)

        assert len(mem.ireqns) <= len(folded.ireqns)

        result = af.call(mem)(">> ")
        assert result == ">> hello worldhello world"
