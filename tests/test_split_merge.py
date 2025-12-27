import pytest
import autoform as af


class TestSplitIr:
    def test_split_basic(self):
        def func(x):
            a = af.concat("Step1: ", x)
            mid = af.sow(a, tag="split", name="checkpoint")
            b = af.concat(mid, " -> Step2")
            return b

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="checkpoint")

        result1 = af.run_ir(ir1, "input")
        assert result1 == "Step1: input"

        result2 = af.run_ir(ir2, "Step1: input")
        assert result2 == "Step1: input -> Step2"

    def test_split_at_first_sow(self):
        def func(x):
            a = af.sow(x, tag="split", name="first")
            b = af.sow(af.concat(a, "!"), tag="split", name="second")
            return b

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="first")

        result1 = af.run_ir(ir1, "hello")
        assert result1 == "hello"

        result2 = af.run_ir(ir2, "hello")
        assert result2 == "hello!"

    def test_split_at_end(self):
        def func(x):
            a = af.concat("prefix: ", x)
            return af.sow(a, tag="split", name="end")

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="end")

        result1 = af.run_ir(ir1, "test")
        assert result1 == "prefix: test"

        result2 = af.run_ir(ir2, "anything")
        assert result2 == "anything"

    def test_split_not_found_returns_original(self):
        def func(x):
            return af.concat("hello", x)

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="missing")

        # ir1 should be the original IR
        assert ir1 is ir
        # ir2 should be empty
        assert ir2.ireqns == []
        assert ir2.in_irtree == ()
        assert ir2.out_irtree == ()

    def test_split_filters_by_tag(self):
        def func(x):
            a = af.sow(x, tag="other", name="checkpoint")
            b = af.sow(a, tag="split", name="checkpoint")
            return b

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="checkpoint")
        assert len(ir1.ireqns) == 2


class TestMergeIr:
    def test_merge_basic(self):
        def step1(x):
            return af.concat("Step1: ", x)

        def step2(x):
            return af.concat(x, " -> Step2")

        ir1 = af.build_ir(step1)("test")
        ir2 = af.build_ir(step2)("test")
        merged = af.merge_ir(ir1, ir2)

        result = af.run_ir(merged, "input")
        assert result == "Step1: input -> Step2"

    def test_merge_is_composition(self):
        def f(x):
            return af.format("f({})", x)

        def g(x):
            return af.format("g({})", x)

        ir_f = af.build_ir(f)("test")
        ir_g = af.build_ir(g)("test")
        merged = af.merge_ir(ir_f, ir_g)

        result = af.run_ir(merged, "x")
        assert result == "g(f(x))"

    def test_merge_many(self):
        def add_a(x):
            return af.concat(x, "a")

        def add_b(x):
            return af.concat(x, "b")

        def add_c(x):
            return af.concat(x, "c")

        ir_a = af.build_ir(add_a)("")
        ir_b = af.build_ir(add_b)("")
        ir_c = af.build_ir(add_c)("")

        merged = af.merge_ir(af.merge_ir(ir_a, ir_b), ir_c)
        result = af.run_ir(merged, "")
        assert result == "abc"

    def test_merge_structure_mismatch_raises(self):
        def single_out(x):
            return af.concat("hello", x)

        def double_in(x, y):
            return af.concat(x, y)

        ir1 = af.build_ir(single_out)("test")
        ir2 = af.build_ir(double_in)("a", "b")

        with pytest.raises(AssertionError, match="mismatch"):
            af.merge_ir(ir1, ir2)

    def test_merge_tree_structure_mismatch_raises(self):
        def tuple_out(x):
            return (af.concat("a", x), af.concat("b", x))

        def list_in(x):
            return af.concat(x[0], x[1])

        ir1 = af.build_ir(tuple_out)("test")
        ir2 = af.build_ir(list_in)(["a", "b"])
        with pytest.raises(AssertionError, match="mismatch"):
            af.merge_ir(ir1, ir2)


class TestSplitMergeRoundtrip:
    def test_split_then_merge_is_identity(self):
        def func(x):
            a = af.concat("Step1: ", x)
            mid = af.sow(a, tag="split", name="checkpoint")
            b = af.concat(mid, " -> Step2")
            return b

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="checkpoint")
        merged = af.merge_ir(ir1, ir2)
        original_result = af.run_ir(ir, "test")
        merged_result = af.run_ir(merged, "test")
        assert original_result == merged_result

    def test_split_for_debugging(self):
        def pipeline(x):
            step1 = af.format("processed: {}", x)
            checkpoint = af.sow(step1, tag="debug", name="after_step1")
            step2 = af.format("[{}]", checkpoint)
            return step2

        ir = af.build_ir(pipeline)("test")

        ir1, ir2 = af.split_ir(ir, tag="debug", name="after_step1")
        intermediate = af.run_ir(ir1, "data")
        assert intermediate == "processed: data"
        final = af.run_ir(ir2, intermediate)
        assert final == "[processed: data]"


class TestSplitMergeWithTransforms:
    def test_split_with_pullback(self):
        def func(x):
            a = af.concat("frozen: ", x)
            mid = af.sow(a, tag="split", name="mid")
            b = af.concat(mid, " trainable")
            return b

        ir = af.build_ir(func)("test")
        ir1, ir2 = af.split_ir(ir, tag="split", name="mid")

        # Only differentiate ir2
        grad_ir2 = af.pullback_ir(ir2)
        result = af.run_ir(grad_ir2, ("primal_mid", "cotangent"))
        primal_out, cotangent_in = result
        assert "trainable" in primal_out

    def test_merged_with_batch(self):
        def f(x):
            return af.concat("f:", x)

        def g(x):
            return af.concat(x, ":g")

        ir_f = af.build_ir(f)("")
        ir_g = af.build_ir(g)("")
        merged = af.merge_ir(ir_f, ir_g)
        batched = af.batch_ir(merged)

        result = af.run_ir(batched, ["a", "b", "c"])
        assert result == ["f:a:g", "f:b:g", "f:c:g"]
