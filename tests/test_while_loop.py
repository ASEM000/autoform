import pytest

import autoform as af
from autoform.core import call, trace
from tests.conftest import TEST_MODEL, requires_llm


class TestWhileLoopImpl:
    def test_cond_initially_false(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        result = af.while_loop(cond_ir, body_ir, "input", max_iters=10)
        assert result == "input"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_cond_initially_false_async(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("")
        result = await af.acall(loop_ir)("input")
        assert result == "input"

    def test_cond_always_true_iterates(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        result = call(loop_ir)("a")
        assert result == "a..."

    @pytest.mark.asyncio(loop_scope="function")
    async def test_cond_always_true_iterates_async(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        result = await af.acall(loop_ir)("a")
        assert result == "a..."

    def test_cond_always_true_exits_first_iter(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, "DONE")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        assert True


class TestWhileLoopBatch:
    def test_batch_with_constant_cond(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b", "c"]
        states = call(batched_ir)(inputs)

        assert states == ["a", "b", "c"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_with_constant_cond_async(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b", "c"]
        states = await af.acall(batched_ir)(inputs)

        assert states == ["a", "b", "c"]

    def test_batch_with_data_dependent_cond(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["", "", "already"]
        states = call(batched_ir)(inputs)

        assert states == ["x", "x", "already"]

    def test_batch_with_always_true_cond(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b"]
        states = call(batched_ir)(inputs)

        assert states == ["a...", "b..."]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_with_always_true_cond_async(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b"]
        states = await af.acall(batched_ir)(inputs)

        assert states == ["a...", "b..."]

    def test_batch_preserves_tuple_container(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ("", "a")
        states = call(batched_ir)(inputs)

        assert states == ("x", "a")
        assert isinstance(states, tuple)

    def test_batch_preserves_list_container(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["", "a"]
        states = call(batched_ir)(inputs)

        assert states == ["x", "a"]
        assert isinstance(states, list)


class TestWhileLoopPullback:
    def test_pullback_no_iterations(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "start"
        out_cotangent = "feedback"

        final_state, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert final_state == "start"
        assert in_cotangent == "feedback"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_no_iterations_async(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "start"
        out_cotangent = "feedback"

        final_state, in_cotangent = await af.acall(pb_ir)((primal_in, out_cotangent))

        assert final_state == "start"
        assert in_cotangent == "feedback"

    def test_pullback_with_iterations(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=2)

        loop_ir = trace(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "a"
        out_cotangent = "g"

        final_state, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert final_state == "a.."
        assert in_cotangent == "g"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_with_iterations_async(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=2)

        loop_ir = trace(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "a"
        out_cotangent = "g"

        final_state, in_cotangent = await af.acall(pb_ir)((primal_in, out_cotangent))

        assert final_state == "a.."
        assert in_cotangent == "g"


class TestWhileLoopWithMark:
    def test_collect_no_iterations(self):
        def cond(x):
            return False

        def body(x):
            new_x = af.concat(x, "x")
            return af.checkpoint(new_x, key="state", collection="trace")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("init")
        with af.collect(collection="trace") as collected:
            result = af.call(loop_ir)("a")

        assert result == "a"
        assert "state" not in collected or collected["state"] == []

    def test_pullback_with_mark(self):
        def cond(x):
            return True

        def body(x):
            new_x = af.concat(x, "x")
            return af.checkpoint(new_x, key="state", collection="trace")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("init")

        with af.collect(collection="trace") as collected:
            result = af.call(loop_ir)("a")
        assert result == "axxx"
        assert collected["state"] == ["ax", "axx", "axxx"]

        pb_ir = af.pullback(loop_ir)
        primal_in = "a"
        out_cotangent = "feedback"
        final_state, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert final_state == "axxx"
        assert in_cotangent == "feedback"

        assert result == "axxx"
        assert collected["state"] == ["ax", "axx", "axxx"]


class TestWhileLoopValidation:
    def test_cond_must_be_ir(self):
        def body(x):
            return x

        body_ir = trace(body)("x")

        try:
            af.while_loop(lambda x: False, body_ir, "init", max_iters=10)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "cond_ir must be an IR" in str(e)

    def test_body_must_be_ir(self):
        def cond(x):
            return False

        cond_ir = trace(cond)("x")

        try:
            af.while_loop(cond_ir, lambda x: x, "init", max_iters=10)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "body_ir must be an IR" in str(e)

    def test_body_input_output_structure_must_match(self):
        def cond(x):
            return False

        def mismatched(x):
            return (x, x)

        cond_ir = trace(cond)("x")
        body_ir = trace(mismatched)("x")

        try:
            af.while_loop(cond_ir, body_ir, "init", max_iters=10)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "identical input/output structure" in str(e)


class TestWhileLoopAdvanced:
    def test_nested_while_loop(self):
        def inner_cond(x):
            return af.match(x, "")

        def inner_body(x):
            return af.concat(x, "i")

        inner_cond_ir = trace(inner_cond)("x")
        inner_body_ir = trace(inner_body)("x")

        def outer_body(x):
            inner_result = af.while_loop(inner_cond_ir, inner_body_ir, "", max_iters=2)
            return af.concat(x, inner_result)

        def outer_cond(x):
            return True

        outer_cond_ir = trace(outer_cond)("x")
        outer_body_ir = trace(outer_body)("x")

        def loop(init):
            return af.while_loop(outer_cond_ir, outer_body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        result = call(loop_ir)("start")
        assert result == "startiii"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_nested_while_loop_async(self):
        def inner_cond(x):
            return af.match(x, "")

        def inner_body(x):
            return af.concat(x, "i")

        inner_cond_ir = trace(inner_cond)("x")
        inner_body_ir = trace(inner_body)("x")

        def outer_body(x):
            inner_result = af.while_loop(inner_cond_ir, inner_body_ir, "", max_iters=2)
            return af.concat(x, inner_result)

        def outer_cond(x):
            return True

        outer_cond_ir = trace(outer_cond)("x")
        outer_body_ir = trace(outer_body)("x")

        def loop(init):
            return af.while_loop(outer_cond_ir, outer_body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        result = await af.acall(loop_ir)("start")
        assert result == "startiii"

    def test_batch_divergent_exit(self):
        def cond(x):
            return af.match(x, "go")

        def body(x):
            return af.concat(x, "!")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["go", "stop", "go"]
        states = call(batched_ir)(inputs)

        assert states[0] == "go!"
        assert states[1] == "stop"
        assert states[2] == "go!"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_divergent_exit_async(self):
        def cond(x):
            return af.match(x, "go")

        def body(x):
            return af.concat(x, "!")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["go", "stop", "go"]
        states = await af.acall(batched_ir)(inputs)

        assert states[0] == "go!"
        assert states[1] == "stop"
        assert states[2] == "go!"

    def test_batch_of_pullback(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=2)

        loop_ir = trace(loop)("")
        pb_ir = af.pullback(loop_ir)
        batched_pb = af.batch(pb_ir, in_axes=(True, True))

        primals = ["a", "b"]
        cotangents = ["g1", "g2"]
        result = call(batched_pb)(primals, cotangents)

        outputs, in_cotangents = result
        assert outputs == ["a..", "b.."]
        assert in_cotangents == ["g1", "g2"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_of_pullback_async(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=2)

        loop_ir = trace(loop)("")
        pb_ir = af.pullback(loop_ir)
        batched_pb = af.batch(pb_ir, in_axes=(True, True))

        primals = ["a", "b"]
        cotangents = ["g1", "g2"]
        result = await af.acall(batched_pb)(primals, cotangents)

        outputs, in_cotangents = result
        assert outputs == ["a..", "b.."]
        assert in_cotangents == ["g1", "g2"]

    def test_max_iters_zero(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        result = af.while_loop(cond_ir, body_ir, "start", max_iters=0)
        assert result == "start"

    def test_single_iteration(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, "!")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=1)

        loop_ir = trace(loop)("")
        result = call(loop_ir)("test")
        assert result == "test!"

    def test_batch_all_exit_immediately(self):
        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b", "c", "d", "e"]
        states = call(batched_ir)(inputs)

        assert states == ["a", "b", "c", "d", "e"]

    def test_batch_staggered_exit(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["", "", "", "done"]
        states = call(batched_ir)(inputs)

        assert states[0] == "x"
        assert states[1] == "x"
        assert states[2] == "x"
        assert states[3] == "done"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_staggered_exit_async(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["", "", "", "done"]
        states = await af.acall(batched_ir)(inputs)

        assert states[0] == "x"
        assert states[1] == "x"
        assert states[2] == "x"
        assert states[3] == "done"

    def test_many_iterations(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=20)

        loop_ir = trace(loop)("")
        result = call(loop_ir)("a")
        assert result == "a" + "." * 20

    @pytest.mark.asyncio(loop_scope="function")
    async def test_many_iterations_async(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=20)

        loop_ir = trace(loop)("")
        result = await af.acall(loop_ir)("a")
        assert result == "a" + "." * 20

    def test_batch_variable_iteration_counts(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["", "", "already", "done"]
        states = call(batched_ir)(inputs)

        assert states[0] == "x"
        assert states[1] == "x"
        assert states[2] == "already"
        assert states[3] == "done"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_variable_iteration_counts_async(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["", "", "already", "done"]
        states = await af.acall(batched_ir)(inputs)

        assert states[0] == "x"
        assert states[1] == "x"
        assert states[2] == "already"
        assert states[3] == "done"

    def test_batch_mixed_continue_exit(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b", "c"]
        states = call(batched_ir)(inputs)

        assert states == ["a...", "b...", "c..."]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_mixed_continue_exit_async(self):
        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["a", "b", "c"]
        states = await af.acall(batched_ir)(inputs)

        assert states == ["a...", "b...", "c..."]

    def test_batch_early_exit_vs_max_iters(self):
        def cond(x):
            check = af.match(x, "go")
            return check

        def body(x):
            return af.concat(x, "!")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["go", "go", "stop", "go"]
        states = call(batched_ir)(inputs)

        assert states[0] == "go!"
        assert states[1] == "go!"
        assert states[2] == "stop"
        assert states[3] == "go!"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_early_exit_vs_max_iters_async(self):
        def cond(x):
            check = af.match(x, "go")
            return check

        def body(x):
            return af.concat(x, "!")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = trace(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=True)

        inputs = ["go", "go", "stop", "go"]
        states = await af.acall(batched_ir)(inputs)

        assert states[0] == "go!"
        assert states[1] == "go!"
        assert states[2] == "stop"
        assert states[3] == "go!"


@pytest.mark.skipif(True, reason="Skipping")
class TestWhileLoopWithLLM:
    @requires_llm
    def test_refine_text_with_traces(self):
        def cond(x):
            return True

        def body(text):
            msgs = [
                {"role": "system", "content": "Make this text more professional."},
                {"role": "user", "content": text},
            ]
            refined = af.lm_call(msgs, model=TEST_MODEL)
            return af.checkpoint(refined, key="step", collection="refinements")

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("text")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = trace(loop)("init")

        with af.collect(collection="refinements") as collected:
            result = af.call(loop_ir)("hey whats up")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "step" in collected
        assert len(collected["step"]) == 3

    @requires_llm
    def test_refine_with_pullback(self):
        def cond(x):
            return True

        def body(text):
            msgs = [
                {"role": "system", "content": "Make this text more professional."},
                {"role": "user", "content": text},
            ]
            return af.lm_call(msgs, model=TEST_MODEL)

        cond_ir = trace(cond)("x")
        body_ir = trace(body)("text")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=2)

        loop_ir = trace(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "hey whats up"
        out_cotangent = "feedback on final output"

        final_state, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert isinstance(final_state, str)
        assert len(final_state) > 0
        assert isinstance(in_cotangent, str)

    @requires_llm
    def test_batched_lm_calls(self):
        def translate(text):
            msgs = [
                {"role": "system", "content": "Translate to French. Return ONLY the translation."},
                {"role": "user", "content": text},
            ]
            return af.lm_call(msgs, model=TEST_MODEL)

        translate_ir = trace(translate)("text")
        batched_ir = af.batch(translate_ir, in_axes=True)

        inputs = ["Hello", "Goodbye", "Thank you"]
        results = call(batched_ir)(inputs)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    @requires_llm
    def test_struct_lm_call(self):
        class Sentiment(af.Struct):
            positive: bool
            confidence: float
            summary: str

        def analyze(text):
            msgs = [{"role": "user", "content": af.format("Analyze sentiment: {}", text)}]
            return af.struct_lm_call(msgs, model=TEST_MODEL, struct=Sentiment)

        analyze_ir = trace(analyze)("text")
        result = call(analyze_ir)("I love this product! It's amazing!")

        assert hasattr(result, "positive")
        assert hasattr(result, "confidence")
        assert hasattr(result, "summary")
        assert isinstance(result.positive, bool)

    @requires_llm
    def test_pushforward_lm(self):
        def improve(text):
            msgs = [
                {"role": "system", "content": "Make more professional."},
                {"role": "user", "content": text},
            ]
            return af.lm_call(msgs, model=TEST_MODEL)

        improve_ir = trace(improve)("text")
        pf_ir = af.pushforward(improve_ir)

        primal = "hey whats up"
        tangent = "tangent direction"

        (out_primal, out_tangent) = call(pf_ir)((primal, tangent))

        assert isinstance(out_primal, str)
        assert len(out_primal) > 0
        assert isinstance(out_tangent, str)

    @requires_llm
    def test_collect_lm_marks(self):
        def process(text):
            step1 = af.lm_call(
                [{"role": "user", "content": af.format("Summarize: {}", text)}],
                model=TEST_MODEL,
            )
            step1 = af.checkpoint(step1, key="summary", collection="steps")

            step2 = af.lm_call(
                [{"role": "user", "content": af.format("Translate to Spanish: {}", step1)}],
                model=TEST_MODEL,
            )
            step2 = af.checkpoint(step2, key="translation", collection="steps")

            return step2

        process_ir = trace(process)("text")
        with af.collect(collection="steps") as collected:
            result = af.call(process_ir)("The quick brown fox jumps over the lazy dog.")

        assert isinstance(result, str)
        assert "summary" in collected
        assert "translation" in collected

    @requires_llm
    def test_refine_then_update(self):
        class QualityCheck(af.Struct):
            needs_improvement: bool
            reason: str

        def cond(text):
            verdict = af.struct_lm_call(
                [
                    {
                        "role": "user",
                        "content": af.format(
                            "Does this text need improvement to be more professional? Text: '{}'",
                            text,
                        ),
                    }
                ],
                model=TEST_MODEL,
                struct=QualityCheck,
            )
            return verdict.needs_improvement

        def body(text):
            refined = af.lm_call(
                [
                    {"role": "system", "content": "Make more professional and formal."},
                    {"role": "user", "content": text},
                ],
                model=TEST_MODEL,
            )
            return af.checkpoint(refined, key="draft", collection="trace")

        cond_ir = trace(cond)("...")
        body_ir = trace(body)("...")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = trace(loop)("...")
        with af.collect(collection="trace") as collected:
            result = af.call(loop_ir)("yo whats good bro")

        assert isinstance(result, str)
        assert "draft" in collected
        assert len(collected["draft"]) <= 5

        if collected["draft"]:
            updated_drafts = [f"[EDITED] {d}" for d in collected["draft"]]
            with af.inject(collection="trace", values={"draft": updated_drafts}):
                updated_result = af.call(loop_ir)("yo whats good bro")

            assert isinstance(updated_result, str)
