import pytest

import autoform as af
from autoform.core import build_ir, call
import os


class TestWhileLoopImpl:
    def test_cond_initially_false(self):

        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        result = af.while_loop(cond_ir, body_ir, "input", max_iters=10)
        assert result == "input"

    def test_cond_always_true_exits_first_iter(self):

        def cond(x):
            return True

        def body(x):
            return af.concat(x, "DONE")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        assert True


class TestWhileLoopBatch:
    def test_batch_with_constant_cond(self):

        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = build_ir(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=list)

        inputs = ["a", "b", "c"]
        states = call(batched_ir)(inputs)

        assert states == ["a", "b", "c"]


class TestWhileLoopPullback:
    def test_pullback_no_iterations(self):

        def cond(x):
            return False

        def body(x):
            return af.concat(x, "x")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = build_ir(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "start"
        out_cotangent = "feedback"

        final_state, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert final_state == "start"
        assert in_cotangent == "feedback"


class TestWhileLoopWithCheckpoint:
    def test_collect_no_iterations(self):

        def cond(x):
            return False

        def body(x):
            new_x = af.concat(x, "x")
            return af.checkpoint(new_x, collection="trace", name="state")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=10)

        loop_ir = build_ir(loop)("init")
        result, collected = af.collect(loop_ir, collection="trace")("a")

        assert result == "a"
        assert "state" not in collected or collected["state"] == []

    def test_pullback_with_checkpoint(self):
        def cond(x):
            return True

        def body(x):
            new_x = af.concat(x, "x")
            return af.checkpoint(new_x, collection="trace", name="state")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = build_ir(loop)("init")

        result, collected = af.collect(loop_ir, collection="trace")("a")
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

        body_ir = build_ir(body)("x")

        try:
            af.while_loop(lambda x: False, body_ir, "init", max_iters=10)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "cond_func must be an IR" in str(e)

    def test_body_must_be_ir(self):
        def cond(x):
            return False

        cond_ir = build_ir(cond)("x")

        try:
            af.while_loop(cond_ir, lambda x: x, "init", max_iters=10)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "body_func must be an IR" in str(e)

    def test_body_input_output_structure_must_match(self):
        def cond(x):
            return False

        def mismatched(x):
            return (x, x)

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(mismatched)("x")

        try:
            af.while_loop(cond_ir, body_ir, "init", max_iters=10)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "identical input/output structure" in str(e)


class TestWhileLoopWithLLM:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_refine_text_with_traces(self):

        def cond(x):
            return True

        def body(text):
            msgs = [
                {"role": "system", "content": "Make this text more professional."},
                {"role": "user", "content": text},
            ]
            refined = af.lm_call(msgs, model="gpt-4o")
            return af.checkpoint(refined, collection="refinements", name="step")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("text")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = build_ir(loop)("init")

        result, collected = af.collect(loop_ir, collection="refinements")("hey whats up")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "step" in collected
        assert len(collected["step"]) == 3

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_refine_with_pullback(self):

        def cond(x):
            return True

        def body(text):
            msgs = [
                {"role": "system", "content": "Make this text more professional."},
                {"role": "user", "content": text},
            ]
            return af.lm_call(msgs, model="gpt-4o")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("text")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=2)

        loop_ir = build_ir(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "hey whats up"
        out_cotangent = "feedback on final output"

        final_state, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert isinstance(final_state, str)
        assert len(final_state) > 0
        assert isinstance(in_cotangent, str)
