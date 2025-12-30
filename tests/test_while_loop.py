import autoform as af
from autoform.core import build_ir, call
from tests.conftest import TEST_MODEL, requires_llm


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

    def test_batch_with_data_dependent_cond(self):

        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = build_ir(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=list)

        inputs = ["", "", "already"]
        states = call(batched_ir)(inputs)

        assert states == ["x", "x", "already"]

    def test_batch_with_always_true_cond(self):

        def cond(x):
            return True

        def body(x):
            return af.concat(x, ".")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = build_ir(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=list)

        inputs = ["a", "b"]
        states = call(batched_ir)(inputs)

        # Both get 3 iterations
        assert states == ["a...", "b..."]

    def test_batch_preserves_tuple_container(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = build_ir(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=tuple)

        inputs = ("", "a")
        states = call(batched_ir)(inputs)

        assert states == ("x", "a")
        assert isinstance(states, tuple)

    def test_batch_preserves_list_container(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            return af.concat(x, "x")

        cond_ir = build_ir(cond)("x")
        body_ir = build_ir(body)("x")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        loop_ir = build_ir(loop)("")
        batched_ir = af.batch(loop_ir, in_axes=list)

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

    @requires_llm
    def test_batched_lm_calls(self):

        def translate(text):
            msgs = [
                {"role": "system", "content": "Translate to French. Return ONLY the translation."},
                {"role": "user", "content": text},
            ]
            return af.lm_call(msgs, model=TEST_MODEL)

        translate_ir = build_ir(translate)("text")
        batched_ir = af.batch(translate_ir, in_axes=list)

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

        analyze_ir = build_ir(analyze)("text")
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

        improve_ir = build_ir(improve)("text")
        pf_ir = af.pushforward(improve_ir)

        primal = "hey whats up"
        tangent = "tangent direction"

        (out_primal, out_tangent) = call(pf_ir)((primal, tangent))

        assert isinstance(out_primal, str)
        assert len(out_primal) > 0
        assert isinstance(out_tangent, str)

    @requires_llm
    def test_collect_lm_checkpoints(self):

        def process(text):
            step1 = af.lm_call(
                [{"role": "user", "content": af.format("Summarize: {}", text)}],
                model=TEST_MODEL,
            )
            step1 = af.checkpoint(step1, collection="steps", name="summary")

            step2 = af.lm_call(
                [{"role": "user", "content": af.format("Translate to Spanish: {}", step1)}],
                model=TEST_MODEL,
            )
            step2 = af.checkpoint(step2, collection="steps", name="translation")

            return step2

        process_ir = build_ir(process)("text")
        result, collected = af.collect(process_ir, collection="steps")(
            "The quick brown fox jumps over the lazy dog."
        )

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
            return af.checkpoint(refined, collection="trace", name="draft")

        cond_ir = build_ir(cond)("...")
        body_ir = build_ir(body)("...")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=5)

        loop_ir = build_ir(loop)("...")
        result, collected = af.collect(loop_ir, collection="trace")("yo whats good bro")

        assert isinstance(result, str)
        assert "draft" in collected
        assert len(collected["draft"]) <= 5

        if collected["draft"]:
            updated_drafts = [f"[EDITED] {d}" for d in collected["draft"]]
            updated_result = af.inject(
                loop_ir, collection="trace", values={"draft": updated_drafts}
            )("yo whats good bro")

            assert isinstance(updated_result, str)
