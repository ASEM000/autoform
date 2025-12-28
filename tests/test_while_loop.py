import autoform as af
from autoform.core import build_ir, call


class TestIterateUntilImpl:
    def test_goal_already_satisfied(self):
        def identity(x):
            return x

        def always_true(x):
            return True

        body = build_ir(identity)("x")
        goal = build_ir(always_true)("x")

        result, status, n = af.while_loop(body, "input", goal, max_iters=10)
        assert result == "input"
        assert status == "goal"
        assert n == 0

    def test_goal_never_satisfied(self):
        def append_x(x):
            return af.concat(x, "x")

        def always_false(x):
            return False

        body = build_ir(append_x)("x")
        goal = build_ir(always_false)("x")

        result, status, n = af.while_loop(body, "a", goal, max_iters=3)
        assert result == "axxx"
        assert status == "max"
        assert n == 3

    def test_max_iters_zero(self):
        def append_x(x):
            return af.concat(x, "x")

        def always_false(x):
            return False

        body = build_ir(append_x)("x")
        goal = build_ir(always_false)("x")

        result, status, n = af.while_loop(body, "input", goal, max_iters=0)
        assert result == "input"
        assert status == "max"
        assert n == 0


class TestIterateUntilBatch:
    def test_batch_all_same_iterations(self):
        def append_x(x):
            return af.concat(x, "x")

        def always_false(x):
            return False

        body = build_ir(append_x)("x")
        goal = build_ir(always_false)("x")

        def loop(init):
            return af.while_loop(body, init, goal, max_iters=2)

        loop_ir = build_ir(loop)("init")
        batched_ir = af.batch(loop_ir, in_axes=list)

        inputs = ["a", "b", "c"]
        states, statuses, iters = call(batched_ir)(inputs)

        assert states == ["axx", "bxx", "cxx"]
        assert statuses == ["max", "max", "max"]
        assert iters == [2, 2, 2]


class TestIterateUntilPullback:
    def test_pullback_preserves_primal(self):
        def append_x(x):
            return af.concat(x, "x")

        def always_false(x):
            return False

        body = build_ir(append_x)("x")
        goal = build_ir(always_false)("x")

        def loop(init):
            return af.while_loop(body, init, goal, max_iters=2)

        loop_ir = build_ir(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "start"
        out_cotangent = ("feedback", "", "")

        (final_state, status, n), in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert final_state == "startxx"
        assert status == "max"
        assert n == 2

    def test_pullback_propagates_feedback(self):
        def append_x(x):
            return af.concat(x, "x")

        def always_false(x):
            return False

        body = build_ir(append_x)("x")
        goal = build_ir(always_false)("x")

        def loop(init):
            return af.while_loop(body, init, goal, max_iters=2)

        loop_ir = build_ir(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "start"
        out_cotangent = ("feedback on final", "", "")

        (_, _, _), in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert in_cotangent == "feedback on final"

    def test_pullback_no_iterations(self):
        def append_x(x):
            return af.concat(x, "x")

        def always_true(x):
            return True

        body = build_ir(append_x)("x")
        goal = build_ir(always_true)("x")

        def loop(init):
            return af.while_loop(body, init, goal, max_iters=10)

        loop_ir = build_ir(loop)("init")
        pb_ir = af.pullback(loop_ir)

        primal_in = "start"
        out_cotangent = ("feedback", "", "")

        (final_state, status, n), in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert final_state == "start"
        assert status == "goal"
        assert n == 0
        assert in_cotangent == "feedback"


class TestIterateUntilWithCheckpoint:
    def test_collect_intermediate_states(self):
        def append_and_checkpoint(x):
            new_x = af.concat(x, "x")
            return af.checkpoint(new_x, collection="trace", name="state")

        def always_false(x):
            return False

        body = build_ir(append_and_checkpoint)("x")
        goal = build_ir(always_false)("x")

        def loop(init):
            return af.while_loop(body, init, goal, max_iters=3)

        loop_ir = build_ir(loop)("init")
        result, collected = af.collect(loop_ir, collection="trace")("a")

        final_state, status, n = result
        assert final_state == "axxx"
        assert status == "max"
        assert n == 3

        assert "state" in collected
        assert collected["state"] == ["ax", "axx", "axxx"]


class TestIterateUntilValidation:
    def test_body_must_be_ir(self):
        def goal(x):
            return True

        goal_ir = build_ir(goal)("x")

        try:
            af.while_loop(lambda x: x, "init", goal_ir, max_iters=5)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "body must be an IR" in str(e)

    def test_goal_must_be_ir(self):
        def body(x):
            return x

        body_ir = build_ir(body)("x")

        try:
            af.while_loop(body_ir, "init", lambda x: True, max_iters=5)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "goal must be an IR" in str(e)

    def test_body_input_output_structure_must_match(self):
        def mismatched(x):
            return (x, x)  # Returns tuple, not string

        body_ir = build_ir(mismatched)("x")

        def goal(x):
            return True

        goal_ir = build_ir(goal)("x")

        try:
            af.while_loop(body_ir, "init", goal_ir, max_iters=5)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "identical input/output structure" in str(e)

    def test_goal_must_return_bool(self):
        def body(x):
            return x

        def bad_goal(x):
            return af.format("yes")

        body_ir = build_ir(body)("x")
        goal_ir = build_ir(bad_goal)("x")

        try:
            af.while_loop(body_ir, "init", goal_ir, max_iters=5)
            assert False, "Should have raised"
        except AssertionError as e:
            assert "goal must return bool" in str(e)
