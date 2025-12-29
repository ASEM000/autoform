import autoform as af
from autoform.core import build_ir, call


class TestMatchBasic:
    def test_match_equal_strings(self):
        assert af.match("yes", "yes") is True

    def test_match_unequal_strings(self):
        assert af.match("yes", "no") is False

    def test_match_empty_strings(self):
        assert af.match("", "") is True

    def test_match_empty_vs_nonempty(self):
        assert af.match("", "x") is False


class TestMatchTraced:
    def test_traced_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        assert call(ir)("yes") is True
        assert call(ir)("no") is False

    def test_traced_match_both_args(self):
        def check(a, b):
            return af.match(a, b)

        ir = build_ir(check)("a", "b")
        assert call(ir)("hello", "hello") is True
        assert call(ir)("hello", "world") is False

    def test_match_with_literal_second_arg(self):
        def check(x):
            return af.match(x, "target")

        ir = build_ir(check)("dummy")
        assert call(ir)("target") is True
        assert call(ir)("other") is False


class TestMatchBatch:
    def test_batch_match_all_equal(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        batched_ir = af.batch(ir, in_axes=list)

        results = call(batched_ir)(["yes", "yes", "yes"])
        assert results == [True, True, True]

    def test_batch_match_mixed(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        batched_ir = af.batch(ir, in_axes=list)

        results = call(batched_ir)(["yes", "no", "yes"])
        assert results == [True, False, True]

    def test_batch_match_both_args_batched(self):
        def check(a, b):
            return af.match(a, b)

        ir = build_ir(check)("a", "b")
        batched_ir = af.batch(ir, in_axes=(list, list))

        results = call(batched_ir)(["a", "b", "c"], ["a", "x", "c"])
        assert results == [True, False, True]

    def test_batch_match_one_arg_broadcast(self):
        def check(a, b):
            return af.match(a, b)

        ir = build_ir(check)("a", "b")
        batched_ir = af.batch(ir, in_axes=(list, None))

        results = call(batched_ir)(["target", "other", "target"], "target")
        assert results == [True, False, True]


class TestMatchPushforward:
    def test_pushforward_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        pf_ir = af.pushforward(ir)

        primal = "yes"
        tangent = "tangent_input"

        out_primal, out_tangent = call(pf_ir)((primal, tangent))

        assert out_primal is True
        assert out_tangent == ("", "")

    def test_pushforward_match_false_case(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        pf_ir = af.pushforward(ir)

        primal = "no"
        tangent = "tangent_input"

        out_primal, out_tangent = call(pf_ir)((primal, tangent))

        assert out_primal is False
        assert out_tangent == ("", "")


class TestMatchPullback:
    def test_pullback_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        pb_ir = af.pullback(ir)

        primal_in = "yes"
        out_cotangent = "feedback"

        out_primal, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert out_primal is True
        assert in_cotangent == ""

    def test_pullback_match_false_case(self):
        def check(x):
            return af.match(x, "yes")

        ir = build_ir(check)("dummy")
        pb_ir = af.pullback(ir)

        primal_in = "no"
        out_cotangent = "feedback"

        out_primal, in_cotangent = call(pb_ir)((primal_in, out_cotangent))

        assert out_primal is False
        assert in_cotangent == ""


class TestMatchComposition:
    def test_match_in_larger_program(self):
        def process(status, text):
            is_active = af.match(status, "active")
            formatted = af.format("Status check: {}", text)
            return is_active, formatted

        ir = build_ir(process)("status", "text")

        is_active, formatted = call(ir)("active", "hello")
        assert is_active is True
        assert formatted == "Status check: hello"

        is_active, formatted = call(ir)("inactive", "hello")
        assert is_active is False
        assert formatted == "Status check: hello"

    def test_batch_match_with_format(self):
        def process(status):
            is_yes = af.match(status, "yes")
            msg = af.format("Input was: {}", status)
            return is_yes, msg

        ir = build_ir(process)("status")
        batched_ir = af.batch(ir, in_axes=list)

        results = call(batched_ir)(["yes", "no", "yes"])
        is_yes_list, msg_list = results

        assert is_yes_list == [True, False, True]
        assert msg_list == ["Input was: yes", "Input was: no", "Input was: yes"]
