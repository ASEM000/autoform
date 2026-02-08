import autoform as af
from autoform.core import Var, call, trace
from autoform.string import eval_match


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

        ir = trace(check)("dummy")
        assert call(ir)("yes") is True
        assert call(ir)("no") is False

    def test_traced_match_both_args(self):
        def check(a, b):
            return af.match(a, b)

        ir = trace(check)("a", "b")
        assert call(ir)("hello", "hello") is True
        assert call(ir)("hello", "world") is False

    def test_match_with_literal_second_arg(self):
        def check(x):
            return af.match(x, "target")

        ir = trace(check)("dummy")
        assert call(ir)("target") is True
        assert call(ir)("other") is False


class TestMatchBatch:
    def test_batch_match_all_equal(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        batched_ir = af.batch(ir, in_axes=True)

        results = call(batched_ir)(["yes", "yes", "yes"])
        assert results == [True, True, True]

    def test_batch_match_mixed(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        batched_ir = af.batch(ir, in_axes=True)

        results = call(batched_ir)(["yes", "no", "yes"])
        assert results == [True, False, True]

    def test_batch_match_both_args_batched(self):
        def check(a, b):
            return af.match(a, b)

        ir = trace(check)("a", "b")
        batched_ir = af.batch(ir, in_axes=(True, True))

        results = call(batched_ir)(["a", "b", "c"], ["a", "x", "c"])
        assert results == [True, False, True]

    def test_batch_match_one_arg_broadcast(self):
        def check(a, b):
            return af.match(a, b)

        ir = trace(check)("a", "b")
        batched_ir = af.batch(ir, in_axes=(True, False))

        results = call(batched_ir)(["target", "other", "target"], "target")
        assert results == [True, False, True]


class TestMatchPushforward:
    def test_pushforward_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pf_ir = af.pushforward(ir)

        out_primal, out_tangent = call(pf_ir)(("yes", "tangent_input"))

        assert out_primal is True
        assert af.ad.is_zero(out_tangent)
        assert out_tangent.type is bool

    def test_pushforward_match_false_case(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pf_ir = af.pushforward(ir)

        out_primal, out_tangent = call(pf_ir)(("no", "tangent_input"))

        assert out_primal is False
        assert af.ad.is_zero(out_tangent)
        assert out_tangent.type is bool


class TestMatchPullback:
    def test_pullback_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pb_ir = af.pullback(ir)

        out_primal, in_cotangent = call(pb_ir)(("yes", "feedback"))

        assert out_primal is True
        assert af.ad.is_zero(in_cotangent)
        assert in_cotangent.type is str

    def test_pullback_match_false_case(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pb_ir = af.pullback(ir)

        out_primal, in_cotangent = call(pb_ir)(("no", "feedback"))

        assert out_primal is False
        assert af.ad.is_zero(in_cotangent)
        assert in_cotangent.type is str


class TestMatchComposition:
    def test_match_in_larger_program(self):
        def process(status, text):
            is_active = af.match(status, "active")
            formatted = af.format("Status check: {}", text)
            return is_active, formatted

        ir = trace(process)("status", "text")

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

        ir = trace(process)("status")
        batched_ir = af.batch(ir, in_axes=True)

        results = call(batched_ir)(["yes", "no", "yes"])
        is_yes_list, msg_list = results

        assert is_yes_list == [True, False, True]
        assert msg_list == ["Input was: yes", "Input was: no", "Input was: yes"]


class TestEvalMatch:
    def test_eval_match_concrete_equal(self):

        result = eval_match(("yes", "yes"))
        assert isinstance(result, Var)
        assert result.type is bool

    def test_eval_match_concrete_unequal(self):

        result = eval_match(("yes", "no"))
        assert isinstance(result, Var)
        assert result.type is bool

    def test_eval_match_with_var_returns_var(self):

        result = eval_match((Var(str), "yes"))
        assert isinstance(result, Var)

        result = eval_match(("yes", Var(str)))
        assert isinstance(result, Var)

        result = eval_match((Var(str), Var(str)))
        assert isinstance(result, Var)
