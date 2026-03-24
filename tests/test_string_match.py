# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import autoform as af
from autoform.core import AVal, trace
from autoform.string import abstract_match


class TestMatchBasic:
    def test_match_equal_strings(self):
        assert af.match("yes", "yes") is True

    def test_match_unequal_strings(self):
        assert af.match("yes", "no") is False

    def test_match_empty_strings(self):
        assert af.match("", "") is True

    def test_match_empty_vs_nonempty(self):
        assert af.match("", "x") is False

    def test_match_rejects_non_string_input(self):
        assert af.match("yes", 1) is False


class TestMatchTraced:
    def test_traced_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        assert ir.call("yes") is True
        assert ir.call("no") is False

    def test_traced_match_both_args(self):
        def check(a, b):
            return af.match(a, b)

        ir = trace(check)("a", "b")
        assert ir.call("hello", "hello") is True
        assert ir.call("hello", "world") is False

    def test_match_with_literal_second_arg(self):
        def check(x):
            return af.match(x, "target")

        ir = trace(check)("dummy")
        assert ir.call("target") is True
        assert ir.call("other") is False

    def test_traced_match_rejects_non_string_input(self):
        def check(a, b):
            return af.match(a, b)

        with pytest.raises(AssertionError, match="`match` expects string inputs"):
            trace(check)("yes", 1)


class TestMatchBatch:
    def test_batch_match_all_equal(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        batched_ir = af.batch(ir, in_axes=True)

        results = batched_ir.call(["yes", "yes", "yes"])
        assert results == [True, True, True]

    def test_batch_match_mixed(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        batched_ir = af.batch(ir, in_axes=True)

        results = batched_ir.call(["yes", "no", "yes"])
        assert results == [True, False, True]

    def test_batch_match_both_args_batched(self):
        def check(a, b):
            return af.match(a, b)

        ir = trace(check)("a", "b")
        batched_ir = af.batch(ir, in_axes=(True, True))

        results = batched_ir.call(["a", "b", "c"], ["a", "x", "c"])
        assert results == [True, False, True]

    def test_batch_match_one_arg_broadcast(self):
        def check(a, b):
            return af.match(a, b)

        ir = trace(check)("a", "b")
        batched_ir = af.batch(ir, in_axes=(True, False))

        results = batched_ir.call(["target", "other", "target"], "target")
        assert results == [True, False, True]


class TestMatchPushforward:
    def test_pushforward_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pf_ir = af.pushforward(ir)

        out_primal, out_tangent = pf_ir.call(("yes", "tangent_input"))

        assert out_primal is True
        assert af.ad.is_zero(out_tangent)
        assert out_tangent.type is bool

    def test_pushforward_match_false_case(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pf_ir = af.pushforward(ir)

        out_primal, out_tangent = pf_ir.call(("no", "tangent_input"))

        assert out_primal is False
        assert af.ad.is_zero(out_tangent)
        assert out_tangent.type is bool


class TestMatchPullback:
    def test_pullback_match(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pb_ir = af.pullback(ir)

        out_primal, in_cotangent = pb_ir.call(("yes", "feedback"))

        assert out_primal is True
        assert af.ad.is_zero(in_cotangent)
        assert in_cotangent.type is str

    def test_pullback_match_false_case(self):
        def check(x):
            return af.match(x, "yes")

        ir = trace(check)("dummy")
        pb_ir = af.pullback(ir)

        out_primal, in_cotangent = pb_ir.call(("no", "feedback"))

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

        is_active, formatted = ir.call("active", "hello")
        assert is_active is True
        assert formatted == "Status check: hello"

        is_active, formatted = ir.call("inactive", "hello")
        assert is_active is False
        assert formatted == "Status check: hello"

    def test_batch_match_with_format(self):
        def process(status):
            is_yes = af.match(status, "yes")
            msg = af.format("Input was: {}", status)
            return is_yes, msg

        ir = trace(process)("status")
        batched_ir = af.batch(ir, in_axes=True)

        results = batched_ir.call(["yes", "no", "yes"])
        is_yes_list, msg_list = results

        assert is_yes_list == [True, False, True]
        assert msg_list == ["Input was: yes", "Input was: no", "Input was: yes"]


class TestAbstractMatch:
    def test_abstract_match_concrete_equal(self):

        result = abstract_match(("yes", "yes"))
        assert isinstance(result, AVal)
        assert result.type is bool

    def test_abstract_match_concrete_unequal(self):

        result = abstract_match(("yes", "no"))
        assert isinstance(result, AVal)
        assert result.type is bool

    def test_abstract_match_with_var_returns_var(self):

        result = abstract_match((AVal(str), "yes"))
        assert isinstance(result, AVal)

        result = abstract_match(("yes", AVal(str)))
        assert isinstance(result, AVal)

        result = abstract_match((AVal(str), AVal(str)))
        assert isinstance(result, AVal)

    def test_abstract_match_rejects_non_string_input(self):
        with pytest.raises(AssertionError, match="`match` expects string inputs"):
            abstract_match(("yes", 1))
