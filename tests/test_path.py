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

import asyncio

import pytest

import autoform as af
import autoform.core as core

delay_p = core.Prim("test_path_delay")


def delay(value: float) -> float:
    return delay_p.bind(value)


def impl_delay(value: float) -> float:
    return value


async def aimpl_delay(value: float) -> float:
    await asyncio.sleep(value)
    return value


def abstract_delay(value):
    return value


core.impl_rules.set(delay_p, impl_delay)
core.impl_rules.aset(delay_p, aimpl_delay)
core.abstract_rules.set(delay_p, abstract_delay)


class TestFactor:
    def test_factor_is_noop_in_normal_execution(self):
        assert af.factor(1.0, name="neutral") is None

    def test_factor_traces_to_primitive(self):
        def program(x: str, weight: float):
            af.factor(weight, name="score")
            return af.concat(x, "!")

        ir = af.trace(program)("x", 1.0)

        assert [eqn.prim.name for eqn in ir.eqns] == ["factor", "concat"]
        assert ir.eqns[0].params["name"] == "score"
        assert ir.eqns[0].out_tree == ()
        assert ir.call("hello", 0.5) == "hello!"

    def test_factor_rejects_negative_weight_in_normal_execution(self):
        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            af.factor(-0.1)

    def test_factor_rejects_bool_weight(self):
        with pytest.raises(AssertionError, match="numeric factor weight"):
            af.factor(True)

    def test_factor_rejects_negative_dynamic_weight_at_runtime(self):
        def program(weight: float):
            af.factor(weight, name="score")
            return "done"

        ir = af.trace(program)(1.0)

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            ir.call(-0.1)

    def test_factor_rejects_negative_traced_literal_at_runtime(self):
        def program():
            af.factor(-0.1, name="score")
            return "done"

        ir = af.trace(program)()

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            ir.call()

    def test_dce_preserves_factor(self):
        def program(x: str, weight: float):
            af.factor(weight, name="score")
            return af.concat(x, "!")

        dced = af.dce(af.trace(program)("x", 1.0))

        assert [eqn.prim.name for eqn in dced.eqns] == ["factor", "concat"]


class TestWeighted:
    def test_weighted_ir_returns_output_and_path_weight(self):
        def program(x: str, weight: float):
            af.factor(weight, name="score")
            return af.concat(x, "!")

        ir = af.trace(program)("x", 1.0)
        weighted_ir = af.weighted(ir)

        output, weight = weighted_ir.call("hello", 0.5)

        assert len(weighted_ir.eqns) == 1
        assert weighted_ir.eqns[0].prim is af.path.weighted_call_p
        assert output == "hello!"
        assert weight == 0.5

    def test_weighted_multiplies_factors(self):
        def program(x: str):
            af.factor(0.5, name="a")
            af.factor(0.25, name="b")
            return x

        output, weight = af.weighted(af.trace(program)("x")).call("done")

        assert output == "done"
        assert weight == pytest.approx(0.125)

    def test_weighted_zero_factor_returns_zero_weight(self):
        def program(x: str):
            af.factor(0.0, name="reject")
            return x

        output, weight = af.weighted(af.trace(program)("x")).call("done")

        assert output == "done"
        assert weight == 0.0

    def test_normal_call_ignores_factor_weight(self):
        def program(x: str):
            af.factor(0.5, name="score")
            return x

        assert af.trace(program)("x").call("done") == "done"

    def test_weighted_intercepts_nested_factor(self):
        def branch():
            af.factor(0.25, name="branch")
            return "hit"

        branches = {"hit": af.trace(branch)()}

        def program(key: str):
            return af.switch(key, branches)

        output, weight = af.weighted(af.trace(program)("hit")).call("hit")

        assert output == "hit"
        assert weight == 0.25

    def test_weighted_validates_factor_weight(self):
        def program():
            af.factor(-0.1, name="bad")
            return "done"

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            af.weighted(af.trace(program)()).call()

    def test_batch_over_weighted_ir_scores_candidate_paths(self):
        def program(candidate: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return candidate

        ir = af.trace(program)("x", 1.0)
        batched = af.batch(af.weighted(ir), in_axes=(True, True))

        outputs, weights = batched.call(["x1", "x2"], [0.9, 0.2])

        assert outputs == ["x1", "x2"]
        assert weights == pytest.approx([0.9, 0.2])

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_batch_over_weighted_collects_in_batch_order(self):
        def program(seconds: float):
            value = delay(seconds)
            value = af.checkpoint(value, key="seen", collection="debug")
            af.factor(1.0, name="score")
            return value

        ir = af.trace(program)(0.0)
        batched = af.batch(af.weighted(ir), in_axes=True)

        with af.collect(collection="debug") as collected:
            outputs, weights = await batched.acall([0.03, 0.01, 0.02])

        assert outputs == [0.03, 0.01, 0.02]
        assert weights == [1.0, 1.0, 1.0]
        assert collected == {"seen": [0.03, 0.01, 0.02]}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_batch_over_weighted_injects_in_batch_order(self):
        def program(seconds: float):
            value = delay(seconds)
            value = af.checkpoint(value, key="seen", collection="cache")
            af.factor(1.0, name="score")
            return value

        ir = af.trace(program)(0.0)
        batched = af.batch(af.weighted(ir), in_axes=True)

        with af.inject(collection="cache", values={"seen": ["a", "b", "c"]}):
            outputs, weights = await batched.acall([0.03, 0.01, 0.02])

        assert outputs == ["a", "b", "c"]
        assert weights == [1.0, 1.0, 1.0]

    def test_posterior_can_be_normalized_outside_core(self):
        def program(candidate: str, likelihood: float):
            af.factor(likelihood, name="e")
            return candidate

        ir = af.trace(program)("x", 1.0)
        outputs, path_weights = af.batch(af.weighted(ir), in_axes=(True, True)).call(
            ["x1", "x2"],
            [0.9, 0.2],
        )
        priors = [0.5, 0.5]
        unnormalized = [prior * path_weight for prior, path_weight in zip(priors, path_weights)]
        total = sum(unnormalized)
        posterior = {
            output: weight / total for output, weight in zip(outputs, unnormalized, strict=True)
        }

        assert posterior == pytest.approx({"x1": 0.45 / 0.55, "x2": 0.10 / 0.55})

    def test_weighted_after_batch_scores_whole_batched_trace(self):
        def program(candidate: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return candidate

        ir = af.trace(program)("x", 1.0)
        batched = af.batch(ir, in_axes=(True, True))

        outputs, path_weight = af.weighted(batched).call(
            ["x1", "x2"],
            [0.9, 0.2],
        )

        assert outputs == ["x1", "x2"]
        assert path_weight == pytest.approx(0.9 * 0.2)

    def test_weighted_after_pushforward_scores_primal_trace_once(self):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.concat(x, "!")

        ir = af.trace(program)("x", 1.0)
        pushforward_ir = af.pushforward(ir)

        (output, tangent), path_weight = af.weighted(pushforward_ir).call(
            ("hello", 0.5),
            ("dhello", 0.0),
        )

        assert output == "hello!"
        assert tangent == "dhello"
        assert path_weight == 0.5

    def test_weighted_after_pullback_scores_forward_trace_once(self):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.concat(x, "!")

        ir = af.trace(program)("x", 1.0)
        pullback_ir = af.pullback(ir)

        (output, cotangents), path_weight = af.weighted(pullback_ir).call(
            ("hello", 0.5),
            "feedback",
        )

        assert output == "hello!"
        assert cotangents[0] == "feedback"
        assert af.ad.is_zero(cotangents[1])
        assert path_weight == 0.5

    def test_pushforward_of_weighted_ir_raises_not_supported(self):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.concat(x, "!")

        weighted_ir = af.weighted(af.trace(program)("x", 1.0))

        with pytest.raises(NotImplementedError, match=r"pushforward\(af\.weighted\(ir\)\)"):
            af.pushforward(weighted_ir).call(("hello", 0.5), ("dhello", 0.0))

    def test_pullback_of_weighted_ir_raises_not_supported(self):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.concat(x, "!")

        weighted_ir = af.weighted(af.trace(program)("x", 1.0))

        with pytest.raises(NotImplementedError, match=r"pullback\(af\.weighted\(ir\)\)"):
            af.pullback(weighted_ir).call(("hello", 0.5), ("feedback", 1.0))

    def test_dce_weighted_ir_optimizes_inner_trace(self):
        def program(x: str, likelihood: float):
            output = af.concat(x, "!")
            af.factor(likelihood, name="evidence")
            return output

        weighted_ir = af.weighted(af.trace(program)("x", 1.0))
        dced = af.dce(weighted_ir, out_used=(False, True))
        inner_ir = dced.eqns[0].params["ir"]

        assert [eqn.prim.name for eqn in inner_ir.eqns] == ["factor"]
