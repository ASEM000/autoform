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
from tests import aexecute, execute


@pytest.fixture
def delay():
    async def aimpl(value):
        await asyncio.sleep(value)
        return value

    prim = af.core.Prim("test_path_delay")
    af.core.impl_rules.set(prim, lambda value: value)
    af.core.abstract_rules.set(prim, lambda value: value)
    af.core.impl_rules.aset(prim, aimpl)
    return prim.bind


class TestFactor:
    def test_factor_is_noop_in_normal_execution(self):
        assert af.factor(1.0, name="neutral") is None

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_factor_traces_to_primitive(self, executor):
        def program(x: str, weight: float):
            af.factor(weight, name="score")
            return af.string.concat(x, "!")

        ir = af.trace(program)("x", 1.0)

        assert [eqn.prim.name for eqn in ir.eqns] == ["factor", "concat"]
        assert ir.eqns[0].out_tree == ()
        assert executor(ir, "hello", 0.5) == "hello!"

    def test_factor_rejects_negative_weight_in_normal_execution(self):
        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            af.factor(-0.1)

    def test_factor_rejects_bool_weight(self):
        with pytest.raises(AssertionError, match="numeric factor weight"):
            af.factor(True)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_factor_rejects_negative_dynamic_weight_at_runtime(self, executor):
        def program(weight: float):
            af.factor(weight, name="score")
            return "done"

        ir = af.trace(program)(1.0)

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            executor(ir, -0.1)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_factor_rejects_negative_traced_literal_at_runtime(self, executor):
        def program():
            af.factor(-0.1, name="score")
            return "done"

        ir = af.trace(program)()

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            executor(ir)

    def test_dce_preserves_factor(self):
        def program(x: str, weight: float):
            af.factor(weight, name="score")
            return af.string.concat(x, "!")

        dced = af.dce(af.trace(program)("x", 1.0))

        assert [eqn.prim.name for eqn in dced.eqns] == ["factor", "concat"]


class TestWeight:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_ir_returns_output_and_path_weight(self, executor):
        def program(x: str, weight: float):
            af.factor(weight, name="score")
            return af.string.concat(x, "!")

        ir = af.trace(program)("x", 1.0)
        weight_ir = af.weight(ir)

        output, weight = executor(weight_ir, "hello", 0.5)

        assert len(weight_ir.eqns) == 1
        assert weight_ir.eqns[0].prim is af.path.weight_call_p
        assert output == "hello!"
        assert weight == 0.5

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_multiplies_factors(self, executor):
        def program(x: str):
            af.factor(0.5, name="a")
            af.factor(0.25, name="b")
            return x

        output, weight = executor(af.weight(af.trace(program)("x")), "done")

        assert output == "done"
        assert weight == pytest.approx(0.125)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_zero_factor_returns_zero_weight(self, executor):
        def program(x: str):
            af.factor(0.0, name="reject")
            return x

        output, weight = executor(af.weight(af.trace(program)("x")), "done")

        assert output == "done"
        assert weight == 0.0

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_normal_call_ignores_factor_weight(self, executor):
        def program(x: str):
            af.factor(0.5, name="score")
            return x

        assert executor(af.trace(program)("x"), "done") == "done"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_intercepts_nested_factor(self, executor):
        def branch():
            af.factor(0.25, name="branch")
            return "hit"

        branches = {"hit": af.trace(branch)()}

        def program(key: str):
            return af.switch(key, branches)

        output, weight = executor(af.weight(af.trace(program)("hit")), "hit")

        assert output == "hit"
        assert weight == 0.25

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_validates_factor_weight(self, executor):
        def program():
            af.factor(-0.1, name="bad")
            return "done"

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            executor(af.weight(af.trace(program)()))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_over_weight_ir_scores_candidate_paths(self, executor):
        def program(candidate: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return candidate

        ir = af.trace(program)("x", 1.0)
        batched = af.batch(af.weight(ir), in_axes=(True, True))

        outputs, weights = executor(batched, ["x1", "x2"], [0.9, 0.2])

        assert outputs == ["x1", "x2"]
        assert weights == pytest.approx([0.9, 0.2])

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_over_weight_collects_in_batch_order(self, executor, delay):
        def program(seconds: float):
            value = delay(seconds)
            value = af.checkpoint(value, key="seen", collection="debug")
            af.factor(1.0, name="score")
            return value

        ir = af.trace(program)(0.0)
        batched = af.batch(af.weight(ir), in_axes=True)

        with af.collect(collection="debug") as collected:
            outputs, weights = executor(batched, [0.03, 0.01, 0.02])

        assert outputs == [0.03, 0.01, 0.02]
        assert weights == [1.0, 1.0, 1.0]
        assert collected == {"seen": [0.03, 0.01, 0.02]}

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_over_weight_injects_in_batch_order(self, executor, delay):
        def program(seconds: float):
            value = delay(seconds)
            value = af.checkpoint(value, key="seen", collection="cache")
            af.factor(1.0, name="score")
            return value

        ir = af.trace(program)(0.0)
        batched = af.batch(af.weight(ir), in_axes=True)

        with af.inject(collection="cache", values={"seen": ["a", "b", "c"]}):
            outputs, weights = executor(batched, [0.03, 0.01, 0.02])

        assert outputs == ["a", "b", "c"]
        assert weights == [1.0, 1.0, 1.0]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_posterior_can_be_normalized_outside_core(self, executor):
        def program(candidate: str, likelihood: float):
            af.factor(likelihood, name="e")
            return candidate

        ir = af.trace(program)("x", 1.0)
        outputs, path_weights = executor(
            af.batch(af.weight(ir), in_axes=(True, True)),
            ["x1", "x2"],
            [0.9, 0.2],
        )
        priors = [0.5, 0.5]
        unnormalized = [prior * path_weight for prior, path_weight in zip(priors, path_weights)]
        total = sum(unnormalized)
        posterior = {out: weight / total for out, weight in zip(outputs, unnormalized, strict=True)}

        assert posterior == pytest.approx({"x1": 0.45 / 0.55, "x2": 0.10 / 0.55})

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_after_batch_scores_whole_batched_trace(self, executor):
        def program(candidate: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return candidate

        ir = af.trace(program)("x", 1.0)
        batched = af.batch(ir, in_axes=(True, True))

        outputs, path_weight = executor(af.weight(batched), ["x1", "x2"], [0.9, 0.2])

        assert outputs == ["x1", "x2"]
        assert path_weight == pytest.approx(0.9 * 0.2)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_after_pushforward_scores_primal_trace_once(self, executor):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.string.concat(x, "!")

        ir = af.trace(program)("x", 1.0)
        pushforward_ir = af.pushforward(ir)

        (output, tangent), path_weight = executor(
            af.weight(pushforward_ir),
            ("hello", 0.5),
            ("dhello", 0.0),
        )

        assert output == "hello!"
        assert tangent == "dhello"
        assert path_weight == 0.5

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_after_pullback_scores_forward_trace_once(self, executor):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.string.concat(x, "!")

        ir = af.trace(program)("x", 1.0)
        pullback_ir = af.pullback(ir)

        (output, cotangents), path_weight = executor(
            af.weight(pullback_ir),
            ("hello", 0.5),
            "feedback",
        )

        assert output == "hello!"
        assert cotangents[0] == "feedback"
        assert isinstance(cotangents[1], af.abstract.Zero)
        assert path_weight == 0.5

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_pushforward_of_weight_ir_raises_not_supported(self, executor):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.string.concat(x, "!")

        weight_ir = af.weight(af.trace(program)("x", 1.0))

        with pytest.raises(NotImplementedError, match=r"pushforward\(af\.weight\(ir\)\)"):
            executor(af.pushforward(weight_ir), ("hello", 0.5), ("dhello", 0.0))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_pullback_of_weight_ir_raises_not_supported(self, executor):
        def program(x: str, likelihood: float):
            af.factor(likelihood, name="evidence")
            return af.string.concat(x, "!")

        weight_ir = af.weight(af.trace(program)("x", 1.0))

        with pytest.raises(NotImplementedError, match=r"pullback\(af\.weight\(ir\)\)"):
            executor(af.pullback(weight_ir), ("hello", 0.5), ("feedback", 1.0))

    def test_dce_weight_ir_optimizes_inner_trace(self):
        def program(x: str, likelihood: float):
            output = af.string.concat(x, "!")
            af.factor(likelihood, name="evidence")
            return output

        weight_ir = af.weight(af.trace(program)("x", 1.0))
        dced = af.dce(weight_ir, out_used=(False, True))
        inner_ir = dced.eqns[0].params["ir"]

        assert [eqn.prim.name for eqn in inner_ir.eqns] == ["factor"]
