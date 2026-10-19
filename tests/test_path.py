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

import autoform.extend as extend
import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.control as control
import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.intercept as intercept
import autoform.src.path as path
import autoform.src.stage as stage
import autoform.src.string as string
from tests import aexecute, execute


@pytest.fixture
def delay():
    async def aimpl(value):
        await asyncio.sleep(value)
        return value

    p_test_path_delay = core.Prim("test_path_delay")
    extend.r_impl.set(p_test_path_delay, lambda value: value)
    extend.r_abstract.set(p_test_path_delay, lambda value: value)
    extend.r_aimpl.set(p_test_path_delay, aimpl)
    return p_test_path_delay.bind


class TestFactor:
    def test_factor_is_noop_in_normal_execution(self):
        assert path.factor(1.0, name="neutral") is None

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize("weight", [1, 1.0], ids=["int", "float"])
    def test_factor_traces_to_primitive(self, executor, weight):
        def program(x: str, weight: float):
            path.factor(weight, name="score")
            return string.concat(x, "!")

        ir = stage.trace(program)("x", weight)

        assert [eqn.prim.name for eqn in ir.eqns] == ["factor", "render"]
        assert ir.eqns[0].out_tree == ()
        assert executor(ir, "hello", weight) == "hello!"

    def test_factor_rejects_negative_weight_in_normal_execution(self):
        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            path.factor(-0.1)

    def test_factor_rejects_bool_weight(self):
        with pytest.raises(AssertionError, match="numeric factor weight"):
            path.factor(True)
        with pytest.raises(AssertionError, match="Expected number"):
            stage.trace(path.factor)(True)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_factor_rejects_negative_dynamic_weight_at_runtime(self, executor):
        def program(weight: float):
            path.factor(weight, name="score")
            return "done"

        ir = stage.trace(program)(1.0)

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            executor(ir, -0.1)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_factor_rejects_negative_traced_literal_at_runtime(self, executor):
        def program():
            path.factor(-0.1, name="score")
            return "done"

        ir = stage.trace(program)()

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            executor(ir)

    def test_dce_preserves_factor(self):
        def program(x: str, weight: float):
            path.factor(weight, name="score")
            return string.concat(x, "!")

        dced = dead.dce(stage.trace(program)("x", 1.0))

        assert [eqn.prim.name for eqn in dced.eqns] == ["factor", "render"]


class TestWeight:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_ir_returns_output_and_path_weight(self, executor):
        def program(x: str, weight: float):
            path.factor(weight, name="score")
            return string.concat(x, "!")

        ir = stage.trace(program)("x", 1.0)
        weight_ir = path.weight(ir)

        output, weight = executor(weight_ir, "hello", 0.5)

        assert len(weight_ir.eqns) == 1
        assert weight_ir.eqns[0].prim is path.p_weight_call
        assert output == "hello!"
        assert weight == 0.5

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_multiplies_factors(self, executor):
        def program(x: str):
            path.factor(0.5, name="a")
            path.factor(0.25, name="b")
            return x

        output, weight = executor(path.weight(stage.trace(program)("x")), "done")

        assert output == "done"
        assert weight == pytest.approx(0.125)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_zero_factor_returns_zero_weight(self, executor):
        def program(x: str):
            path.factor(0.0, name="reject")
            return x

        output, weight = executor(path.weight(stage.trace(program)("x")), "done")

        assert output == "done"
        assert weight == 0.0

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_normal_call_ignores_factor_weight(self, executor):
        def program(x: str):
            path.factor(0.5, name="score")
            return x

        assert executor(stage.trace(program)("x"), "done") == "done"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_intercepts_nested_factor(self, executor):
        def branch():
            path.factor(0.25, name="branch")
            return "hit"

        branches = {"hit": stage.trace(branch)()}

        def program(key: str):
            return control.switch(key, branches)

        output, weight = executor(path.weight(stage.trace(program)("hit")), "hit")

        assert output == "hit"
        assert weight == 0.25

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_validates_factor_weight(self, executor):
        def program():
            path.factor(-0.1, name="bad")
            return "done"

        with pytest.raises(AssertionError, match="finite non-negative factor weight"):
            executor(path.weight(stage.trace(program)()))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_over_weight_ir_scores_candidate_paths(self, executor):
        def program(candidate: str, likelihood: float):
            path.factor(likelihood, name="evidence")
            return candidate

        ir = stage.trace(program)("x", 1.0)
        batched = axis.batch(path.weight(ir), in_axes=(True, True))

        outputs, weights = executor(batched, ["x1", "x2"], [0.9, 0.2])

        assert outputs == ["x1", "x2"]
        assert weights == pytest.approx([0.9, 0.2])

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_over_weight_collects_in_batch_order(self, executor, delay):
        def program(seconds: float):
            value = delay(seconds)
            value = intercept.checkpoint(value, key="seen", collection="debug")
            path.factor(1.0, name="score")
            return value

        ir = stage.trace(program)(0.0)
        batched = axis.batch(path.weight(ir), in_axes=True)

        with intercept.collect(collection="debug") as collected:
            outputs, weights = executor(batched, [0.03, 0.01, 0.02])

        assert outputs == [0.03, 0.01, 0.02]
        assert weights == [1.0, 1.0, 1.0]
        assert collected == {"seen": [0.03, 0.01, 0.02]}

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_over_weight_injects_in_batch_order(self, executor, delay):
        def program(seconds: float):
            value = delay(seconds)
            value = intercept.checkpoint(value, key="seen", collection="cache")
            path.factor(1.0, name="score")
            return value

        ir = stage.trace(program)(0.0)
        batched = axis.batch(path.weight(ir), in_axes=True)

        with intercept.inject(collection="cache", values={"seen": [3.0, 1.0, 2.0]}):
            outputs, weights = executor(batched, [0.03, 0.01, 0.02])

        assert outputs == [3.0, 1.0, 2.0]
        assert weights == [1.0, 1.0, 1.0]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_posterior_can_be_normalized_outside_core(self, executor):
        def program(candidate: str, likelihood: float):
            path.factor(likelihood, name="e")
            return candidate

        ir = stage.trace(program)("x", 1.0)
        outputs, path_weights = executor(
            axis.batch(path.weight(ir), in_axes=(True, True)),
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
            path.factor(likelihood, name="evidence")
            return candidate

        ir = stage.trace(program)("x", 1.0)
        batched = axis.batch(ir, in_axes=(True, True))

        outputs, path_weight = executor(path.weight(batched), ["x1", "x2"], [0.9, 0.2])

        assert outputs == ["x1", "x2"]
        assert path_weight == pytest.approx(0.9 * 0.2)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_after_pushforward_scores_primal_trace_once(self, executor):
        def program(x: str, likelihood: float):
            path.factor(likelihood, name="evidence")
            return string.concat(x, "!")

        ir = stage.trace(program)("x", 1.0)
        pushforward_ir = ad.pushforward(ir)

        (output, t), path_weight = executor(
            path.weight(pushforward_ir),
            ("hello", 0.5),
            ("dhello", 0.0),
        )

        assert output == "hello!"
        assert t == "dhello"
        assert path_weight == 0.5

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_weight_after_pullback_scores_forward_trace_once(self, executor):
        def program(x: str, likelihood: float):
            path.factor(likelihood, name="evidence")
            return string.concat(x, "!")

        ir = stage.trace(program)("x", 1.0)
        pullback_ir = ad.pullback(ir)

        (output, c), path_weight = executor(
            path.weight(pullback_ir),
            ("hello", 0.5),
            "feedback",
        )

        assert output == "hello!"
        assert c[0] == "feedback"
        assert isinstance(c[1], core.Zero)
        assert path_weight == 0.5

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_pushforward_of_weight_ir_raises_not_supported(self, executor):
        def program(x: str, likelihood: float):
            path.factor(likelihood, name="evidence")
            return string.concat(x, "!")

        weight_ir = path.weight(stage.trace(program)("x", 1.0))

        with pytest.raises(NotImplementedError, match=r"pushforward\(af\.weight\(ir\)\)"):
            executor(ad.pushforward(weight_ir), ("hello", 0.5), ("dhello", 0.0))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_pullback_of_weight_ir_raises_not_supported(self, executor):
        def program(x: str, likelihood: float):
            path.factor(likelihood, name="evidence")
            return string.concat(x, "!")

        weight_ir = path.weight(stage.trace(program)("x", 1.0))

        with pytest.raises(NotImplementedError, match=r"pullback\(af\.weight\(ir\)\)"):
            executor(ad.pullback(weight_ir), ("hello", 0.5), ("feedback", 1.0))

    def test_dce_weight_ir_optimizes_inner_trace(self):
        def program(x: str, likelihood: float):
            output = string.concat(x, "!")
            path.factor(likelihood, name="evidence")
            return output

        weight_ir = path.weight(stage.trace(program)("x", 1.0))
        dced = dead.dce(weight_ir, out_used=(False, True))
        inner_ir = dced.eqns[0].params["ir"]

        assert [eqn.prim.name for eqn in inner_ir.eqns] == ["factor"]
