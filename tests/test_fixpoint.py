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

import optree
import pytest

import autoform as af

tree = optree.pytree.reexport(namespace=af.PYTREE_NAMESPACE)


class TestFixpointImpl:
    def test_constant_map_converges(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("draft", "done")

        assert af.fixpoint(step_ir, "draft", "done", max_iters=10) == "done"

    def test_max_iters_bounds_nonconvergent_step(self):
        def step(state, instruction):
            del instruction
            return af.concat(state, ".")

        step_ir = af.trace(step)("x", "unused")

        assert af.fixpoint(step_ir, "x", "unused", max_iters=4) == "x...."

    def test_custom_equiv_ir(self):
        def step(state, instruction):
            del instruction
            return af.concat(state, "!")

        step_ir = af.trace(step)("x", "unused")
        equiv_ir = af.trace(lambda prev, new: af.match(new, "x!!"))("a", "b")

        assert af.fixpoint(step_ir, "x", "unused", max_iters=10, equiv_ir=equiv_ir) == "x!!"

    def test_max_iters_validation(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("draft", "done")

        with pytest.raises(AssertionError, match="max_iters must be >= 1"):
            af.fixpoint(step_ir, "draft", "done", max_iters=0)

    def test_arity_validation(self):
        step_ir = af.trace(lambda state: state)("draft")

        with pytest.raises(AssertionError, match="exactly two"):
            af.fixpoint(step_ir, "draft", "done", max_iters=1)

    def test_state_structure_validation(self):
        step_ir = af.trace(lambda state, instruction: (state, instruction))("draft", "done")

        with pytest.raises(AssertionError, match="identical state"):
            af.fixpoint(step_ir, "draft", "done", max_iters=1)


class TestFixpointTraced:
    def test_trace_and_call(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("draft", "done")

        assert "fixpoint" in repr(ir)
        assert ir.call("draft", "done") == "done"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_acall(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("draft", "done")

        assert await ir.acall("draft", "done") == "done"


class TestFixpointPullback:
    def test_constant_map_feedback_flows_to_theta_not_init(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10, adj_iters=2)

        ir = af.trace(program)("draft", "done")
        out, (c_init, c_theta) = af.pullback(ir).call(("draft", "done"), "feedback")

        assert out == "done"
        assert af.ad.is_zero(c_init)
        assert c_theta == "feedback"

    def test_adjoint_iterations_accumulate_state_feedback(self):
        def step(state, instruction):
            return af.concat(state, instruction)

        step_ir = af.trace(step)("s", "c")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=2, adj_iters=1)

        ir = af.trace(program)("s", "c")
        out, (c_init, c_theta) = af.pullback(ir).call(("s", "c"), "g")

        assert out == "scc"
        assert af.ad.is_zero(c_init)
        assert c_theta == "gg"

    def test_zero_output_cotangent_short_circuits(self):
        def step(state, instruction):
            return af.concat(state, instruction)

        step_ir = af.trace(step)("s", "c")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=2)

        ir = af.trace(program)("s", "c")
        zero = af.ad.zeroof("g")
        _, (c_init, c_theta) = af.pullback(ir).call(("s", "c"), zero)

        assert af.ad.is_zero(c_init)
        assert af.ad.is_zero(c_theta)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_pullback(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("draft", "done")
        out, (c_init, c_theta) = await af.pullback(ir).acall(("draft", "done"), "feedback")

        assert out == "done"
        assert af.ad.is_zero(c_init)
        assert c_theta == "feedback"


class TestFixpointBatch:
    def test_batched_init_broadcast_theta(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("x", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("x", "done")
        batched = af.batch(ir, in_axes=(True, False))

        assert batched.call(["a", "b"], "done") == ["done", "done"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_batch(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("x", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("x", "done")
        batched = af.batch(ir, in_axes=(True, False))

        assert await batched.acall(["a", "b"], "done") == ["done", "done"]

    def test_batch_of_pullback(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("x", "done")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("x", "done")
        composed = af.batch(af.pullback(ir), in_axes=((True, False), True))
        out, (c_init, c_theta) = composed.call((["a", "b"], "done"), ["g1", "g2"])

        assert out == ["done", "done"]
        assert all(af.ad.is_zero(c) for c in c_init)
        assert c_theta == ["g1", "g2"]

    def test_unbatched_fallback_preserves_pytree_out_batched(self):
        def step(state, instruction):
            left, right = state
            return af.concat(left, instruction), right

        step_ir = af.trace(step)(("x", "y"), "!")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=1)

        ir = af.trace(program)(("x", "y"), "!")
        batched = af.batch(ir, in_axes=(False, False))

        assert batched.call(("x", "y"), "!") == ("x!", "y")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unbatched_fallback_preserves_pytree_out_batched_async(self):
        def step(state, instruction):
            left, right = state
            return af.concat(left, instruction), right

        step_ir = af.trace(step)(("x", "y"), "!")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=1)

        ir = af.trace(program)(("x", "y"), "!")
        batched = af.batch(ir, in_axes=(False, False))

        assert await batched.acall(("x", "y"), "!") == ("x!", "y")

    def test_batched_preserves_custom_state_container(self):
        @tree.dataclasses.dataclass
        class State:
            text: str
            status: str
            label: str = tree.dataclasses.field(pytree_node=False)

        def step(state, instruction):
            return State(
                text=af.concat(state.text, instruction),
                status=state.status,
                label=state.label,
            )

        step_ir = af.trace(step)(State("x", "keep", label="state"), "!")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=1)

        ir = af.trace(program)(State("x", "keep", label="state"), "!")
        batched = af.batch(ir, in_axes=(State(True, False, label="state"), False))
        out = batched.call(State(["a", "b"], "keep", label="state"), "!")

        assert isinstance(out, State)
        assert out == State(text=["a!", "b!"], status=["keep", "keep"], label="state")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_batched_preserves_custom_state_container(self):
        @tree.dataclasses.dataclass
        class State:
            text: str
            status: str
            label: str = tree.dataclasses.field(pytree_node=False)

        def step(state, instruction):
            return State(
                text=af.concat(state.text, instruction),
                status=state.status,
                label=state.label,
            )

        step_ir = af.trace(step)(State("x", "keep", label="state"), "!")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=1)

        ir = af.trace(program)(State("x", "keep", label="state"), "!")
        batched = af.batch(ir, in_axes=(State(True, False, label="state"), False))
        out = await batched.acall(State(["a", "b"], "keep", label="state"), "!")

        assert isinstance(out, State)
        assert out == State(text=["a!", "b!"], status=["keep", "keep"], label="state")


class TestEquivIR:
    def test_judged_convergence_counts(self):
        counters = dict(step=0, judge=0)

        @af.custom
        def step(state, instruction):
            del instruction
            counters["step"] += 1
            return af.concat(state, ".")

        @af.custom
        def probe(prev, new):
            del prev
            counters["judge"] += 1
            return new

        step_ir = af.trace(lambda state, instruction: step(state, instruction))("x", "unused")
        equiv_ir = af.trace(lambda prev, new: af.match(probe(prev, new), "x.."))("a", "b")
        counters["step"] = counters["judge"] = 0

        assert af.fixpoint(step_ir, "x", "unused", max_iters=10, equiv_ir=equiv_ir) == "x.."
        assert counters == dict(step=2, judge=2)

    def test_batched_equiv_ir(self):
        def step(state, instruction):
            del instruction
            return af.concat(state, "!")

        step_ir = af.trace(step)("x", "unused")
        equiv_ir = af.trace(lambda prev, new: af.match(new, af.concat(prev, "!")))("a", "b")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10, equiv_ir=equiv_ir)

        ir = af.trace(program)("x", "unused")
        batched = af.batch(ir, in_axes=(True, False))

        assert batched.call(["a", "b"], "unused") == ["a!", "b!"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_equiv_ir(self):
        def step(state, instruction):
            del state
            return instruction

        step_ir = af.trace(step)("x", "done")
        equiv_ir = af.trace(lambda prev, new: af.match(new, prev))("a", "b")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10, equiv_ir=equiv_ir)

        ir = af.trace(program)("x", "done")

        assert await ir.acall("x", "done") == "done"

    def test_equiv_ir_validation(self):
        step_ir = af.trace(lambda state, instruction: af.concat(state, instruction))("x", "!")
        one_arg = af.trace(lambda prev: af.match(prev, "x"))("a")

        with pytest.raises(AssertionError, match="two positional"):
            af.fixpoint(step_ir, "x", "!", max_iters=3, equiv_ir=one_arg)

        wrong_struct = af.trace(lambda prev, new: af.match(prev[0], new[0]))(("a", "b"), ("c", "d"))
        with pytest.raises(AssertionError, match="state structure"):
            af.fixpoint(step_ir, "x", "!", max_iters=3, equiv_ir=wrong_struct)

    def test_params_memoize_with_equiv_ir(self):
        counters = dict(step=0)

        @af.custom
        def step(state, instruction):
            del state
            counters["step"] += 1
            return instruction

        step_ir = af.trace(lambda state, instruction: step(state, instruction))("x", "done")
        equiv_ir = af.trace(lambda prev, new: af.match(new, prev))("a", "b")

        def program(init, instruction):
            return af.fixpoint(step_ir, init, instruction, max_iters=10, equiv_ir=equiv_ir)

        ir = af.trace(program)("x", "done")

        with af.memoize():
            first = ir.call("x", "done")
            after_first = counters["step"]
            second = ir.call("x", "done")

        assert first == second == "done"
        assert counters["step"] == after_first
