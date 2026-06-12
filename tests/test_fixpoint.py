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


class TestFixpointImpl:
    def test_constant_map_converges(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("draft", "done")

        assert af.fixpoint(f_ir, "draft", "done", max_iters=10) == "done"

    def test_max_iters_bounds_nonconvergent_step(self):
        def step(state, instruction):
            del instruction
            return af.concat(state, ".")

        f_ir = af.trace(step)("x", "unused")

        assert af.fixpoint(f_ir, "x", "unused", max_iters=4) == "x...."

    def test_custom_equiv(self):
        def step(state, instruction):
            del instruction
            return af.concat(state, "!")

        def enough(prev, new):
            del prev
            return new.endswith("!!")

        f_ir = af.trace(step)("x", "unused")

        assert af.fixpoint(f_ir, "x", "unused", max_iters=10, equiv=enough) == "x!!"

    def test_max_iters_validation(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("draft", "done")

        with pytest.raises(AssertionError, match="max_iters must be >= 1"):
            af.fixpoint(f_ir, "draft", "done", max_iters=0)

    def test_arity_validation(self):
        f_ir = af.trace(lambda state: state)("draft")

        with pytest.raises(AssertionError, match="exactly two"):
            af.fixpoint(f_ir, "draft", "done", max_iters=1)

    def test_state_structure_validation(self):
        f_ir = af.trace(lambda state, instruction: (state, instruction))("draft", "done")

        with pytest.raises(AssertionError, match="identical state"):
            af.fixpoint(f_ir, "draft", "done", max_iters=1)


class TestFixpointTraced:
    def test_trace_and_call(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("draft", "done")

        assert "fixpoint" in repr(ir)
        assert ir.call("draft", "done") == "done"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_acall(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("draft", "done")

        assert await ir.acall("draft", "done") == "done"


class TestFixpointPullback:
    def test_constant_map_feedback_flows_to_theta_not_init(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10, adj_iters=2)

        ir = af.trace(program)("draft", "done")
        out, (c_init, c_theta) = af.pullback(ir).call(("draft", "done"), "feedback")

        assert out == "done"
        assert af.ad.is_zero(c_init)
        assert c_theta == "feedback"

    def test_adjoint_iterations_accumulate_state_feedback(self):
        def step(state, instruction):
            return af.concat(state, instruction)

        f_ir = af.trace(step)("s", "c")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=2, adj_iters=1)

        ir = af.trace(program)("s", "c")
        out, (c_init, c_theta) = af.pullback(ir).call(("s", "c"), "g")

        assert out == "scc"
        assert af.ad.is_zero(c_init)
        assert c_theta == "gg"

    def test_zero_output_cotangent_short_circuits(self):
        def step(state, instruction):
            return af.concat(state, instruction)

        f_ir = af.trace(step)("s", "c")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=2)

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

        f_ir = af.trace(step)("draft", "done")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("draft", "done")
        out, (c_init, c_theta) = await af.pullback(ir).acall(("draft", "done"), "feedback")

        assert out == "done"
        assert af.ad.is_zero(c_init)
        assert c_theta == "feedback"


class TestFixpointBatch:
    def test_batched_init_broadcast_theta(self):
        def step(state, instruction):
            del instruction
            return af.concat(state, "!")

        def enough(prev, new):
            del prev
            return new.endswith("!!")

        f_ir = af.trace(step)("x", "unused")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10, equiv=enough)

        ir = af.trace(program)("x", "unused")
        batched = af.batch(ir, in_axes=(True, False))

        assert batched.call(["a!", "b"], "unused") == ["a!!", "b!!"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_batch(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("x", "done")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10)

        ir = af.trace(program)("x", "done")
        batched = af.batch(ir, in_axes=(True, False))

        assert await batched.acall(["a", "b"], "done") == ["done", "done"]

    def test_batch_of_pullback(self):
        def step(state, instruction):
            del state
            return instruction

        f_ir = af.trace(step)("x", "done")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=10)

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

        f_ir = af.trace(step)(("x", "y"), "!")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=1)

        ir = af.trace(program)(("x", "y"), "!")
        batched = af.batch(ir, in_axes=(False, False))

        assert batched.call(("x", "y"), "!") == ("x!", "y")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_unbatched_fallback_preserves_pytree_out_batched_async(self):
        def step(state, instruction):
            left, right = state
            return af.concat(left, instruction), right

        f_ir = af.trace(step)(("x", "y"), "!")

        def program(init, instruction):
            return af.fixpoint(f_ir, init, instruction, max_iters=1)

        ir = af.trace(program)(("x", "y"), "!")
        batched = af.batch(ir, in_axes=(False, False))

        assert await batched.acall(("x", "y"), "!") == ("x!", "y")
