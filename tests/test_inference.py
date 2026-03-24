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


def len_score(text: str) -> float:
    return float(len(text))


class TestFactor:
    def test_factor_returns_local_score(self):
        def program(x):
            return af.factor(x, judge=len_score)

        ir = af.trace(program)("x")
        assert ir.call("hello") == 5.0

    def test_factor_batches_elementwise(self):
        def program(x):
            return af.factor(x, judge=len_score)

        ir = af.trace(program)("x")
        batched_ir = af.batch(ir)
        assert batched_ir.call(["a", "bbb"]) == [1.0, 3.0]

    def test_factor_pushforward_is_unsupported(self):
        def program(x):
            return af.factor(x, judge=len_score)

        ir = af.trace(program)("x")
        pf_ir = af.pushforward(ir)

        with pytest.raises(NotImplementedError, match="no default pushforward rule"):
            pf_ir.call(("hello",), ("delta",))

    def test_factor_pullback_is_unsupported(self):
        def program(x):
            return af.factor(x, judge=len_score)

        ir = af.trace(program)("x")
        pb_ir = af.pullback(ir)

        with pytest.raises(NotImplementedError, match="no default pullback rule"):
            pb_ir.call(("hello",), 1.0)


class TestWeight:
    def test_weight_sums_executed_factors(self):
        def program(x):
            y = af.concat(x, "!")
            af.factor(x, judge=len_score)
            af.factor(y, judge=len_score)
            return y

        ir = af.trace(program)("x")
        weight_ir = af.weight(ir)

        output, total = weight_ir.call("ab")
        assert output == "ab!"
        assert total == 5.0

    def test_weight_only_counts_taken_switch_branch(self):
        def left(x):
            out = af.concat("L:", x)
            af.factor(out, judge=len_score)
            return out

        def right(x):
            out = af.concat("R:", x)
            af.factor(x, judge=len_score)
            return out

        branches = {
            "left": af.trace(left)("x"),
            "right": af.trace(right)("x"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.trace(program)("left", "x")
        weight_ir = af.weight(ir)

        left_out, left_total = weight_ir.call("left", "go")
        right_out, right_total = weight_ir.call("right", "go")

        assert left_out == "L:go"
        assert left_total == 4.0
        assert right_out == "R:go"
        assert right_total == 2.0

    def test_weight_counts_loop_iterations(self):
        def cond(x):
            return af.match(x, "")

        def body(x):
            new_x = af.concat(x, "x")
            af.factor(new_x, judge=len_score)
            return new_x

        cond_ir = af.trace(cond)("")
        body_ir = af.trace(body)("")

        def loop(init):
            return af.while_loop(cond_ir, body_ir, init, max_iters=3)

        ir = af.trace(loop)("")
        weight_ir = af.weight(ir)

        out1, total1 = weight_ir.call("")
        out2, total2 = weight_ir.call("seed")

        assert out1 == "x"
        assert total1 == 1.0
        assert out2 == "seed"
        assert total2 == 0.0

    def test_batch_of_weight_returns_per_trajectory_totals(self):
        def program(x):
            y = af.concat(x, "!")
            af.factor(y, judge=len_score)
            return y

        ir = af.trace(program)("x")
        weight_ir = af.weight(ir)
        batched_ir = af.batch(weight_ir)

        outputs, totals = batched_ir.call(["a", "bbb"])
        assert outputs == ["a!", "bbb!"]
        assert totals == [2.0, 4.0]

    def test_dce_keeps_factor_equations(self):
        def program(x):
            y = af.concat(x, "!")
            af.factor(y, judge=len_score)
            return x

        ir = af.trace(program)("x")
        dced = af.dce(ir)
        weight_ir = af.weight(dced)

        assert len(dced.ir_eqns) == 2
        output, total = weight_ir.call("ab")
        assert output == "ab"
        assert total == 3.0

    @pytest.mark.asyncio(loop_scope="function")
    async def test_weight_async(self):
        def program(x):
            af.factor(x, judge=len_score)
            return af.concat(x, "!")

        ir = af.trace(program)("x")
        weight_ir = af.weight(ir)

        output, total = await weight_ir.acall("abc")
        assert output == "abc!"
        assert total == 3.0
