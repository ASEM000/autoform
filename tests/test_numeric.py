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

import autoform as af


def test_numeric_dunders_form_one_scalar_program():
    def program(x, y):
        score = -(x + 2) * y / 2
        return score, x - y, x < y, x != y

    ir = af.trace(program)(1.0, 2.0)

    assert [eqn.prim for eqn in ir.eqns] == [
        af.numeric.add_p,
        af.numeric.neg_p,
        af.numeric.mul_p,
        af.numeric.div_p,
        af.numeric.sub_p,
        af.numeric.lt_p,
        af.numeric.ne_p,
    ]
    assert ir.call(3.0, 4.0) == (-10.0, -1.0, True, True)


def test_reverse_numeric_dunders_promote_integer_literals():
    def program(x):
        return 2 + x, 2 - x, 2 * x, 8 / x

    ir = af.trace(program)(2.0)

    assert ir.call(4.0) == (6.0, -2.0, 8.0, 2.0)


def test_arithmetic_composes_under_ad():
    ir = af.trace(lambda x: x * x + x / 2)(2.0)

    primal, tangent = af.pushforward(ir).call((2.0,), (1.0,))
    assert primal == 5.0
    assert tangent == 4.5

    primal, (cotangent,) = af.pullback(ir).call((2.0,), 1.0)
    assert primal == 5.0
    assert cotangent == 4.5


def test_numeric_batching_composes_unary_and_binary_primitives():
    ir = af.trace(lambda x, scale: x * x + x * scale)(1.0, 1.0)
    batched_ir = af.batch(ir, in_axes=(True, False))

    result = batched_ir.call([1.0, 2.0, 3.0], 2.0)

    assert result == [3.0, 8.0, 15.0]


def test_comparison_blocks_derivatives():
    ir = af.trace(lambda x: x >= 0)(1.0)

    primal, tangent = af.pushforward(ir).call((1.0,), (1.0,))
    assert primal is True
    assert af.ad.is_zero(tangent)
    assert tangent.aval == af.core.BoolAVal()

    primal, (cotangent,) = af.pullback(ir).call((1.0,), True)
    assert primal is True
    assert af.ad.is_zero(cotangent)
    assert cotangent.aval == af.core.FloatAVal()
