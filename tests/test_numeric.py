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

from collections import namedtuple

import pytest

import autoform as af
from tests import aexecute, execute


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


def test_comparison_blocks_pushforward():
    ir = af.pushforward(af.trace(lambda x: x >= 0)(1.0))
    primal, derivative = ir.call((1.0,), (1.0,))
    assert primal is True
    assert isinstance(derivative, af.core.Zero)
    assert derivative.aval == af.numeric.BoolAVal()


def test_comparison_blocks_pullback():
    ir = af.pullback(af.trace(lambda x: x >= 0)(1.0))
    primal, (derivative,) = ir.call((1.0,), True)
    assert primal is True
    assert isinstance(derivative, af.core.Zero)
    assert derivative.aval == af.numeric.FloatAVal()


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "operation, primal, tangent",
    [
        pytest.param(af.numeric.add, 3.0, 3.0, id="add"),
        pytest.param(af.numeric.sub, 1.0, -1.0, id="sub"),
        pytest.param(af.numeric.mul, 2.0, 5.0, id="mul"),
        pytest.param(af.numeric.div, 2.0, -3.0, id="div"),
    ],
)
def test_binary_pushforward(executor, operation, primal, tangent):
    ir = af.pushforward(af.trace(operation)(2.0, 1.0))
    actual = executor(ir, (2.0, 1.0), (1.0, 2.0))
    assert actual == (primal, tangent)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "operation, primal, cotangents",
    [
        pytest.param(af.numeric.add, 3.0, (1.0, 1.0), id="add"),
        pytest.param(af.numeric.sub, 1.0, (1.0, -1.0), id="sub"),
        pytest.param(af.numeric.mul, 2.0, (1.0, 2.0), id="mul"),
        pytest.param(af.numeric.div, 2.0, (1.0, -2.0), id="div"),
    ],
)
def test_binary_pullback(executor, operation, primal, cotangents):
    ir = af.pullback(af.trace(operation)(2.0, 1.0))
    actual = executor(ir, (2.0, 1.0), 1.0)
    assert actual == (primal, cotangents)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "operation, expected",
    [
        pytest.param(af.numeric.add, 3.0, id="add"),
        pytest.param(af.numeric.sub, 1.0, id="sub"),
        pytest.param(af.numeric.mul, 2.0, id="mul"),
        pytest.param(af.numeric.div, 2.0, id="div"),
        pytest.param(af.numeric.eq, False, id="eq"),
        pytest.param(af.numeric.ne, True, id="ne"),
        pytest.param(af.numeric.lt, False, id="lt"),
        pytest.param(af.numeric.le, False, id="le"),
        pytest.param(af.numeric.gt, True, id="gt"),
        pytest.param(af.numeric.ge, True, id="ge"),
    ],
)
def test_binary_promotion_and_batching(executor, operation, expected):
    assert operation(2, 1) == expected
    ir = af.batch(af.trace(operation)(2.0, 1.0), in_axes=(True, False))
    args = ([2.0], 1.0)
    actual = executor(ir, *args)
    assert actual == [expected]


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected",
    [
        pytest.param(af.pushforward, ((2.0,), (1.0,)), (-2.0, -1.0), id="pushforward"),
        pytest.param(af.pullback, ((2.0,), 1.0), (-2.0, (-1.0,)), id="pullback"),
        pytest.param(af.batch, ([1.0, 2.0, 3.0],), [-1.0, -2.0, -3.0], id="batch"),
    ],
)
def test_negation(executor, transform, args, expected):
    assert af.numeric.neg(2) == -2.0
    ir = transform(af.trace(af.numeric.neg)(2.0))
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "container",
    [list, tuple, namedtuple("Batch", ["x", "y"])._make, lambda xs: dict(zip(("x", "y"), xs))],
    ids=["list", "tuple", "namedtuple", "dict"],
)
def test_negation_batch_container(executor, container):
    ir = af.trace(af.numeric.neg)(0.0)
    xs, coefficients = container([1.0, 2.0]), container([3.0, 4.0])
    values, gradients = container([-1.0, -2.0]), container([-3.0, -4.0])
    assert executor(af.batch(ir), xs) == values
    assert executor(af.batch(af.pullback(ir)), (xs,), coefficients) == (values, (gradients,))
    assert executor(af.batch(af.pushforward(ir)), (xs,), (coefficients,)) == (values, gradients)
    assert executor(af.pullback(af.batch(ir)), (xs,), coefficients) == (values, (gradients,))
    assert executor(af.pushforward(af.batch(ir)), (xs,), (coefficients,)) == (values, gradients)
