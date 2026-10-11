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

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.core as core
import autoform.src.lm as lm
import autoform.src.numeric as numeric
import autoform.src.stage as stage
from tests import aexecute, execute


@pytest.mark.parametrize(
    "x",
    [
        pytest.param(core.Zero(numeric.FloatAVal()), id="symbolic-zero"),
        pytest.param(lm.Float(), id="schema"),
    ],
)
def test_comparison_requires_concrete_floats(x):
    with pytest.raises(AssertionError, match="Expected floats"):
        numeric.eq(x, x)


def test_numeric_dunders_form_one_scalar_program():
    def program(x, y):
        score = -(x + 2) * y / 2
        return score, x - y, x < y, x != y

    ir = stage.trace(program)(1.0, 2.0)

    assert [eqn.prim for eqn in ir.eqns] == [
        numeric.add_p,
        numeric.neg_p,
        numeric.mul_p,
        numeric.div_p,
        numeric.sub_p,
        numeric.lt_p,
        numeric.ne_p,
    ]
    assert ir.call(3.0, 4.0) == (-10.0, -1.0, True, True)


def test_reverse_numeric_dunders_promote_integer_literals():
    def program(x):
        return 2 + x, 2 - x, 2 * x, 8 / x

    ir = stage.trace(program)(2.0)

    assert ir.call(4.0) == (6.0, -2.0, 8.0, 2.0)


def test_comparison_blocks_pushforward():
    ir = ad.pushforward(stage.trace(lambda x: x >= 0)(1.0))
    p, derivative = ir.call((1.0,), (1.0,))
    assert p is True
    assert isinstance(derivative, core.Zero)
    assert derivative.aval == numeric.BoolAVal()


def test_comparison_blocks_pullback():
    ir = ad.pullback(stage.trace(lambda x: x >= 0)(1.0))
    p, (derivative,) = ir.call((1.0,), True)
    assert p is True
    assert isinstance(derivative, core.Zero)
    assert derivative.aval == numeric.FloatAVal()


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "operation, p, t",
    [
        pytest.param(numeric.add, 3.0, 3.0, id="add"),
        pytest.param(numeric.sub, 1.0, -1.0, id="sub"),
        pytest.param(numeric.mul, 2.0, 5.0, id="mul"),
        pytest.param(numeric.div, 2.0, -3.0, id="div"),
    ],
)
def test_binary_pushforward(executor, operation, p, t):
    ir = ad.pushforward(stage.trace(operation)(2.0, 1.0))
    actual = executor(ir, (2.0, 1.0), (1.0, 2.0))
    assert actual == (p, t)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "operation, p, c",
    [
        pytest.param(numeric.add, 3.0, (1.0, 1.0), id="add"),
        pytest.param(numeric.sub, 1.0, (1.0, -1.0), id="sub"),
        pytest.param(numeric.mul, 2.0, (1.0, 2.0), id="mul"),
        pytest.param(numeric.div, 2.0, (1.0, -2.0), id="div"),
    ],
)
def test_binary_pullback(executor, operation, p, c):
    ir = ad.pullback(stage.trace(operation)(2.0, 1.0))
    actual = executor(ir, (2.0, 1.0), 1.0)
    assert actual == (p, c)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "operation, expected",
    [
        pytest.param(numeric.add, 3.0, id="add"),
        pytest.param(numeric.sub, 1.0, id="sub"),
        pytest.param(numeric.mul, 2.0, id="mul"),
        pytest.param(numeric.div, 2.0, id="div"),
        pytest.param(numeric.eq, False, id="eq"),
        pytest.param(numeric.ne, True, id="ne"),
        pytest.param(numeric.lt, False, id="lt"),
        pytest.param(numeric.le, False, id="le"),
        pytest.param(numeric.gt, True, id="gt"),
        pytest.param(numeric.ge, True, id="ge"),
    ],
)
def test_binary_promotion_and_batching(executor, operation, expected):
    assert operation(2, 1) == expected
    ir = axis.batch(stage.trace(operation)(2.0, 1.0), in_axes=(True, False))
    args = ([2.0], 1.0)
    actual = executor(ir, *args)
    assert actual == [expected]


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, expected",
    [
        pytest.param(ad.pushforward, ((2.0,), (1.0,)), (-2.0, -1.0), id="pushforward"),
        pytest.param(ad.pullback, ((2.0,), 1.0), (-2.0, (-1.0,)), id="pullback"),
        pytest.param(axis.batch, ([1.0, 2.0, 3.0],), [-1.0, -2.0, -3.0], id="batch"),
    ],
)
def test_negation(executor, transform, args, expected):
    assert numeric.neg(2) == -2.0
    ir = transform(stage.trace(numeric.neg)(2.0))
    actual = executor(ir, *args)
    assert actual == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "container",
    [list, tuple, namedtuple("Batch", ["x", "y"])._make, lambda xs: dict(zip(("x", "y"), xs))],
    ids=["list", "tuple", "namedtuple", "dict"],
)
def test_negation_batch_container(executor, container):
    ir = stage.trace(numeric.neg)(0.0)
    xs, coefficients = container([1.0, 2.0]), container([3.0, 4.0])
    values, gradients = container([-1.0, -2.0]), container([-3.0, -4.0])
    assert executor(axis.batch(ir), xs) == values
    assert executor(axis.batch(ad.pullback(ir)), (xs,), coefficients) == (values, (gradients,))
    assert executor(axis.batch(ad.pushforward(ir)), (xs,), (coefficients,)) == (values, gradients)
    assert executor(ad.pullback(axis.batch(ir)), (xs,), coefficients) == (values, (gradients,))
    assert executor(ad.pushforward(axis.batch(ir)), (xs,), (coefficients,)) == (values, gradients)
