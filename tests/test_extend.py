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

import functools as ft

import autoform as af
import autoform.extend as afe


def make_box_domain():
    class Box:
        __slots__ = ["value"]

        def __init__(self, value: int):
            self.value = value

        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value

    class BoxAVal(afe.AVal):
        __slots__ = []

        def __eq__(self, other):
            return type(self) is type(other)

        def __hash__(self):
            return hash(type(self))

    return Box, BoxAVal


def test_register_trace_type():
    Box, BoxAVal = make_box_domain()
    aval_rule = lambda value: BoxAVal()
    assert afe.register_trace_type(Box, aval_rule) is aval_rule

    ir = af.trace(lambda x: x)(Box(1))

    assert afe.primal_s.avalof(Box(1)) == BoxAVal()
    assert ir.in_tree[0].aval == BoxAVal()


def test_scalar_aval_reexports():
    assert afe.StrAVal is af.core.StrAVal
    assert afe.IntAVal is af.core.IntAVal
    assert afe.FloatAVal is af.core.FloatAVal
    assert afe.BoolAVal is af.core.BoolAVal
    assert afe.Space is af.core.Space
    assert afe.primal_s is af.core.primal_s


def test_register_zero_and_cotangent_accumulator():
    Box, BoxAVal = make_box_domain()
    afe.register_trace_type(Box, lambda value: BoxAVal())
    zero_rule = lambda aval: Box(0)
    cot_acc_rule = lambda cotangents, aval: Box(sum(c.value for c in cotangents))
    assert afe.register_zero(BoxAVal, zero_rule) is zero_rule
    assert afe.register_cotangent_accumulator(BoxAVal, cot_acc_rule) is cot_acc_rule

    assert afe.materialize(afe.Zero(BoxAVal())) == Box(0)
    assert af.ad.cot_acc([Box(1), Box(2)]) == Box(3)


def test_register_add_with_primitive_rules():
    Box, BoxAVal = make_box_domain()
    box_add_p = afe.Prim("test_box_add")

    def box_add(x, y):
        return box_add_p.bind((x, y))

    def impl_add(in_tree):
        x, y = in_tree
        return Box(x.value + y.value)

    def abstract_add(in_tree):
        del in_tree
        return BoxAVal()

    afe.register_trace_type(Box, lambda value: BoxAVal())
    afe.register_add(BoxAVal, box_add)
    afe.impl_rules.set(box_add_p, impl_add)
    afe.abstract_rules.set(box_add_p, abstract_add)

    ir = af.trace(lambda x, y: x + y)(Box(1), Box(2))

    assert ir.call(Box(3), Box(4)) == Box(7)


def test_operator_registration_helpers():
    _, BoxAVal = make_box_domain()

    def rule(x, y):
        return x, y

    cases = [
        (afe.register_add, af.core.trace_add_rules),
        (afe.register_sub, af.core.trace_sub_rules),
        (afe.register_mul, af.core.trace_mul_rules),
        (afe.register_div, af.core.trace_truediv_rules),
        (afe.register_matmul, af.core.trace_matmul_rules),
        (afe.register_eq, af.core.trace_eq_rules),
    ]

    for register, registry in cases:
        assert register(BoxAVal, rule) is rule
        assert registry[BoxAVal] is rule


def test_registration_helpers_work_as_decorators():
    Box, BoxAVal = make_box_domain()

    @ft.partial(afe.register_trace_type, Box)
    def aval_rule(value):
        return BoxAVal()

    @ft.partial(afe.register_zero, BoxAVal)
    def zero_rule(aval):
        return Box(0)

    @ft.partial(afe.register_add, BoxAVal)
    def add_rule(x, y):
        return x, y

    assert isinstance(afe.primal_s.avalof(Box(1)), BoxAVal)
    assert af.ad.zero_rules[BoxAVal] is zero_rule
    assert af.core.trace_add_rules[BoxAVal] is add_rule
