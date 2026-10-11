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


import autoform.src.ad as ad
import autoform.src.core as core
import autoform.src.extend as extend
import autoform.src.stage as stage


def make_box_domain():
    class Box:
        __slots__ = ["value"]

        def __init__(self, value: int):
            self.value = value

        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value

    class BoxAVal(core.AVal):
        __slots__ = []

        def __eq__(self, other):
            return type(self) is type(other)

        def __hash__(self):
            return hash(type(self))

        def zero(self):
            return Box(0)

        def accum(self, c):
            return Box(sum(c.value for c in c))

    core.aval_types[BoxAVal] = lambda aval: aval
    return Box, BoxAVal


def test_register_trace_type():
    Box, BoxAVal = make_box_domain()
    extend.register_trace_type(Box, lambda _: BoxAVal())

    ir = stage.trace(lambda x: x)(Box(1))

    assert core.avalof(Box(1)) == BoxAVal()
    assert ir.in_tree[0].aval == BoxAVal()


def test_aval_zero_and_accumulation():
    Box, BoxAVal = make_box_domain()
    extend.register_trace_type(Box, lambda value: BoxAVal())

    assert core.materialize_zeros(core.Zero(BoxAVal())) == Box(0)
    assert ad.cot_accum([Box(1), Box(2)]) == Box(3)


def test_register_dunder_with_primitive_rules():
    Box, BoxAVal = make_box_domain()

    def box_add(x, y):
        return box_add_p.bind((x, y))

    def impl_add(in_tree):
        x, y = in_tree
        return Box(x.value + y.value)

    extend.register_trace_type(Box, lambda value: BoxAVal())
    extend.register_dunder(stage.Dunder.ADD, BoxAVal, box_add)
    box_add_p = core.Prim("test_box_add")
    extend.register_impl(box_add_p, impl_add)
    extend.register_abstract(box_add_p, lambda _: BoxAVal())

    ir = stage.trace(lambda x, y: x + y)(Box(1), Box(2))

    assert ir.call(Box(3), Box(4)) == Box(7)


def test_register_dunder_with_static_python_protocol():
    Box, BoxAVal = make_box_domain()
    extend.register_trace_type(Box, lambda value: BoxAVal())
    extend.register_dunder(stage.Dunder.LEN, BoxAVal, lambda value: 1)

    ir = stage.trace(lambda x: len(x))(Box(1))

    assert ir.call(Box(2)) == 1
