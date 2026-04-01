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
from autoform.analysis import ir_tree_ir_vars, ir_var_producers


class TestIrTreeIrVars:
    def test_returns_input_ir_vars_in_leaf_order(self):
        def program(payload):
            head, (left, right) = payload
            return af.format("{} {} {}", head, left, right)

        ir = af.trace(program)(("head", ("left", "right")))
        (payload,) = ir.in_ir_tree
        head, pair = payload

        assert ir_tree_ir_vars(ir.in_ir_tree) == (head, pair[0], pair[1])

    def test_filters_static_input_literals(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        assert ir_tree_ir_vars(ir.in_ir_tree) == (ir.in_ir_tree[1],)

    def test_returns_output_ir_vars_in_leaf_order(self):
        def program(x):
            left = af.concat(x, "1")
            right = af.concat(x, "2")
            return ({"left": left}, (right, "const"))

        ir = af.trace(program)("seed")
        left_tree, right_tree = ir.out_ir_tree

        assert ir_tree_ir_vars(ir.out_ir_tree) == (left_tree["left"], right_tree[0])

    def test_filters_literal_outputs(self):
        def program(x):
            return ("const", {"value": x})

        ir = af.trace(program)("seed")

        assert ir_tree_ir_vars(ir.out_ir_tree) == (ir.out_ir_tree[1]["value"],)


class TestIrVarProducers:
    def test_maps_each_output_ir_var_to_its_producer(self):
        def program(x):
            left = af.concat(x, "1")
            right = af.concat(left, "2")
            return left, right

        ir = af.trace(program)("seed")
        first_eqn, second_eqn = ir.ir_eqns
        left, right = ir.out_ir_tree

        assert ir_var_producers(ir) == {left: first_eqn, right: second_eqn}

    def test_includes_all_ir_vars_from_tree_outputs(self):
        def program(x):
            pair = af.concat(x, "!")
            return {"value": pair, "original": x}

        ir = af.trace(program)("seed")
        producers = ir_var_producers(ir)
        produced = ir.out_ir_tree["value"]

        assert producers == {produced: ir.ir_eqns[0]}

    def test_errors_if_same_ir_var_is_produced_twice(self):
        shared = af.core.IRVar.fresh(type=str)
        eqn_a = af.core.IREqn(af.core.Prim("a"), None, (), shared, {})
        eqn_b = af.core.IREqn(af.core.Prim("b"), None, (), shared, {})
        ir = af.core.IR([eqn_a, eqn_b], in_ir_tree=(), out_ir_tree=shared)

        with pytest.raises(AssertionError, match="produced by multiple equations"):
            ir_var_producers(ir)
