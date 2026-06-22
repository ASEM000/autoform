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
from autoform.analysis import (
    eqn_graph,
    ir_liveness,
    var_leaves,
    var_producers,
)


class TestIrVarLeaves:
    def test_returns_input_vars_in_leaf_order(self):
        def program(payload):
            head, (left, right) = payload
            return af.format("{} {} {}", head, left, right)

        ir = af.trace(program)(("head", ("left", "right")))
        (payload,) = ir.in_tree
        head, pair = payload

        assert var_leaves(ir.in_tree) == [head, pair[0], pair[1]]

    def test_filters_static_input_literals(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        assert var_leaves(ir.in_tree) == [ir.in_tree[1]]

    def test_returns_output_vars_in_leaf_order(self):
        def program(x):
            left = af.concat(x, "1")
            right = af.concat(x, "2")
            return ({"left": left}, (right, "const"))

        ir = af.trace(program)("seed")
        left_tree, right_tree = ir.out_tree

        assert var_leaves(ir.out_tree) == [left_tree["left"], right_tree[0]]

    def test_filters_literal_outputs(self):
        def program(x):
            return ("const", {"value": x})

        ir = af.trace(program)("seed")

        assert var_leaves(ir.out_tree) == [ir.out_tree[1]["value"]]


class TestIrVarProducers:
    def test_maps_each_output_var_to_its_producer(self):
        def program(x):
            left = af.concat(x, "1")
            right = af.concat(left, "2")
            return left, right

        ir = af.trace(program)("seed")
        first_eqn, second_eqn = ir.eqns
        left, right = ir.out_tree

        assert var_producers(ir) == {left: first_eqn, right: second_eqn}

    def test_includes_all_vars_from_tree_outputs(self):
        def program(x):
            pair = af.concat(x, "!")
            return {"value": pair, "original": x}

        ir = af.trace(program)("seed")
        producers = var_producers(ir)
        produced = ir.out_tree["value"]

        assert producers == {produced: ir.eqns[0]}

    def test_errors_if_same_var_is_produced_twice(self):
        shared = af.core.Var.fresh(aval=af.core.StrAVal())
        eqn_a = af.core.Eqn(af.core.Prim("a"), (), shared, {})
        eqn_b = af.core.Eqn(af.core.Prim("b"), (), shared, {})
        ir = af.core.IR([eqn_a, eqn_b], in_tree=(), out_tree=shared)

        with pytest.raises(AssertionError):
            var_producers(ir)


class TestIrEqnDependencyGraph:
    def test_returns_empty_graph_for_empty_ir(self):
        def program(x):
            return x

        ir = af.trace(program)("seed")

        assert eqn_graph(ir) == {}

    def test_includes_independent_equations_with_empty_children(self):
        def program(a, b):
            left = af.format("{}", a)
            right = af.format("{}", b)
            return left, right

        ir = af.trace(program)("a", "b")
        left_eqn, right_eqn = ir.eqns

        assert eqn_graph(ir) == {left_eqn: [], right_eqn: []}

    def test_maps_parent_equations_to_children(self):
        def program(x):
            a = af.format("{}", x)
            b = af.concat(a, "!")
            c = af.concat(b, "?")
            return c

        ir = af.trace(program)("seed")
        a_eqn, b_eqn, c_eqn = ir.eqns

        assert eqn_graph(ir) == {a_eqn: [b_eqn], b_eqn: [c_eqn], c_eqn: []}

    def test_dedupes_repeated_input_dependencies(self):
        def program(x):
            a = af.format("{}", x)
            b = af.concat(a, a)
            return b

        ir = af.trace(program)("seed")
        a_eqn, b_eqn = ir.eqns

        assert eqn_graph(ir) == {a_eqn: [b_eqn], b_eqn: []}


class TestIrLiveness:
    def test_empty_ir_returns_single_boundary(self):
        def program(x):
            return x

        ir = af.trace(program)("seed")
        (x,) = ir.in_tree

        assert ir_liveness(ir) == [{x}]

    def test_empty_ir_respects_partial_output_mask(self):
        def program(x, y):
            return x, y

        ir = af.trace(program)("x", "y")
        x, y = ir.in_tree

        assert ir_liveness(ir, out_used=(True, False)) == [{x}]
        assert ir_liveness(ir, out_used=(False, True)) == [{y}]
        assert ir_liveness(ir, out_used=(False, False)) == [set()]

    def test_chain_returns_boundary_liveness(self):
        def program(x):
            a = af.format("{}", x)
            b = af.concat(a, "!")
            c = af.concat(b, "?")
            return c

        ir = af.trace(program)("seed")
        (x,) = ir.in_tree
        a, b, c = (eqn.out_tree for eqn in ir.eqns)

        assert ir_liveness(ir) == [{x}, {a}, {b}, {c}]

    def test_parallel_equations_keep_suffix_live_ins(self):
        def program(a, b):
            left = af.format("{}", a)
            right = af.format("{}", b)
            return left, right

        ir = af.trace(program)("left", "right")
        a, b = ir.in_tree
        left, right = (eqn.out_tree for eqn in ir.eqns)

        assert ir_liveness(ir) == [{a, b}, {b, left}, {left, right}]

    def test_partial_output_mask_reduces_output_boundary_liveness(self):
        def program(x):
            a = af.concat(x, "a")
            b = af.concat(x, "b")
            return a, b

        ir = af.trace(program)("seed")
        (x,) = ir.in_tree
        a, b = (eqn.out_tree for eqn in ir.eqns)

        assert ir_liveness(ir, out_used=(True, False)) == [{x}, {x, a}, {a}]

    def test_static_inputs_do_not_become_live_vars(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")
        out_var = ir.out_tree

        assert ir_liveness(ir) == [{ir.in_tree[1]}, {out_var}]
