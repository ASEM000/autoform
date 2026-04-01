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
from autoform.autoform.analysis import ir_tree_ir_vars


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
