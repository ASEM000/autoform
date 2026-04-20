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
from autoform.axes import GatheredAVal


class TestNamedAxisBasics:
    def test_axis_size(self):
        def program(x):
            return af.axis_size(axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert batched.call(["a", "b", "c"]) == [3, 3, 3]

    def test_axis_index(self):
        def program(x):
            return af.axis_index(axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert batched.call(["a", "b", "c"]) == [0, 1, 2]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_axis_index_async(self):
        def program(x):
            return af.axis_index(axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert await batched.acall(["a", "b", "c"]) == [0, 1, 2]

    def test_unbound_axis_name_raises(self):
        def program(x):
            return af.axis_size(axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir)

        with pytest.raises(AssertionError, match="unbound axis name"):
            batched.call(["a", "b"])

    def test_mismatched_axis_name_raises(self):
        def program(x):
            return af.axis_index(axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="other")

        with pytest.raises(AssertionError, match="non-current axis"):
            batched.call(["a", "b"])

    def test_axis_name_must_be_keyword_argument(self):
        with pytest.raises(TypeError):
            af.axis_size("items")

        with pytest.raises(TypeError):
            af.axis_index("items")


class TestNamedAxisCollectives:
    def test_axis_gather_batched_leaf(self):
        def program(x):
            return af.axis_gather(x, axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert batched.call(["a", "b", "c"]) == [
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["a", "b", "c"],
        ]

    def test_axis_gather_unbatched_leaf(self):
        def program(x):
            return af.axis_gather("fixed", axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert batched.call(["a", "b"]) == [["fixed", "fixed"], ["fixed", "fixed"]]

    def test_axis_gather_unbatched_leaf_preserves_axis_container(self):
        def program(x):
            return af.axis_gather("fixed", axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert batched.call(("a", "b")) == (("fixed", "fixed"), ("fixed", "fixed"))

    def test_axis_gather_tree(self):
        def program(x, y):
            return af.axis_gather((x, y), axis_name="items")

        ir = af.trace(program)("x", 1)
        batched = af.batch(ir, axis_name="items")

        assert batched.call(["a", "b"], [1, 2]) == (
            [["a", "b"], ["a", "b"]],
            [[1, 2], [1, 2]],
        )

    def test_axis_gather_abstract_preserves_leaf_aval(self):
        def program(x):
            return af.axis_gather(x, axis_name="items")

        ir = af.trace(program)("x")

        assert isinstance(ir.out_ir_tree, af.core.IRVar)
        assert isinstance(ir.out_ir_tree.aval, GatheredAVal)
        assert isinstance(ir.out_ir_tree.aval.elem_aval, af.core.TypedAVal)
        assert ir.out_ir_tree.aval.elem_aval.type is str

    def test_axis_gather_axis_name_must_be_keyword_argument(self):
        with pytest.raises(TypeError):
            af.axis_gather("x", "items")


class TestNamedAxisComposition:
    def test_axis_name_param_is_preserved(self):
        def program(x):
            return af.axis_size(axis_name="items")

        ir = af.trace(program)("x")
        batched = af.batch(ir, axis_name="items")

        assert batched.ir_eqns[0].params["axis_name"] == "items"

    def test_sched_preserves_named_axis(self):
        def program(x):
            i = af.axis_index(axis_name="items")
            return af.format("{}:{}", i, x)

        ir = af.trace(program)("x")
        scheduled = af.sched(ir)
        batched = af.batch(scheduled, axis_name="items")

        assert batched.call(["a", "b"]) == ["0:a", "1:b"]
