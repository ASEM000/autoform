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

import autoform.src.ad as ad
import autoform.src.core as core
import autoform.src.stage as stage
import autoform.src.tree as tree
import autoform.src.untyped as untyped


def test_untyped_aval_is_one_type():
    aval = untyped.AUntyped()
    assert aval == untyped.AUntyped() and len({aval, untyped.AUntyped()}) == 1
    assert repr(aval) == "AUntyped()"
    for space in (core.s_primal, core.s_tangent, core.s_cotangent):
        assert space.map(aval) == aval


def test_untyped_aval_join():
    aval = untyped.AUntyped()
    assert aval.join(untyped.AUntyped()) is aval
    assert aval.join(untyped.Untyped(None)) is aval
    assert aval.join(core.Zero(aval)) is aval
    for wrong in (core.avalof(""), core.Zero(core.avalof("")), object(), 1.0, ("raw",)):
        with pytest.raises(TypeError, match="Cannot join AUntyped"):
            aval.join(wrong)


def test_untyped_hides_its_value():
    box = untyped.Untyped({"private": [object()]})
    assert core.avalof(box) == untyped.AUntyped()
    assert repr(box) == "Untyped()" and "private" not in repr(box)
    assert tree.leaves(box) == box.value["private"]


def test_trace_untyped_is_one_var():
    box = untyped.Untyped(("private", [1.0, 2.0]))
    ir = stage.trace(lambda x, o: (x, o))("seed", box)

    x, o = ir.in_tree
    assert stage.is_var(o) and o.aval == untyped.AUntyped()
    assert ir.out_tree == (x, o)
    assert ir.call("hello", box) == ("hello", box)
    with pytest.raises(TypeError, match="Cannot join AUntyped"):
        ir.call("hello", ("private", [1.0, 2.0]))


def test_untyped_hides_traced_values_and_passes_gradients():
    ir = stage.trace(lambda x: untyped.untyped(x))("seed")
    var = ir.out_tree
    assert stage.is_var(var) and var.aval == untyped.AUntyped()
    assert ir.call("hello").value == "hello"

    unbox = core.Prim("unbox")
    core.r_impl.set(unbox, lambda box: box.value)
    core.r_abstract.set(unbox, lambda _: core.avalof(""))
    core.r_pull_fwd.set(unbox, lambda box: (box.value, None))
    core.r_pull_bwd.set(unbox, lambda in_tree: untyped.Untyped(in_tree[1]))
    ir = stage.trace(lambda x: unbox.bind(untyped.untyped(x)) + "!")("seed")
    assert ad.pullback(ir).call(("hello",), "df") == ("hello!", ("df",))

    with pytest.raises(TypeError, match="Expected a typed value"):
        untyped.untyped(untyped.Untyped(None))
