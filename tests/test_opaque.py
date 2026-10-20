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

import autoform.src.core as core
import autoform.src.opaque as opaque
import autoform.src.stage as stage
import autoform.src.utils as utils


def test_opaque_aval_is_one_per_pair():
    lhs, rhs = opaque.AOpaque(object()), opaque.AOpaque(object())
    sealed = opaque.Opaque(lhs.key, {"private": [object()]})

    assert lhs == opaque.AOpaque(lhs.key) and lhs != rhs
    assert len({lhs, opaque.AOpaque(lhs.key), rhs}) == 2
    assert repr(lhs) == f"AOpaque({lhs.key})"
    assert core.avalof(sealed) == lhs
    assert repr(sealed) == f"Opaque({lhs.key})"
    assert "private" not in repr(sealed)
    for space in (core.s_primal, core.s_tangent, core.s_cotangent):
        assert space.map(lhs) == lhs


def test_opaque_aval_join():
    aval, other = opaque.AOpaque(object()), opaque.AOpaque(object())
    assert aval.join(opaque.AOpaque(aval.key)) is aval
    assert aval.join(opaque.Opaque(aval.key, None)) is aval
    assert aval.join(core.Zero(aval)) is aval
    for wrong in (other, opaque.Opaque(other.key, None), core.Zero(other), object(), 1.0):
        with pytest.raises(TypeError, match="Cannot join AOpaque"):
            aval.join(wrong)


def test_trace_opaque_is_one_var_consumed_by_its_pair():
    aval = opaque.AOpaque(object())
    sealed = opaque.Opaque(aval.key, ("private", [1.0, 2.0]))

    ir = stage.trace(lambda x, o: (x, o))("seed", sealed)

    x, o = ir.in_tree
    assert stage.is_var(o) and o.aval == aval
    assert ir.out_tree == (x, o)
    assert ir.call("hello", sealed) == ("hello", sealed)
    assert utils.tree.leaves(sealed) == ["private", 1.0, 2.0]
    with pytest.raises(TypeError, match="Cannot join AOpaque"):
        ir.call("hello", opaque.Opaque(object(), None))


def test_seal_and_unseal():
    aval = opaque.AOpaque(object())
    raw = ("private", [1.0, 2.0])

    sealed = opaque.seal(raw, key=aval.key)
    assert isinstance(sealed, opaque.Opaque) and sealed.key == aval.key and sealed.value is raw
    assert opaque.unseal(sealed, key=aval.key) is raw
    assert opaque.seal(core.Zero(opaque.AOpaque()), key=aval.key) == core.Zero(aval)
    assert opaque.unseal(core.Zero(aval), key=aval.key) == core.Zero(opaque.AOpaque())
    with pytest.raises(TypeError, match="Expected a value sealed with"):
        opaque.unseal(sealed, key=object())
    with pytest.raises(AssertionError, match="Opaque requires a key"):
        opaque.Opaque(None, raw)


def test_seal_and_unseal_check_their_input_types():
    key = object()
    abstract_seal = core.r_abstract.get(opaque.p_seal)
    abstract_unseal = core.r_abstract.get(opaque.p_unseal)

    assert abstract_seal(opaque.AOpaque(), key=key) == opaque.AOpaque(key)
    assert abstract_unseal(opaque.AOpaque(key), key=key) == opaque.AOpaque()
    for wrong in (opaque.AOpaque(key), core.avalof("")):
        with pytest.raises(TypeError, match=r"Expected AOpaque\(\)"):
            abstract_seal(wrong, key=key)
    for wrong in (opaque.AOpaque(object()), opaque.AOpaque(), core.avalof("")):
        with pytest.raises(TypeError, match="Expected AOpaque"):
            abstract_unseal(wrong, key=key)
    with pytest.raises(TypeError, match="Expected an unsealed value"):
        opaque.seal(opaque.Opaque(key, None), key=key)


def test_unkeyed_opaque_aval_accepts_raw_values_only():
    raw = opaque.AOpaque()
    sealed = opaque.AOpaque(object())

    assert raw == opaque.AOpaque() and repr(raw) == "AOpaque()"
    assert raw.join(("private", object())) is raw and raw.join(core.Zero(raw)) is raw
    for wrong in (sealed, core.Zero(sealed), opaque.Opaque(sealed.key, None), core.avalof("")):
        with pytest.raises(TypeError, match="Cannot join AOpaque"):
            raw.join(wrong)
