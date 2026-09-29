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

import json
from collections import namedtuple

import pytest

import autoform as af
from autoform.utils import tree
from tests import aexecute, execute


@tree.dataclasses.dataclass
class Record:
    x: float
    y: str
    label: str = tree.dataclasses.field(pytree_node=False)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "x",
    [
        pytest.param("hello", id="str"),
        pytest.param(0.5, id="float"),
        pytest.param(2, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="none"),
        pytest.param([], id="empty"),
        pytest.param({"x": [1.0, "a"], "y": (None, [], {})}, id="nested"),
        pytest.param({0: "a", "0": "b", "0_": "c"}, id="colliding-keys"),
        pytest.param(Record(0.5, "hello", "static"), id="custom-pytree"),
    ],
)
def test_roundtrip(executor, x):
    def program(x):
        return af.json.decode(af.json.encode(x))

    ir = af.trace(program)(x)
    assert [eqn.prim for eqn in ir.eqns] == [af.json.encode_p, af.json.decode_p]
    assert executor(ir, x) == x
    encoded = af.json.encode(x)
    assert af.core.avalof(encoded) == ir.eqns[0].out_tree.aval
    assert hash(encoded) == hash(af.json.encode(x))


def test_payload_omits_static_metadata():
    x = Record(0.5, "hello", "static")
    encoded = af.json.encode({"record": x, "empty": (None, [], {})})
    assert json.loads(encoded.text) == {"record": {"x": 0.5, "y": "hello"}}
    assert af.json.decode(encoded) == {"record": x, "empty": (None, [], {})}


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("operation", [af.json.encode, af.json.decode], ids=["encode", "decode"])
def test_pushforward_and_pullback(executor, operation):
    x, y = Record(0.5, "hello", "static"), Record(2.0, "change", "static")
    encoded_x, encoded_y = af.json.encode(x), af.json.encode(y)
    if operation is af.json.encode:
        in_value, in_change, out_value, out_change = x, y, encoded_x, encoded_y
    else:
        in_value, in_change, out_value, out_change = encoded_x, encoded_y, x, y
    ir = af.trace(operation)(in_value)
    assert executor(af.pushforward(ir), (in_value,), (in_change,)) == (out_value, out_change)
    assert executor(af.pullback(ir), (in_value,), out_change) == (out_value, (in_change,))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_encoded_cotangents_accumulate_by_field(executor):
    def program(x):
        y = af.json.encode(x)
        return y, y

    x, y, z = {"x": 0.5, "y": "hello"}, {"x": 2.0, "y": "a"}, {"x": 3.0, "y": "b"}
    encoded = af.json.encode(x)
    ir = af.pullback(af.trace(program)(x))
    assert executor(ir, (x,), (af.json.encode(y), af.json.encode(z))) == (
        (encoded, encoded),
        ({"x": 5.0, "y": "ab"},),
    )


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("operation", [af.json.encode, af.json.decode], ids=["encode", "decode"])
def test_symbolic_zero(executor, operation):
    x = {"x": 0.5, "y": "hello"}
    encoded = af.json.encode(x)
    zeros = tree.map(lambda x: af.core.Zero(af.core.avalof(x)), x)
    encoded_zero = af.core.Zero(af.core.avalof(encoded))
    assert af.json.encode(zeros) == encoded_zero
    assert af.json.decode(encoded_zero) == zeros
    assert af.core.materialize_zeros(encoded_zero) == af.json.encode({"x": 0.0, "y": ""})
    if operation is af.json.encode:
        in_value, in_zero, out_value, out_zero = x, zeros, encoded, encoded_zero
    else:
        in_value, in_zero, out_value, out_zero = encoded, encoded_zero, x, zeros
    ir = af.trace(operation)(in_value)
    assert executor(af.pushforward(ir), (in_value,), (in_zero,)) == (out_value, out_zero)
    assert executor(af.pullback(ir), (in_value,), out_zero) == (out_value, (in_zero,))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("operation", [af.json.encode, af.json.decode], ids=["encode", "decode"])
@pytest.mark.parametrize(
    "container",
    [list, tuple, namedtuple("Batch", ["x", "y"])._make, lambda xs: dict(zip(("x", "y"), xs))],
    ids=["list", "tuple", "namedtuple", "dict"],
)
def test_batch_transform_orders(executor, operation, container):
    xs, changes = container([1.0, 2.0]), container([3.0, 4.0])
    values = container([af.json.encode(1.0), af.json.encode(2.0)])
    derivatives = container([af.json.encode(3.0), af.json.encode(4.0)])
    seed = 0.0
    if operation is af.json.decode:
        seed = af.json.encode(seed)
        xs, values, changes, derivatives = values, xs, derivatives, changes
    ir = af.trace(operation)(seed)
    assert executor(af.batch(ir), xs) == values
    assert executor(af.batch(af.pushforward(ir)), (xs,), (changes,)) == (values, derivatives)
    assert executor(af.pushforward(af.batch(ir)), (xs,), (changes,)) == (values, derivatives)
    assert executor(af.batch(af.pullback(ir)), (xs,), derivatives) == (values, (changes,))
    assert executor(af.pullback(af.batch(ir)), (xs,), derivatives) == (values, (changes,))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_batch_broadcast(executor):
    def program(x, y):
        return af.json.encode({"x": x, "y": y}), af.json.decode(af.json.encode(y))

    ir = af.batch(af.trace(program)(0.0, "hello"), in_axes=(True, False))
    expected = [af.json.encode({"x": x, "y": "hello"}) for x in [1.0, 2.0]]
    assert executor(ir, [1.0, 2.0], "hello") == (expected, ["hello", "hello"])


def test_space_mapping_preserves_spec():
    x = af.core.avalof(0.0)
    y = af.core.avalof("")
    aval = af.json.JsonAVal(tree.structure({"x": 0.0}), [x])
    space = af.core.Space("feedback")
    space.set(type(x), lambda _: y)
    assert af.json.map_json_aval(space, aval) == af.json.JsonAVal(aval.spec, [y])
    for space in (af.core.primal_s, af.core.tangent_s, af.core.cotangent_s):
        assert space.map(aval) == aval


def test_pushforward_requires_matching_static_metadata():
    x, y = Record(0.5, "x", "original"), Record(1.0, "dx", "changed")
    with pytest.raises(ValueError, match="identical pytree specs"):
        af.json.pushforward_encode((x, y))


@pytest.mark.parametrize(
    "payload, error",
    [
        pytest.param("{}", ValueError, id="missing-field"),
        pytest.param('{"x": 0.5, "y": 1}', ValueError, id="extra-field"),
        pytest.param("[]", ValueError, id="wrong-container"),
        pytest.param('{"x": "wrong"}', TypeError, id="wrong-leaf"),
        pytest.param('{"x": true}', TypeError, id="bool-not-float"),
        pytest.param('{"x": NaN}', ValueError, id="nan"),
        pytest.param('{"x": Infinity}', ValueError, id="infinity"),
        pytest.param("not json", ValueError, id="invalid-json"),
    ],
)
def test_decode_validates_payload(payload, error):
    aval = af.core.avalof(af.json.encode({"x": 0.5}))
    with pytest.raises(error):
        af.json.decode(af.json.Json(payload, aval))


@pytest.mark.parametrize("x", [float("nan"), float("inf"), -float("inf")])
def test_encode_rejects_non_finite_numbers(x):
    with pytest.raises(ValueError):
        af.json.encode(x)


def test_decode_requires_typed_json():
    with pytest.raises(TypeError, match="Expected JsonAVal"):
        af.json.decode("{}")


def test_cotangent_accumulation_requires_matching_types():
    x, y = af.json.encode({"x": 0.5}), af.json.encode({"x": "wrong"})
    with pytest.raises(TypeError, match="matching specs and leaf types"):
        af.ad.cot_acc([x, y])
