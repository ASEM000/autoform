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
import autoform.extend as afe


def test_pullback_stages_cotangent_fan_in():
    ir = af.trace(lambda x: af.concat(x, x))("x")

    transformed = af.pullback(ir)

    assert [eqn.prim.name for eqn in transformed.eqns] == [
        "concat",
        "cotangent_accumulate",
    ]
    assert transformed.call(("p",), "c") == ("pp", ("cc",))


def test_staged_transforms_preserve_equation_tags():
    def program(x):
        with af.tag("source"):
            return af.concat(x, "!")

    ir = af.trace(program)("x")

    assert all(eqn.tags == frozenset({"source"}) for eqn in af.pushforward(ir).eqns)
    assert all(eqn.tags == frozenset({"source"}) for eqn in af.pullback(ir).eqns)


def test_higher_order_pushforward_stays_flat():
    ir = af.trace(lambda x: af.concat(x, "!"))("x")

    transformed = af.pushforward(af.pushforward(ir))

    assert [eqn.prim.name for eqn in transformed.eqns] == ["concat"] * 4


def test_transforms_are_not_primitive_keys():
    assert not hasattr(afe, "pushforward_call_p")
    assert not hasattr(afe, "pullback_call_p")
