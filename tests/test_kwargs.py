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

import pytest

import autoform as af

greet_p = af.core.Prim("greet")


def greet(name: str, *, greeting: str = "Hello", punctuation: str = "!") -> str:
    return greet_p.bind((name, dict(greeting=greeting, punctuation=punctuation)))


@ft.partial(af.core.impl_rules.set, greet_p)
def impl_greet(in_tree) -> str:
    match in_tree:
        case (name, {"greeting": greeting, "punctuation": punctuation}):
            return f"{greeting}, {name}{punctuation}"


@ft.partial(af.core.abstract_rules.set, greet_p)
def abstract_greet(in_tree) -> af.core.AVal:
    return af.core.AVal(str)


@ft.partial(af.core.push_rules.set, greet_p)
def pushforward_greet(in_tree):
    primals, tangents = in_tree
    tangents = af.ad.materialize(tangents)
    return impl_greet(primals), impl_greet(tangents)


@ft.partial(af.core.pull_fwd_rules.set, greet_p)
def pullback_fwd_greet(in_tree):
    out = impl_greet(in_tree)
    residuals = in_tree
    return out, residuals


@ft.partial(af.core.pull_bwd_rules.set, greet_p)
def pullback_bwd_greet(residuals, out_cotangent):
    match residuals:
        case (_, {"greeting": greeting, "punctuation": punct}):
            return (
                out_cotangent,
                {
                    "greeting": af.ad.Zero(type(greeting)),
                    "punctuation": af.ad.Zero(type(punct)),
                },
            )


class TestKwargsBuildIR:
    def test_kwargs_in_ir(self):
        def program(name):
            return greet(name, greeting="Hi", punctuation="?")

        ir = af.trace(program)("World")
        assert len(ir.ir_eqns) == 1
        eqn = ir.ir_eqns[0]
        assert eqn.prim == greet_p
        match eqn.in_ir_tree:
            case (name, {"greeting": greeting, "punctuation": punctuation}):
                assert isinstance(name, af.core.IRVar)
                assert isinstance(greeting, af.core.IRLit)
                assert isinstance(punctuation, af.core.IRLit)
            case _:
                pytest.fail("Unexpected in_ir_tree structure")

    def test_kwargs_execution(self):
        def program(name):
            return greet(name, greeting="Hi", punctuation="?")

        ir = af.trace(program)("World")
        result = ir.call("World")
        assert result == "Hi, World?"


class TestKeywordArgumentBoundary:
    def test_trace_rejects_kwargs(self):
        def program(x, *, repeat=1):
            return greet(x, greeting="Hi", punctuation="!" * repeat)

        with pytest.raises(AssertionError, match="trace.*keyword arguments"):
            af.trace(program)("A", repeat=3)

    def test_call_rejects_kwargs(self):
        def program(name, punctuation):
            return af.format("Hello, {}{}", name, punctuation)

        ir = af.trace(program)("World", "!")
        with pytest.raises(AssertionError, match="call.*keyword arguments"):
            ir.call("World", punctuation="?")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_acall_rejects_kwargs(self):
        def program(name, punctuation):
            return af.format("Hello, {}{}", name, punctuation)

        ir = af.trace(program)("World", "!")
        with pytest.raises(AssertionError, match="acall.*keyword arguments"):
            await ir.acall("World", punctuation="?")

    def test_switch_rejects_kwargs(self):
        branches = {"a": af.trace(lambda x: af.concat("A:", x))("X")}

        with pytest.raises(AssertionError, match="switch.*keyword arguments"):
            af.switch("a", branches, x="test")


class TestKwargsPullback:
    def test_pullback_greet_ir_structure(self):
        def program(name):
            return greet(name, greeting="Hi", punctuation="?")

        ir = af.trace(program)("World")
        pb_ir = af.pullback(ir)
        assert len(pb_ir.in_ir_tree) == 2
        assert len(pb_ir.out_ir_tree) == 2
