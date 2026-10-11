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
import autoform.src.axis as axis
import autoform.src.check as check
import autoform.src.core as core
import autoform.src.extend as extend
import autoform.src.numeric as numeric
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import Blob, BlobAVal, aexecute, execute


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(numeric.FloatAVal(), id="aval"),
        pytest.param(core.Zero(numeric.FloatAVal()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(numeric.FloatAVal(), id="aval"),
        pytest.param(core.Zero(numeric.FloatAVal()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
def test_check_representations(expected, value):
    core.avalof(expected).check(value)
    with pytest.raises(TypeError, match="Expected StrAVal"):
        string.StrAVal().check(value)


def test_avalof_requires_explicit_subclass_registration():
    class FloatAVal(numeric.FloatAVal): ...

    x = FloatAVal()
    with pytest.raises(TypeError, match="No aval rule registered"):
        core.avalof(x)

    core.aval_types[FloatAVal] = lambda aval: aval
    assert core.avalof(x) is x
    assert core.avalof(core.Zero(x)) is x
    core.avalof(x).check(x)
    with pytest.raises(TypeError, match="Expected FloatAVal"):
        core.avalof(x).check(3.0)


class TestAVal:
    def test_default_equality_and_hash(self):
        class X(core.AVal): ...

        class Y(core.AVal): ...

        assert X() == X()
        assert X() != Y()
        assert X() != object()
        assert len({X(), X(), Y()}) == 2
        assert {X(): "x"}[X()] == "x"

    @pytest.mark.parametrize(
        "aval, wrap",
        [
            pytest.param(BlobAVal(3), lambda x: x, id="value"),
            pytest.param(BlobAVal(3), lambda x: ad.PushforwardBox(None, x, x), id="pushforward"),
            pytest.param(BlobAVal(3), lambda x: ad.PullbackFwdBox(None, x), id="pullback-forward"),
            pytest.param(BlobAVal(3), lambda x: ad.PullbackBwdBox(None, x), id="pullback-backward"),
            pytest.param(BlobAVal(3), lambda x: axis.BatchBox(None, x, False), id="broadcast"),
            pytest.param(BlobAVal(3), lambda x: axis.BatchBox(None, [x], True), id="batch"),
        ],
    )
    def test_check_respects_metadata_equality(self, aval, wrap):
        core.avalof(aval).check(wrap(core.Zero(BlobAVal(3))))
        with pytest.raises(TypeError, match="Expected"):
            core.avalof(aval).check(wrap(core.Zero(BlobAVal(4))))


class TestSpace:
    def test_registration_and_replacement(self):
        space = core.Space("blob")
        rule = lambda value: BlobAVal(value.size)
        replacement = lambda value: BlobAVal(value.size + 1)
        space.set(BlobAVal, rule)
        assert space.map(BlobAVal(3)) == BlobAVal(3)
        with pytest.raises(AssertionError, match="already defined"):
            space.set(BlobAVal, replacement)
        space.set(BlobAVal, replacement, replace=True)
        assert space.map(BlobAVal(3)) == BlobAVal(4)
        zero = core.Zero(space.map(BlobAVal(3)))
        assert isinstance(zero, core.Zero)
        assert zero.aval == BlobAVal(4)
        with pytest.raises(AssertionError, match="No concrete zero defined"):
            core.materialize_zeros(zero)

    @pytest.mark.parametrize(
        "value_type, rule, replace, message",
        [
            pytest.param(Blob(3), lambda x: x, False, "Expected type", id="type"),
            pytest.param(BlobAVal, BlobAVal(3), False, "Expected callable", id="callable"),
            pytest.param(BlobAVal, lambda x: x, 1, "Expected bool for replace", id="replace"),
        ],
    )
    def test_invalid_registration(self, value_type, rule, replace, message):
        with pytest.raises(AssertionError, match=message):
            core.Space("blob").set(value_type, rule, replace=replace)

    def test_missing_rule(self):
        with pytest.raises(TypeError, match="No empty aval rule registered"):
            core.Space("empty").map(BlobAVal(3))

    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(core.primal_s, id="primal"),
            pytest.param(core.tangent_s, id="tangent"),
            pytest.param(core.cotangent_s, id="cotangent"),
        ],
    )
    @pytest.mark.parametrize(
        "aval",
        [
            pytest.param(string.StrAVal(), id="string"),
            pytest.param(numeric.FloatAVal(), id="float"),
            pytest.param(numeric.BoolAVal(), id="boolean"),
        ],
    )
    def test_builtin_ad_spaces_preserve_aval(self, space, aval):
        assert space.map(aval) is aval

    def test_custom_ad_spaces(self):
        class TextAVal(core.AVal): ...

        class TextEditAVal(core.AVal): ...

        class TextFeedbackAVal(core.AVal): ...

        tangent_s = core.Space("tangent")
        cotangent_s = core.Space("cotangent")
        tangent_s.set(TextAVal, lambda _: TextEditAVal())
        tangent_s.set(TextEditAVal, lambda aval: aval)
        cotangent_s.set(TextAVal, lambda _: TextFeedbackAVal())
        cotangent_s.set(TextFeedbackAVal, lambda aval: aval)

        t = tangent_s.map(TextAVal())
        c = cotangent_s.map(TextAVal())

        assert isinstance(t, TextEditAVal)
        assert isinstance(c, TextFeedbackAVal)
        assert tangent_s.map(t) is t
        assert cotangent_s.map(c) is c
        assert isinstance(core.Zero(tangent_s.map(TextAVal())).aval, TextEditAVal)
        assert isinstance(core.Zero(cotangent_s.map(TextAVal())).aval, TextFeedbackAVal)

    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(core.tangent_s, id="tangent"),
            pytest.param(core.cotangent_s, id="cotangent"),
        ],
    )
    def test_missing_ad_space_rule(self, space):
        class UnknownAVal(core.AVal): ...

        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            space.map(UnknownAVal())
        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            core.Zero(space.map(UnknownAVal()))


class TestZero:
    def test_zero_string_contract(self):
        z = core.Zero(string.StrAVal())
        assert isinstance(z, core.Zero)
        assert z.aval == string.StrAVal()
        assert z == core.Zero(string.StrAVal())
        assert z != core.Zero(numeric.BoolAVal())
        assert core.Zero(core.primal_s.map(core.avalof(z))) == z
        assert core.materialize_zeros(z) == ""
        assert core.avalof(z) == string.StrAVal()
        assert not stage.is_traceable(z)

    def test_zero_requires_aval(self):
        with pytest.raises(AssertionError, match="Expected AVal"):
            core.Zero(str)

    def test_zero_non_differentiable_type(self):
        z = core.Zero(numeric.BoolAVal())
        assert isinstance(z, core.Zero)
        assert z.aval == numeric.BoolAVal()
        with pytest.raises(AssertionError, match="No concrete zero defined"):
            core.materialize_zeros(z)

    def test_zero_materializes_with_aval_metadata(self):
        class BlobAVal(core.AVal):
            __slots__ = ["size"]

            def __init__(self, size):
                self.size = size

            def zero(self):
                return "zero", self.size

        assert core.materialize_zeros(core.Zero(BlobAVal(3))) == ("zero", 3)


class TestPrimitive:
    def test_creation(self):
        p = core.Prim("test_prim")
        assert p.name == "test_prim"
        assert repr(p) == "test_prim"


class TestBind:
    def test_bind_using(self):
        p = core.Prim("custom_bind")

        def impl(in_tree, *, multiplier):
            return in_tree * multiplier

        def abstract_rule(in_tree, *, multiplier):
            return string.StrAVal()

        extend.register_impl(p, impl)
        extend.register_abstract(p, abstract_rule)

        def func(x):
            return p.bind(x, multiplier=3)

        ir = stage.trace(func)("A")
        result = ir.call("B")
        assert result == "BBB"


def test_interpreter_context_restores_default():
    assert type(core.active_interpreter.get()) is core.EvalInterpreter
    tracer = stage.TraceInterpreter()
    with core.using_interpreter(tracer) as active:
        assert active is tracer
        string.format("Hello, {value}!", value=stage.Var.fresh(aval=string.StrAVal()))
        assert len(tracer.eqns) == 1
    assert type(core.active_interpreter.get()) is core.EvalInterpreter
    assert string.concat("a", "b") == "ab"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_custom_rule_interpreter(executor):
    rules = core.Rule("operation_count")
    arules = core.Rule("aoperation_count")
    rules.set(
        string.concat_p, lambda values: sum(v if isinstance(v, int) else 0 for v in values) + 1
    )
    arules.set(string.concat_p, utils.asyncify(rules.get(string.concat_p)))
    rules.set(check.typecheck_p, check.impl_check)
    arules.set(check.typecheck_p, utils.asyncify(check.impl_check))

    class OperationCountInterpreter(core.Interpreter):
        def interpret(self, prim, in_tree, /, **params):
            return rules.get(prim)(in_tree, **params)

        async def ainterpret(self, prim, in_tree, /, **params):
            return await arules.get(prim)(in_tree, **params)

    ir = stage.trace(lambda x, y: (x + y) + y)("x", "y")
    with core.using_interpreter(OperationCountInterpreter()):
        with pytest.raises(TypeError, match="Expected StrAVal"):
            executor(ir, "x", "y")
    assert executor(ir, "x", "y") == "xyy"
