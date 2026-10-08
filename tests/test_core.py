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
from tests import Blob, BlobAVal, aexecute, execute


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(af.numeric.FloatAVal(), id="aval"),
        pytest.param(af.core.Zero(af.numeric.FloatAVal()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(af.numeric.FloatAVal(), id="aval"),
        pytest.param(af.core.Zero(af.numeric.FloatAVal()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
def test_check_representations(expected, value):
    af.core.avalof(expected).check(value)
    with pytest.raises(TypeError, match="Expected StrAVal"):
        af.string.StrAVal().check(value)


def test_avalof_requires_explicit_subclass_registration():
    class FloatAVal(af.numeric.FloatAVal): ...

    x = FloatAVal()
    with pytest.raises(TypeError, match="No aval rule registered"):
        af.core.avalof(x)

    af.core.aval_types[FloatAVal] = lambda aval: aval
    assert af.core.avalof(x) is x
    assert af.core.avalof(af.core.Zero(x)) is x
    af.core.avalof(x).check(x)
    with pytest.raises(TypeError, match="Expected FloatAVal"):
        af.core.avalof(x).check(3.0)


class TestAVal:
    def test_default_equality_and_hash(self):
        class X(af.core.AVal): ...

        class Y(af.core.AVal): ...

        assert X() == X()
        assert X() != Y()
        assert X() != object()
        assert len({X(), X(), Y()}) == 2
        assert {X(): "x"}[X()] == "x"

    @pytest.mark.parametrize(
        "aval, wrap",
        [
            pytest.param(BlobAVal(3), lambda x: x, id="value"),
            pytest.param(BlobAVal(3), lambda x: af.ad.PushforwardBox(None, x, x), id="pushforward"),
            pytest.param(
                BlobAVal(3), lambda x: af.ad.PullbackFwdBox(None, x), id="pullback-forward"
            ),
            pytest.param(
                BlobAVal(3), lambda x: af.ad.PullbackBwdBox(None, x), id="pullback-backward"
            ),
            pytest.param(BlobAVal(3), lambda x: af.axis.BatchBox(None, x, False), id="broadcast"),
            pytest.param(BlobAVal(3), lambda x: af.axis.BatchBox(None, [x], True), id="batch"),
            pytest.param(
                af.axis.BatchAVal(BlobAVal(3)),
                lambda x: af.axis.BatchBox(None, [x], False),
                id="broadcast-batch",
            ),
            pytest.param(
                af.axis.BatchAVal(BlobAVal(3)),
                lambda x: af.axis.BatchBox(None, [[x]], True),
                id="nested-batch",
            ),
        ],
    )
    def test_check_respects_metadata_equality(self, aval, wrap):
        af.core.avalof(aval).check(wrap(af.core.Zero(BlobAVal(3))))
        with pytest.raises(TypeError, match="Expected"):
            af.core.avalof(aval).check(wrap(af.core.Zero(BlobAVal(4))))


class TestSpace:
    def test_registration_and_replacement(self):
        space = af.core.Space("blob")
        rule = lambda value: BlobAVal(value.size)
        replacement = lambda value: BlobAVal(value.size + 1)
        space.set(BlobAVal, rule)
        assert space.map(BlobAVal(3)) == BlobAVal(3)
        with pytest.raises(AssertionError, match="already defined"):
            space.set(BlobAVal, replacement)
        space.set(BlobAVal, replacement, replace=True)
        assert space.map(BlobAVal(3)) == BlobAVal(4)
        zero = af.core.Zero(space.map(BlobAVal(3)))
        assert isinstance(zero, af.core.Zero)
        assert zero.aval == BlobAVal(4)
        with pytest.raises(AssertionError, match="No concrete zero defined"):
            af.core.materialize_zeros(zero)

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
            af.core.Space("blob").set(value_type, rule, replace=replace)

    def test_missing_rule(self):
        with pytest.raises(TypeError, match="No empty aval rule registered"):
            af.core.Space("empty").map(BlobAVal(3))

    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(af.core.primal_s, id="primal"),
            pytest.param(af.core.tangent_s, id="tangent"),
            pytest.param(af.core.cotangent_s, id="cotangent"),
        ],
    )
    @pytest.mark.parametrize(
        "aval",
        [
            pytest.param(af.string.StrAVal(), id="string"),
            pytest.param(af.numeric.FloatAVal(), id="float"),
            pytest.param(af.numeric.BoolAVal(), id="boolean"),
        ],
    )
    def test_builtin_ad_spaces_preserve_aval(self, space, aval):
        assert space.map(aval) is aval

    def test_custom_ad_spaces(self):
        class TextAVal(af.core.AVal): ...

        class TextEditAVal(af.core.AVal): ...

        class TextFeedbackAVal(af.core.AVal): ...

        tangent_s = af.core.Space("tangent")
        cotangent_s = af.core.Space("cotangent")
        tangent_s.set(TextAVal, lambda _: TextEditAVal())
        tangent_s.set(TextEditAVal, lambda aval: aval)
        cotangent_s.set(TextAVal, lambda _: TextFeedbackAVal())
        cotangent_s.set(TextFeedbackAVal, lambda aval: aval)

        tangent = tangent_s.map(TextAVal())
        cotangent = cotangent_s.map(TextAVal())

        assert isinstance(tangent, TextEditAVal)
        assert isinstance(cotangent, TextFeedbackAVal)
        assert tangent_s.map(tangent) is tangent
        assert cotangent_s.map(cotangent) is cotangent
        assert isinstance(af.core.Zero(tangent_s.map(TextAVal())).aval, TextEditAVal)
        assert isinstance(af.core.Zero(cotangent_s.map(TextAVal())).aval, TextFeedbackAVal)

    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(af.core.tangent_s, id="tangent"),
            pytest.param(af.core.cotangent_s, id="cotangent"),
        ],
    )
    def test_missing_ad_space_rule(self, space):
        class UnknownAVal(af.core.AVal): ...

        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            space.map(UnknownAVal())
        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            af.core.Zero(space.map(UnknownAVal()))


class TestZero:
    def test_zero_string_contract(self):
        z = af.core.Zero(af.string.StrAVal())
        assert isinstance(z, af.core.Zero)
        assert z.aval == af.string.StrAVal()
        assert z == af.core.Zero(af.string.StrAVal())
        assert z != af.core.Zero(af.numeric.BoolAVal())
        assert af.core.Zero(af.core.primal_s.map(af.core.avalof(z))) == z
        assert af.core.materialize_zeros(z) == ""
        assert af.core.avalof(z) == af.string.StrAVal()
        assert not af.stage.is_traceable(z)

    def test_zero_requires_aval(self):
        with pytest.raises(AssertionError, match="Expected AVal"):
            af.core.Zero(str)

    def test_zero_non_differentiable_type(self):
        z = af.core.Zero(af.numeric.BoolAVal())
        assert isinstance(z, af.core.Zero)
        assert z.aval == af.numeric.BoolAVal()
        with pytest.raises(AssertionError, match="No concrete zero defined"):
            af.core.materialize_zeros(z)

    def test_zero_materializes_with_aval_metadata(self):
        class BlobAVal(af.core.AVal):
            __slots__ = ["size"]

            def __init__(self, size):
                self.size = size

            def zero(self):
                return "zero", self.size

        assert af.core.materialize_zeros(af.core.Zero(BlobAVal(3))) == ("zero", 3)


class TestPrimitive:
    def test_creation(self):
        p = af.core.Prim("test_prim")
        assert p.name == "test_prim"
        assert repr(p) == "test_prim"


class TestBind:
    def test_bind_using(self):
        p = af.core.Prim("custom_bind")

        def impl(in_tree, *, multiplier):
            return in_tree * multiplier

        def abstract_rule(in_tree, *, multiplier):
            return af.string.StrAVal()

        af.extend.register_impl(p, impl)
        af.extend.register_abstract(p, abstract_rule)

        def func(x):
            return p.bind(x, multiplier=3)

        ir = af.trace(func)("A")
        result = ir.call("B")
        assert result == "BBB"


def test_interpreter_context_restores_default():
    assert type(af.core.active_interpreter.get()) is af.core.EvalInterpreter
    tracer = af.stage.TraceInterpreter()
    with af.core.using_interpreter(tracer) as active:
        assert active is tracer
        af.string.format("Hello, {value}!", value=af.stage.Var.fresh(aval=af.string.StrAVal()))
        assert len(tracer.eqns) == 1
    assert type(af.core.active_interpreter.get()) is af.core.EvalInterpreter
    assert af.string.concat("a", "b") == "ab"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_custom_rule_interpreter(executor):
    rules = af.extend.Rule("operation_count")
    arules = af.extend.Rule("aoperation_count")
    rules.set(
        af.extend.concat_p, lambda values: sum(v if isinstance(v, int) else 0 for v in values) + 1
    )
    arules.set(af.extend.concat_p, af.utils.asyncify(rules.get(af.extend.concat_p)))

    class OperationCountInterpreter(af.core.Interpreter):
        def interpret(self, prim, in_tree, /, **params):
            return rules.get(prim)(in_tree, **params)

        async def ainterpret(self, prim, in_tree, /, **params):
            return await arules.get(prim)(in_tree, **params)

    ir = af.trace(lambda x, y: (x + y) + y)("x", "y")
    with af.core.using_interpreter(OperationCountInterpreter()):
        with pytest.raises(TypeError, match="Expected StrAVal"):
            executor(ir, "x", "y")
    assert executor(ir, "x", "y") == "xyy"
