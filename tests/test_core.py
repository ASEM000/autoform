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

import asyncio

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
from tests import AValBlob, Blob, aexecute, execute


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(numeric.AValFloat(), id="aval"),
        pytest.param(core.Zero(numeric.AValFloat()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(numeric.AValFloat(), id="aval"),
        pytest.param(core.Zero(numeric.AValFloat()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
def test_check_representations(expected, value):
    check.typecheck(value, core.avalof(expected))
    with pytest.raises(TypeError, match="Expected AValStr"):
        check.typecheck(value, string.AValStr())


def test_avalof_requires_explicit_subclass_registration():
    class AValFloat(numeric.AValFloat): ...

    x = AValFloat()
    with pytest.raises(TypeError, match="No aval rule registered"):
        core.avalof(x)
    with pytest.raises(TypeError, match="No aval rule registered"):
        core.domainof(x).accepts(x)

    core.aval_types[AValFloat] = lambda aval: aval
    assert core.avalof(x) is x
    assert core.avalof(core.Zero(x)) is x
    check.typecheck(x, core.avalof(x))
    with pytest.raises(TypeError, match="Expected AValFloat"):
        check.typecheck(3.0, core.avalof(x))


class TestAVal:
    def test_default_equality_and_hash(self):
        class AValX(core.AVal): ...

        class AValY(core.AVal): ...

        assert AValX() == AValX()
        assert AValX() != AValY()
        assert AValX() != object()
        assert len({AValX(), AValX(), AValY()}) == 2
        assert {AValX(): "x"}[AValX()] == "x"

    @pytest.mark.parametrize(
        "aval, wrap",
        [
            pytest.param(AValBlob(3), lambda x: x, id="value"),
            pytest.param(AValBlob(3), lambda x: ad.PushforwardBox(None, x, x), id="pushforward"),
            pytest.param(AValBlob(3), lambda x: ad.PullbackFwdBox(None, x), id="pullback-forward"),
            pytest.param(AValBlob(3), lambda x: ad.PullbackBwdBox(None, x), id="pullback-backward"),
            pytest.param(AValBlob(3), lambda x: axis.BatchBox(None, x, False), id="broadcast"),
            pytest.param(AValBlob(3), lambda x: axis.BatchBox(None, [x], True), id="batch"),
        ],
    )
    def test_check_respects_metadata_equality(self, aval, wrap):
        check.typecheck(wrap(core.Zero(AValBlob(3))), core.avalof(aval))
        with pytest.raises(TypeError, match="Expected"):
            check.typecheck(wrap(core.Zero(AValBlob(4))), core.avalof(aval))


class TestSpace:
    def test_registration_and_replacement(self):
        s_blob = core.Space("blob")
        rule = lambda value: AValBlob(value.size)
        replacement = lambda value: AValBlob(value.size + 1)
        s_blob.set(AValBlob, rule)
        assert s_blob.map(AValBlob(3)) == AValBlob(3)
        with pytest.raises(AssertionError, match="already defined"):
            s_blob.set(AValBlob, replacement)
        s_blob.set(AValBlob, replacement, replace=True)
        assert s_blob.map(AValBlob(3)) == AValBlob(4)
        zero = core.Zero(s_blob.map(AValBlob(3)))
        assert isinstance(zero, core.Zero)
        assert zero.aval == AValBlob(4)
        with pytest.raises(TypeError, match="No algebra registered"):
            core.algebraof(zero.aval).zero()

    @pytest.mark.parametrize(
        "value_type, rule, replace, message",
        [
            pytest.param(Blob(3), lambda x: x, False, "Expected type", id="type"),
            pytest.param(AValBlob, AValBlob(3), False, "Expected callable", id="callable"),
            pytest.param(AValBlob, lambda x: x, 1, "Expected bool for replace", id="replace"),
        ],
    )
    def test_invalid_registration(self, value_type, rule, replace, message):
        with pytest.raises(AssertionError, match=message):
            core.Space("blob").set(value_type, rule, replace=replace)

    def test_missing_rule(self):
        with pytest.raises(TypeError, match="No empty aval rule registered"):
            core.Space("empty").map(AValBlob(3))

    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(core.s_primal, id="primal"),
            pytest.param(core.s_tangent, id="tangent"),
            pytest.param(core.s_cotangent, id="cotangent"),
        ],
    )
    @pytest.mark.parametrize(
        "aval",
        [
            pytest.param(string.AValStr(), id="string"),
            pytest.param(numeric.AValFloat(), id="float"),
            pytest.param(numeric.AValBool(), id="boolean"),
        ],
    )
    def test_builtin_ad_spaces_preserve_aval(self, space, aval):
        assert space.map(aval) is aval

    def test_custom_ad_spaces(self):
        class AValText(core.AVal): ...

        class AValTextEdit(core.AVal): ...

        class AValTextFeedback(core.AVal): ...

        s_tangent = core.Space("tangent")
        s_cotangent = core.Space("cotangent")
        s_tangent.set(AValText, lambda _: AValTextEdit())
        s_tangent.set(AValTextEdit, lambda aval: aval)
        s_cotangent.set(AValText, lambda _: AValTextFeedback())
        s_cotangent.set(AValTextFeedback, lambda aval: aval)

        t = s_tangent.map(AValText())
        c = s_cotangent.map(AValText())

        assert isinstance(t, AValTextEdit)
        assert isinstance(c, AValTextFeedback)
        assert s_tangent.map(t) is t
        assert s_cotangent.map(c) is c
        assert isinstance(core.Zero(s_tangent.map(AValText())).aval, AValTextEdit)
        assert isinstance(core.Zero(s_cotangent.map(AValText())).aval, AValTextFeedback)

    @pytest.mark.parametrize(
        "space",
        [
            pytest.param(core.s_tangent, id="tangent"),
            pytest.param(core.s_cotangent, id="cotangent"),
        ],
    )
    def test_missing_ad_space_rule(self, space):
        class AValUnknown(core.AVal): ...

        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            space.map(AValUnknown())
        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            core.Zero(space.map(AValUnknown()))


class TestZero:
    def test_zero_string_contract(self):
        z = core.Zero(string.AValStr())
        assert isinstance(z, core.Zero)
        assert z.aval == string.AValStr()
        assert z == core.Zero(string.AValStr())
        assert z != core.Zero(numeric.AValBool())
        assert core.Zero(core.s_primal.map(core.avalof(z))) == z
        assert core.algebraof(z.aval).zero() == ""
        assert core.avalof(z) == string.AValStr()
        assert not stage.is_traceable(z)

    def test_zero_requires_aval(self):
        with pytest.raises(AssertionError, match="Expected AVal"):
            core.Zero(str)

    def test_zero_non_differentiable_type(self):
        z = core.Zero(numeric.AValBool())
        assert isinstance(z, core.Zero)
        assert z.aval == numeric.AValBool()
        with pytest.raises(TypeError, match="No algebra registered"):
            core.algebraof(z.aval).zero()

    def test_zero_materializes_with_aval_metadata(self):
        class AValBlob(core.AVal):
            __slots__ = ["size"]

            def __init__(self, size):
                self.size = size

        class AlgebraBlob(core.Algebra):
            __slots__ = []

            def zero(self):
                return "zero", self.aval.size

            def accum(self, values, /):
                return "accum", self.aval.size, values

        extend.register_algebra(AValBlob, lambda aval: AlgebraBlob(aval))

        assert core.algebraof(AValBlob(3)).zero() == ("zero", 3)
        assert core.algebraof(AValBlob(4)).accum([1, 2]) == ("accum", 4, [1, 2])

    def test_algebra_registration_is_not_inherited(self):
        class AValText(string.AValStr):
            __slots__ = []

        class Text(str): ...

        aval = AValText()
        core.aval_types[Text] = lambda _: aval
        with pytest.raises(TypeError, match="No algebra registered"):
            core.algebraof(aval)
        with pytest.raises(TypeError, match="No algebra registered"):
            ad.cot_accum([Text("a"), Text("b")])
        extend.register_algebra(AValText, lambda aval: string.AlgebraStr(aval))
        assert core.algebraof(aval).aval is aval
        assert core.algebraof(aval).zero() == ""
        assert ad.cot_accum([Text("a"), Text("b")]) == "ab"


class TestPrimitive:
    def test_creation(self):
        p_test_prim = core.Prim("test_prim")
        assert p_test_prim.name == "test_prim"
        assert repr(p_test_prim) == "test_prim"


class TestRule:
    def test_nested_overrides_restore_after_exception(self):
        rules = core.Rule("text")
        x, y, z = core.Prim("x"), core.Prim("y"), core.Prim("z")
        rules.set(x, str)
        rules.set(y, str)

        with rules.override((x, repr), (z, repr)) as active:
            assert active is rules
            assert rules.get(x) is repr
            assert rules.get(y) is str
            assert rules.get(z) is repr
            rules.set(y, ascii)
            assert rules.get(y) is ascii
            with pytest.raises(ValueError, match="stop"):
                with rules.override((x, ascii)):
                    assert rules.get(x) is ascii
                    assert rules.get(z) is repr
                    raise ValueError("stop")
            assert rules.get(x) is repr

        assert rules.get(x) is str
        assert rules.get(y) is ascii
        with pytest.raises(KeyError):
            rules.get(z)

    @pytest.mark.parametrize("prim, rule", [("invalid", str), (core.Prim("x"), None)])
    def test_invalid_override(self, prim, rule):
        rules = core.Rule("text")
        with pytest.raises(AssertionError), rules.override((prim, rule)):
            pytest.fail("Invalid overrides must not enter the context")

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_execution_uses_overrides(self, executor):
        ir = stage.trace(lambda x, y: x + y)("x", "y")
        reverse = lambda values: "".join(reversed(values))
        with (
            core.r_impl.override((string.p_concat, reverse)),
            core.r_aimpl.override((string.p_concat, utils.asyncify(reverse))),
        ):
            assert executor(ir, "a", "b") == "ba"
        assert executor(ir, "a", "b") == "ab"

    def test_async_overrides_are_isolated(self):
        async def run():
            left, right, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def worker(value, entered):
                rule = utils.asyncify(lambda values: value)
                with core.r_aimpl.override((string.p_concat, rule)):
                    entered.set()
                    await release.wait()
                    return await string.p_concat.abind(("a", "b"))

            tasks = [
                asyncio.create_task(worker("left", left)),
                asyncio.create_task(worker("right", right)),
            ]
            await left.wait()
            await right.wait()
            assert await string.p_concat.abind(("a", "b")) == "ab"
            release.set()
            assert await asyncio.gather(*tasks) == ["left", "right"]
            assert await string.p_concat.abind(("a", "b")) == "ab"

        asyncio.run(asyncio.wait_for(run(), timeout=5))


class TestBind:
    def test_bind_using(self):
        p_custom_bind = core.Prim("custom_bind")

        def impl(in_tree, *, multiplier):
            return in_tree * multiplier

        def abstract_rule(in_tree, *, multiplier):
            return string.AValStr()

        extend.register_impl(p_custom_bind, impl)
        extend.register_abstract(p_custom_bind, abstract_rule)

        def func(x):
            return p_custom_bind.bind(x, multiplier=3)

        ir = stage.trace(func)("A")
        result = ir.call("B")
        assert result == "BBB"


def test_interpreter_context_restores_default():
    assert type(core.active_interpreter.get()) is core.EvalInterpreter
    tracer = stage.TraceInterpreter()
    with core.using_interpreter(tracer) as active:
        assert active is tracer
        string.format("Hello, {value}!", value=stage.Var.fresh(aval=string.AValStr()))
        assert len(tracer.eqns) == 1
    assert type(core.active_interpreter.get()) is core.EvalInterpreter
    assert string.concat("a", "b") == "ab"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_custom_rule_interpreter(executor):
    rules = core.Rule("operation_count")
    arules = core.Rule("aoperation_count")
    rules.set(
        string.p_concat, lambda values: sum(v if isinstance(v, int) else 0 for v in values) + 1
    )
    arules.set(string.p_concat, utils.asyncify(rules.get(string.p_concat)))
    rules.set(check.p_typecheck, check.impl_check)
    arules.set(check.p_typecheck, utils.asyncify(check.impl_check))

    class OperationCountInterpreter(core.Interpreter):
        def interpret(self, prim, in_tree, /, **params):
            return rules.get(prim)(in_tree, **params)

        async def ainterpret(self, prim, in_tree, /, **params):
            return await arules.get(prim)(in_tree, **params)

    ir = stage.trace(lambda x, y: (x + y) + y)("x", "y")
    with core.using_interpreter(OperationCountInterpreter()):
        with pytest.raises(TypeError, match="Expected AValStr"):
            executor(ir, "x", "y")
    assert executor(ir, "x", "y") == "xyy"


class TestDomain:
    def test_default_acceptance_and_join(self):
        x = AValBlob(3)
        domain = core.domainof(x)
        assert domain.base() is x
        assert domain.accepts(core.Zero(AValBlob(3)))
        assert not domain.accepts(core.Zero(AValBlob(4)))
        assert domain.join(AValBlob(3)) is x
        with pytest.raises(TypeError, match="No common domain"):
            domain.join(AValBlob(4))

    def test_exact_registration(self):
        class AValText(core.AVal): ...

        class AValOther(AValText): ...

        class DomainText(core.Domain):
            def accepts(self, value):
                return isinstance(value, str) or super().accepts(value)

        core.aval_types[AValText] = lambda aval: aval
        extend.register_domain(AValText, lambda aval: DomainText(aval))
        x, y = AValText(), AValOther()
        assert core.domainof(x).aval is x
        assert core.domainof(x).accepts("x")
        assert core.domainof(x).accepts(core.Zero(x))
        assert core.domainof(x).join(x) is x
        assert type(core.domainof(y)) is core.Domain
        assert not core.domainof(y).accepts("x")
        with pytest.raises(TypeError, match="No algebra registered"):
            core.algebraof(x)

    def test_batch_domain(self):
        x = axis.AValBatch(string.AValStr())
        domain = core.domainof(x)
        assert domain.base() == x
        assert domain.accepts(["x", "y"])
        assert domain.accepts([])
        assert not domain.accepts(["x", 1.0])
        assert domain.join(x) == x
        with pytest.raises(TypeError, match="No common domain"):
            domain.join(axis.AValBatch(numeric.AValFloat()))
