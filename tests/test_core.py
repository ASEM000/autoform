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
import functools as ft

import pytest

import autoform.extend as extend
import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.check as check
import autoform.src.core as core
import autoform.src.impl as impl
import autoform.src.numeric as numeric
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import AValBlob, Blob, aexecute, execute


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(numeric.AFloat(), id="aval"),
        pytest.param(core.Zero(numeric.AFloat()), id="zero"),
        pytest.param(3.0, id="concrete"),
    ],
)
def test_check_representations(value):
    assert check.typecheck(value, numeric.AFloat()) is value
    with pytest.raises(TypeError, match="AStr"):
        check.typecheck(value, string.AStr())


def test_avalof_requires_explicit_subclass_registration():
    class AFloat(numeric.AFloat): ...

    x = AFloat()
    with pytest.raises(TypeError, match="No aval rule registered"):
        core.avalof(x)
    core.aval_types[AFloat] = lambda aval: aval
    assert core.avalof(x) is x
    assert core.avalof(core.Zero(x)) is x


class TestAVal:
    @pytest.mark.parametrize(
        "method, args",
        [
            ("zero", ()),
            ("add", (1, 2)),
            ("join", (core.AVal(),)),
            ("join", (core.Zero(core.AVal()),)),
        ],
        ids=["zero", "add", "join-aval", "join-zero"],
    )
    def test_method_requires_implementation(self, method, args):
        with pytest.raises(NotImplementedError):
            getattr(core.AVal(), method)(*args)

    @pytest.mark.parametrize(
        "aval, values, expected",
        [
            pytest.param(numeric.AFloat(), [], 0.0, id="empty-float"),
            pytest.param(numeric.AFloat(), [1.0, 2.0, 3.0], 6.0, id="float"),
            pytest.param(string.AStr(), [], "", id="empty-string"),
            pytest.param(string.AStr(), ["a", "b", "c"], "abc", id="string"),
        ],
    )
    def test_add_reduction(self, aval, values, expected):
        x = ft.reduce(aval.add, values, aval.zero())
        assert x == expected
        assert aval.join(x) == aval

    def test_default_equality_and_hash(self):
        class AValX(core.AVal): ...

        class AValY(core.AVal): ...

        assert AValX() == AValX()
        assert AValX() != AValY()
        assert AValX() != object()
        assert len({AValX(), AValX(), AValY()}) == 2
        assert {AValX(): "x"}[AValX()] == "x"

    def test_check_respects_metadata_equality(self):
        aval = AValBlob(3)
        check.typecheck(core.Zero(AValBlob(3)), aval)
        with pytest.raises(TypeError, match="AValBlob"):
            check.typecheck(core.Zero(AValBlob(4)), aval)


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
        with pytest.raises(NotImplementedError):
            zero.aval.zero()

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
            pytest.param(string.AStr(), id="string"),
            pytest.param(numeric.AFloat(), id="float"),
            pytest.param(numeric.ABool(), id="boolean"),
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
        z = core.Zero(string.AStr())
        assert isinstance(z, core.Zero)
        assert z.aval == string.AStr()
        assert z == core.Zero(string.AStr())
        assert z != core.Zero(numeric.ABool())
        assert core.Zero(core.s_primal.map(core.avalof(z))) == z
        assert z.aval.zero() == ""
        assert core.avalof(z) == string.AStr()
        assert not stage.is_traceable(z)

    def test_zero_requires_aval(self):
        with pytest.raises(AssertionError, match="Expected AVal"):
            core.Zero(str)

    def test_zero_non_differentiable_type(self):
        z = core.Zero(numeric.ABool())
        assert isinstance(z, core.Zero)
        assert z.aval == numeric.ABool()
        with pytest.raises(NotImplementedError):
            z.aval.zero()

    def test_zero_materializes_with_aval_metadata(self):
        class AValBlob(core.AVal):
            __slots__ = ["size"]

            def __init__(self, size):
                self.size = size

            def zero(self):
                return "zero", self.size

            def add(self, left, right, /):
                return "add", self.size, left, right

        assert AValBlob(3).zero() == ("zero", 3)
        assert AValBlob(4).add(1, 2) == ("add", 4, 1, 2)

    def test_zero_and_add_are_inherited(self):
        class AValText(string.AStr):
            __slots__ = []

        class Text(str): ...

        aval = AValText()
        core.aval_types[Text] = lambda _: aval
        assert aval.zero() == ""
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
            return string.AStr()

        extend.r_impl.set(p_custom_bind, impl)
        extend.r_abstract.set(p_custom_bind, abstract_rule)

        def func(x):
            return p_custom_bind.bind(x, multiplier=3)

        ir = stage.trace(func)("A")
        result = ir.call("B")
        assert result == "BBB"


def test_interpreter_context_restores_default():
    assert type(core.active_interp.get()) is impl.ImplInterp
    tracer = stage.TraceInterp()
    with core.using_interp(tracer) as active:
        assert active is tracer
        string.format("Hello, {value}!", value=stage.Var.fresh(aval=string.AStr()))
        assert len(tracer.eqns) == 1
    assert type(core.active_interp.get()) is impl.ImplInterp
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

    class OperationCountInterp(core.Interp):
        def interpret(self, prim, in_tree, /, **params):
            return rules.get(prim)(in_tree, **params)

        async def ainterpret(self, prim, in_tree, /, **params):
            return await arules.get(prim)(in_tree, **params)

    ir = stage.trace(lambda x, y: (x + y) + y)("x", "y")
    with core.using_interp(OperationCountInterp()):
        with pytest.raises(TypeError, match="Cannot join AStr"):
            executor(ir, "x", "y")
    assert executor(ir, "x", "y") == "xyy"


class TestChecking:
    @pytest.mark.parametrize(
        "aval, value, expected",
        [
            pytest.param(numeric.AFloat(), 1.0, True, id="float"),
            pytest.param(numeric.AFloat(), "x", False, id="wrong-float"),
            pytest.param(numeric.AFloat(), numeric.AFloat(), True, id="aval"),
            pytest.param(numeric.AFloat(), core.Zero(numeric.AFloat()), True, id="zero"),
            pytest.param(numeric.AInt(), 1, True, id="int"),
            pytest.param(numeric.AInt(), True, False, id="bool-is-not-int"),
            pytest.param(numeric.ABool(), True, True, id="bool"),
            pytest.param(numeric.ABool(), 1, False, id="int-is-not-bool"),
            pytest.param(string.AStr(), "x", True, id="string"),
        ],
    )
    def test_scalar(self, aval, value, expected):
        if expected:
            assert check.typecheck(value, aval) is value
        else:
            with pytest.raises(TypeError):
                check.typecheck(value, aval)

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(["x", "y"], True, id="strings"),
            pytest.param([], True, id="empty"),
            pytest.param(["x", 1.0], False, id="mixed"),
            pytest.param([string.AStr()], True, id="aval-element"),
            pytest.param([core.Zero(string.AStr())], True, id="zero-element"),
            pytest.param(axis.ABatch(string.AStr()), True, id="aval"),
            pytest.param(core.Zero(axis.ABatch(string.AStr())), True, id="zero"),
        ],
    )
    def test_batch(self, value, expected):
        aval = axis.ABatch(string.AStr())
        if expected:
            assert check.typecheck(value, aval) is value
        else:
            with pytest.raises(TypeError):
                check.typecheck(value, aval)


class TestJoin:
    @pytest.mark.parametrize(
        "cls, value",
        [
            (numeric.AFloat, 1.0),
            (numeric.AInt, 1),
            (numeric.ABool, True),
            (string.AStr, "x"),
        ],
        ids=["float", "int", "bool", "string"],
    )
    def test_subclasses_inherit_join(self, cls, value):
        class AValDerived(cls): ...

        x, y = AValDerived(), cls()
        assert x.join(x) is x
        assert x.join(value) is x
        assert x.join(y) is x
        assert y.join(x) is y
        assert x.join(core.Zero(y)) is x
        assert y.join(core.Zero(x)) is y
        if type(value) is not bool:

            class Value(type(value)): ...

            assert x.join(Value(value)) is x
            assert y.join(Value(value)) is y

    @pytest.mark.parametrize("error", [TypeError("join failed"), KeyError("join failed")])
    def test_handler_error_propagates(self, error):
        class AValBound(AValBlob):
            def join(self, other):
                raise error

        with pytest.raises(type(error)) as caught:
            check.impl_check(core.Zero(AValBlob(3)), aval=AValBound(3))
        assert caught.value is error

    def test_metadata(self):
        x = AValBlob(3)
        assert x.join(AValBlob(3)) is x
        with pytest.raises(TypeError):
            x.join(AValBlob(4))

    def test_join_uses_left_method(self):
        class AValBound(numeric.AFloat):
            def join(self, other):
                match other:
                    case (numeric.AFloat() as aval) | core.Zero(aval=numeric.AFloat() as aval):
                        return aval
                    case float():
                        return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        x, y = AValBound(), numeric.AFloat()
        assert x.join(y) is y
        assert y.join(x) is y
        with pytest.raises(TypeError, match="Expected"):
            check.impl_check(core.Zero(y), aval=x)

    @pytest.mark.parametrize(
        "x, y",
        [
            pytest.param(numeric.AFloat(), string.AStr(), id="different-types"),
            pytest.param(numeric.AInt(), core.Zero(numeric.ABool()), id="wrong-zero"),
        ],
    )
    def test_unsupported_join(self, x, y):
        with pytest.raises(TypeError):
            x.join(y)

    @pytest.mark.parametrize("wrap", [lambda x: x, axis.ABatch], ids=["single", "nested"])
    @pytest.mark.parametrize("make", [lambda x: x, core.Zero], ids=["aval", "zero"])
    def test_batch_join(self, wrap, make):
        x = wrap(axis.ABatch(string.AStr()))
        assert x.join(make(x)) == x
        with pytest.raises(TypeError):
            x.join(make(wrap(axis.ABatch(numeric.AFloat()))))
