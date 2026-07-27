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


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class EchoRouter:
    def completion(self, *, messages: list[dict], model: str, **kwargs):
        del kwargs
        return FakeResponse(f"{model}|{messages[-1]['content']}")

    async def acompletion(self, *, messages: list[dict], model: str, **kwargs):
        return self.completion(messages=messages, model=model, **kwargs)


class TestPrimitive:
    def test_creation(self):
        p = af.core.Prim("test_prim")
        assert p.name == "test_prim"
        assert repr(p) == "test_prim"

    def test_def_impl_decorator(self):
        p = af.core.Prim("test_impl")

        @ft.partial(af.core.impl_rules.set, p)
        def impl(x):
            return x

        assert af.core.impl_rules.get(p) is impl

    def test_def_abstract_decorator(self):
        p = af.core.Prim("test_abstract")

        @ft.partial(af.core.abstract_rules.set, p)
        def abstract_rule(x):
            return af.core.StrAVal()

        assert af.core.abstract_rules.get(p) is abstract_rule

    def test_def_batch_decorator(self):
        p = af.core.Prim("test_batch")

        @ft.partial(af.core.batch_rules.set, p)
        def batch_rule(in_tree):
            return in_tree[2], True

        assert af.core.batch_rules.get(p) is batch_rule

    def test_def_pushforward_decorator(self):
        p = af.core.Prim("test_pushforward")

        @ft.partial(af.core.push_rules.set, p)
        def pf_rule(in_tree):
            return in_tree

        assert af.core.push_rules.get(p) is pf_rule

    def test_def_pullback_forward_decorator(self):
        p = af.core.Prim("test_pullback_fwd")

        @ft.partial(af.core.pull_fwd_rules.set, p)
        def pb_fwd_rule(in_tree):
            return in_tree, in_tree

        assert af.core.pull_fwd_rules.get(p) is pb_fwd_rule

    def test_def_pullback_backward_decorator(self):
        p = af.core.Prim("test_pullback_bwd")

        @ft.partial(af.core.pull_bwd_rules.set, p)
        def pb_bwd_rule(residuals, out_cotangent):
            return out_cotangent

        assert af.core.pull_bwd_rules.get(p) is pb_bwd_rule


class TestVar:
    def test_var_aval_returns_aval(self):
        var = af.core.Var(aval=af.core.StrAVal())

        assert af.core.is_var(var)
        assert isinstance(var.aval, af.core.AVal)
        assert var.aval == af.core.StrAVal()

    def test_len_on_trace_box_has_informative_error(self):
        tracer = af.core.TraceInterpreter()
        var = af.core.Var(aval=af.core.StrAVal())
        traced = tracer.box(var)

        with pytest.raises(
            TypeError,
            match=r"Cannot use length on a traced value\..*"
            r"Python length needs a concrete runtime value",
        ):
            len(traced)

    def test_plain_literal_is_not_irvar(self):
        assert not af.core.is_var("hello")


class TestFormatPrimitive:
    def test_basic_format(self):
        result = af.format("Hello, {}!", "World")
        assert result == "Hello, World!"

    def test_multiple_placeholders(self):
        result = af.format("{} + {} = {}", "1", "2", "3")
        assert result == "1 + 2 = 3"

    def test_format_ir(self):
        def func(x):
            return af.format("Value: {}", x)

        ir = af.trace(func)("test")
        assert len(ir.eqns) == 1
        assert ir.eqns[0].prim.name == "format"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_format_ir_async(self):
        def func(x):
            return af.format("Value: {}", x)

        ir = af.trace(func)("test")
        result = await ir.acall("hello")
        assert result == "Value: hello"


class TestConcatPrimitive:
    def test_basic_concat(self):
        result = af.concat("Hello", " ", "World")
        assert result == "Hello World"

    def test_two_args(self):
        result = af.concat("A", "B")
        assert result == "AB"

    def test_concat_rejects_non_string_input(self):
        with pytest.raises(TypeError):
            af.concat("A", 1)

    def test_concat_ir(self):
        def func(x, y):
            return af.concat(x, y)

        ir = af.trace(func)("a", "b")
        assert len(ir.eqns) == 1
        assert ir.eqns[0].prim.name == "concat"

    def test_traced_string_add_lowers_to_concat(self):
        def func(x):
            return x + "!"

        ir = af.trace(func)("a")

        assert len(ir.eqns) == 1
        assert ir.eqns[0].prim is af.string.concat_p
        assert ir.call("hello") == "hello!"

    def test_traced_string_radd_lowers_to_concat(self):
        def func(x):
            return "Hello, " + x

        ir = af.trace(func)("a")

        assert len(ir.eqns) == 1
        assert ir.eqns[0].prim is af.string.concat_p
        assert ir.call("world") == "Hello, world"

    def test_traced_string_add_both_args_lowers_to_concat(self):
        def func(a, b):
            return a + b

        ir = af.trace(func)("a", "b")

        assert len(ir.eqns) == 1
        assert ir.eqns[0].prim is af.string.concat_p
        assert ir.call("left", "right") == "leftright"

    def test_traced_string_add_unsupported_operand_raises_concat_error(self):
        def func(x):
            return x + 1

        with pytest.raises(AssertionError, match="Expected strings"):
            af.trace(func)("a")

    def test_traced_string_radd_unsupported_operand_raises_concat_error(self):
        def func(x):
            return 1 + x

        with pytest.raises(AssertionError, match="Expected strings"):
            af.trace(func)("a")

    def test_traced_non_string_add_raises_rule_error(self):
        def func(x):
            return x + 1

        with pytest.raises(TypeError, match=r"No trace rule for \+ on values of type IntAVal\(\)"):
            af.trace(func)(1)

    def test_concat_trace_rejects_non_string_input(self):
        def func(x, y, z):
            return af.concat(x, y, z)

        with pytest.raises(AssertionError, match="Expected strings"):
            af.trace(func)("a", "b", 1)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_concat_ir_async(self):
        def func(x, y):
            return af.concat(x, y)

        ir = af.trace(func)("a", "b")
        result = await ir.acall("hello", " world")
        assert result == "hello world"


class TestLMPrimitive:
    def test_lm_call_model_is_traced_as_input(self):
        def program(prompt: str, model: str):
            return af.lm_call([{"role": "user", "content": prompt}], model=model)

        ir = af.trace(program)("test", "gpt-5.5")
        eqn = ir.eqns[0]

        assert eqn.prim.name == "lm_call"
        assert eqn.params == {"roles": ["user"]}
        assert isinstance(eqn.in_tree[0][0], af.core.Var)
        assert isinstance(eqn.in_tree[1], af.core.Var)

    def test_lm_call_only_traces_messages_and_model(self):
        def program(prompt: str, model: str):
            return af.lm_call([{"role": "user", "content": prompt}], model=model)

        ir = af.trace(program)("test", "gpt-5.5")
        eqn = ir.eqns[0]

        assert eqn.params == {"roles": ["user"]}
        assert len(eqn.in_tree) == 2
        assert isinstance(eqn.in_tree[0][0], af.core.Var)
        assert isinstance(eqn.in_tree[1], af.core.Var)

    def test_lm_call_leaves_litellm_params_to_active_client(self):
        class ConfiguredRouter:
            def __init__(self):
                self.litellm_params = {"m1": {"temperature": 0.7, "max_tokens": 128}}

            def completion(self, *, messages: list[dict], model: str, **kwargs):
                assert kwargs == {}
                params = self.litellm_params[model]
                return FakeResponse(
                    f"{model}|{params['temperature']}|{params['max_tokens']}|"
                    f"{messages[-1]['content']}"
                )

            async def acompletion(self, *, messages: list[dict], model: str, **kwargs):
                return self.completion(messages=messages, model=model, **kwargs)

        def program(prompt: str, model: str):
            return af.lm_call([{"role": "user", "content": prompt}], model=model)

        ir = af.trace(program)("test", "gpt-5.5")

        with af.lm_client(ConfiguredRouter()):
            result = ir.call("hello", "m1")

        assert result == "m1|0.7|128|hello"

    def test_batch_lm_call_supports_variable_models(self):
        def program(prompt: str, model: str):
            return af.lm_call([{"role": "user", "content": prompt}], model=model)

        ir = af.trace(program)("test", "gpt-5.5")
        batched_ir = af.batch(ir, in_axes=(True, True))

        with af.lm_client(EchoRouter()):
            result = batched_ir.call(["hello", "goodbye"], ["m1", "m2"])

        assert result == ["m1|hello", "m2|goodbye"]

    def test_pullback_lm_call_zeroes_model_cotangent(self):
        def program(prompt: str, model: str):
            return af.lm_call([{"role": "user", "content": prompt}], model=model)

        ir = af.trace(program)("test", "gpt-5.5")
        pb_ir = af.pullback(ir)

        with af.lm_client(EchoRouter()):
            out, cotangent = pb_ir.call(("hello", "m1"), "feedback")

        assert out == "m1|hello"
        assert isinstance(cotangent[0], str)
        assert cotangent[1] == af.ad.Zero(af.core.StrAVal())


class TestBind:
    def test_bind_using(self):
        p = af.core.Prim("custom_bind")

        @ft.partial(af.core.impl_rules.set, p)
        def impl(in_tree, *, multiplier):
            return in_tree * multiplier

        @ft.partial(af.core.abstract_rules.set, p)
        def abstract_rule(in_tree, *, multiplier):
            return af.core.StrAVal()

        def func(x):
            return p.bind(x, multiplier=3)

        ir = af.trace(func)("A")
        assert ir.eqns[0].params["multiplier"] == 3
        result = ir.call("B")
        assert result == "BBB"


class TestInterpreter:
    def test_eval_interpreter_is_default(self):
        result = af.concat("a", "b")
        assert result == "ab"

    def test_use_interpreter_context(self):
        tracer = af.core.TraceInterpreter()
        with af.core.using_interpreter(tracer) as t:
            assert t is tracer
            af.format("Hello, {}!", af.core.Var.fresh(aval=af.core.StrAVal()))
            assert len(tracer.eqns) == 1
        result = af.concat("a", "b")
        assert result == "ab"

    def test_tracing_interpreter_creates_eqns(self):
        tracer = af.core.TraceInterpreter()
        with af.core.using_interpreter(tracer):
            af.format("Hello, {}!", af.core.Var.fresh(aval=af.core.StrAVal()))
        assert len(tracer.eqns) == 1


class TestStopGradient:
    def test_impl_is_identity(self):
        result = af.stop_gradient("hello")
        assert result == "hello"

    def test_ir_build(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)("test")
        assert len(ir.eqns) == 1
        assert ir.eqns[0].prim.name == "stop_gradient"

    def test_run_ir(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)("test")
        result = ir.call("hello")
        assert result == "hello"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_run_ir_async(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)("test")
        result = await ir.acall("hello")
        assert result == "hello"

    def test_pushforward_zeros_tangent(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)("a")
        pf_ir = af.pushforward(ir)
        primal_out, tangent_out = pf_ir.call(("primal",), ("tangent",))
        assert primal_out == "primal"
        assert af.ad.is_zero(tangent_out)

    def test_pullback_zeros_cotangent(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)("a")
        pb_ir = af.pullback(ir)
        primal_out, cotangent_in = pb_ir.call(("primal",), "cotangent")
        assert primal_out == "primal"
        assert af.ad.is_zero(cotangent_in[0])

    def test_batch(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)("a")
        batched_ir = af.batch(ir)
        result = batched_ir.call(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_in_chain_stops_gradient(self):
        def func(x, y):
            stopped = af.stop_gradient(x)
            return af.concat(stopped, y)

        ir = af.trace(func)("a", "b")
        pb_ir = af.pullback(ir)
        _, (cotangent_x, cotangent_y) = pb_ir.call(("a", "b"), "grad")
        assert af.ad.is_zero(cotangent_x)
        assert cotangent_y == "grad"

    def test_chained_with_format(self):
        def func(x):
            stopped = af.stop_gradient(x)
            return af.format("[{}]", stopped)

        ir = af.trace(func)("test")
        result = ir.call("hello")
        assert result == "[hello]"

    def test_tree_input(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)(("a", "b"))
        result = ir.call(("hello", "world"))
        assert result == ("hello", "world")

    def test_tree_pullback_zeros_all(self):
        def func(x):
            return af.stop_gradient(x)

        ir = af.trace(func)(("a", "b"))
        pb_ir = af.pullback(ir)
        _, cotangent_in = pb_ir.call((("p1", "p2"),), ("c1", "c2"))
        assert af.ad.is_zero(cotangent_in[0][0])
        assert af.ad.is_zero(cotangent_in[0][1])


class TestRunIRInline:
    """Tests for run_ir inlining behavior when called inside traced functions."""

    def test_run_ir_inlines_operations(self):
        """run_ir inside a traced function inlines the inner IR's operations."""
        inner_ir = af.trace(lambda x: af.format("[{}]", x))("X")

        def outer(x):
            return inner_ir.call(x)

        outer_ir = af.trace(outer)("X")

        assert len(outer_ir.eqns) == 1
        assert outer_ir.eqns[0].prim.name == "format"

    def test_run_ir_inline_executes_correctly(self):
        """Inlined run_ir produces correct output."""
        inner_ir = af.trace(lambda x: af.format("<{}>", x))("X")

        def outer(x):
            return inner_ir.call(x)

        outer_ir = af.trace(outer)("X")
        result = outer_ir.call("test")
        assert result == "<test>"

    def test_run_ir_inline_with_multiple_ops(self):
        """Multiple operations are all inlined."""

        def inner(x):
            a = af.concat(x, "!")
            b = af.format("[{}]", a)
            return b

        inner_ir = af.trace(inner)("X")

        def outer(x):
            return inner_ir.call(x)

        outer_ir = af.trace(outer)("X")
        assert len(outer_ir.eqns) == 2
        result = outer_ir.call("hello")
        assert result == "[hello!]"

    def test_nested_run_ir_inlines(self):
        """Nested run_ir calls all get inlined."""
        ir1 = af.trace(lambda x: af.concat(x, "1"))("X")
        ir2 = af.trace(lambda x: af.concat(x, "2"))("X")

        def outer(x):
            r1 = ir1.call(x)
            return ir2.call(r1)

        outer_ir = af.trace(outer)("X")
        assert len(outer_ir.eqns) == 2
        result = outer_ir.call("start")
        assert result == "start12"

    def test_pushforward_on_inlined_run_ir(self):
        """Pushforward works on inlined run_ir."""
        inner_ir = af.trace(lambda x: af.concat(x, "!"))("X")

        def outer(x):
            return inner_ir.call(x)

        outer_ir = af.trace(outer)("X")
        pf_ir = af.pushforward(outer_ir)
        (p_out, t_out) = pf_ir.call(("primal",), ("tangent",))
        assert p_out == "primal!"

        assert t_out == "tangent"

    def test_pullback_on_inlined_run_ir(self):
        """Pullback works on inlined run_ir."""
        inner_ir = af.trace(lambda x: af.concat(x, "!"))("X")

        def outer(x):
            return inner_ir.call(x)

        outer_ir = af.trace(outer)("X")
        pb_ir = af.pullback(outer_ir)
        _, cotangent = pb_ir.call(("hello",), "grad")
        assert cotangent == ("grad",)

    def test_batch_on_inlined_run_ir(self):
        """Batch works on inlined run_ir."""
        inner_ir = af.trace(lambda x: af.format("[{}]", x))("X")

        def outer(x):
            return inner_ir.call(x)

        outer_ir = af.trace(outer)("X")
        batched_ir = af.batch(outer_ir, in_axes=True)
        result = batched_ir.call(["a", "b", "c"])
        assert result == ["[a]", "[b]", "[c]"]


class TestTransformWrapperAvals:
    def test_pushforward_wrapper_preserves_aval(self):
        class TaggedAVal(af.core.AVal):
            __slots__ = ["tag"]

            def __init__(self, tag):
                self.tag = tag

        aval = TaggedAVal("pf")
        var = af.core.Var(aval=aval)
        ir = af.core.IR([], (var,), (var,))

        pf_ir = af.pushforward(ir)
        primals_in, tangents_in = pf_ir.in_tree
        primals_out, tangents_out = pf_ir.out_tree

        assert primals_in[0].aval is aval
        assert tangents_in[0].aval is aval
        assert primals_out[0].aval is aval
        assert tangents_out[0].aval is aval

    def test_pullback_wrapper_preserves_aval(self):
        class TaggedAVal(af.core.AVal):
            __slots__ = ["tag"]

            def __init__(self, tag):
                self.tag = tag

        aval = TaggedAVal("pb")
        var = af.core.Var(aval=aval)
        ir = af.core.IR([], (var,), (var,))

        pb_ir = af.pullback(ir)
        primals_in, cotangents_in = pb_ir.in_tree
        primals_out, cotangents_out = pb_ir.out_tree

        assert primals_in[0].aval is aval
        assert cotangents_in[0].aval is aval
        assert primals_out[0].aval is aval
        assert cotangents_out[0].aval is aval


class TestCotangentHelpers:
    def test_zero_str(self):
        z = af.ad.Zero(af.core.StrAVal())
        assert af.ad.is_zero(z)
        assert z.aval == af.core.StrAVal()
        assert af.ad.materialize(z) == ""

    def test_zero_requires_aval(self):
        with pytest.raises(AssertionError, match="Expected AVal"):
            af.ad.Zero(str)

    def test_zero_non_differentiable_type(self):
        z = af.ad.Zero(af.core.BoolAVal())
        assert af.ad.is_zero(z)
        assert z.aval == af.core.BoolAVal()
        with pytest.raises(TypeError):
            af.ad.materialize(z)

    def test_zero_equality(self):
        assert af.ad.Zero(af.core.StrAVal()) == af.ad.Zero(af.core.StrAVal())
        assert af.ad.Zero(af.core.StrAVal()) != af.ad.Zero(af.core.BoolAVal())

    def test_zeroof_is_idempotent(self):
        z = af.ad.Zero(af.core.StrAVal())
        assert af.ad.zeroof(z) is z
        assert af.ad.materialize(af.ad.zeroof(z)) == ""

    def test_zero_has_aval_but_is_not_traceable(self):
        z = af.ad.Zero(af.core.StrAVal())
        assert af.core.avalof(z) == af.core.StrAVal()
        assert not af.core.is_traceable(z)

    def test_zero_materializes_with_registered_aval_rule(self):
        class BlobAVal(af.core.AVal):
            __slots__ = ["size"]

            def __init__(self, size):
                self.size = size

            def __eq__(self, other):
                return type(self) is type(other) and self.size == other.size

            def __hash__(self):
                return hash((type(self), self.size))

        af.ad.zero_rules[BlobAVal] = lambda aval: ("zero", aval.size)
        try:
            assert af.ad.materialize(af.ad.Zero(BlobAVal(3))) == ("zero", 3)
        finally:
            del af.ad.zero_rules[BlobAVal]

    def test_cot_acc_single(self):
        result = af.ad.cot_acc(["hello"])
        assert result == "hello"

    def test_cot_acc_strings(self):
        result = af.ad.cot_acc(["a", "b", "c"])
        assert result == "abc"

    def test_cot_acc_is_traceable(self):
        ir = af.trace(lambda x, y: af.ad.cot_acc([x, y]))("a", "b")
        assert [eqn.prim for eqn in ir.eqns] == [af.ad.cot_acc_p]
        assert ir.call("c", "d") == "cd"

    def test_cot_acc_transform_rules(self):
        ir = af.trace(lambda x, y: af.ad.cot_acc([x, y]))("a", "b")
        assert af.pushforward(ir).call(("a", "b"), ("da", "db")) == ("ab", "dadb")
        assert af.pullback(ir).call(("a", "b"), "g") == ("ab", ("g", "g"))
        assert af.batch(ir).call(["a", "b"], ["c", "d"]) == ["ac", "bd"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_cot_acc_transform_rules_async(self):
        ir = af.trace(lambda x, y: af.ad.cot_acc([x, y]))("a", "b")
        assert await af.pushforward(ir).acall(("a", "b"), ("da", "db")) == ("ab", "dadb")
        assert await af.pullback(ir).acall(("a", "b"), "g") == ("ab", ("g", "g"))
        assert await af.batch(ir).acall(["a", "b"], ["c", "d"]) == ["ac", "bd"]

    def test_cot_acc_lists_elementwise(self):
        result = af.ad.cot_acc([["a", "b"], ["c", "d"]])
        assert result == ["ac", "bd"]

    def test_cot_acc_nested_containers_elementwise(self):
        result = af.ad.cot_acc([
            {"x": [1, af.ad.Zero(af.core.StrAVal())], "y": "a"},
            {"x": [2, "b"], "y": "c"},
        ])
        assert result == {"x": [3, "b"], "y": "ac"}

    def test_cot_acc_all_zeros(self):
        result = af.ad.cot_acc([
            af.ad.Zero(af.core.StrAVal()),
            af.ad.Zero(af.core.StrAVal()),
        ])
        assert af.ad.is_zero(result)

    def test_cot_acc_all_zeros_must_match_aval(self):
        with pytest.raises(AssertionError):
            af.ad.cot_acc([
                af.ad.Zero(af.core.StrAVal()),
                af.ad.Zero(af.core.IntAVal()),
            ])

    def test_cot_acc_ints_use_registered_rule(self):
        result = af.ad.cot_acc([1, 2, 3])
        assert result == 6

    def test_cot_acc_registered_val_without_rule_raises(self):
        with pytest.raises(
            TypeError, match=r"No cotangent accumulator registered for BoolAVal\(\)"
        ):
            af.ad.cot_acc([True, False])

    def test_cot_acc_unregistered_leaf_raises(self):
        class Blob:
            pass

        with pytest.raises(TypeError, match="Unsupported input leaf type"):
            af.ad.cot_acc([Blob(), Blob()])

    def test_cot_acc_uses_registered_aval_rule(self):
        class Blob:
            __slots__ = ["text"]

            def __init__(self, text):
                self.text = text

        class BlobAVal(af.core.AVal):
            __slots__ = []

        af.core.aval_rules[Blob] = lambda _: BlobAVal()
        af.ad.cot_acc_rules[BlobAVal] = lambda cs, aval: Blob("|".join(c.text for c in cs))
        try:
            result = af.ad.cot_acc([Blob("a"), Blob("b")])
            assert isinstance(result, Blob)
            assert result.text == "a|b"
        finally:
            del af.core.aval_rules[Blob]
            del af.ad.cot_acc_rules[BlobAVal]


class TestLiteralZeroing:
    def test_pushforward_zeros_literal_input_tangent(self):
        lit = "constant"
        var = af.core.Var(aval=af.core.StrAVal())
        in_tree = (lit, var)
        out_tree = (var,)
        ir = af.core.IR([], in_tree, out_tree)

        tangent_ir = af.pushforward(ir)

        _, tangent_in = tangent_ir.in_tree
        t_lit, t_var = tangent_in

        assert af.ad.is_zero(t_lit)
        assert t_lit.aval == af.core.StrAVal()
        assert isinstance(t_var, af.core.Var)

    def test_pushforward_zeros_literal_output_tangent(self):
        def f(x):
            return x, "constant_output"

        ir = af.trace(f)("input")

        res_var, res_lit = ir.out_tree
        assert res_lit == "constant_output"

        tangent_ir = af.pushforward(ir)

        _, tangent_out = tangent_ir.out_tree
        t_out_var, t_out_lit = tangent_out

        assert af.ad.is_zero(t_out_lit)
        assert isinstance(t_out_var, af.core.Var)

    def test_pullback_zeros_literal_output_cotangent(self):
        def f(x):
            return x, "constant_output"

        ir = af.trace(f)("input")
        adjoint_ir = af.pullback(ir)

        _, cotangent_out = adjoint_ir.in_tree
        c_out_var, c_out_lit = cotangent_out

        assert af.ad.is_zero(c_out_lit)

    def test_pullback_zeros_literal_input_cotangent(self):
        lit = "constant_input"
        var = af.core.Var(aval=af.core.StrAVal())
        in_tree = (lit, var)
        out_tree = (var,)
        ir = af.core.IR([], in_tree, out_tree)

        adjoint_ir = af.pullback(ir)

        _, cotangent_in = adjoint_ir.out_tree
        c_in_lit, c_in_var = cotangent_in

        assert af.ad.is_zero(c_in_lit)


class TestStaticTransformInputs:
    def test_pushforward_static_input_literal_is_not_boxed(self):
        def label(prefix, value):
            return af.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        pf_ir = af.pushforward(ir)

        out, tangent = pf_ir.call(("Q", "x"), (af.ad.zeroof("Q"), "dx"))

        assert out == "Qx"
        assert tangent == "dx"
        with pytest.raises(AssertionError, match="Static input mismatch"):
            pf_ir.call(("R", "x"), (af.ad.zeroof("R"), "dx"))

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pushforward_static_input_literal_is_not_boxed_async(self):
        def label(prefix, value):
            return af.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        pf_ir = af.pushforward(ir)

        out, tangent = await pf_ir.acall(("Q", "x"), (af.ad.zeroof("Q"), "dx"))

        assert out == "Qx"
        assert tangent == "dx"
        with pytest.raises(AssertionError, match="Static input mismatch"):
            await pf_ir.acall(("R", "x"), (af.ad.zeroof("R"), "dx"))

    def test_pullback_static_input_literal_is_not_boxed(self):
        def label(prefix, value):
            return af.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        pb_ir = af.pullback(ir)

        out, (c_prefix, c_value) = pb_ir.call(("Q", "x"), "g")

        assert out == "Qx"
        assert af.ad.is_zero(c_prefix)
        assert c_value == "g"
        with pytest.raises(AssertionError, match="Static input mismatch"):
            pb_ir.call(("R", "x"), "g")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_pullback_static_input_literal_is_not_boxed_async(self):
        def label(prefix, value):
            return af.concat(prefix, value)

        ir = af.trace(label, static=(True, False))("Q", "x")
        pb_ir = af.pullback(ir)

        out, (c_prefix, c_value) = await pb_ir.acall(("Q", "x"), "g")

        assert out == "Qx"
        assert af.ad.is_zero(c_prefix)
        assert c_value == "g"
        with pytest.raises(AssertionError, match="Static input mismatch"):
            await pb_ir.acall(("R", "x"), "g")
