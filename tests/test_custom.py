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

import importlib

import pytest

import autoform as af


class TestCustomFunction:
    def test_only_custom_is_exported(self):
        custom_module = importlib.import_module("autoform.custom")

        assert af.custom is custom_module.custom
        assert not hasattr(af, "CustomFunction")
        assert not hasattr(custom_module, "CustomFunction")

    def test_call_stages_custom_call_with_python_function(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        ir = af.trace(lambda x: af.concat(bracket(x), "!"))("seed")
        custom_eqn = ir.ir_eqns[0]

        assert ir.ir_eqns[0].prim.name.endswith("bracket")
        assert [eqn.prim.name.split(":")[0] for eqn in ir.ir_eqns] == ["custom_call", "concat"]
        assert set(custom_eqn.params) == {"call"}
        assert custom_eqn.params["call"] is bracket.func
        assert "func" not in custom_eqn.params
        assert "ir" not in custom_eqn.params
        assert "key" not in custom_eqn.params
        assert ir.call("hello") == "[hello]!"

    def test_rules_are_stored_in_rule_mappings(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            primals, tangents = in_tree
            (dx,) = tangents
            return call(*primals), dx

        rule = af.core.push_rules.get(bracket.prim)

        assert not hasattr(bracket, "pushforward_rule")
        assert rule is bracket_pushforward

    def test_direct_call_behaves_like_function(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        assert bracket("hello") == "[hello]"

    def test_undefined_pushforward_falls_back_to_body_ir(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        ir = af.trace(lambda x: bracket(x))("seed")
        out, tangent = af.pushforward(ir).call(("hello",), ("change",))

        assert out == "[hello]"
        assert tangent == "[change]"

    def test_undefined_pullback_falls_back_to_body_ir(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        ir = af.trace(lambda x: bracket(x))("seed")
        out, cotangent = af.pullback(ir).call(("hello",), "feedback")

        assert out == "[hello]"
        assert cotangent == ("feedback",)

    def test_undefined_batch_falls_back_to_body_ir(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert batched.call(["a", "b"]) == ["[a]", "[b]"]


class TestCustomPushforward:
    def test_custom_pushforward_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_upper(x):
            return x.upper()

        with pytest.raises(AttributeError):
            af.trace(lambda x: python_only_upper(x))("seed")

        @af.custom
        def pair_program(x, y):
            left = af.format("left {}", x)
            right = af.concat(y, "!")
            return left, right

        @pair_program.set_pushforward
        def pair_program_pushforward(in_tree, /, *, call):
            del call
            primals, tangents = in_tree
            x, y = primals
            dx, dy = tangents
            return (
                python_only_upper(x),
                python_only_upper(y),
            ), (
                python_only_upper(dx),
                python_only_upper(dy),
            )

        ir = af.trace(lambda x, y: pair_program(x, y))("x", "y")
        out, tangent = af.pushforward(ir).call(("hello", "world"), ("small", "change"))

        assert out == ("HELLO", "WORLD")
        assert tangent == ("SMALL", "CHANGE")

    def test_custom_pushforward_rule_matches_mapping_signature(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            primals, tangents = in_tree
            (dx,) = tangents
            return call(*primals), af.format("custom delta: {}", af.ad.materialize(dx))

        ir = af.trace(lambda x: af.concat(bracket(x), "!"))("seed")
        out, tangent = af.pushforward(ir).call(("hello",), ("small change",))

        assert out == "[hello]!"
        assert tangent == "custom delta: small change"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_custom_pushforward_async(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.aset_pushforward
        async def bracket_pushforward(in_tree, /, *, call):
            primals, tangents = in_tree
            (dx,) = tangents
            return call(*primals), af.format("async delta: {}", af.ad.materialize(dx))

        ir = af.trace(lambda x: bracket(x))("seed")
        out, tangent = await af.pushforward(ir).acall(("hello",), ("change",))

        assert out == "[hello]"
        assert tangent == "async delta: change"

    def test_set_pushforward_replaces_default_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            primals, tangents = in_tree
            (dx,) = tangents
            return call(*primals), af.format("push {}", af.ad.materialize(dx))

        ir = af.trace(lambda x: bracket(x))("seed")
        _, tangent = af.pushforward(ir).call(("hello",), ("change",))

        assert tangent == "push change"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_set_pushforward_does_not_replace_async_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            primals, tangents = in_tree
            (dx,) = tangents
            return call(*primals), af.format("sync push {}", af.ad.materialize(dx))

        ir = af.trace(lambda x: bracket(x))("seed")
        _, tangent = await af.pushforward(ir).acall(("hello",), ("change",))

        assert tangent == "[change]"


class TestCustomPullback:
    def test_custom_pullback_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_lower(x):
            return x.lower()

        with pytest.raises(AttributeError):
            af.trace(lambda x: python_only_lower(x))("seed")

        @af.custom
        def pair_program(x, y):
            left = af.format("left {}", x)
            right = af.concat(y, "!")
            return left, right

        @pair_program.set_pullback
        def pair_program_pullback(in_tree, /, *, call):
            del call
            (primals, output), cotangents = in_tree
            x, y = primals
            left_out, right_out = output
            left_cotangent, right_cotangent = cotangents
            return (
                f"{python_only_lower(x)} <- {python_only_lower(left_out)} <- {left_cotangent}",
                f"{python_only_lower(y)} <- {python_only_lower(right_out)} <- {right_cotangent}",
            )

        ir = af.trace(lambda x, y: pair_program(x, y))("x", "y")
        out, cotangents = af.pullback(ir).call(("HELLO", "WORLD"), ("L", "R"))

        assert out == ("left HELLO", "WORLD!")
        assert cotangents == ("hello <- left hello <- L", "world <- world! <- R")

    def test_custom_pullback_rule_uses_mlx_argument_order(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /, *, call):
            del call
            (primals, output), cotangent = in_tree
            (x,) = primals
            return (af.format("{} via {} from {}", cotangent, output, x),)

        ir = af.trace(lambda x: af.concat(bracket(x), "!"))("seed")
        out, cotangent = af.pullback(ir).call(("hello",), "feedback")

        assert out == "[hello]!"
        assert cotangent == ("feedback via [hello] from hello",)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_custom_pullback_async(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.aset_pullback
        async def bracket_pullback(in_tree, /, *, call):
            del call
            (primals, output), cotangent = in_tree
            del primals
            return (af.format("async {} via {}", cotangent, output),)

        ir = af.trace(lambda x: bracket(x))("seed")
        out, cotangent = await af.pullback(ir).acall(("hello",), "feedback")

        assert out == "[hello]"
        assert cotangent == ("async feedback via [hello]",)

    def test_set_pullback_replaces_default_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /, *, call):
            del call
            (primals, output), cotangent = in_tree
            del primals
            return (af.format("pull {} {}", output, cotangent),)

        ir = af.trace(lambda x: bracket(x))("seed")
        _, cotangent = af.pullback(ir).call(("hello",), "feedback")

        assert cotangent == ("pull [hello] feedback",)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_set_pullback_does_not_replace_async_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /, *, call):
            del call
            (primals, output), cotangent = in_tree
            del primals
            return (af.format("sync pull {} {}", output, cotangent),)

        ir = af.trace(lambda x: bracket(x))("seed")
        _, cotangent = await af.pullback(ir).acall(("hello",), "feedback")

        assert cotangent == ("feedback",)


class TestCustomBatch:
    def test_custom_batch_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_title(x):
            return x.title()

        with pytest.raises(AttributeError):
            af.trace(lambda x: python_only_title(x))("seed")

        @af.custom
        def pair_program(x, y):
            left = af.format("left {}", x)
            right = af.concat(y, "!")
            return left, right

        @pair_program.set_batch
        def pair_program_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            xs, ys = values
            x_axis, y_axis = axes
            assert batch_size == 2
            assert x_axis is True
            assert y_axis is True
            return (
                [python_only_title(x) for x in xs],
                [python_only_title(y) for y in ys],
            ), (True, True)

        ir = af.trace(lambda x, y: pair_program(x, y))("x", "y")
        batched = af.batch(ir)

        assert batched.call(["hello", "goodbye"], ["world", "moon"]) == (
            ["Hello", "Goodbye"],
            ["World", "Moon"],
        )

    def test_custom_batch_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_batch
        def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return [af.format("<{}>", x) for x in xs], True

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert batched.call(["a", "b"]) == ["<a>", "<b>"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_custom_batch_async(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.aset_batch
        async def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return [af.format("async <{}>", x) for x in xs], True

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert await batched.acall(["a", "b"]) == ["async <a>", "async <b>"]

    def test_set_batch_replaces_default_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_batch
        def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return [af.format("batch <{}>", x) for x in xs], True

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert batched.call(["a", "b"]) == ["batch <a>", "batch <b>"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_set_batch_does_not_replace_async_rule(self):
        @af.custom
        def bracket(x):
            return af.format("[{}]", x)

        @bracket.set_batch
        def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return [af.format("sync batch <{}>", x) for x in xs], True

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert await batched.acall(["a", "b"]) == ["[a]", "[b]"]
