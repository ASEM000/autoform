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
from tests import aexecute, execute


class TestCustomFunction:
    def test_traced_call_composes_with_primitives(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        ir = af.trace(lambda x: af.string.concat(bracket(x), "!"))("seed")
        assert ir.call("hello") == "[hello]!"

    def test_direct_call_behaves_like_function(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        assert bracket("hello") == "[hello]"

    def test_undefined_pushforward_falls_back_to_body_ir(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        ir = af.trace(lambda x: bracket(x))("seed")
        out, t = af.pushforward(ir).call(("hello",), ("change",))

        assert out == "[hello]"
        assert t == "change"

    def test_undefined_pullback_falls_back_to_body_ir(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        ir = af.trace(lambda x: bracket(x))("seed")
        out, c = af.pullback(ir).call(("hello",), "feedback")

        assert out == "[hello]"
        assert c == ("feedback",)

    def test_undefined_batch_falls_back_to_body_ir(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

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
            left = af.string.format("left {x}", x=x)
            right = af.string.concat(y, "!")
            return left, right

        @pair_program.set_pushforward
        def pair_program_pushforward(in_tree, /, *, call):
            del call
            p, t = in_tree
            x, y = p
            dx, dy = t
            return (
                python_only_upper(x),
                python_only_upper(y),
            ), (
                python_only_upper(dx),
                python_only_upper(dy),
            )

        ir = af.trace(lambda x, y: pair_program(x, y))("x", "y")
        out, t = af.pushforward(ir).call(("hello", "world"), ("small", "change"))

        assert out == ("HELLO", "WORLD")
        assert t == ("SMALL", "CHANGE")

    def test_custom_pushforward_rule_matches_mapping_signature(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            p, t = in_tree
            (dx,) = t
            return call(*p), af.string.format(
                "custom delta: {value}",
                value=af.core.materialize_zeros(dx),
            )

        ir = af.trace(lambda x: af.string.concat(bracket(x), "!"))("seed")
        out, t = af.pushforward(ir).call(("hello",), ("small change",))

        assert out == "[hello]!"
        assert t == "custom delta: small change"

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, "change", id="sync"),
            pytest.param(aexecute, "async delta: change", id="async"),
        ],
    )
    def test_aset_pushforward_dispatches_by_executor(self, executor, expected):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.aset_pushforward
        async def bracket_pushforward(in_tree, /, *, call):
            p, t = in_tree
            (dx,) = t
            return call(*p), af.string.format(
                "async delta: {value}",
                value=af.core.materialize_zeros(dx),
            )

        ir = af.trace(lambda x: bracket(x))("seed")
        out, t = executor(af.pushforward(ir), ("hello",), ("change",))

        assert out == "[hello]"
        assert t == expected

    def test_set_pushforward_replaces_default_rule(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            p, t = in_tree
            (dx,) = t
            return call(*p), af.string.format("push {value}", value=af.core.materialize_zeros(dx))

        ir = af.trace(lambda x: bracket(x))("seed")
        _, t = af.pushforward(ir).call(("hello",), ("change",))

        assert t == "push change"

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, "sync push change", id="sync"),
            pytest.param(aexecute, "change", id="async"),
        ],
    )
    def test_set_pushforward_dispatches_by_executor(self, executor, expected):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /, *, call):
            p, t = in_tree
            (dx,) = t
            return call(*p), af.string.format(
                "sync push {value}",
                value=af.core.materialize_zeros(dx),
            )

        ir = af.trace(lambda x: bracket(x))("seed")
        _, t = executor(af.pushforward(ir), ("hello",), ("change",))

        assert t == expected


class TestCustomPullback:
    def test_custom_pullback_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_lower(x):
            return x.lower()

        with pytest.raises(AttributeError):
            af.trace(lambda x: python_only_lower(x))("seed")

        @af.custom
        def pair_program(x, y):
            left = af.string.format("left {x}", x=x)
            right = af.string.concat(y, "!")
            return left, right

        @pair_program.set_pullback
        def pair_program_pullback(in_tree, /, *, call):
            del call
            (p, output), c = in_tree
            x, y = p
            out_left, out_right = output
            c_left, c_right = c
            return (
                f"{python_only_lower(x)} <- {python_only_lower(out_left)} <- {c_left}",
                f"{python_only_lower(y)} <- {python_only_lower(out_right)} <- {c_right}",
            )

        ir = af.trace(lambda x, y: pair_program(x, y))("x", "y")
        out, c = af.pullback(ir).call(("HELLO", "WORLD"), ("L", "R"))

        assert out == ("left HELLO", "WORLD!")
        assert c == ("hello <- left hello <- L", "world <- world! <- R")

    def test_custom_pullback_rule_uses_mlx_argument_order(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /, *, call):
            del call
            (p, output), c = in_tree
            (x,) = p
            return (
                af.string.format(
                    "{cotangent} via {output} from {x}",
                    cotangent=c,
                    output=output,
                    x=x,
                ),
            )

        ir = af.trace(lambda x: af.string.concat(bracket(x), "!"))("seed")
        out, c = af.pullback(ir).call(("hello",), "feedback")

        assert out == "[hello]!"
        assert c == ("feedback via [hello] from hello",)

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ("feedback",), id="sync"),
            pytest.param(aexecute, ("async feedback via [hello]",), id="async"),
        ],
    )
    def test_aset_pullback_dispatches_by_executor(self, executor, expected):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.aset_pullback
        async def bracket_pullback(in_tree, /, *, call):
            del call
            (p, output), c = in_tree
            del p
            return (
                af.string.format(
                    "async {cotangent} via {output}",
                    cotangent=c,
                    output=output,
                ),
            )

        ir = af.trace(lambda x: bracket(x))("seed")
        out, c = executor(af.pullback(ir), ("hello",), "feedback")

        assert out == "[hello]"
        assert c == expected

    def test_set_pullback_replaces_default_rule(self):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /, *, call):
            del call
            (p, output), c = in_tree
            del p
            return (af.string.format("pull {output} {cotangent}", output=output, cotangent=c),)

        ir = af.trace(lambda x: bracket(x))("seed")
        _, c = af.pullback(ir).call(("hello",), "feedback")

        assert c == ("pull [hello] feedback",)

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ("sync pull [hello] feedback",), id="sync"),
            pytest.param(aexecute, ("feedback",), id="async"),
        ],
    )
    def test_set_pullback_dispatches_by_executor(self, executor, expected):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /, *, call):
            del call
            (p, output), c = in_tree
            del p
            return (
                af.string.format(
                    "sync pull {output} {cotangent}",
                    output=output,
                    cotangent=c,
                ),
            )

        ir = af.trace(lambda x: bracket(x))("seed")
        _, c = executor(af.pullback(ir), ("hello",), "feedback")

        assert c == expected


class TestCustomBatch:
    def test_custom_batch_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_title(x):
            return x.title()

        with pytest.raises(AttributeError):
            af.trace(lambda x: python_only_title(x))("seed")

        @af.custom
        def pair_program(x, y):
            left = af.string.format("left {x}", x=x)
            right = af.string.concat(y, "!")
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

    @pytest.mark.parametrize(
        "template, expected_a, expected_b",
        [
            pytest.param("<{x}>", "<a>", "<b>", id="custom_batch_rule"),
            pytest.param(
                "batch <{x}>",
                "batch <a>",
                "batch <b>",
                id="set_batch_replaces_default_rule",
            ),
        ],
    )
    def test_custom_batch_rule(self, template, expected_a, expected_b):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_batch
        def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return ([af.string.format(template, x=x) for x in xs], True)

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)
        assert batched.call(["a", "b"]) == [expected_a, expected_b]

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ["[a]", "[b]"], id="sync"),
            pytest.param(aexecute, ["async <a>", "async <b>"], id="async"),
        ],
    )
    def test_aset_batch_dispatches_by_executor(self, executor, expected):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.aset_batch
        async def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return [af.string.format("async <{x}>", x=x) for x in xs], True

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert executor(batched, ["a", "b"]) == expected

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ["sync batch <a>", "sync batch <b>"], id="sync"),
            pytest.param(aexecute, ["[a]", "[b]"], id="async"),
        ],
    )
    def test_set_batch_dispatches_by_executor(self, executor, expected):
        @af.custom
        def bracket(x):
            return af.string.format("[{x}]", x=x)

        @bracket.set_batch
        def bracket_batch(in_tree, /, *, call):
            del call
            batch_size, axes, values = in_tree
            assert batch_size == 2
            (xs,) = values
            (x_axis,) = axes
            assert x_axis is True
            return [af.string.format("sync batch <{x}>", x=x) for x in xs], True

        ir = af.trace(lambda x: bracket(x))("seed")
        batched = af.batch(ir)

        assert executor(batched, ["a", "b"]) == expected
