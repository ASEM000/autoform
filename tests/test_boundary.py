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
import autoform.src.boundary as boundary
import autoform.src.core as core
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.tree as tree
import autoform.src.utils as utils
from tests import aexecute, execute


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param(["x", "y"], (["x!", "y!"], ["!", "!"]), id="list"),
        pytest.param(("x", "y"), (("x!", "y!"), ("!", "!")), id="tuple"),
        pytest.param(
            {"y": "x", "x": "y"},
            ({"y": "x!", "x": "y!"}, {"y": "!", "x": "!"}),
            id="dict",
        ),
    ],
)
@pytest.mark.parametrize("out_axes", [(True, True), (True, False)], ids=["selected", "shared"])
def test_custom_batch_rule_receives_common_spec(executor, values, expected, out_axes):
    @boundary.custom
    def program(x, y):
        return string.concat(x, y), y

    @program.set_batch
    def rule(in_tree):
        spec, in_axes, inputs = in_tree
        assert isinstance(spec, tree.PyTreeSpec)
        assert spec.num_leaves == 2
        assert spec == tree.structure(values)
        assert in_axes == (True, False)
        items = tree.unstack(spec, in_axes, inputs)
        results = ((x + y, y) for x, y in items)
        out = tree.stack(spec, out_axes, results)
        if not out_axes[1]:
            assert out[1] == "!"
        return out, out_axes

    program.aset_batch(utils.asyncify(rule))
    ir = axis.batch(stage.trace(program)("x", "!"), in_axes=(True, False))
    assert tree.equal(executor(ir, values, "!"), expected)


class TestCustomFunction:
    def test_traced_call_composes_with_primitives(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        ir = stage.trace(lambda x: string.concat(bracket(x), "!"))("seed")
        assert ir.call("hello") == "[hello]!"

    def test_direct_call_behaves_like_function(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        assert bracket("hello") == "[hello]"

    def test_undefined_pushforward_falls_back_to_body_ir(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        ir = stage.trace(lambda x: bracket(x))("seed")
        out, t = ad.pushforward(ir).call(("hello",), ("change",))

        assert out == "[hello]"
        assert t == "change"

    def test_undefined_pullback_falls_back_to_body_ir(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        ir = stage.trace(lambda x: bracket(x))("seed")
        out, c = ad.pullback(ir).call(("hello",), "feedback")

        assert out == "[hello]"
        assert c == ("feedback",)

    def test_undefined_batch_falls_back_to_body_ir(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        ir = stage.trace(lambda x: bracket(x))("seed")
        batched = axis.batch(ir)

        assert batched.call(["a", "b"]) == ["[a]", "[b]"]


class TestCustomPushforward:
    def test_custom_pushforward_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_upper(x):
            return x.upper()

        with pytest.raises(AttributeError):
            stage.trace(lambda x: python_only_upper(x))("seed")

        @boundary.custom
        def pair_program(x, y):
            left = string.format("left {x}", x=x)
            right = string.concat(y, "!")
            return left, right

        @pair_program.set_pushforward
        def pair_program_pushforward(in_tree, /):
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

        ir = stage.trace(lambda x, y: pair_program(x, y))("x", "y")
        out, t = ad.pushforward(ir).call(("hello", "world"), ("small", "change"))

        assert out == ("HELLO", "WORLD")
        assert t == ("SMALL", "CHANGE")

    def test_custom_pushforward_rule_matches_mapping_signature(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /):
            p, t = in_tree
            (dx,) = t
            return bracket(*p), string.format(
                "custom delta: {value}",
                value=(dx.aval.zero() if isinstance(dx, core.Zero) else dx),
            )

        ir = stage.trace(lambda x: string.concat(bracket(x), "!"))("seed")
        out, t = ad.pushforward(ir).call(("hello",), ("small change",))

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
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.aset_pushforward
        async def bracket_pushforward(in_tree, /):
            p, t = in_tree
            (dx,) = t
            return bracket(*p), string.format(
                "async delta: {value}",
                value=(dx.aval.zero() if isinstance(dx, core.Zero) else dx),
            )

        ir = stage.trace(lambda x: bracket(x))("seed")
        out, t = executor(ad.pushforward(ir), ("hello",), ("change",))

        assert out == "[hello]"
        assert t == expected

    def test_set_pushforward_replaces_default_rule(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /):
            p, t = in_tree
            (dx,) = t
            return bracket(*p), string.format(
                "push {value}",
                value=(dx.aval.zero() if isinstance(dx, core.Zero) else dx),
            )

        ir = stage.trace(lambda x: bracket(x))("seed")
        _, t = ad.pushforward(ir).call(("hello",), ("change",))

        assert t == "push change"

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, "sync push change", id="sync"),
            pytest.param(aexecute, "change", id="async"),
        ],
    )
    def test_set_pushforward_dispatches_by_executor(self, executor, expected):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_pushforward
        def bracket_pushforward(in_tree, /):
            p, t = in_tree
            (dx,) = t
            return bracket(*p), string.format(
                "sync push {value}",
                value=(dx.aval.zero() if isinstance(dx, core.Zero) else dx),
            )

        ir = stage.trace(lambda x: bracket(x))("seed")
        _, t = executor(ad.pushforward(ir), ("hello",), ("change",))

        assert t == expected


class TestCustomPullback:
    def test_custom_pullback_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_lower(x):
            return x.lower()

        with pytest.raises(AttributeError):
            stage.trace(lambda x: python_only_lower(x))("seed")

        @boundary.custom
        def pair_program(x, y):
            left = string.format("left {x}", x=x)
            right = string.concat(y, "!")
            return left, right

        @pair_program.set_pullback
        def pair_program_pullback(in_tree, /):
            (p, output), c = in_tree
            x, y = p
            out_left, out_right = output
            c_left, c_right = c
            return (
                f"{python_only_lower(x)} <- {python_only_lower(out_left)} <- {c_left}",
                f"{python_only_lower(y)} <- {python_only_lower(out_right)} <- {c_right}",
            )

        ir = stage.trace(lambda x, y: pair_program(x, y))("x", "y")
        out, c = ad.pullback(ir).call(("HELLO", "WORLD"), ("L", "R"))

        assert out == ("left HELLO", "WORLD!")
        assert c == ("hello <- left hello <- L", "world <- world! <- R")

    def test_custom_pullback_rule_uses_mlx_argument_order(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /):
            (p, output), c = in_tree
            (x,) = p
            return (
                string.format(
                    "{cotangent} via {output} from {x}",
                    cotangent=c,
                    output=output,
                    x=x,
                ),
            )

        ir = stage.trace(lambda x: string.concat(bracket(x), "!"))("seed")
        out, c = ad.pullback(ir).call(("hello",), "feedback")

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
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.aset_pullback
        async def bracket_pullback(in_tree, /):
            (p, output), c = in_tree
            del p
            return (
                string.format(
                    "async {cotangent} via {output}",
                    cotangent=c,
                    output=output,
                ),
            )

        ir = stage.trace(lambda x: bracket(x))("seed")
        out, c = executor(ad.pullback(ir), ("hello",), "feedback")

        assert out == "[hello]"
        assert c == expected

    def test_set_pullback_replaces_default_rule(self):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /):
            (p, output), c = in_tree
            del p
            return (string.format("pull {output} {cotangent}", output=output, cotangent=c),)

        ir = stage.trace(lambda x: bracket(x))("seed")
        _, c = ad.pullback(ir).call(("hello",), "feedback")

        assert c == ("pull [hello] feedback",)

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ("sync pull [hello] feedback",), id="sync"),
            pytest.param(aexecute, ("feedback",), id="async"),
        ],
    )
    def test_set_pullback_dispatches_by_executor(self, executor, expected):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_pullback
        def bracket_pullback(in_tree, /):
            (p, output), c = in_tree
            del p
            return (
                string.format(
                    "sync pull {output} {cotangent}",
                    output=output,
                    cotangent=c,
                ),
            )

        ir = stage.trace(lambda x: bracket(x))("seed")
        _, c = executor(ad.pullback(ir), ("hello",), "feedback")

        assert c == expected

    def test_pullback_survives_retrace_between_halves(self):
        @boundary.custom
        def bracket(x):
            return string.concat("[", x, "]")

        fwd, bwd = ad.fwd_bwd(stage.trace(lambda x: bracket(x))("seed"))
        _, residual = fwd.call("hello")
        for _ in range(256):
            stage.trace(boundary.custom(lambda x: string.concat(x, "!")))("x")
        assert bwd.call(residual, "feedback") == ("feedback",)


class TestCustomBatch:
    def test_custom_batch_rule_can_use_untraceable_python_for_multi_primitive_body(self):
        def python_only_title(x):
            return x.title()

        with pytest.raises(AttributeError):
            stage.trace(lambda x: python_only_title(x))("seed")

        @boundary.custom
        def pair_program(x, y):
            left = string.format("left {x}", x=x)
            right = string.concat(y, "!")
            return left, right

        @pair_program.set_batch
        def pair_program_batch(in_tree, /):
            spec, in_axes, values = in_tree
            xs, ys = values
            x_axis, y_axis = in_axes
            assert spec.num_leaves == 2
            assert x_axis is True
            assert y_axis is True
            return (
                [python_only_title(x) for x in xs],
                [python_only_title(y) for y in ys],
            ), (True, True)

        ir = stage.trace(lambda x, y: pair_program(x, y))("x", "y")
        batched = axis.batch(ir)

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
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_batch
        def bracket_batch(in_tree, /):
            spec, in_axes, values = in_tree
            assert spec.num_leaves == 2
            (xs,) = values
            (x_axis,) = in_axes
            assert x_axis is True
            return ([string.format(template, x=x) for x in xs], True)

        ir = stage.trace(lambda x: bracket(x))("seed")
        batched = axis.batch(ir)
        assert batched.call(["a", "b"]) == [expected_a, expected_b]

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ["[a]", "[b]"], id="sync"),
            pytest.param(aexecute, ["async <a>", "async <b>"], id="async"),
        ],
    )
    def test_aset_batch_dispatches_by_executor(self, executor, expected):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.aset_batch
        async def bracket_batch(in_tree, /):
            spec, in_axes, values = in_tree
            assert spec.num_leaves == 2
            (xs,) = values
            (x_axis,) = in_axes
            assert x_axis is True
            return [string.format("async <{x}>", x=x) for x in xs], True

        ir = stage.trace(lambda x: bracket(x))("seed")
        batched = axis.batch(ir)

        assert executor(batched, ["a", "b"]) == expected

    @pytest.mark.parametrize(
        "executor, expected",
        [
            pytest.param(execute, ["sync batch <a>", "sync batch <b>"], id="sync"),
            pytest.param(aexecute, ["[a]", "[b]"], id="async"),
        ],
    )
    def test_set_batch_dispatches_by_executor(self, executor, expected):
        @boundary.custom
        def bracket(x):
            return string.format("[{x}]", x=x)

        @bracket.set_batch
        def bracket_batch(in_tree, /):
            spec, in_axes, values = in_tree
            assert spec.num_leaves == 2
            (xs,) = values
            (x_axis,) = in_axes
            assert x_axis is True
            return [string.format("sync batch <{x}>", x=x) for x in xs], True

        ir = stage.trace(lambda x: bracket(x))("seed")
        batched = axis.batch(ir)

        assert executor(batched, ["a", "b"]) == expected


class TestCustomParams:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_default_rules_use_params(self, executor):
        @boundary.custom
        def wrap(x, *, left, right):
            return string.concat(string.concat(left, x), right)

        def program(x):
            return wrap(x, left="<", right=">")

        ir = stage.trace(program)("seed")
        assert executor(ir, "a") == "<a>"
        assert executor(ad.pushforward(ir), ("a",), ("t",)) == ("<a>", "t")
        assert executor(ad.pullback(ir), ("a",), "g") == ("<a>", ("g",))
        assert executor(axis.batch(ir), ["a", "b"]) == ["<a>", "<b>"]

    def test_params_are_not_shared_between_calls(self):
        @boundary.custom
        def wrap(x, *, left):
            return string.concat(left, x)

        def program(x):
            return wrap(x, left="<"), wrap(x, left="[")

        ir = stage.trace(program)("seed")
        assert ir.call("a") == ("<a", "[a")

    def test_rules_receive_params(self):
        @boundary.custom
        def wrap(x, *, mark):
            return string.concat(x, mark)

        @wrap.set_pushforward
        def wrap_pushforward(in_tree, /, *, mark):
            (x,), (t,) = in_tree
            return wrap(x, mark=mark), string.concat(t, mark)

        @wrap.set_pullback
        def wrap_pullback(in_tree, /, *, mark):
            _, c = in_tree
            return (string.concat(c, mark),)

        @wrap.set_batch
        def wrap_batch(in_tree, /, *, mark):
            _, _, (xs,) = in_tree
            return [wrap(x, mark=mark) for x in xs], True

        def program(x):
            return wrap(x, mark="!")

        ir = stage.trace(program)("seed")
        assert ad.pushforward(ir).call(("a",), ("t",)) == ("a!", "t!")
        assert ad.pullback(ir).call(("a",), "g") == ("a!", ("g!",))
        assert axis.batch(ir).call(["a", "b"]) == ["a!", "b!"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_body_is_traced_once(self, executor):
        traces = []

        @boundary.custom
        def wrap(x, *, mark):
            traces.append(mark)
            return string.concat(x, mark)

        def program(x):
            return wrap(x, mark="!")

        ir = stage.trace(program)("seed")
        assert executor(ad.pushforward(ir), ("a",), ("t",)) == ("a!", "t")
        assert executor(ad.pullback(ir), ("a",), "g") == ("a!", ("g",))
        assert executor(ad.pullback(ir), ("b",), "h") == ("b!", ("h",))
        assert executor(axis.batch(ir), ["a", "b"]) == ["a!", "b!"]
        assert traces == ["!"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_default_batch_with_shared_inputs(self, executor):
        @boundary.custom
        def wrap(x):
            return string.concat(x, "!")

        def program(x, y):
            return string.concat(wrap(y), x)

        ir = stage.trace(program)("a", "b")
        batched = axis.batch(ir, in_axes=(True, False))
        assert executor(batched, ["a", "b"], "c") == ["c!a", "c!b"]
