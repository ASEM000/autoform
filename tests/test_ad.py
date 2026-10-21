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

import autoform.extend as extend
import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.boundary as boundary
import autoform.src.check as check
import autoform.src.control as control
import autoform.src.core as core
import autoform.src.intercept as intercept
import autoform.src.numeric as numeric
import autoform.src.order as order
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import AValBlob, aexecute, angle_text, append_bang, bracket_text, cube, execute, square


def test_pushforward_checker_rejects_nested_tangent_without_equations():
    active = core.active_interp.get()
    parent = ad.PushInterp(parent=active)
    pusher = ad.PushInterp(parent=parent)
    x = pusher.box((parent.box(("x", "dx")), parent.box(("dy", 1.0))))
    ir = stage.trace(lambda x: x)("x")

    with core.using_interp(pusher):
        with pytest.raises(TypeError, match="Cannot join AStr"):
            next(ir.walk(x))
    assert core.active_interp.get() is active


@pytest.mark.parametrize("use_in_equation", [False, True], ids=["final-output", "equation-input"])
def test_pushforward_checker_rechecks_mutated_tangent(use_in_equation):
    def program(x):
        intercept.checkpoint("pause", key="pause")
        return intercept.checkpoint(x, key="next") if use_in_equation else x

    active = core.active_interp.get()
    pusher = ad.PushInterp(parent=active)
    x = pusher.box(("x", "dx"))
    ir = stage.trace(program)("x")
    gen = ir.walk(x)
    with core.using_interp(pusher):
        eqn, inputs = next(gen)
        x.tangent = 1.0
        output = eqn.bind(inputs, **eqn.params)

        with pytest.raises(TypeError, match="Cannot join AStr"):
            gen.send(output)
    assert core.active_interp.get() is active


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "t",
    [
        pytest.param(1.0, id="float"),
        pytest.param(core.Zero(numeric.AFloat()), id="wrong-zero"),
    ],
)
def test_pushforward_rejects_invalid_intermediate_tangent(executor, t):
    p_bad_tangent = core.Prim("bad_tangent")
    forward = lambda args: (args[0], t)
    extend.r_abstract.set(p_bad_tangent, lambda x: x)
    extend.r_push.set(p_bad_tangent, forward)
    extend.r_apush.set(p_bad_tangent, utils.asyncify(forward))
    ir = ad.pushforward(stage.trace(lambda x: control.stop_gradient(p_bad_tangent.bind(x)))("x"))

    with pytest.raises(TypeError, match="AStr"):
        executor(ir, ("x",), ("dx",))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_checks_concrete_primal(executor):
    class AValNonemptyStr(string.AStr):
        def join(self, other):
            match other:
                case core.Zero(aval=string.AStr() as aval):
                    return self.join(aval)
            if type(other) is str:
                return self if other else string.AStr()
            if type(other) is string.AStr:
                return other
            if isinstance(other, (str, string.AStr)):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    core.aval_types[AValNonemptyStr] = lambda aval: aval
    core.s_primal.set(AValNonemptyStr, lambda aval: aval)
    core.s_cotangent.set(AValNonemptyStr, lambda _: string.AStr())
    x = stage.Var(aval=AValNonemptyStr())
    ir = ad.pullback(stage.IR([], (x,), x))

    assert executor(ir, ("x",), "df") == ("x", ("df",))
    with pytest.raises(TypeError, match="Expected AValNonemptyStr"):
        executor(ir, ("",), "df")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, message",
    [
        pytest.param(["x", "y"], "Cannot join AStr", id="list"),
        pytest.param(1.0, "Cannot join AStr", id="float"),
        pytest.param(core.Zero(numeric.AFloat()), "Cannot join AStr", id="wrong-zero"),
    ],
)
def test_pullback_rejects_incompatible_cotangent(executor, c, message):
    ir = ad.pullback(stage.trace(lambda x: x)("x"))
    with pytest.raises(TypeError, match=message):
        executor(ir, ("x",), c)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_rejects_invalid_intermediate_cotangent(executor):
    p_bad_cotangent = core.Prim("bad_cotangent")
    forward = lambda x: (x, None)
    backward = lambda args: 1.0
    extend.r_abstract.set(p_bad_cotangent, lambda x: x)
    extend.r_pull_fwd.set(p_bad_cotangent, forward)
    extend.r_apull_fwd.set(p_bad_cotangent, utils.asyncify(forward))
    extend.r_pull_bwd.set(p_bad_cotangent, backward)
    extend.r_apull_bwd.set(p_bad_cotangent, utils.asyncify(backward))
    ir = ad.pullback(stage.trace(lambda x: p_bad_cotangent.bind(control.stop_gradient(x)))("x"))

    with pytest.raises(TypeError, match="Cannot join AStr"):
        executor(ir, ("x",), "df")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_rejects_invalid_accumulated_cotangent(executor):
    class Value: ...

    class Feedback: ...

    class AValValue(core.AVal):
        def __eq__(self, other) -> bool:
            return type(self) is type(other)

        def __hash__(self) -> int:
            return hash(type(self))

        def join(self, other):
            match other:
                case core.Zero(aval=AValValue() as aval):
                    return self.join(aval)
            if type(other) in (Value, type(self)):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    class AValFeedback(core.AVal):
        __slots__ = []

        def __eq__(self, other) -> bool:
            return type(self) is type(other)

        def __hash__(self) -> int:
            return hash(type(self))

        def zero(self):
            return Feedback()

        def add(self, left, right):
            return 1.0

        def join(self, other):
            match other:
                case core.Zero(aval=AValFeedback() as aval):
                    return self.join(aval)
            if type(other) in (Feedback, type(self)):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    extend.register_trace_type(Value, lambda _: AValValue())
    core.aval_types[AValValue] = lambda aval: aval
    core.aval_types[Feedback] = lambda _: AValFeedback()
    core.aval_types[AValFeedback] = lambda aval: aval
    core.s_primal.set(AValValue, lambda aval: aval)
    core.s_cotangent.set(AValValue, lambda _: AValFeedback())

    def program(x):
        y = control.stop_gradient(x)
        return y, y

    ir = ad.pullback(stage.trace(program)(Value()))
    with pytest.raises(TypeError, match="Cannot join .*AValFeedback"):
        executor(ir, (Value(),), (Feedback(), Feedback()))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, expected",
    [
        pytest.param(("a", "b"), "ab", id="accum"),
        pytest.param(("a", core.Zero(string.AStr())), "a", id="mixed-zero"),
        pytest.param(
            (core.Zero(string.AStr()), core.Zero(string.AStr())),
            core.Zero(string.AStr()),
            id="all-zero",
        ),
    ],
)
def test_pullback_accumulates_cotangents_and_zeros_unused_inputs(executor, c, expected):
    ir = ad.pullback(stage.trace(lambda x, y: (x, x))("x", "y"))
    assert executor(ir, ("x", "y"), c) == (
        ("x", "x"),
        (expected, core.Zero(string.AStr())),
    )


def test_pullback_rechecks_mutated_contribution():
    seed = core.Zero(string.AStr())
    p_mutate_cotangent = core.Prim("mutate_cotangent")

    def backward(args):
        seed.aval = numeric.AFloat()
        return args[1]

    extend.r_abstract.set(p_mutate_cotangent, lambda x: x)
    extend.r_pull_fwd.set(p_mutate_cotangent, lambda x: (x, None))
    extend.r_pull_bwd.set(p_mutate_cotangent, backward)

    def program(x):
        y = x + "!"
        return y, p_mutate_cotangent.bind(y)

    ir = ad.pullback(stage.trace(program)("x"))
    with pytest.raises(TypeError, match="Cannot join AStr"):
        ir.call(("x",), (seed, "df"))


def test_pullback_maps_cotangent_once():
    class AValValue(core.AVal):
        def __eq__(self, other) -> bool:
            return type(self) is type(other)

        def __hash__(self) -> int:
            return hash(type(self))

        def join(self, other):
            match other:
                case core.Zero(aval=AValValue() as aval):
                    return self.join(aval)
            if type(other) is type(self):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    class AValFeedback(AValValue): ...

    class AValHigherFeedback(AValValue): ...

    for cls in (AValValue, AValFeedback, AValHigherFeedback):
        core.aval_types[cls] = lambda a: a
    core.s_primal.set(AValValue, lambda aval: aval)
    core.s_cotangent.set(AValValue, lambda _: AValFeedback())
    core.s_cotangent.set(AValFeedback, lambda _: AValHigherFeedback())
    x, y = (stage.Var(aval=AValValue()) for _ in range(2))
    ir = stage.IR([], (x, y), x)
    parent = core.active_interp.get()
    no_stage = stage.no_stage_flag.get()
    value = core.Zero(AValValue())
    feedback = core.Zero(AValFeedback())
    assert ad.impl_pullback(((value, value), feedback), ir=ir) == (
        value,
        (feedback, feedback),
    )
    with pytest.raises(TypeError, match="Cannot join"):
        ad.impl_pullback(((value, value), core.Zero(AValHigherFeedback())), ir=ir)
    assert core.active_interp.get() is parent
    assert stage.no_stage_flag.get() is no_stage


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_preserves_original_ir(executor):
    ir = stage.trace(lambda x: x * x + 1.0)(2.0)
    pullback = ad.pullback(ir)
    for x in (2.0, 3.0):
        assert executor(pullback, (x,), 1.0) == (x * x + 1.0, (2.0 * x,))
        assert executor(ir, x) == x * x + 1.0


def test_pullback_checks_under_caller_interpreter():
    parent = core.active_interp.get()
    no_stage = stage.no_stage_flag.get()
    x, y = (stage.Var(aval=string.AStr()) for _ in range(2))
    ir = stage.IR([], (x, y), (x, x))

    with core.using_interp(parent):
        _, c = ad.impl_pullback((("x", "y"), ("a", "b")), ir=ir)
        assert c == ("ab", core.Zero(string.AStr()))
        with pytest.raises(TypeError, match="Cannot join AStr"):
            ad.impl_pullback((("x", "y"), ("a", 1.0)), ir=ir)
    assert core.active_interp.get() is parent
    assert stage.no_stage_flag.get() is no_stage


def test_pullback_preserves_parent_pushforward_values():
    parent = core.active_interp.get()
    pusher = ad.PushInterp(parent=parent)
    x, y = (stage.Var(aval=string.AStr()) for _ in range(2))
    ir = stage.IR([], (x, y), (x, x))
    zero = core.Zero(string.AStr())

    with core.using_interp(pusher):
        values = pusher.box((("a", "b"), ("da", "db")))
        _, c = ad.impl_pullback((("x", "y"), values), ir=ir)
        assert pusher.unbox(c) == (("ab", zero), ("dadb", zero))
    assert core.active_interp.get() is parent


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_batched_pullback_rejects_shared_list_cotangent(executor):
    ir = ad.pullback(stage.trace(lambda x, y: x + y)("x", "y"))
    ir = axis.batch(ir, in_axes=((False, True), False))
    with pytest.raises(TypeError, match="Cannot join AStr"):
        executor(ir, ("x", ["y1", "y2"]), ["o1", "o2"])
    assert executor(ir, ("x", ["y1", "y2"]), "o") == (
        ["xy1", "xy2"],
        (["o", "o"], ["o", "o"]),
    )


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, error",
    [
        pytest.param("o", "Cannot join ABatch", id="scalar"),
        pytest.param(["o", 1.0], "Cannot join AStr", id="mixed-elements"),
        pytest.param(core.Zero(axis.ABatch(numeric.AFloat())), "Cannot join AStr", id="wrong-zero"),
    ],
)
def test_pullback_of_batch_rejects_incompatible_cotangent(executor, c, error):
    ir = ad.pullback(axis.batch(stage.trace(lambda x: x)("x")))
    with pytest.raises(TypeError, match=error):
        executor(ir, (["x", "y"],), c)


@pytest.mark.parametrize(
    "box_type, kwargs",
    [
        pytest.param(ad.PushforwardBox, {"tangent": object()}, id="pushforward"),
        pytest.param(ad.PullbackFwdBox, {}, id="pullback-forward"),
        pytest.param(ad.PullbackBwdBox, {}, id="pullback-backward"),
    ],
)
@pytest.mark.parametrize(
    "value, aval",
    [
        pytest.param("x", string.AStr(), id="string"),
        pytest.param(1.5, numeric.AFloat(), id="float"),
    ],
)
def test_ad_box_avalof(box_type, kwargs, value, aval):
    box = box_type(object(), value, **kwargs)

    assert core.avalof(box) == aval


@pytest.mark.parametrize(
    "box_type, kwargs",
    [
        pytest.param(ad.PushforwardBox, {"tangent": object()}, id="pushforward"),
        pytest.param(ad.PullbackFwdBox, {}, id="pullback-forward"),
        pytest.param(ad.PullbackBwdBox, {}, id="pullback-backward"),
    ],
)
def test_ad_box_avalof_preserves_zero_metadata(box_type, kwargs):
    aval = AValBlob(3)
    box = box_type(object(), core.Zero(aval), **kwargs)

    assert core.avalof(box) is aval


def test_avalof_nested_ad_and_trace_boxes():
    aval = AValBlob(3)
    x = stage.TraceBox(owner=stage.TraceInterp(), var=stage.Var(aval=aval))
    x = ad.PullbackFwdBox(object(), x)
    x = ad.PushforwardBox(object(), x, object())
    x = ad.PullbackBwdBox(object(), x)

    assert core.avalof(x) is aval


@pytest.mark.parametrize(
    "transform",
    [ad.pushforward, ad.pullback],
    ids=["pushforward", "pullback"],
)
def test_int_input_is_not_differentiable(transform):
    with pytest.raises(TypeError):
        transform(stage.trace(lambda x: x)(1))


@pytest.mark.parametrize(
    "transform, space, change",
    [
        pytest.param(
            ad.pushforward,
            core.s_tangent,
            "replace hello",
            id="tangent",
        ),
        pytest.param(
            ad.pullback,
            core.s_cotangent,
            "be clearer",
            id="cotangent",
        ),
    ],
)
def test_wrapper_uses_derivative_space(transform, space, change):
    class Text:
        def __init__(self, value):
            self.value = value

    class Change:
        def __init__(self, value):
            self.value = value

    class AValText(core.AVal):
        def __eq__(self, other) -> bool:
            return type(self) is type(other)

        def __hash__(self) -> int:
            return hash(type(self))

        def join(self, other):
            match other:
                case core.Zero(aval=AValText() as aval):
                    return self.join(aval)
            if type(other) in (Text, type(self)):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    class AValChange(core.AVal):
        def __eq__(self, other) -> bool:
            return type(self) is type(other)

        def __hash__(self) -> int:
            return hash(type(self))

        def join(self, other):
            match other:
                case core.Zero(aval=AValChange() as aval):
                    return self.join(aval)
            if type(other) in (Change, type(self)):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    core.aval_types[Text] = lambda _: AValText()
    core.aval_types[AValText] = lambda aval: aval
    stage.trace_types.add(Text)
    core.aval_types[Change] = lambda _: AValChange()
    core.aval_types[AValChange] = lambda aval: aval
    stage.trace_types.add(Change)
    core.s_primal.set(AValText, lambda aval: aval)
    core.s_primal.set(AValChange, lambda aval: aval)
    space.set(AValText, lambda _: AValChange())
    space.set(AValChange, lambda aval: aval)
    aval = AValText()
    var = stage.Var(aval=aval)
    source = stage.IR([], (var,), (var,))
    ir = transform(source)
    for p, derivatives in (ir.in_tree, ir.out_tree):
        assert p[0].aval is aval
        assert isinstance(derivatives[0].aval, AValChange)

    text, delta = Text("hello"), Change(change)
    owner = object()
    assert isinstance(core.avalof(ad.PushforwardBox(owner, text, delta)), AValText)
    assert isinstance(core.avalof(ad.PullbackFwdBox(owner, text)), AValText)
    assert isinstance(core.avalof(ad.PullbackBwdBox(owner, delta)), AValChange)
    assert ir.call((text,), (delta,)) == ((text,), (delta,))
    assert isinstance(core.Zero(space.map(core.avalof(text))).aval, AValChange)
    nested_derivatives = transform(ir).in_tree[1]
    assert all(isinstance(v.aval, AValChange) for v in utils.tree.leaves(nested_derivatives))


class TestCotangentHelpers:
    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param(["hello"], "hello", id="single"),
            pytest.param(["a", "b", "c"], "abc", id="strings"),
            pytest.param([["a", "b"], ["c", "d"]], ["ac", "bd"], id="lists"),
            pytest.param(
                [
                    {"x": ["1", core.Zero(string.AStr())], "y": "a"},
                    {"x": ["2", "b"], "y": "c"},
                ],
                {"x": ["12", "b"], "y": "ac"},
                id="nested-with-zero",
            ),
            pytest.param(
                [core.Zero(string.AStr()), core.Zero(string.AStr())],
                core.Zero(string.AStr()),
                id="all-zero",
            ),
        ],
    )
    def test_cotangent_accumulation(self, values, expected):
        assert ad.cot_accum(values) == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "transform, args, expected",
        [
            pytest.param(lambda ir: ir, ("c", "d"), "cd", id="execution"),
            pytest.param(
                ad.pushforward,
                (("a", "b"), ("da", "db")),
                ("ab", "dadb"),
                id="pushforward",
            ),
            pytest.param(ad.pullback, (("a", "b"), "g"), ("ab", ("g", "g")), id="pullback"),
            pytest.param(axis.batch, (["a", "b"], ["c", "d"]), ["ac", "bd"], id="batch"),
        ],
    )
    def test_cot_accum_transforms(self, executor, transform, args, expected):
        source = stage.trace(lambda x, y: ad.cot_accum([x, y]))("a", "b")
        assert [eqn.prim for eqn in source.eqns] == [ad.p_cot_accum]
        ir = transform(source)
        actual = executor(ir, *args)
        assert actual == expected

    def test_cot_accum_all_zeros_must_match_aval(self):
        with pytest.raises(AssertionError):
            ad.cot_accum([
                core.Zero(string.AStr()),
                core.Zero(numeric.AInt()),
            ])

    def test_cot_accum_unsupported_type_raises(self):
        with pytest.raises(NotImplementedError):
            ad.cot_accum([True, False])
        ir = stage.trace(lambda x, y: ad.cot_accum([x, y]))(True, False)
        with pytest.raises(NotImplementedError):
            ir.call(True, False)

    def test_cot_accum_unregistered_leaf_raises(self):
        class Blob: ...

        with pytest.raises(TypeError, match="No aval rule registered"):
            ad.cot_accum([Blob(), Blob()])

    def test_cot_accum_uses_aval_method(self):
        class Text:
            def __init__(self, value):
                self.value = value

        class AValText(core.AVal):
            __slots__ = []

            def __eq__(self, other) -> bool:
                return type(self) is type(other)

            def __hash__(self) -> int:
                return hash(type(self))

            def zero(self):
                return Text("")

            def add(self, left, right):
                return Text("|".join(x.value for x in (left, right) if x.value))

            def join(self, other):
                match other:
                    case core.Zero(aval=AValText() as aval):
                        return self.join(aval)
                if type(other) in (Text, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        core.aval_types[Text] = lambda _: AValText()
        stage.trace_types.add(Text)
        result = ad.cot_accum([Text("a"), Text("b")])
        assert isinstance(result, Text)
        assert result.value == "a|b"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_pullback_accumulates_custom_cotangent_space(self, executor):
        class Text:
            def __init__(self, value):
                self.value = value

        class TextFeedback:
            def __init__(self, value):
                self.value = value

        class AValText(core.AVal):
            def __eq__(self, other) -> bool:
                return type(self) is type(other)

            def __hash__(self) -> int:
                return hash(type(self))

            def join(self, other):
                match other:
                    case core.Zero(aval=AValText() as aval):
                        return self.join(aval)
                if type(other) in (Text, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        class AValTextFeedback(core.AVal):
            __slots__ = []

            def __eq__(self, other) -> bool:
                return type(self) is type(other)

            def __hash__(self) -> int:
                return hash(type(self))

            def zero(self):
                return TextFeedback("")

            def add(self, left, right):
                return TextFeedback(" | ".join(x.value for x in (left, right) if x.value))

            def join(self, other):
                match other:
                    case core.Zero(aval=AValTextFeedback() as aval):
                        return self.join(aval)
                if type(other) in (TextFeedback, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        class AValDerivedFeedback(AValTextFeedback): ...

        core.aval_types[Text] = lambda _: AValText()
        core.aval_types[AValText] = lambda aval: aval
        stage.trace_types.add(Text)
        core.aval_types[TextFeedback] = lambda _: AValDerivedFeedback()
        core.aval_types[AValDerivedFeedback] = lambda aval: aval
        stage.trace_types.add(TextFeedback)
        core.s_primal.set(AValText, lambda aval: aval)
        core.s_cotangent.set(AValText, lambda _: AValDerivedFeedback())
        var = stage.Var(aval=AValText())
        ir = stage.IR([], (var,), (var, var))
        text = Text("hello")
        out_p, in_c = executor(
            ad.pullback(ir), (text,), (TextFeedback("left"), TextFeedback("right"))
        )
        assert out_p == (text, text)
        assert isinstance(in_c[0], TextFeedback)
        assert in_c[0].value == "left | right"
        zero = core.Zero(core.s_cotangent.map(core.avalof(text)))
        assert zero.aval.zero().value == ""


@pytest.mark.parametrize(
    "transform, literal, derivative_side",
    [
        pytest.param(ad.pushforward, "constant", "in_tree", id="pushforward"),
        pytest.param(ad.pullback, "constant_input", "out_tree", id="pullback"),
    ],
)
def test_literal_input_derivatives_are_zero(transform, literal, derivative_side):
    var = stage.Var(aval=string.AStr())
    ir = stage.IR([], (literal, var), (var,))
    zero, variable = getattr(transform(ir), derivative_side)[1]
    assert isinstance(zero, core.Zero)
    assert zero.aval == string.AStr()
    assert isinstance(variable, stage.Var)


@pytest.mark.parametrize(
    "transform, derivative_side",
    [
        pytest.param(ad.pushforward, "out_tree", id="pushforward"),
        pytest.param(ad.pullback, "in_tree", id="pullback"),
    ],
)
def test_literal_output_derivatives_are_zero(transform, derivative_side):
    ir = stage.trace(lambda x: (x, "constant_output"))("input")
    variable, zero = getattr(transform(ir), derivative_side)[1]
    assert isinstance(zero, core.Zero)
    assert zero.aval == string.AStr()
    assert isinstance(variable, stage.Var)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, feedback, expected",
    [
        pytest.param(
            ad.pushforward,
            (core.Zero(core.s_primal.map(core.avalof("Q"))), "dx"),
            "dx",
            id="pushforward",
        ),
        pytest.param(
            ad.pullback,
            "g",
            (core.Zero(core.s_primal.map(core.avalof("Q"))), "g"),
            id="pullback",
        ),
    ],
)
def test_static_input_literal_is_not_boxed(executor, transform, feedback, expected):
    ir = transform(stage.trace(string.concat, static=(True, False))("Q", "x"))
    args = (("Q", "x"), feedback)
    actual = executor(ir, *args)
    assert actual == ("Qx", expected)
    invalid = (("R", "x"), feedback)
    with pytest.raises(AssertionError, match="Static input mismatch"):
        executor(ir, *invalid)


def polynomial(x):
    return x * x + x / 2


def formatted_bang(x):
    y = string.format("Value: {x}", x=x)
    z = string.concat(y, "!")
    return z


def test_alternating_pushforward_pullback():
    ir = ad.pushforward(ad.pullback(ad.pushforward(ad.pullback(stage.trace(bracket_text)("x")))))
    p = ((("x",), "x"), (("x",), "x"))
    feedback = (("x", ("x",)), ("x", ("x",)))
    args = ((p, feedback), (p, feedback))
    expected = (
        ((("[x]", ("x",)), ("x", ("x",))), p),
        (feedback, p),
    )
    assert ir.call(*args) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "program, transform, trace_args, args, expected",
    [
        pytest.param(
            polynomial,
            ad.pushforward,
            (2.0,),
            ((2.0,), (1.0,)),
            (5.0, 4.5),
            id="numeric-pushforward",
        ),
        pytest.param(
            polynomial,
            ad.pullback,
            (2.0,),
            ((2.0,), 1.0),
            (5.0, (4.5,)),
            id="numeric-pullback",
        ),
        pytest.param(
            polynomial,
            lambda ir: ad.pushforward(ad.pullback(ir)),
            (2.0,),
            (((2.0,), 1.0), ((1.0,), 0.0)),
            ((5.0, (4.5,)), (4.5, (2.0,))),
            id="numeric-pushforward-pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pullback(ad.pullback(ad.pullback(ir))),
            ("x",),
            (((("a",), "g1"), ("g2_p", ("g2_c",))), (("g3_pp", ("g3_pc",)), (("g3_cp",), "g3_cc"))),
            (
                (("a!", ("g1",)), (("g2_p",), "g2_c")),
                ((("g3_pp",), "g3_pc"), ("g3_cp", ("g3_cc",))),
            ),
            id="triple-pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: axis.batch(ad.pushforward(ad.pullback(ir)), in_axes=(True, True)),
            ("x",),
            (((["a", "b"],), ["g1", "g2"]), ((["ta", "tb"],), ["tg1", "tg2"])),
            ((["a!", "b!"], (["g1", "g2"],)), (["ta", "tb"], (["tg1", "tg2"],))),
            id="batch-pushforward-pullback",
        ),
        pytest.param(
            bracket_text,
            lambda ir: ad.pushforward(axis.batch(ad.pullback(ir), in_axes=(True, True))),
            ("x",),
            (((["a", "b"],), ["g1", "g2"]), ((["ta", "tb"],), ["tg1", "tg2"])),
            ((["[a]", "[b]"], (["g1", "g2"],)), (["ta", "tb"], (["tg1", "tg2"],))),
            id="pushforward-batch-pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pullback(ad.pushforward(axis.batch(ir))),
            ("x",),
            (((["a", "b"],), (["ta", "tb"],)), (["g1", "g2"], ["tg1", "tg2"])),
            ((["a!", "b!"], ["ta", "tb"]), ((["g1", "g2"],), (["tg1", "tg2"],))),
            id="pullback-pushforward-batch",
        ),
        pytest.param(
            lambda a, b: string.format("{a}-{b}", a=a, b=b),
            lambda ir: ad.pushforward(axis.batch(ir)),
            ("a", "b"),
            ((["a1", "a2"], ["b1", "b2"]), (["ta1", "ta2"], ["tb1", "tb2"])),
            (["a1-b1", "a2-b2"], ["ta1tb1", "ta2tb2"]),
            id="pushforward-batch-two-inputs",
        ),
        pytest.param(
            string.concat,
            lambda ir: ad.pullback(axis.batch(axis.batch(ir))),
            ("a", "b"),
            (([["a1", "a2"], ["a3"]], [["b1", "b2"], ["b3"]]), [["g1", "g2"], ["g3"]]),
            ([["a1b1", "a2b2"], ["a3b3"]], ([["g1", "g2"], ["g3"]], [["g1", "g2"], ["g3"]])),
            id="pullback-double-batch-two-inputs",
        ),
        pytest.param(
            lambda x: string.concat(string.format("[{x}]", x=x), "!"),
            lambda ir: ad.pushforward(axis.batch(axis.batch(ir))),
            ("x",),
            (([["a", "b"], ["c"]],), ([["ta", "tb"], ["tc"]],)),
            ([["[a]!", "[b]!"], ["[c]!"]], [["ta", "tb"], ["tc"]]),
            id="pushforward-double-batch-chain",
        ),
        pytest.param(
            lambda x: string.concat(string.format("({x}", x=x), ")"),
            lambda ir: axis.batch(
                ad.pushforward(axis.batch(ad.pullback(ir), in_axes=(True, True))),
                in_axes=(True, True),
            ),
            ("x",),
            ((([["a", "b"]],), [["g1", "g2"]]), (([["ta", "tb"]],), [["tg1", "tg2"]])),
            (([["(a)", "(b)"]], ([["g1", "g2"]],)), ([["ta", "tb"]], ([["tg1", "tg2"]],))),
            id="batch-pushforward-batch-pullback",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: axis.batch(ad.pushforward(ir), in_axes=(True, True)),
            ("x",),
            ((["a", "b", "c"],), (["da", "db", "dc"],)),
            (["Value: a!", "Value: b!", "Value: c!"], ["da", "db", "dc"]),
            id="batch_of_pushforward",
        ),
        pytest.param(
            append_bang,
            lambda ir: axis.batch(ad.pushforward(ir), in_axes=(True, True)),
            ("x",),
            ((["a"],), (["da"],)),
            (["a!"], ["da"]),
            id="batch_of_pushforward_single_element",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: axis.batch(ad.pullback(ir), in_axes=(True, True)),
            ("x",),
            ((["a", "b", "c"],), ["g1", "g2", "g3"]),
            (["Value: a!", "Value: b!", "Value: c!"], (["g1", "g2", "g3"],)),
            id="batch_of_pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: axis.batch(ad.pullback(ir), in_axes=(True, True)),
            ("x",),
            ((["a"],), ["g"]),
            (["a!"], (["g"],)),
            id="batch_of_pullback_single_element",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: ad.pushforward(axis.batch(ir)),
            ("x",),
            ((["a", "b"],), (["da", "db"],)),
            (["Value: a!", "Value: b!"], ["da", "db"]),
            id="pushforward_of_batch",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pushforward(axis.batch(ir)),
            ("x",),
            ((["a"],), (["da"],)),
            (["a!"], ["da"]),
            id="pushforward_of_batch_single_element",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: ad.pullback(axis.batch(ir)),
            ("x",),
            ((["a", "b"],), ["g1", "g2"]),
            (["Value: a!", "Value: b!"], (["g1", "g2"],)),
            id="pullback_of_batch",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pullback(axis.batch(ir)),
            ("x",),
            ((["a"],), ["g"]),
            (["a!"], (["g"],)),
            id="pullback_of_batch_single_element",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pushforward(ad.pushforward(axis.batch(ir))),
            ("x",),
            (((["a", "b"],), (["t1a", "t1b"],)), ((["t2a", "t2b"],), (["t2t1a", "t2t1b"],))),
            ((["a!", "b!"], ["t1a", "t1b"]), (["t2a", "t2b"], ["t2t1a", "t2t1b"])),
            id="pushforward_of_pushforward_of_batch",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pushforward(ad.pushforward(ad.pushforward(ir))),
            ("x",),
            (
                ((("a",), ("t1",)), (("t2p",), ("t2t",))),
                ((("t3pp",), ("t3pt",)), (("t3tp",), ("t3tt",))),
            ),
            ((("a!", "t1"), ("t2p", "t2t")), (("t3pp", "t3pt"), ("t3tp", "t3tt"))),
            id="triple_pushforward",
        ),
        pytest.param(
            bracket_text,
            lambda ir: ad.pushforward(ad.pushforward(ad.pushforward(ad.pushforward(ir)))),
            ("x",),
            (
                (((("a",), ("b",)), (("c",), ("d",))), ((("e",), ("f",)), (("g",), ("h",)))),
                (((("i",), ("j",)), (("k",), ("l",))), ((("m",), ("n",)), (("o",), ("p",)))),
            ),
            (
                ((("[a]", "b"), ("c", "d")), (("e", "f"), ("g", "h"))),
                ((("i", "j"), ("k", "l")), (("m", "n"), ("o", "p"))),
            ),
            id="quadruple_pushforward",
        ),
        pytest.param(
            angle_text,
            lambda ir: axis.batch(
                axis.batch(ad.pushforward(ir), in_axes=(True, True)),
                in_axes=(True, True),
            ),
            ("x",),
            (([["a", "b"], ["c"]],), ([["ta", "tb"], ["tc"]],)),
            ([["<a>", "<b>"], ["<c>"]], [["ta", "tb"], ["tc"]]),
            id="batch_batch_pushforward",
        ),
        pytest.param(
            append_bang,
            lambda ir: axis.batch(
                ad.pushforward(axis.batch(ad.pushforward(ir), in_axes=(True, True))),
                in_axes=(True, True),
            ),
            ("x",),
            (
                *(
                    (([["a", "b"], ["c", "d"]],), ([["ta", "tb"], ["tc", "td"]],)),
                    (([["qa", "qb"], ["qc", "qd"]],), ([["qta", "qtb"], ["qtc", "qtd"]],)),
                ),
            ),
            (
                ([["a!", "b!"], ["c!", "d!"]], [["ta", "tb"], ["tc", "td"]]),
                ([["qa", "qb"], ["qc", "qd"]], [["qta", "qtb"], ["qtc", "qtd"]]),
            ),
            id="batch_pf_batch_pf",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pushforward(axis.batch(axis.batch(axis.batch(ir)))),
            ("x",),
            (([[["a"]]],), ([[["t"]]],)),
            ([[["a!"]]], [[["t"]]]),
            id="single_element_deep",
        ),
        pytest.param(
            append_bang,
            lambda ir: axis.batch(ad.pushforward(ir), in_axes=(True, False)),
            ("x",),
            ((["a", "b", "c"],), ("t",)),
            (["a!", "b!", "c!"], ["t", "t", "t"]),
            id="pushforward_batch_primals_batched_tangents_unbatched",
        ),
        pytest.param(
            append_bang,
            lambda ir: axis.batch(ad.pullback(ir), in_axes=(True, False)),
            ("x",),
            ((["a", "b", "c"],), "g"),
            (["a!", "b!", "c!"], (["g", "g", "g"],)),
            id="pullback_batch_primals_batched_cotangents_unbatched",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pullback(ad.pushforward(ir)),
            ("x",),
            ((("hello",), ("world",)), ("grad_p", "grad_t")),
            (("hello!", "world"), (("grad_p",), ("grad_t",))),
            id="pullback-pushforward",
        ),
        pytest.param(
            bracket_text,
            lambda ir: ad.pullback(ad.pushforward(ir)),
            ("x",),
            ((("a",), ("ta",)), ("gp", "gt")),
            (("[a]", "ta"), (("gp",), ("gt",))),
            id="pullback-pushforward-format",
        ),
        pytest.param(
            string.concat,
            lambda ir: ad.pullback(ad.pushforward(ir)),
            ("a", "b"),
            ((("hello", " world"), ("t1", "t2")), ("gp", "gt")),
            (("hello world", "t1t2"), (("gp", "gp"), ("gt", "gt"))),
            id="pullback-pushforward-two-inputs",
        ),
        pytest.param(
            append_bang,
            lambda ir: ad.pullback(ad.pullback(ir)),
            ("x",),
            ((("hello",), "grad"), ("gg_p", ("gg_c",))),
            (("hello!", ("grad",)), (("gg_p",), "gg_c")),
            id="pullback-pullback",
        ),
        pytest.param(
            angle_text,
            lambda ir: ad.pullback(ad.pullback(ir)),
            ("x",),
            ((("a",), "g"), ("cp", ("cc",))),
            (("<a>", ("g",)), (("cp",), "cc")),
            id="pullback-pullback-format",
        ),
        pytest.param(
            string.concat,
            lambda ir: ad.pullback(ad.pullback(ir)),
            ("a", "b"),
            ((("hello", " world"), "grad"), ("cp", ("cc_x", "cc_y"))),
            (("hello world", ("grad", "grad")), (("cp", "cp"), "cc_xcc_y")),
            id="pullback-pullback-two-inputs",
        ),
    ],
)
def test_composition(executor, program, transform, trace_args, args, expected):
    ir = transform(stage.trace(program)(*trace_args))
    result = executor(ir, *args)
    assert result == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pushforward_preserves_batched_zero_structure(executor):
    double = axis.batch(stage.trace(lambda x: x + x)(1.0))
    add = axis.batch(stage.trace(lambda x, y: x + y)(1.0, 1.0))
    x, y = [1.0, 2.0], [3.0, 4.0]
    ir = stage.trace(lambda x, y: add.call(double.call(x), y))(x, y)
    zero = core.Zero(numeric.AFloat())

    assert executor(ad.pushforward(ir), (x, y), ([zero, zero], [1.0, 1.0])) == (
        [5.0, 8.0],
        [1.0, 1.0],
    )


@pytest.mark.parametrize(
    "executor, forward",
    [
        (execute, ad.impl_pushforward_call),
        (
            aexecute,
            lambda args, *, ir: asyncio.run(ad.aimpl_pushforward_call(args, ir=ir)),
        ),
    ],
    ids=["sync", "async"],
)
def test_traced_pushforward_preserves_batched_zero_structure(executor, forward):
    double = axis.batch(stage.trace(lambda x: x + x)(1.0))
    add = axis.batch(stage.trace(lambda x, y: x + y)(1.0, 1.0))
    x, y = [1.0, 2.0], [3.0, 4.0]
    source = stage.trace(lambda x, y: add.call(double.call(x), y))(x, y)
    zero = core.Zero(numeric.AFloat())
    ir = stage.trace(lambda x, y: forward(((x, y), ([zero, zero], [1.0, 1.0])), ir=source))(x, y)

    assert executor(ir, x, y) == ([5.0, 8.0], [1.0, 1.0])


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_unused_batched_output(executor):
    double = axis.batch(stage.trace(lambda x: x + x)(1.0))

    def program(x, y):
        double.call(x)
        return y

    x, y = [1.0, 2.0], [3.0, 4.0]
    ir = ad.pullback(stage.trace(program)(x, y))
    zero = core.Zero(numeric.AFloat())

    assert executor(ir, (x, y), [1.0, 1.0]) == (y, ([zero, zero], [1.0, 1.0]))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("transform", [ad.pushforward, ad.pullback], ids=["pf", "pb"])
def test_nonidentity_primal_space(executor, transform):

    class AValValue(core.AVal):
        def __eq__(self, other) -> bool:
            return type(self) is type(other)

        def __hash__(self) -> int:
            return hash(type(self))

        def join(self, other):
            match other:
                case core.Zero(aval=AValValue() as aval):
                    return self.join(aval)
            if type(other) is type(self):
                return self
            raise TypeError(f"Cannot join {self!r} and {other!r}")

    core.aval_types[AValValue] = lambda aval: aval
    core.s_primal.set(AValValue, lambda _: numeric.AFloat())
    core.s_tangent.set(AValValue, lambda _: string.AStr())
    core.s_cotangent.set(AValValue, lambda _: string.AStr())
    p_mapped_primal = core.Prim("mapped_primal")

    def forward(args):
        p, t = args
        return p * 2.0, t if isinstance(t, core.Zero) else t + t

    extend.r_impl.set(p_mapped_primal, lambda x: x * 2.0)
    extend.r_aimpl.set(p_mapped_primal, utils.asyncify(lambda x: x * 2.0))
    extend.r_abstract.set(p_mapped_primal, lambda _: AValValue())
    extend.r_push.set(p_mapped_primal, forward)
    extend.r_apush.set(p_mapped_primal, utils.asyncify(forward))
    extend.r_pull_fwd.set(p_mapped_primal, lambda x: (x * 2.0, None))
    extend.r_apull_fwd.set(p_mapped_primal, utils.asyncify(lambda x: (x * 2.0, None)))
    extend.r_pull_bwd.set(p_mapped_primal, lambda args: args[1] + args[1])
    extend.r_apull_bwd.set(p_mapped_primal, utils.asyncify(lambda args: args[1] + args[1]))
    x, y, z = (stage.Var(aval=AValValue()) for _ in range(3))
    source = stage.IR(
        [
            stage.Eqn(check.p_typecheck, x, y, dict(aval=AValValue())),
            stage.Eqn(p_mapped_primal, y, z, {}),
        ],
        (x,),
        (z, "literal"),
    )
    ir = transform(source)
    zero = core.Zero(string.AStr())
    assert ir.in_tree[0][0].aval == numeric.AFloat()
    assert ir.out_tree[0][0].aval == numeric.AFloat()
    if transform is ad.pushforward:
        args = ((2.0,), ("dx",))
        expected = ((4.0, "literal"), ("dxdx", zero))
        assert executor(ir, (2.0,), (zero,)) == ((4.0, "literal"), (zero, zero))
    else:
        args = ((2.0,), ("dy", zero))
        expected = ((4.0, "literal"), ("dydy",))
    assert executor(ir, *args) == expected
    static = (False, (False, True)) if transform is ad.pullback else False
    traced = stage.trace(lambda p, d: executor(ir, p, d), static=static)(*args)
    assert traced.out_tree[0][0].aval == numeric.AFloat()
    assert executor(traced, *args) == expected
    with pytest.raises(TypeError, match="Cannot join AFloat"):
        executor(ir, ("bad",), args[1])


executors = pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])


def zero_like(value):
    return core.Zero(core.avalof(value))


@executors
def test_hvp_with_symbolic_zero_cotangent_tangent(executor):
    pf_pb = ad.pushforward(ad.pullback(stage.trace(cube)(2.0)))
    assert executor(pf_pb, ((2.0,), 1.0), ((1.0,), zero_like(2.0))) == (
        (8.0, (12.0,)),
        (12.0, (12.0,)),
    )


@executors
def test_hvp_through_cot_accum(executor):
    def program(x):
        return ad.cot_accum([x * x, x * x * x])

    pf_pb = ad.pushforward(ad.pullback(stage.trace(program)(2.0)))
    assert executor(pf_pb, ((2.0,), 1.0), ((1.0,), zero_like(2.0))) == (
        (12.0, (16.0,)),
        (16.0, (14.0,)),
    )


@executors
def test_pushforward_bwd_with_residual_from_plain_fwd(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))
    _, residual = executor(fwd, 2.0)
    tangent = (zero_like(residual), 1.0)
    assert executor(ad.pushforward(bwd), (residual, 1.0), tangent) == ((12.0,), (12.0,))


@executors
def test_pushforward_bwd_after_zero_tangent_fwd(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))
    (_, residual), (_, residual_t) = executor(ad.pushforward(fwd), (2.0,), (zero_like(2.0),))
    assert executor(ad.pushforward(bwd), (residual, 1.0), (residual_t, 1.0)) == ((12.0,), (12.0,))


@executors
def test_pullback_through_traced_halves_with_unused_primal(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))

    def program(x, g):
        _, residual = fwd.call(x)
        (dx,) = bwd.call(residual, g)
        return dx

    ir = stage.trace(program)(2.0, 1.0)
    assert executor(ad.pullback(ir), (2.0, 1.0), 1.0) == (12.0, (12.0, 12.0))


@executors
def test_pullback_accumulates_shared_residual(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))

    def program(x, g1, g2):
        _, residual = fwd.call(x)
        (a,) = bwd.call(residual, g1)
        (b,) = bwd.call(residual, g2)
        return a, b

    ir = stage.trace(program)(2.0, 1.0, 1.0)
    expected = ((12.0, 12.0), (24.0, 12.0, 12.0))
    assert executor(ad.pullback(ir), (2.0, 1.0, 1.0), (1.0, 1.0)) == expected


@executors
def test_pushforward_through_split_pullback(executor):
    fwd, bwd = ad.fwd_bwd(ad.pullback(stage.trace(cube)(2.0)))
    (out, residual), (out_t, residual_t) = executor(
        ad.pushforward(fwd), ((2.0,), 1.0), ((1.0,), 0.0)
    )
    assert (out, out_t) == ((8.0, (12.0,)), (12.0, (12.0,)))
    in_c = executor(ad.pushforward(bwd), (residual, (1.0, (1.0,))), (residual_t, (0.0, (0.0,))))
    assert in_c == (((24.0,), 12.0), ((18.0,), 12.0))


@executors
def test_pushforward_pullback_fanout(executor):
    ir = stage.trace(
        lambda x: order.p_fanout.bind(
            [(x,), (x,)], irs=[stage.trace(cube)(2.0), stage.trace(square)(2.0)]
        )
    )(2.0)
    out = executor(ad.pushforward(ad.pullback(ir)), ((2.0,), [1.0, 0.0]), ((1.0,), [0.0, 0.0]))
    assert out == (([8.0, 4.0], (12.0,)), ([12.0, 4.0], (12.0,)))


@executors
def test_pushforward_halves_of_while_loop(executor):
    # 2 -> 4 -> 16, so f(x) = x**4: f' = 32, f'' = 48. the predicate must stay a
    # plain bool under pushforward, otherwise the loop runs to max_iters.
    cond = stage.trace(lambda x: x < 10.0)(2.0)
    body = stage.trace(square)(2.0)
    ir = stage.trace(lambda x: control.while_loop(cond, body, x, max_iters=5))(2.0)
    fwd, bwd = ad.fwd_bwd(ir)
    (out, residual), (out_t, residual_t) = executor(ad.pushforward(fwd), (2.0,), (1.0,))
    assert (out, out_t) == (16.0, 32.0)
    assert executor(ad.pushforward(bwd), (residual, 1.0), (residual_t, 0.0)) == ((32.0,), (48.0,))


@executors
def test_pushforward_pullback_switch_constant_key(executor):
    ir = stage.trace(
        lambda x: control.switch(
            "a", {"a": stage.trace(cube)(2.0), "b": stage.trace(square)(2.0)}, x
        )
    )(2.0)
    out = executor(ad.pushforward(ad.pullback(ir)), ((2.0,), 1.0), ((1.0,), 0.0))
    assert out == ((8.0, (12.0,)), (12.0, (12.0,)))


@executors
def test_pushforward_pullback_switch_input_key(executor):
    branches = {"a": stage.trace(cube)(2.0), "b": stage.trace(square)(2.0)}
    ir = stage.trace(lambda k, x: control.switch(k, branches, x))("a", 2.0)
    key_t = core.Zero(core.s_tangent.map(core.avalof("a")))
    (_, (_, dx)), (_, (_, dx_t)) = executor(
        ad.pushforward(ad.pullback(ir)), (("a", 2.0), 1.0), ((key_t, 1.0), 0.0)
    )
    assert (dx, dx_t) == (12.0, 12.0)


@executors
def test_trace_with_concrete_residual_keeps_one_variable(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))
    _, residual = fwd.call(2.0)
    traced = stage.trace(lambda r, g: bwd.call(r, g))(residual, 1.0)
    residual_var, _ = traced.in_tree
    assert stage.is_var(residual_var) and residual_var.aval == core.avalof(residual)
    assert executor(traced, residual, 1.0) == (12.0,)


@executors
def test_fwd_bwd_residual_opens_after_other_transforms(executor):
    ir = stage.trace(cube)(2.0)
    fwd, _ = ad.fwd_bwd(ir)
    _, residual = executor(fwd, 2.0)
    for scale in range(256):
        ad.fwd_bwd(stage.trace(lambda x, scale=float(scale): x * scale)(1.0))
    _, bwd = ad.fwd_bwd(ir)
    assert executor(bwd, residual, 1.0) == (12.0,)


@executors
def test_fwd_bwd_halves_stay_ssa_when_spliced(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))
    out_p, _ = fwd.out_tree
    _, out_c = bwd.in_tree
    ir = stage.IR([*fwd.eqns, *bwd.eqns], (fwd.in_tree, out_c), (out_p, bwd.out_tree))
    out = executor(ad.pullback(ir), ((2.0,), 1.0), (1.0, (1.0,)))
    assert out == ((8.0, (12.0,)), ((24.0,), 12.0))


@executors
def test_fwd_bwd_survives_sched(executor):
    fwd, bwd = ad.fwd_bwd(stage.trace(cube)(2.0))
    _, residual = executor(order.sched(fwd), 2.0)
    assert executor(order.sched(bwd), residual, 1.0) == (12.0,)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("nest", ["fanout", "switch", "batch", "while_loop", "custom"])
def test_pullback_does_not_recompute_nested_forward(executor, nest):
    drafts = iter(f"draft{i}" for i in range(10))
    seen = []
    p_generate = core.Prim("generate")

    def impl(x, /):
        return x + ":" + next(drafts)

    def pull_fwd(x, /):
        out = p_generate.bind(x)
        return out, out

    def pull_bwd(in_tree, /):
        output, c = in_tree
        seen.append(output)
        return c

    core.r_impl.set(p_generate, impl)
    core.r_aimpl.set(p_generate, utils.asyncify(impl))
    core.r_abstract.set(p_generate, lambda _: string.AStr())
    core.r_pull_fwd.set(p_generate, pull_fwd)
    core.r_apull_fwd.set(p_generate, utils.asyncify(pull_fwd))
    core.r_pull_bwd.set(p_generate, pull_bwd)
    core.r_apull_bwd.set(p_generate, utils.asyncify(pull_bwd))
    core.r_batch.set(p_generate, lambda t: ([impl(x) for x in t[2]], True))
    core.r_abatch.set(p_generate, utils.asyncify(lambda t: ([impl(x) for x in t[2]], True)))

    inner = stage.trace(p_generate.bind)("seed")
    if nest == "fanout":
        ir = order.sched(stage.trace(lambda x: (p_generate.bind(x), p_generate.bind(x)))("seed"))
        args = (("x",), ("c", "c"))
    elif nest == "switch":
        ir = stage.trace(lambda k, x: control.switch(k, {True: inner, False: inner}, x))(
            True, "seed"
        )
        args = ((True, "x"), "c")
    elif nest == "batch":
        ir = axis.batch(inner)
        args = ((["x", "y"],), ["c", "c"])
    elif nest == "while_loop":
        cond = stage.trace(lambda s: string.match(s, "x"))("seed")
        ir = stage.trace(lambda x: control.while_loop(cond, inner, x, max_iters=3))("seed")
        args = (("x",), "c")
    else:
        ir = stage.trace(boundary.custom(lambda x: p_generate.bind(x)))("seed")
        args = (("x",), "c")
    out, _ = executor(ad.pullback(ir), *args)

    assert sorted(seen) == sorted(utils.tree.leaves(out))
