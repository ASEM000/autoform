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
import autoform.src.control as control
import autoform.src.core as core
import autoform.src.extend as extend
import autoform.src.intercept as intercept
import autoform.src.numeric as numeric
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import BlobAVal, aexecute, angle_text, append_bang, bracket_text, execute


def test_pushforward_checker_rejects_nested_tangent_without_equations():
    active = core.active_interpreter.get()
    parent = ad.PushforwardInterpreter(parent=active)
    pusher = ad.PushforwardInterpreter(parent=parent)
    x = pusher.box((parent.box(("x", "dx")), parent.box(("dy", 1.0))))
    ir = stage.trace(lambda x: x)("x")

    with core.using_interpreter(pusher):
        with pytest.raises(TypeError, match="Expected StrAVal"):
            next(ir.walk(x))
    assert core.active_interpreter.get() is active


@pytest.mark.parametrize("use_in_equation", [False, True], ids=["final-output", "equation-input"])
def test_pushforward_checker_rechecks_mutated_tangent(use_in_equation):
    def program(x):
        intercept.checkpoint("pause", key="pause")
        return intercept.checkpoint(x, key="next") if use_in_equation else x

    active = core.active_interpreter.get()
    pusher = ad.PushforwardInterpreter(parent=active)
    x = pusher.box(("x", "dx"))
    ir = stage.trace(program)("x")
    gen = ir.walk(x)
    with core.using_interpreter(pusher):
        eqn, inputs = next(gen)
        x.tangent = 1.0
        output = eqn.bind(inputs, **eqn.params)

        with pytest.raises(TypeError, match="Expected StrAVal"):
            gen.send(output)
    assert core.active_interpreter.get() is active


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "t",
    [
        pytest.param(1.0, id="float"),
        pytest.param(core.Zero(numeric.FloatAVal()), id="wrong-zero"),
    ],
)
def test_pushforward_rejects_invalid_intermediate_tangent(executor, t):
    bad = core.Prim("bad_tangent")
    forward = lambda args: (args[0], t)
    extend.register_abstract(bad, lambda x: x)
    extend.register_pushforward(bad, forward)
    extend.register_apushforward(bad, utils.asyncify(forward))
    ir = ad.pushforward(stage.trace(lambda x: control.stop_gradient(bad.bind(x)))("x"))

    with pytest.raises(TypeError, match="Expected StrAVal"):
        executor(ir, ("x",), ("dx",))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_checks_concrete_primal(executor):
    class NonemptyStrAVal(string.StrAVal):
        def check(self, value):
            if not isinstance(value, str) or not value:
                raise TypeError("Expected a nonempty string")

    core.aval_types[NonemptyStrAVal] = lambda aval: aval
    core.primal_s.set(NonemptyStrAVal, lambda aval: aval)
    core.cotangent_s.set(NonemptyStrAVal, lambda _: string.StrAVal())
    x = stage.Var(aval=NonemptyStrAVal())
    ir = ad.pullback(stage.IR([], (x,), x))

    assert executor(ir, ("x",), "df") == ("x", ("df",))
    with pytest.raises(TypeError, match="Expected a nonempty string"):
        executor(ir, ("",), "df")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, message",
    [
        pytest.param(["x", "y"], "No aval rule registered", id="list"),
        pytest.param(1.0, "Expected StrAVal", id="float"),
        pytest.param(core.Zero(numeric.FloatAVal()), "Expected StrAVal", id="wrong-zero"),
    ],
)
def test_pullback_rejects_incompatible_cotangent(executor, c, message):
    ir = ad.pullback(stage.trace(lambda x: x)("x"))
    with pytest.raises(TypeError, match=message):
        executor(ir, ("x",), c)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_rejects_invalid_intermediate_cotangent(executor):
    bad = core.Prim("bad_cotangent")
    forward = lambda x: (x, None)
    backward = lambda args: 1.0
    extend.register_abstract(bad, lambda x: x)
    extend.register_pullback_fwd(bad, forward)
    extend.register_apullback_fwd(bad, utils.asyncify(forward))
    extend.register_pullback_bwd(bad, backward)
    extend.register_apullback_bwd(bad, utils.asyncify(backward))
    ir = ad.pullback(stage.trace(lambda x: bad.bind(control.stop_gradient(x)))("x"))

    with pytest.raises(TypeError, match="Expected StrAVal"):
        executor(ir, ("x",), "df")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_rejects_invalid_accumulated_cotangent(executor):
    class Value: ...

    class Feedback: ...

    class ValueAVal(core.AVal): ...

    class FeedbackAVal(core.AVal):
        def accum(self, c):
            return 1.0

    extend.register_trace_type(Value, lambda _: ValueAVal())
    core.aval_types[ValueAVal] = lambda aval: aval
    core.aval_types[Feedback] = lambda _: FeedbackAVal()
    core.aval_types[FeedbackAVal] = lambda aval: aval
    core.primal_s.set(ValueAVal, lambda aval: aval)
    core.cotangent_s.set(ValueAVal, lambda _: FeedbackAVal())

    def program(x):
        y = control.stop_gradient(x)
        return y, y

    ir = ad.pullback(stage.trace(program)(Value()))
    with pytest.raises(TypeError, match="Expected"):
        executor(ir, (Value(),), (Feedback(), Feedback()))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, expected",
    [
        pytest.param(("a", "b"), "ab", id="accum"),
        pytest.param(("a", core.Zero(string.StrAVal())), "a", id="mixed-zero"),
        pytest.param(
            (core.Zero(string.StrAVal()), core.Zero(string.StrAVal())),
            core.Zero(string.StrAVal()),
            id="all-zero",
        ),
    ],
)
def test_pullback_accumulates_cotangents_and_zeros_unused_inputs(executor, c, expected):
    ir = ad.pullback(stage.trace(lambda x, y: (x, x))("x", "y"))
    assert executor(ir, ("x", "y"), c) == (
        ("x", "x"),
        (expected, core.Zero(string.StrAVal())),
    )


def test_pullback_rechecks_mutated_contribution():
    seed = core.Zero(string.StrAVal())
    mutate = core.Prim("mutate_cotangent")

    def backward(args):
        seed.aval = numeric.FloatAVal()
        return args[1]

    extend.register_abstract(mutate, lambda x: x)
    extend.register_pullback_fwd(mutate, lambda x: (x, None))
    extend.register_pullback_bwd(mutate, backward)

    def program(x):
        y = x + "!"
        return y, mutate.bind(y)

    ir = ad.pullback(stage.trace(program)("x"))
    with pytest.raises(TypeError, match="Expected StrAVal"):
        ir.call(("x",), (seed, "df"))


def test_pullback_maps_cotangent_once():
    class ValueAVal(core.AVal): ...

    class FeedbackAVal(core.AVal): ...

    class HigherFeedbackAVal(core.AVal): ...

    for cls in (ValueAVal, FeedbackAVal, HigherFeedbackAVal):
        core.aval_types[cls] = lambda a: a
    core.primal_s.set(ValueAVal, lambda aval: aval)
    core.cotangent_s.set(ValueAVal, lambda _: FeedbackAVal())
    core.cotangent_s.set(FeedbackAVal, lambda _: HigherFeedbackAVal())
    x, y = (stage.Var(aval=ValueAVal()) for _ in range(2))
    ir = stage.IR([], (x, y), x)
    parent = core.active_interpreter.get()
    no_stage = stage.no_stage_flag.get()
    value = core.Zero(ValueAVal())
    feedback = core.Zero(FeedbackAVal())
    assert ad.impl_pullback_call(((value, value), feedback), ir=ir) == (
        value,
        (feedback, feedback),
    )
    with pytest.raises(TypeError, match="Expected"):
        ad.impl_pullback_call(((value, value), core.Zero(HigherFeedbackAVal())), ir=ir)
    assert core.active_interpreter.get() is parent
    assert stage.no_stage_flag.get() is no_stage


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_preserves_original_ir(executor):
    ir = stage.trace(lambda x: x * x + 1.0)(2.0)
    pullback = ad.pullback(ir)
    for x in (2.0, 3.0):
        assert executor(pullback, (x,), 1.0) == (x * x + 1.0, (2.0 * x,))
        assert executor(ir, x) == x * x + 1.0


def test_pullback_checks_under_caller_interpreter():
    parent = core.active_interpreter.get()
    no_stage = stage.no_stage_flag.get()
    x, y = (stage.Var(aval=string.StrAVal()) for _ in range(2))
    ir = stage.IR([], (x, y), (x, x))

    with core.using_interpreter(parent):
        _, c = ad.impl_pullback_call((("x", "y"), ("a", "b")), ir=ir)
        assert c == ("ab", core.Zero(string.StrAVal()))
        with pytest.raises(TypeError, match="Expected StrAVal"):
            ad.impl_pullback_call((("x", "y"), ("a", 1.0)), ir=ir)
    assert core.active_interpreter.get() is parent
    assert stage.no_stage_flag.get() is no_stage


def test_pullback_preserves_parent_pushforward_values():
    parent = core.active_interpreter.get()
    pusher = ad.PushforwardInterpreter(parent=parent)
    x, y = (stage.Var(aval=string.StrAVal()) for _ in range(2))
    ir = stage.IR([], (x, y), (x, x))
    zero = core.Zero(string.StrAVal())

    with core.using_interpreter(pusher):
        values = pusher.box((("a", "b"), ("da", "db")))
        _, c = ad.impl_pullback_call((("x", "y"), values), ir=ir)
        assert pusher.unbox(c) == (("ab", zero), ("dadb", zero))
    assert core.active_interpreter.get() is parent


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_batched_pullback_rejects_shared_list_cotangent(executor):
    ir = ad.pullback(stage.trace(lambda x, y: x + y)("x", "y"))
    ir = axis.batch(ir, in_axes=((False, True), False))
    with pytest.raises(TypeError, match="No aval rule registered"):
        executor(ir, ("x", ["y1", "y2"]), ["o1", "o2"])
    assert executor(ir, ("x", ["y1", "y2"]), "o") == (
        ["xy1", "xy2"],
        (["o", "o"], ["o", "o"]),
    )


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, error",
    [
        pytest.param("o", "BatchAVal", id="scalar"),
        pytest.param(["o", 1.0], "StrAVal", id="mixed-elements"),
        pytest.param(core.Zero(axis.BatchAVal(numeric.FloatAVal())), "StrAVal", id="wrong-zero"),
    ],
)
def test_pullback_of_batch_rejects_incompatible_cotangent(executor, c, error):
    ir = ad.pullback(axis.batch(stage.trace(lambda x: x)("x")))
    with pytest.raises(TypeError, match=f"Expected {error}"):
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
        pytest.param("x", string.StrAVal(), id="string"),
        pytest.param(1.5, numeric.FloatAVal(), id="float"),
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
    aval = BlobAVal(3)
    box = box_type(object(), core.Zero(aval), **kwargs)

    assert core.avalof(box) is aval


def test_avalof_nested_ad_and_trace_boxes():
    aval = BlobAVal(3)
    x = stage.TraceBox(owner=stage.TraceInterpreter(), var=stage.Var(aval=aval))
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
            core.tangent_s,
            "replace hello",
            id="tangent",
        ),
        pytest.param(
            ad.pullback,
            core.cotangent_s,
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

    class TextAVal(core.AVal): ...

    class ChangeAVal(core.AVal): ...

    core.aval_types[Text] = lambda _: TextAVal()
    core.aval_types[TextAVal] = lambda aval: aval
    stage.trace_types.add(Text)
    core.aval_types[Change] = lambda _: ChangeAVal()
    core.aval_types[ChangeAVal] = lambda aval: aval
    stage.trace_types.add(Change)
    core.primal_s.set(TextAVal, lambda aval: aval)
    core.primal_s.set(ChangeAVal, lambda aval: aval)
    space.set(TextAVal, lambda _: ChangeAVal())
    space.set(ChangeAVal, lambda aval: aval)
    aval = TextAVal()
    var = stage.Var(aval=aval)
    source = stage.IR([], (var,), (var,))
    ir = transform(source)
    for p, derivatives in (ir.in_tree, ir.out_tree):
        assert p[0].aval is aval
        assert isinstance(derivatives[0].aval, ChangeAVal)

    text, delta = Text("hello"), Change(change)
    owner = object()
    assert isinstance(core.avalof(ad.PushforwardBox(owner, text, delta)), TextAVal)
    assert isinstance(core.avalof(ad.PullbackFwdBox(owner, text)), TextAVal)
    assert isinstance(core.avalof(ad.PullbackBwdBox(owner, delta)), ChangeAVal)
    assert ir.call((text,), (delta,)) == ((text,), (delta,))
    assert isinstance(core.Zero(space.map(core.avalof(text))).aval, ChangeAVal)
    nested_derivatives = transform(ir).in_tree[1]
    assert all(isinstance(v.aval, ChangeAVal) for v in utils.tree.leaves(nested_derivatives))


class TestCotangentHelpers:
    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param(["hello"], "hello", id="single"),
            pytest.param(["a", "b", "c"], "abc", id="strings"),
            pytest.param([["a", "b"], ["c", "d"]], ["ac", "bd"], id="lists"),
            pytest.param(
                [
                    {"x": ["1", core.Zero(string.StrAVal())], "y": "a"},
                    {"x": ["2", "b"], "y": "c"},
                ],
                {"x": ["12", "b"], "y": "ac"},
                id="nested-with-zero",
            ),
            pytest.param(
                [core.Zero(string.StrAVal()), core.Zero(string.StrAVal())],
                core.Zero(string.StrAVal()),
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
        assert [eqn.prim for eqn in source.eqns] == [ad.cot_accum_p]
        ir = transform(source)
        actual = executor(ir, *args)
        assert actual == expected

    def test_cot_accum_all_zeros_must_match_aval(self):
        with pytest.raises(AssertionError):
            ad.cot_accum([
                core.Zero(string.StrAVal()),
                core.Zero(numeric.IntAVal()),
            ])

    def test_cot_accum_unsupported_type_raises(self):
        with pytest.raises(
            AssertionError,
            match=r"No accumulation defined for BoolAVal\(\)",
        ):
            ad.cot_accum([True, False])
        ir = stage.trace(lambda x, y: ad.cot_accum([x, y]))(True, False)
        with pytest.raises(AssertionError, match="No accumulation defined"):
            ir.call(True, False)

    def test_cot_accum_unregistered_leaf_raises(self):
        class Blob: ...

        with pytest.raises(TypeError, match="No aval rule registered"):
            ad.cot_accum([Blob(), Blob()])

    def test_cot_accum_uses_aval_method(self):
        class Text:
            def __init__(self, value):
                self.value = value

        class TextAVal(core.AVal):
            def accum(self, c):
                return Text("|".join(c.value for c in c))

        core.aval_types[Text] = lambda _: TextAVal()
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

        class TextAVal(core.AVal): ...

        class TextFeedbackAVal(core.AVal):
            def zero(self):
                return TextFeedback("")

            def accum(self, c):
                return TextFeedback(" | ".join(c.value for c in c))

        class DerivedFeedbackAVal(TextFeedbackAVal): ...

        core.aval_types[Text] = lambda _: TextAVal()
        core.aval_types[TextAVal] = lambda aval: aval
        stage.trace_types.add(Text)
        core.aval_types[TextFeedback] = lambda _: DerivedFeedbackAVal()
        core.aval_types[DerivedFeedbackAVal] = lambda aval: aval
        stage.trace_types.add(TextFeedback)
        core.primal_s.set(TextAVal, lambda aval: aval)
        core.cotangent_s.set(TextAVal, lambda _: DerivedFeedbackAVal())
        var = stage.Var(aval=TextAVal())
        ir = stage.IR([], (var,), (var, var))
        text = Text("hello")
        out_p, in_c = executor(
            ad.pullback(ir), (text,), (TextFeedback("left"), TextFeedback("right"))
        )
        assert out_p == (text, text)
        assert isinstance(in_c[0], TextFeedback)
        assert in_c[0].value == "left | right"
        zero = core.Zero(core.cotangent_s.map(core.avalof(text)))
        assert core.materialize_zeros(zero).value == ""


@pytest.mark.parametrize(
    "transform, literal, derivative_side",
    [
        pytest.param(ad.pushforward, "constant", "in_tree", id="pushforward"),
        pytest.param(ad.pullback, "constant_input", "out_tree", id="pullback"),
    ],
)
def test_literal_input_derivatives_are_zero(transform, literal, derivative_side):
    var = stage.Var(aval=string.StrAVal())
    ir = stage.IR([], (literal, var), (var,))
    zero, variable = getattr(transform(ir), derivative_side)[1]
    assert isinstance(zero, core.Zero)
    assert zero.aval == string.StrAVal()
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
    assert zero.aval == string.StrAVal()
    assert isinstance(variable, stage.Var)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, feedback, expected",
    [
        pytest.param(
            ad.pushforward,
            (core.Zero(core.primal_s.map(core.avalof("Q"))), "dx"),
            "dx",
            id="pushforward",
        ),
        pytest.param(
            ad.pullback,
            "g",
            (core.Zero(core.primal_s.map(core.avalof("Q"))), "g"),
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
    zero = core.Zero(numeric.FloatAVal())

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
    zero = core.Zero(numeric.FloatAVal())
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
    zero = core.Zero(numeric.FloatAVal())

    assert executor(ir, (x, y), [1.0, 1.0]) == (y, ([zero, zero], [1.0, 1.0]))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("transform", [ad.pushforward, ad.pullback], ids=["pf", "pb"])
def test_nonidentity_primal_space(executor, transform):

    class ValueAVal(core.AVal): ...

    core.aval_types[ValueAVal] = lambda aval: aval
    core.primal_s.set(ValueAVal, lambda _: numeric.FloatAVal())
    core.tangent_s.set(ValueAVal, lambda _: string.StrAVal())
    core.cotangent_s.set(ValueAVal, lambda _: string.StrAVal())
    prim = core.Prim("mapped_primal")

    def forward(args):
        p, t = args
        return p * 2.0, t if isinstance(t, core.Zero) else t + t

    extend.register_impl(prim, lambda x: x * 2.0)
    extend.register_aimpl(prim, utils.asyncify(lambda x: x * 2.0))
    extend.register_abstract(prim, lambda _: ValueAVal())
    extend.register_pushforward(prim, forward)
    extend.register_apushforward(prim, utils.asyncify(forward))
    extend.register_pullback_fwd(prim, lambda x: (x * 2.0, None))
    extend.register_apullback_fwd(prim, utils.asyncify(lambda x: (x * 2.0, None)))
    extend.register_pullback_bwd(prim, lambda args: args[1] + args[1])
    extend.register_apullback_bwd(prim, utils.asyncify(lambda args: args[1] + args[1]))
    x, y, z = (stage.Var(aval=ValueAVal()) for _ in range(3))
    source = stage.IR(
        [
            stage.Eqn(check.typecheck_p, x, y, dict(aval=ValueAVal())),
            stage.Eqn(prim, y, z, {}),
        ],
        (x,),
        (z, "literal"),
    )
    ir = transform(source)
    zero = core.Zero(string.StrAVal())
    assert ir.in_tree[0][0].aval == numeric.FloatAVal()
    assert ir.out_tree[0][0].aval == numeric.FloatAVal()
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
    assert traced.out_tree[0][0].aval == numeric.FloatAVal()
    assert executor(traced, *args) == expected
    with pytest.raises(TypeError, match="Expected FloatAVal"):
        executor(ir, ("bad",), args[1])
