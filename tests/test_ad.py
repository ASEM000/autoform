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

import autoform as af
from tests import BlobAVal, aexecute, angle_text, append_bang, bracket_text, execute


def test_pushforward_checker_rejects_nested_tangent_without_equations():
    active = af.core.active_interpreter.get()
    parent = af.ad.PushforwardInterpreter(parent=active)
    pusher = af.ad.PushforwardInterpreter(parent=parent)
    x = pusher.box((parent.box(("x", "dx")), parent.box(("dy", 1.0))))
    ir = af.trace(lambda x: x)("x")

    with af.core.using_interpreter(pusher):
        with pytest.raises(TypeError, match="Expected StrAVal"):
            next(ir.walk(x))
    assert af.core.active_interpreter.get() is active


@pytest.mark.parametrize("use_in_equation", [False, True], ids=["final-output", "equation-input"])
def test_pushforward_checker_rechecks_mutated_tangent(use_in_equation):
    def program(x):
        af.checkpoint("pause", key="pause")
        return af.checkpoint(x, key="next") if use_in_equation else x

    active = af.core.active_interpreter.get()
    pusher = af.ad.PushforwardInterpreter(parent=active)
    x = pusher.box(("x", "dx"))
    ir = af.trace(program)("x")
    gen = ir.walk(x)
    with af.core.using_interpreter(pusher):
        eqn, inputs = next(gen)
        x.tangent = 1.0
        output = eqn.bind(inputs, **eqn.params)

        with pytest.raises(TypeError, match="Expected StrAVal"):
            gen.send(output)
    assert af.core.active_interpreter.get() is active


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "t",
    [
        pytest.param(1.0, id="float"),
        pytest.param(af.core.Zero(af.numeric.FloatAVal()), id="wrong-zero"),
    ],
)
def test_pushforward_rejects_invalid_intermediate_tangent(executor, t):
    bad = af.extend.Prim("bad_tangent")
    forward = lambda args: (args[0], t)
    af.extend.register_abstract(bad, lambda x: x)
    af.extend.register_pushforward(bad, forward)
    af.extend.register_apushforward(bad, af.utils.asyncify(forward))
    ir = af.pushforward(af.trace(lambda x: af.stop_gradient(bad.bind(x)))("x"))

    with pytest.raises(TypeError, match="Expected StrAVal"):
        executor(ir, ("x",), ("dx",))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_checks_concrete_primal(executor):
    class NonemptyStrAVal(af.string.StrAVal):
        def check(self, value):
            if not isinstance(value, str) or not value:
                raise TypeError("Expected a nonempty string")

    af.core.aval_types[NonemptyStrAVal] = lambda aval: aval
    af.core.primal_s.set(NonemptyStrAVal, lambda aval: aval)
    af.core.cotangent_s.set(NonemptyStrAVal, lambda _: af.string.StrAVal())
    x = af.stage.Var(aval=NonemptyStrAVal())
    ir = af.pullback(af.stage.IR([], (x,), x))

    assert executor(ir, ("x",), "df") == ("x", ("df",))
    with pytest.raises(TypeError, match="Expected a nonempty string"):
        executor(ir, ("",), "df")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, message",
    [
        pytest.param(["x", "y"], "No aval rule registered", id="list"),
        pytest.param(1.0, "Expected StrAVal", id="float"),
        pytest.param(af.core.Zero(af.numeric.FloatAVal()), "Expected StrAVal", id="wrong-zero"),
    ],
)
def test_pullback_rejects_incompatible_cotangent(executor, c, message):
    ir = af.pullback(af.trace(lambda x: x)("x"))
    with pytest.raises(TypeError, match=message):
        executor(ir, ("x",), c)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_rejects_invalid_intermediate_cotangent(executor):
    bad = af.extend.Prim("bad_cotangent")
    forward = lambda x: (x, None)
    backward = lambda args: 1.0
    af.extend.register_abstract(bad, lambda x: x)
    af.extend.register_pullback_fwd(bad, forward)
    af.extend.register_apullback_fwd(bad, af.utils.asyncify(forward))
    af.extend.register_pullback_bwd(bad, backward)
    af.extend.register_apullback_bwd(bad, af.utils.asyncify(backward))
    ir = af.pullback(af.trace(lambda x: bad.bind(af.stop_gradient(x)))("x"))

    with pytest.raises(TypeError, match="Expected StrAVal"):
        executor(ir, ("x",), "df")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_rejects_invalid_accumulated_cotangent(executor):
    class Value: ...

    class Feedback: ...

    class ValueAVal(af.core.AVal): ...

    class FeedbackAVal(af.core.AVal):
        def accum(self, c):
            return 1.0

    af.extend.register_trace_type(Value, lambda _: ValueAVal())
    af.core.aval_types[ValueAVal] = lambda aval: aval
    af.core.aval_types[Feedback] = lambda _: FeedbackAVal()
    af.core.aval_types[FeedbackAVal] = lambda aval: aval
    af.core.primal_s.set(ValueAVal, lambda aval: aval)
    af.core.cotangent_s.set(ValueAVal, lambda _: FeedbackAVal())

    def program(x):
        y = af.stop_gradient(x)
        return y, y

    ir = af.pullback(af.trace(program)(Value()))
    with pytest.raises(TypeError, match="Expected"):
        executor(ir, (Value(),), (Feedback(), Feedback()))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "c, expected",
    [
        pytest.param(("a", "b"), "ab", id="accum"),
        pytest.param(("a", af.core.Zero(af.string.StrAVal())), "a", id="mixed-zero"),
        pytest.param(
            (af.core.Zero(af.string.StrAVal()), af.core.Zero(af.string.StrAVal())),
            af.core.Zero(af.string.StrAVal()),
            id="all-zero",
        ),
    ],
)
def test_pullback_accumulates_cotangents_and_zeros_unused_inputs(executor, c, expected):
    ir = af.pullback(af.trace(lambda x, y: (x, x))("x", "y"))
    assert executor(ir, ("x", "y"), c) == (
        ("x", "x"),
        (expected, af.core.Zero(af.string.StrAVal())),
    )


def test_pullback_rechecks_mutated_contribution():
    seed = af.core.Zero(af.string.StrAVal())
    mutate = af.extend.Prim("mutate_cotangent")

    def backward(args):
        seed.aval = af.numeric.FloatAVal()
        return args[1]

    af.extend.register_abstract(mutate, lambda x: x)
    af.extend.register_pullback_fwd(mutate, lambda x: (x, None))
    af.extend.register_pullback_bwd(mutate, backward)

    def program(x):
        y = x + "!"
        return y, mutate.bind(y)

    ir = af.pullback(af.trace(program)("x"))
    with pytest.raises(TypeError, match="Expected StrAVal"):
        ir.call(("x",), (seed, "df"))


def test_pullback_maps_cotangent_once():
    class ValueAVal(af.core.AVal): ...

    class FeedbackAVal(af.core.AVal): ...

    class HigherFeedbackAVal(af.core.AVal): ...

    for cls in (ValueAVal, FeedbackAVal, HigherFeedbackAVal):
        af.core.aval_types[cls] = lambda a: a
    af.core.primal_s.set(ValueAVal, lambda aval: aval)
    af.core.cotangent_s.set(ValueAVal, lambda _: FeedbackAVal())
    af.core.cotangent_s.set(FeedbackAVal, lambda _: HigherFeedbackAVal())
    x, y = (af.stage.Var(aval=ValueAVal()) for _ in range(2))
    ir = af.stage.IR([], (x, y), x)
    parent = af.core.active_interpreter.get()
    no_stage = af.stage.no_stage_flag.get()
    value = af.core.Zero(ValueAVal())
    feedback = af.core.Zero(FeedbackAVal())
    assert af.ad.impl_pullback_call(((value, value), feedback), ir=ir) == (
        value,
        (feedback, feedback),
    )
    with pytest.raises(TypeError, match="Expected"):
        af.ad.impl_pullback_call(((value, value), af.core.Zero(HigherFeedbackAVal())), ir=ir)
    assert af.core.active_interpreter.get() is parent
    assert af.stage.no_stage_flag.get() is no_stage


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_preserves_original_ir(executor):
    ir = af.trace(lambda x: x * x + 1.0)(2.0)
    pullback = af.pullback(ir)
    for x in (2.0, 3.0):
        assert executor(pullback, (x,), 1.0) == (x * x + 1.0, (2.0 * x,))
        assert executor(ir, x) == x * x + 1.0


def test_pullback_checks_under_caller_interpreter():
    parent = af.core.active_interpreter.get()
    no_stage = af.stage.no_stage_flag.get()
    x, y = (af.stage.Var(aval=af.string.StrAVal()) for _ in range(2))
    ir = af.stage.IR([], (x, y), (x, x))

    with af.core.using_interpreter(parent):
        _, c = af.ad.impl_pullback_call((("x", "y"), ("a", "b")), ir=ir)
        assert c == ("ab", af.core.Zero(af.string.StrAVal()))
        with pytest.raises(TypeError, match="Expected StrAVal"):
            af.ad.impl_pullback_call((("x", "y"), ("a", 1.0)), ir=ir)
    assert af.core.active_interpreter.get() is parent
    assert af.stage.no_stage_flag.get() is no_stage


def test_pullback_preserves_parent_pushforward_values():
    parent = af.core.active_interpreter.get()
    pusher = af.ad.PushforwardInterpreter(parent=parent)
    x, y = (af.stage.Var(aval=af.string.StrAVal()) for _ in range(2))
    ir = af.stage.IR([], (x, y), (x, x))
    zero = af.core.Zero(af.string.StrAVal())

    with af.core.using_interpreter(pusher):
        values = pusher.box((("a", "b"), ("da", "db")))
        _, c = af.ad.impl_pullback_call((("x", "y"), values), ir=ir)
        assert pusher.unbox(c) == (("ab", zero), ("dadb", zero))
    assert af.core.active_interpreter.get() is parent


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_batched_pullback_rejects_shared_list_cotangent(executor):
    ir = af.pullback(af.trace(lambda x, y: x + y)("x", "y"))
    ir = af.batch(ir, in_axes=((False, True), False))
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
        pytest.param(
            af.core.Zero(af.axis.BatchAVal(af.numeric.FloatAVal())), "StrAVal", id="wrong-zero"
        ),
    ],
)
def test_pullback_of_batch_rejects_incompatible_cotangent(executor, c, error):
    ir = af.pullback(af.batch(af.trace(lambda x: x)("x")))
    with pytest.raises(TypeError, match=f"Expected {error}"):
        executor(ir, (["x", "y"],), c)


@pytest.mark.parametrize(
    "box_type, kwargs",
    [
        pytest.param(af.ad.PushforwardBox, {"tangent": object()}, id="pushforward"),
        pytest.param(af.ad.PullbackFwdBox, {}, id="pullback-forward"),
        pytest.param(af.ad.PullbackBwdBox, {}, id="pullback-backward"),
    ],
)
@pytest.mark.parametrize(
    "value, aval",
    [
        pytest.param("x", af.string.StrAVal(), id="string"),
        pytest.param(1.5, af.numeric.FloatAVal(), id="float"),
    ],
)
def test_ad_box_avalof(box_type, kwargs, value, aval):
    box = box_type(object(), value, **kwargs)

    assert af.core.avalof(box) == aval


@pytest.mark.parametrize(
    "box_type, kwargs",
    [
        pytest.param(af.ad.PushforwardBox, {"tangent": object()}, id="pushforward"),
        pytest.param(af.ad.PullbackFwdBox, {}, id="pullback-forward"),
        pytest.param(af.ad.PullbackBwdBox, {}, id="pullback-backward"),
    ],
)
def test_ad_box_avalof_preserves_zero_metadata(box_type, kwargs):
    aval = BlobAVal(3)
    box = box_type(object(), af.core.Zero(aval), **kwargs)

    assert af.core.avalof(box) is aval


def test_avalof_nested_ad_and_trace_boxes():
    aval = BlobAVal(3)
    x = af.stage.TraceBox(owner=af.stage.TraceInterpreter(), var=af.stage.Var(aval=aval))
    x = af.ad.PullbackFwdBox(object(), x)
    x = af.ad.PushforwardBox(object(), x, object())
    x = af.ad.PullbackBwdBox(object(), x)

    assert af.core.avalof(x) is aval


@pytest.mark.parametrize(
    "transform",
    [af.pushforward, af.pullback],
    ids=["pushforward", "pullback"],
)
def test_int_input_is_not_differentiable(transform):
    with pytest.raises(TypeError):
        transform(af.trace(lambda x: x)(1))


@pytest.mark.parametrize(
    "transform, space, change",
    [
        pytest.param(
            af.pushforward,
            af.core.tangent_s,
            "replace hello",
            id="tangent",
        ),
        pytest.param(
            af.pullback,
            af.core.cotangent_s,
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

    class TextAVal(af.core.AVal): ...

    class ChangeAVal(af.core.AVal): ...

    af.core.aval_types[Text] = lambda _: TextAVal()
    af.core.aval_types[TextAVal] = lambda aval: aval
    af.stage.trace_types.add(Text)
    af.core.aval_types[Change] = lambda _: ChangeAVal()
    af.core.aval_types[ChangeAVal] = lambda aval: aval
    af.stage.trace_types.add(Change)
    af.core.primal_s.set(TextAVal, lambda aval: aval)
    af.core.primal_s.set(ChangeAVal, lambda aval: aval)
    space.set(TextAVal, lambda _: ChangeAVal())
    space.set(ChangeAVal, lambda aval: aval)
    aval = TextAVal()
    var = af.stage.Var(aval=aval)
    source = af.stage.IR([], (var,), (var,))
    ir = transform(source)
    for p, derivatives in (ir.in_tree, ir.out_tree):
        assert p[0].aval is aval
        assert isinstance(derivatives[0].aval, ChangeAVal)

    text, delta = Text("hello"), Change(change)
    owner = object()
    assert isinstance(af.core.avalof(af.ad.PushforwardBox(owner, text, delta)), TextAVal)
    assert isinstance(af.core.avalof(af.ad.PullbackFwdBox(owner, text)), TextAVal)
    assert isinstance(af.core.avalof(af.ad.PullbackBwdBox(owner, delta)), ChangeAVal)
    assert ir.call((text,), (delta,)) == ((text,), (delta,))
    assert isinstance(af.core.Zero(space.map(af.core.avalof(text))).aval, ChangeAVal)
    nested_derivatives = transform(ir).in_tree[1]
    assert all(isinstance(v.aval, ChangeAVal) for v in af.utils.tree.leaves(nested_derivatives))


class TestCotangentHelpers:
    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param(["hello"], "hello", id="single"),
            pytest.param(["a", "b", "c"], "abc", id="strings"),
            pytest.param([["a", "b"], ["c", "d"]], ["ac", "bd"], id="lists"),
            pytest.param(
                [
                    {"x": ["1", af.core.Zero(af.string.StrAVal())], "y": "a"},
                    {"x": ["2", "b"], "y": "c"},
                ],
                {"x": ["12", "b"], "y": "ac"},
                id="nested-with-zero",
            ),
            pytest.param(
                [af.core.Zero(af.string.StrAVal()), af.core.Zero(af.string.StrAVal())],
                af.core.Zero(af.string.StrAVal()),
                id="all-zero",
            ),
        ],
    )
    def test_cotangent_accumulation(self, values, expected):
        assert af.ad.cot_accum(values) == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "transform, args, expected",
        [
            pytest.param(lambda ir: ir, ("c", "d"), "cd", id="execution"),
            pytest.param(
                af.pushforward,
                (("a", "b"), ("da", "db")),
                ("ab", "dadb"),
                id="pushforward",
            ),
            pytest.param(af.pullback, (("a", "b"), "g"), ("ab", ("g", "g")), id="pullback"),
            pytest.param(af.batch, (["a", "b"], ["c", "d"]), ["ac", "bd"], id="batch"),
        ],
    )
    def test_cot_accum_transforms(self, executor, transform, args, expected):
        source = af.trace(lambda x, y: af.ad.cot_accum([x, y]))("a", "b")
        assert [eqn.prim for eqn in source.eqns] == [af.ad.cot_accum_p]
        ir = transform(source)
        actual = executor(ir, *args)
        assert actual == expected

    def test_cot_accum_all_zeros_must_match_aval(self):
        with pytest.raises(AssertionError):
            af.ad.cot_accum([
                af.core.Zero(af.string.StrAVal()),
                af.core.Zero(af.numeric.IntAVal()),
            ])

    def test_cot_accum_unsupported_type_raises(self):
        with pytest.raises(
            AssertionError,
            match=r"No accumulation defined for BoolAVal\(\)",
        ):
            af.ad.cot_accum([True, False])
        ir = af.trace(lambda x, y: af.ad.cot_accum([x, y]))(True, False)
        with pytest.raises(AssertionError, match="No accumulation defined"):
            ir.call(True, False)

    def test_cot_accum_unregistered_leaf_raises(self):
        class Blob: ...

        with pytest.raises(TypeError, match="No aval rule registered"):
            af.ad.cot_accum([Blob(), Blob()])

    def test_cot_accum_uses_aval_method(self):
        class Text:
            def __init__(self, value):
                self.value = value

        class TextAVal(af.core.AVal):
            def accum(self, c):
                return Text("|".join(c.value for c in c))

        af.core.aval_types[Text] = lambda _: TextAVal()
        af.stage.trace_types.add(Text)
        result = af.ad.cot_accum([Text("a"), Text("b")])
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

        class TextAVal(af.core.AVal): ...

        class TextFeedbackAVal(af.core.AVal):
            def zero(self):
                return TextFeedback("")

            def accum(self, c):
                return TextFeedback(" | ".join(c.value for c in c))

        class DerivedFeedbackAVal(TextFeedbackAVal): ...

        af.core.aval_types[Text] = lambda _: TextAVal()
        af.core.aval_types[TextAVal] = lambda aval: aval
        af.stage.trace_types.add(Text)
        af.core.aval_types[TextFeedback] = lambda _: DerivedFeedbackAVal()
        af.core.aval_types[DerivedFeedbackAVal] = lambda aval: aval
        af.stage.trace_types.add(TextFeedback)
        af.core.primal_s.set(TextAVal, lambda aval: aval)
        af.core.cotangent_s.set(TextAVal, lambda _: DerivedFeedbackAVal())
        var = af.stage.Var(aval=TextAVal())
        ir = af.stage.IR([], (var,), (var, var))
        text = Text("hello")
        out_p, in_c = executor(
            af.pullback(ir), (text,), (TextFeedback("left"), TextFeedback("right"))
        )
        assert out_p == (text, text)
        assert isinstance(in_c[0], TextFeedback)
        assert in_c[0].value == "left | right"
        zero = af.core.Zero(af.core.cotangent_s.map(af.core.avalof(text)))
        assert af.core.materialize_zeros(zero).value == ""


@pytest.mark.parametrize(
    "transform, literal, derivative_side",
    [
        pytest.param(af.pushforward, "constant", "in_tree", id="pushforward"),
        pytest.param(af.pullback, "constant_input", "out_tree", id="pullback"),
    ],
)
def test_literal_input_derivatives_are_zero(transform, literal, derivative_side):
    var = af.stage.Var(aval=af.string.StrAVal())
    ir = af.stage.IR([], (literal, var), (var,))
    zero, variable = getattr(transform(ir), derivative_side)[1]
    assert isinstance(zero, af.core.Zero)
    assert zero.aval == af.string.StrAVal()
    assert isinstance(variable, af.stage.Var)


@pytest.mark.parametrize(
    "transform, derivative_side",
    [
        pytest.param(af.pushforward, "out_tree", id="pushforward"),
        pytest.param(af.pullback, "in_tree", id="pullback"),
    ],
)
def test_literal_output_derivatives_are_zero(transform, derivative_side):
    ir = af.trace(lambda x: (x, "constant_output"))("input")
    variable, zero = getattr(transform(ir), derivative_side)[1]
    assert isinstance(zero, af.core.Zero)
    assert zero.aval == af.string.StrAVal()
    assert isinstance(variable, af.stage.Var)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, feedback, expected",
    [
        pytest.param(
            af.pushforward,
            (af.core.Zero(af.core.primal_s.map(af.core.avalof("Q"))), "dx"),
            "dx",
            id="pushforward",
        ),
        pytest.param(
            af.pullback,
            "g",
            (af.core.Zero(af.core.primal_s.map(af.core.avalof("Q"))), "g"),
            id="pullback",
        ),
    ],
)
def test_static_input_literal_is_not_boxed(executor, transform, feedback, expected):
    ir = transform(af.trace(af.string.concat, static=(True, False))("Q", "x"))
    args = (("Q", "x"), feedback)
    actual = executor(ir, *args)
    assert actual == ("Qx", expected)
    invalid = (("R", "x"), feedback)
    with pytest.raises(AssertionError, match="Static input mismatch"):
        executor(ir, *invalid)


def polynomial(x):
    return x * x + x / 2


def formatted_bang(x):
    y = af.string.format("Value: {x}", x=x)
    z = af.string.concat(y, "!")
    return z


def test_alternating_pushforward_pullback():
    ir = af.pushforward(af.pullback(af.pushforward(af.pullback(af.trace(bracket_text)("x")))))
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
            af.pushforward,
            (2.0,),
            ((2.0,), (1.0,)),
            (5.0, 4.5),
            id="numeric-pushforward",
        ),
        pytest.param(
            polynomial,
            af.pullback,
            (2.0,),
            ((2.0,), 1.0),
            (5.0, (4.5,)),
            id="numeric-pullback",
        ),
        pytest.param(
            polynomial,
            lambda ir: af.pushforward(af.pullback(ir)),
            (2.0,),
            (((2.0,), 1.0), ((1.0,), 0.0)),
            ((5.0, (4.5,)), (4.5, (2.0,))),
            id="numeric-pushforward-pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pullback(af.pullback(af.pullback(ir))),
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
            lambda ir: af.batch(af.pushforward(af.pullback(ir)), in_axes=(True, True)),
            ("x",),
            (((["a", "b"],), ["g1", "g2"]), ((["ta", "tb"],), ["tg1", "tg2"])),
            ((["a!", "b!"], (["g1", "g2"],)), (["ta", "tb"], (["tg1", "tg2"],))),
            id="batch-pushforward-pullback",
        ),
        pytest.param(
            bracket_text,
            lambda ir: af.pushforward(af.batch(af.pullback(ir), in_axes=(True, True))),
            ("x",),
            (((["a", "b"],), ["g1", "g2"]), ((["ta", "tb"],), ["tg1", "tg2"])),
            ((["[a]", "[b]"], (["g1", "g2"],)), (["ta", "tb"], (["tg1", "tg2"],))),
            id="pushforward-batch-pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pullback(af.pushforward(af.batch(ir))),
            ("x",),
            (((["a", "b"],), (["ta", "tb"],)), (["g1", "g2"], ["tg1", "tg2"])),
            ((["a!", "b!"], ["ta", "tb"]), ((["g1", "g2"],), (["tg1", "tg2"],))),
            id="pullback-pushforward-batch",
        ),
        pytest.param(
            lambda a, b: af.string.format("{a}-{b}", a=a, b=b),
            lambda ir: af.pushforward(af.batch(ir)),
            ("a", "b"),
            ((["a1", "a2"], ["b1", "b2"]), (["ta1", "ta2"], ["tb1", "tb2"])),
            (["a1-b1", "a2-b2"], ["ta1tb1", "ta2tb2"]),
            id="pushforward-batch-two-inputs",
        ),
        pytest.param(
            af.string.concat,
            lambda ir: af.pullback(af.batch(af.batch(ir))),
            ("a", "b"),
            (([["a1", "a2"], ["a3"]], [["b1", "b2"], ["b3"]]), [["g1", "g2"], ["g3"]]),
            ([["a1b1", "a2b2"], ["a3b3"]], ([["g1", "g2"], ["g3"]], [["g1", "g2"], ["g3"]])),
            id="pullback-double-batch-two-inputs",
        ),
        pytest.param(
            lambda x: af.string.concat(af.string.format("[{x}]", x=x), "!"),
            lambda ir: af.pushforward(af.batch(af.batch(ir))),
            ("x",),
            (([["a", "b"], ["c"]],), ([["ta", "tb"], ["tc"]],)),
            ([["[a]!", "[b]!"], ["[c]!"]], [["ta", "tb"], ["tc"]]),
            id="pushforward-double-batch-chain",
        ),
        pytest.param(
            lambda x: af.string.concat(af.string.format("({x}", x=x), ")"),
            lambda ir: af.batch(
                af.pushforward(af.batch(af.pullback(ir), in_axes=(True, True))),
                in_axes=(True, True),
            ),
            ("x",),
            ((([["a", "b"]],), [["g1", "g2"]]), (([["ta", "tb"]],), [["tg1", "tg2"]])),
            (([["(a)", "(b)"]], ([["g1", "g2"]],)), ([["ta", "tb"]], ([["tg1", "tg2"]],))),
            id="batch-pushforward-batch-pullback",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: af.batch(af.pushforward(ir), in_axes=(True, True)),
            ("x",),
            ((["a", "b", "c"],), (["da", "db", "dc"],)),
            (["Value: a!", "Value: b!", "Value: c!"], ["da", "db", "dc"]),
            id="batch_of_pushforward",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.batch(af.pushforward(ir), in_axes=(True, True)),
            ("x",),
            ((["a"],), (["da"],)),
            (["a!"], ["da"]),
            id="batch_of_pushforward_single_element",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: af.batch(af.pullback(ir), in_axes=(True, True)),
            ("x",),
            ((["a", "b", "c"],), ["g1", "g2", "g3"]),
            (["Value: a!", "Value: b!", "Value: c!"], (["g1", "g2", "g3"],)),
            id="batch_of_pullback",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.batch(af.pullback(ir), in_axes=(True, True)),
            ("x",),
            ((["a"],), ["g"]),
            (["a!"], (["g"],)),
            id="batch_of_pullback_single_element",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: af.pushforward(af.batch(ir)),
            ("x",),
            ((["a", "b"],), (["da", "db"],)),
            (["Value: a!", "Value: b!"], ["da", "db"]),
            id="pushforward_of_batch",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pushforward(af.batch(ir)),
            ("x",),
            ((["a"],), (["da"],)),
            (["a!"], ["da"]),
            id="pushforward_of_batch_single_element",
        ),
        pytest.param(
            formatted_bang,
            lambda ir: af.pullback(af.batch(ir)),
            ("x",),
            ((["a", "b"],), ["g1", "g2"]),
            (["Value: a!", "Value: b!"], (["g1", "g2"],)),
            id="pullback_of_batch",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pullback(af.batch(ir)),
            ("x",),
            ((["a"],), ["g"]),
            (["a!"], (["g"],)),
            id="pullback_of_batch_single_element",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pushforward(af.pushforward(af.batch(ir))),
            ("x",),
            (((["a", "b"],), (["t1a", "t1b"],)), ((["t2a", "t2b"],), (["t2t1a", "t2t1b"],))),
            ((["a!", "b!"], ["t1a", "t1b"]), (["t2a", "t2b"], ["t2t1a", "t2t1b"])),
            id="pushforward_of_pushforward_of_batch",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pushforward(af.pushforward(af.pushforward(ir))),
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
            lambda ir: af.pushforward(af.pushforward(af.pushforward(af.pushforward(ir)))),
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
            lambda ir: af.batch(
                af.batch(af.pushforward(ir), in_axes=(True, True)),
                in_axes=(True, True),
            ),
            ("x",),
            (([["a", "b"], ["c"]],), ([["ta", "tb"], ["tc"]],)),
            ([["<a>", "<b>"], ["<c>"]], [["ta", "tb"], ["tc"]]),
            id="batch_batch_pushforward",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.batch(
                af.pushforward(af.batch(af.pushforward(ir), in_axes=(True, True))),
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
            lambda ir: af.pushforward(af.batch(af.batch(af.batch(ir)))),
            ("x",),
            (([[["a"]]],), ([[["t"]]],)),
            ([[["a!"]]], [[["t"]]]),
            id="single_element_deep",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.batch(af.pushforward(ir), in_axes=(True, False)),
            ("x",),
            ((["a", "b", "c"],), ("t",)),
            (["a!", "b!", "c!"], ["t", "t", "t"]),
            id="pushforward_batch_primals_batched_tangents_unbatched",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.batch(af.pullback(ir), in_axes=(True, False)),
            ("x",),
            ((["a", "b", "c"],), "g"),
            (["a!", "b!", "c!"], (["g", "g", "g"],)),
            id="pullback_batch_primals_batched_cotangents_unbatched",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pullback(af.pushforward(ir)),
            ("x",),
            ((("hello",), ("world",)), ("grad_p", "grad_t")),
            (("hello!", "world"), (("grad_p",), ("grad_t",))),
            id="pullback-pushforward",
        ),
        pytest.param(
            bracket_text,
            lambda ir: af.pullback(af.pushforward(ir)),
            ("x",),
            ((("a",), ("ta",)), ("gp", "gt")),
            (("[a]", "ta"), (("gp",), ("gt",))),
            id="pullback-pushforward-format",
        ),
        pytest.param(
            af.string.concat,
            lambda ir: af.pullback(af.pushforward(ir)),
            ("a", "b"),
            ((("hello", " world"), ("t1", "t2")), ("gp", "gt")),
            (("hello world", "t1t2"), (("gp", "gp"), ("gt", "gt"))),
            id="pullback-pushforward-two-inputs",
        ),
        pytest.param(
            append_bang,
            lambda ir: af.pullback(af.pullback(ir)),
            ("x",),
            ((("hello",), "grad"), ("gg_p", ("gg_c",))),
            (("hello!", ("grad",)), (("gg_p",), "gg_c")),
            id="pullback-pullback",
        ),
        pytest.param(
            angle_text,
            lambda ir: af.pullback(af.pullback(ir)),
            ("x",),
            ((("a",), "g"), ("cp", ("cc",))),
            (("<a>", ("g",)), (("cp",), "cc")),
            id="pullback-pullback-format",
        ),
        pytest.param(
            af.string.concat,
            lambda ir: af.pullback(af.pullback(ir)),
            ("a", "b"),
            ((("hello", " world"), "grad"), ("cp", ("cc_x", "cc_y"))),
            (("hello world", ("grad", "grad")), (("cp", "cp"), "cc_xcc_y")),
            id="pullback-pullback-two-inputs",
        ),
    ],
)
def test_composition(executor, program, transform, trace_args, args, expected):
    ir = transform(af.trace(program)(*trace_args))
    result = executor(ir, *args)
    assert result == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pushforward_preserves_batched_zero_structure(executor):
    double = af.batch(af.trace(lambda x: x + x)(1.0))
    add = af.batch(af.trace(lambda x, y: x + y)(1.0, 1.0))
    x, y = [1.0, 2.0], [3.0, 4.0]
    ir = af.trace(lambda x, y: add.call(double.call(x), y))(x, y)
    zero = af.core.Zero(af.numeric.FloatAVal())

    assert executor(af.pushforward(ir), (x, y), ([zero, zero], [1.0, 1.0])) == (
        [5.0, 8.0],
        [1.0, 1.0],
    )


@pytest.mark.parametrize(
    "executor, forward",
    [
        (execute, af.ad.impl_pushforward_call),
        (
            aexecute,
            lambda args, *, ir: asyncio.run(af.ad.aimpl_pushforward_call(args, ir=ir)),
        ),
    ],
    ids=["sync", "async"],
)
def test_traced_pushforward_preserves_batched_zero_structure(executor, forward):
    double = af.batch(af.trace(lambda x: x + x)(1.0))
    add = af.batch(af.trace(lambda x, y: x + y)(1.0, 1.0))
    x, y = [1.0, 2.0], [3.0, 4.0]
    source = af.trace(lambda x, y: add.call(double.call(x), y))(x, y)
    zero = af.core.Zero(af.numeric.FloatAVal())
    ir = af.trace(lambda x, y: forward(((x, y), ([zero, zero], [1.0, 1.0])), ir=source))(x, y)

    assert executor(ir, x, y) == ([5.0, 8.0], [1.0, 1.0])


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_pullback_unused_batched_output(executor):
    double = af.batch(af.trace(lambda x: x + x)(1.0))

    def program(x, y):
        double.call(x)
        return y

    x, y = [1.0, 2.0], [3.0, 4.0]
    ir = af.pullback(af.trace(program)(x, y))
    zero = af.core.Zero(af.numeric.FloatAVal())

    assert executor(ir, (x, y), [1.0, 1.0]) == (y, ([zero, zero], [1.0, 1.0]))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("transform", [af.pushforward, af.pullback], ids=["pf", "pb"])
def test_nonidentity_primal_space(executor, transform):
    import autoform.check as check

    class ValueAVal(af.core.AVal): ...

    af.core.aval_types[ValueAVal] = lambda aval: aval
    af.core.primal_s.set(ValueAVal, lambda _: af.numeric.FloatAVal())
    af.core.tangent_s.set(ValueAVal, lambda _: af.string.StrAVal())
    af.core.cotangent_s.set(ValueAVal, lambda _: af.string.StrAVal())
    prim = af.extend.Prim("mapped_primal")

    def forward(args):
        p, t = args
        return p * 2.0, t if isinstance(t, af.core.Zero) else t + t

    af.extend.register_impl(prim, lambda x: x * 2.0)
    af.extend.register_aimpl(prim, af.utils.asyncify(lambda x: x * 2.0))
    af.extend.register_abstract(prim, lambda _: ValueAVal())
    af.extend.register_pushforward(prim, forward)
    af.extend.register_apushforward(prim, af.utils.asyncify(forward))
    af.extend.register_pullback_fwd(prim, lambda x: (x * 2.0, None))
    af.extend.register_apullback_fwd(prim, af.utils.asyncify(lambda x: (x * 2.0, None)))
    af.extend.register_pullback_bwd(prim, lambda args: args[1] + args[1])
    af.extend.register_apullback_bwd(prim, af.utils.asyncify(lambda args: args[1] + args[1]))
    x, y, z = (af.stage.Var(aval=ValueAVal()) for _ in range(3))
    source = af.stage.IR(
        [
            af.stage.Eqn(check.typecheck_p, x, y, dict(aval=ValueAVal())),
            af.stage.Eqn(prim, y, z, {}),
        ],
        (x,),
        (z, "literal"),
    )
    ir = transform(source)
    zero = af.core.Zero(af.string.StrAVal())
    assert ir.in_tree[0][0].aval == af.numeric.FloatAVal()
    assert ir.out_tree[0][0].aval == af.numeric.FloatAVal()
    if transform is af.pushforward:
        args = ((2.0,), ("dx",))
        expected = ((4.0, "literal"), ("dxdx", zero))
        assert executor(ir, (2.0,), (zero,)) == ((4.0, "literal"), (zero, zero))
    else:
        args = ((2.0,), ("dy", zero))
        expected = ((4.0, "literal"), ("dydy",))
    assert executor(ir, *args) == expected
    static = (False, (False, True)) if transform is af.pullback else False
    traced = af.trace(lambda p, d: executor(ir, p, d), static=static)(*args)
    assert traced.out_tree[0][0].aval == af.numeric.FloatAVal()
    assert executor(traced, *args) == expected
    with pytest.raises(TypeError, match="Expected FloatAVal"):
        executor(ir, ("bad",), args[1])
