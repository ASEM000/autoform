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
from tests import aexecute, angle_text, append_bang, bracket_text, execute


@pytest.mark.parametrize(
    "transform",
    [af.pushforward, af.pullback],
    ids=["pushforward", "pullback"],
)
def test_int_input_is_not_differentiable(transform):
    with pytest.raises(TypeError):
        transform(af.trace(lambda x: x)(1))


@pytest.mark.parametrize(
    "transform, space, zeroof, change",
    [
        pytest.param(
            af.pushforward,
            af.core.tangent_s,
            af.ad.tangent_zeroof,
            "replace hello",
            id="tangent",
        ),
        pytest.param(
            af.pullback,
            af.core.cotangent_s,
            af.ad.cotangent_zeroof,
            "be clearer",
            id="cotangent",
        ),
    ],
)
def test_wrapper_uses_derivative_space(transform, space, zeroof, change):
    class Text:
        def __init__(self, value):
            self.value = value

    class Change:
        def __init__(self, value):
            self.value = value

    class TextAVal(af.core.AVal): ...

    class ChangeAVal(af.core.AVal): ...

    af.core.primal_s.set(Text, lambda _: TextAVal())
    af.core.primal_s.set(Change, lambda _: ChangeAVal())
    space.set(TextAVal, lambda _: ChangeAVal())
    space.set(ChangeAVal, lambda aval: aval)
    aval = TextAVal()
    var = af.core.Var(aval=aval)
    source = af.core.IR([], (var,), (var,))
    ir = transform(source)
    for primals, derivatives in (ir.in_tree, ir.out_tree):
        assert primals[0].aval is aval
        assert isinstance(derivatives[0].aval, ChangeAVal)

    text, delta = Text("hello"), Change(change)
    assert ir.call((text,), (delta,)) == ((text,), (delta,))
    assert isinstance(zeroof(text).aval, ChangeAVal)
    nested_derivatives = transform(ir).in_tree[1]
    assert all(isinstance(v.aval, ChangeAVal) for v in af.utils.tree.leaves(nested_derivatives))


class TestCotangentHelpers:
    def test_zero_string_contract(self):
        z = af.ad.Zero(af.core.StrAVal())
        assert af.ad.is_zero(z)
        assert z.aval == af.core.StrAVal()
        assert z == af.ad.Zero(af.core.StrAVal())
        assert z != af.ad.Zero(af.core.BoolAVal())
        assert af.ad.zeroof(z) is z
        assert af.ad.materialize(z) == ""
        assert af.core.primal_s.avalof(z) == af.core.StrAVal()
        assert not af.core.is_traceable(z)

    def test_zero_requires_aval(self):
        with pytest.raises(AssertionError, match="Expected AVal"):
            af.ad.Zero(str)

    def test_zero_non_differentiable_type(self):
        z = af.ad.Zero(af.core.BoolAVal())
        assert af.ad.is_zero(z)
        assert z.aval == af.core.BoolAVal()
        with pytest.raises(TypeError):
            af.ad.materialize(z)

    def test_zero_materializes_with_registered_aval_rule(self):
        class BlobAVal(af.core.AVal):
            __slots__ = ["size"]

            def __init__(self, size):
                self.size = size

        af.ad.zero_rules[BlobAVal] = lambda aval: ("zero", aval.size)

        assert af.ad.materialize(af.ad.Zero(BlobAVal(3))) == ("zero", 3)

    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param(["hello"], "hello", id="single"),
            pytest.param(["a", "b", "c"], "abc", id="strings"),
            pytest.param([["a", "b"], ["c", "d"]], ["ac", "bd"], id="lists"),
            pytest.param(
                [
                    {"x": ["1", af.ad.Zero(af.core.StrAVal())], "y": "a"},
                    {"x": ["2", "b"], "y": "c"},
                ],
                {"x": ["12", "b"], "y": "ac"},
                id="nested-with-zero",
            ),
            pytest.param(
                [af.ad.Zero(af.core.StrAVal()), af.ad.Zero(af.core.StrAVal())],
                af.ad.Zero(af.core.StrAVal()),
                id="all-zero",
            ),
        ],
    )
    def test_cotangent_accumulation(self, values, expected):
        assert af.ad.cot_acc(values) == expected

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
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_cot_acc_transforms(self, transform, args, expected, executor):
        source = af.trace(lambda x, y: af.ad.cot_acc([x, y]))("a", "b")
        assert [eqn.prim for eqn in source.eqns] == [af.ad.cot_acc_p]
        ir = transform(source)
        actual = executor(ir, *args)
        assert actual == expected

    def test_cot_acc_all_zeros_must_match_aval(self):
        with pytest.raises(AssertionError):
            af.ad.cot_acc([
                af.ad.Zero(af.core.StrAVal()),
                af.ad.Zero(af.core.IntAVal()),
            ])

    def test_cot_acc_registered_val_without_rule_raises(self):
        with pytest.raises(
            TypeError,
            match=r"No cotangent accumulator registered for BoolAVal\(\)",
        ):
            af.ad.cot_acc([True, False])

    def test_cot_acc_unregistered_leaf_raises(self):
        class Blob: ...

        with pytest.raises(TypeError, match="No primal aval rule registered"):
            af.ad.cot_acc([Blob(), Blob()])

    def test_cot_acc_uses_registered_aval_rule(self):
        class Text:
            def __init__(self, value):
                self.value = value

        class TextAVal(af.core.AVal): ...

        af.core.primal_s.set(Text, lambda _: TextAVal())
        af.ad.cot_acc_rules[TextAVal] = lambda cs, aval: Text("|".join(c.value for c in cs))
        result = af.ad.cot_acc([Text("a"), Text("b")])
        assert isinstance(result, Text)
        assert result.value == "a|b"

    def test_pullback_accumulates_custom_cotangent_space(self):
        class Text:
            def __init__(self, value):
                self.value = value

        class TextFeedback:
            def __init__(self, value):
                self.value = value

        class TextAVal(af.core.AVal): ...

        class TextFeedbackAVal(af.core.AVal): ...

        af.core.primal_s.set(Text, lambda _: TextAVal())
        af.core.primal_s.set(TextFeedback, lambda _: TextFeedbackAVal())
        af.core.cotangent_s.set(TextAVal, lambda _: TextFeedbackAVal())
        af.ad.zero_rules[TextFeedbackAVal] = lambda _: TextFeedback("")
        af.ad.cot_acc_rules[TextFeedbackAVal] = lambda cs, _: TextFeedback(
            " | ".join(c.value for c in cs)
        )
        var = af.core.Var(aval=TextAVal())
        ir = af.core.IR([], (var,), (var, var))
        text = Text("hello")
        p_out, c_in = af.pullback(ir).call((text,), (TextFeedback("left"), TextFeedback("right")))
        assert p_out == (text, text)
        assert isinstance(c_in[0], TextFeedback)
        assert c_in[0].value == "left | right"
        assert af.ad.materialize(af.ad.cotangent_zeroof(text)).value == ""


@pytest.mark.parametrize(
    "transform, literal, derivative_side",
    [
        pytest.param(af.pushforward, "constant", "in_tree", id="pushforward"),
        pytest.param(af.pullback, "constant_input", "out_tree", id="pullback"),
    ],
)
def test_literal_input_derivatives_are_zero(transform, literal, derivative_side):
    var = af.core.Var(aval=af.core.StrAVal())
    ir = af.core.IR([], (literal, var), (var,))
    zero, variable = getattr(transform(ir), derivative_side)[1]
    assert af.ad.is_zero(zero)
    assert zero.aval == af.core.StrAVal()
    assert isinstance(variable, af.core.Var)


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
    assert af.ad.is_zero(zero)
    assert zero.aval == af.core.StrAVal()
    assert isinstance(variable, af.core.Var)


@pytest.mark.parametrize(
    "transform, feedback, expected",
    [
        pytest.param(af.pushforward, (af.ad.zeroof("Q"), "dx"), "dx", id="pushforward"),
        pytest.param(af.pullback, "g", (af.ad.zeroof("Q"), "g"), id="pullback"),
    ],
)
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_static_input_literal_is_not_boxed(transform, feedback, expected, executor):
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
    primals = ((("x",), "x"), (("x",), "x"))
    feedback = (("x", ("x",)), ("x", ("x",)))
    args = ((primals, feedback), (primals, feedback))
    expected = (
        ((("[x]", ("x",)), ("x", ("x",))), primals),
        (feedback, primals),
    )
    assert ir.call(*args) == expected


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
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_composition(program, transform, trace_args, args, expected, executor):
    ir = transform(af.trace(program)(*trace_args))
    result = executor(ir, *args)
    assert result == expected
