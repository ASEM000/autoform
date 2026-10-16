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
import math
import operator
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import autoform.extend as extend
import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.core as core
import autoform.src.intercept as intercept
import autoform.src.lm as lm
import autoform.src.memo as memo
import autoform.src.numeric as numeric
import autoform.src.order as order
import autoform.src.stage as stage
import autoform.src.string as string
import autoform.src.utils as utils
from tests import (
    AValBlob,
    Blob,
    CountingInterp,
    aexecute,
    angle_text,
    append_bang,
    bracket_text,
    execute,
    prefix_name,
)


@dataclass(frozen=True)
class Label:
    name: str


@dataclass(frozen=True)
class CostTag: ...


class TestBuildIR:
    @pytest.mark.parametrize(
        "traced, runtime, expected, aval",
        [
            pytest.param(1, 2, 2, numeric.AValInt(), id="integer"),
            pytest.param(1.5, 2.5, 2.5, numeric.AValFloat(), id="float"),
            pytest.param(True, False, False, numeric.AValBool(), id="boolean"),
        ],
    )
    def test_trace_scalar_input_is_dynamic(self, traced, runtime, expected, aval):
        def program(x):
            return x

        ir = stage.trace(program)(traced)
        assert isinstance(ir.in_tree, tuple)
        assert len(ir.in_tree) == 1
        assert isinstance(ir.in_tree[0], stage.Var)
        assert ir.in_tree[0].aval == aval
        assert ir.call(runtime) == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "aval, value, invalid",
        [
            pytest.param(extend.AValFloat(), 1.5, 1, id="float"),
            pytest.param(extend.AValInt(), 1, True, id="integer"),
            pytest.param(extend.AValStr(), "x", 1, id="string"),
            pytest.param(extend.AValBool(), True, 1, id="boolean"),
            pytest.param(axis.AValBatch(extend.AValStr()), ["a", "b"], ["a", 1], id="batch"),
            pytest.param(
                axis.AValBatch(axis.AValBatch(extend.AValStr())),
                [["a"], ["b", "c"]],
                ["a"],
                id="nested-batch",
            ),
        ],
    )
    def test_trace_abstract_input(self, executor, aval, value, invalid):
        ir = stage.trace(lambda x: x)(aval)
        assert ir.in_tree[0].aval == ir.out_tree.aval == aval
        assert executor(ir, value) == value
        with pytest.raises(TypeError):
            executor(ir, invalid)

    def test_trace_dict_input_with_scalar_leaves(self):
        def program(payload):
            return payload["name"], payload["count"], payload["score"], payload["active"]

        ir = stage.trace(program)({"name": "cats", "count": 1, "score": 1.5, "active": True})
        result = ir.call({"name": "dogs", "count": 2, "score": 2.5, "active": False})
        assert result == ("dogs", 2, 2.5, False)

    def test_trace_unsupported_input_leaf_errors(self):
        class Opaque: ...

        def program(x):
            return x

        with pytest.raises(AssertionError, match="Unsupported input leaf type"):
            stage.trace(program)(Opaque())

    def test_trace_uses_registered_aval_rule(self):
        class TraceBlob(Blob): ...

        def program(x):
            return x

        core.aval_types[TraceBlob] = lambda x: AValBlob(x.size)
        stage.trace_types.add(TraceBlob)

        ir = stage.trace(program)(TraceBlob(3))

        assert ir.in_tree[0].aval == AValBlob(3)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_avalof_live_trace_values(self, executor):
        def program(x):
            assert core.avalof(x) is x.aval
            y = append_bang(x)
            assert core.avalof(y) is y.aval
            return y

        ir = stage.trace(program)("x")

        assert len(ir.eqns) == 1
        assert executor(ir, "y") == "y!"

    def test_trace_static_unhashable_input_errors(self):
        class Unhashable:
            __hash__ = None

        def program(x):
            return x

        with pytest.raises(TypeError):
            stage.trace(program, static=True)(Unhashable())

    def test_traces_literal_and_variable(self):
        def program(name):
            return string.concat("Hello, ", name)

        ir = stage.trace(program)("x0")
        assert len(ir.eqns) == 1
        assert isinstance(ir.in_tree, tuple)
        assert len(ir.in_tree) == 1
        assert isinstance(ir.in_tree[0], stage.Var)
        eqn = ir.eqns[0]
        assert len(eqn.in_tree) == 2
        lit_candidate = eqn.in_tree[0]
        assert lit_candidate == "Hello, "
        assert isinstance(eqn.in_tree[1], stage.Var)

    def test_tracing_unhashable_literal_leaf_errors(self):
        class Unhashable:
            __hash__ = None

        literal = Unhashable()

        def program(x):
            return intercept.checkpoint((literal, x), key="value")

        with pytest.raises(TypeError):
            stage.trace(program)("x")

    def test_traced_literal_container_is_detached_from_source_mutation(self):
        parts = ["a", "b"]

        def program(x):
            return intercept.checkpoint((parts, x), key="parts")

        ir = stage.trace(program)("x")
        eqn = ir.eqns[0]
        saved_parts, _ = eqn.in_tree

        assert saved_parts == ["a", "b"]
        assert saved_parts is not parts
        assert ir.call("z") == (["a", "b"], "z")

        parts.append("c")

        saved_parts, _ = eqn.in_tree
        assert saved_parts == ["a", "b"]
        assert ir.call("z") == (["a", "b"], "z")


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform, args, message",
    [
        pytest.param(lambda ir: ir, (["x"],), "Cannot join AValStr", id="primal"),
        pytest.param(ad.pushforward, (("x",), (["dx"],)), "Cannot join AValStr", id="tangent"),
        pytest.param(axis.batch, (["x", 1.0],), "Cannot join AValStr", id="batch"),
    ],
)
def test_call_rejects_incompatible_input_aval(executor, transform, args, message):
    ir = transform(stage.trace(lambda x: x)("x"))
    with pytest.raises(TypeError, match=message):
        executor(ir, *args)


def test_walk_rejects_incompatible_input_aval():
    ir = stage.trace(lambda x: x)("x")
    with pytest.raises(TypeError, match="Cannot join AValStr"):
        next(ir.walk(["x"]))


class TestTraceStatic:
    def test_static_inputs_become_literals(self):
        ir = stage.trace(prefix_name, static=(True, False))("Hello", "World")

        assert ir.in_tree[0] == "Hello"
        assert isinstance(ir.in_tree[1], stage.Var)
        assert ir.call("Hello", "x0") == "Hello x0"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_static_input_mismatch_errors_before_execution(self, executor):
        ir = stage.trace(prefix_name, static=(True, False))("Hello", "World")
        with pytest.raises(AssertionError, match="Static input mismatch"):
            executor(ir, "Hi", "x0")

    def test_static_input_check_is_separate_from_walk(self):
        ir = stage.trace(prefix_name, static=(True, False))("Hello", "World")

        with pytest.raises(AssertionError, match="Static input mismatch"):
            stage.check_static_inputs(ir.in_tree, ("Hi", "x0"))

        gen = ir.walk("Hi", "x0")

        eqn, in_values = next(gen)
        assert in_values == ("Hello", " ", "x0")
        done, out = gen.send(eqn.bind(in_values, **eqn.params))

        assert done is None
        assert out == "Hello x0"

    def test_static_spec_must_match_input_tree(self):
        with pytest.raises(ValueError):
            stage.trace(prefix_name, static=(True, False, True))("Hello", "World")

    def test_static_bool_specializes_python_branch(self):
        def program(flag, name):
            if flag:
                return string.format("Hello {name}", name=name)
            return string.format("Bye {name}", name=name)

        ir = stage.trace(program, static=(True, False))(True, "World")

        assert ir.in_tree[0] is True
        assert isinstance(ir.in_tree[1], stage.Var)
        assert ir.call(True, "x0") == "Hello x0"


class TestTags:
    def test_trace_snapshots_tags_per_equation(self):
        def program(x):
            head = string.concat(x, "!")
            with stage.tag(Label("planner")):
                mid = string.concat(head, "?")
                with stage.tag(Label("draft"), CostTag()):
                    tail = string.concat(mid, ".")
            return tail

        ir = stage.trace(program)("seed")

        assert ir.eqns[0].tags == frozenset()
        assert ir.eqns[1].tags == frozenset({Label("planner")})
        assert ir.eqns[2].tags == frozenset({
            Label("planner"),
            Label("draft"),
            CostTag(),
        })

    def test_tag_accepts_plain_hashable_values(self):
        with stage.tag("draft", 1) as active:
            assert active == ("draft", 1)
            assert stage.active_tags.get() == frozenset({"draft", 1})

        assert stage.active_tags.get() == frozenset()

    def test_tag_rejects_unhashable_values(self):
        with pytest.raises(TypeError, match="Tags must be hashable"):
            with stage.tag(["draft"]):
                ...

    def test_tag_unions_active_tags_and_restores_on_exit(self):
        assert stage.active_tags.get() == frozenset()

        with stage.tag(Label("outer")) as outer_tags:
            assert outer_tags == (Label("outer"),)
            assert stage.active_tags.get() == frozenset({Label("outer")})

            with stage.tag(Label("inner")) as inner_tags:
                assert inner_tags == (Label("inner"),)
                assert stage.active_tags.get() == frozenset({Label("outer"), Label("inner")})

            assert stage.active_tags.get() == frozenset({Label("outer")})

        assert stage.active_tags.get() == frozenset()

    def test_ireqn_tags_input_is_frozenset(self):
        p_tag_set = core.Prim("tag_set")
        eqn = stage.Eqn(p_tag_set, (), (), None, frozenset({Label("draft")}))

        assert eqn.tags == frozenset({Label("draft")})

        with pytest.raises(AssertionError):
            stage.Eqn(p_tag_set, (), (), None, (Label("draft"),))

    def test_bind_reinstalls_equation_tags(self):
        def abstract_probe(x):
            del x
            return string.AValStr()

        def impl_probe(x):
            names = sorted(tag.name for tag in stage.active_tags.get() if isinstance(tag, Label))
            return f"{','.join(names)}|{x}"

        p_probe = core.Prim("tag_probe")
        extend.r_impl.set(p_probe, impl_probe)
        extend.r_abstract.set(p_probe, abstract_probe)

        def program(x):
            with stage.tag(Label("draft"), Label("cost")):
                return p_probe.bind(x)

        ir = stage.trace(program)("seed")

        assert ir.call("hello") == "cost,draft|hello"

        with stage.tag(Label("runtime")):
            assert ir.call("hello") == "cost,draft,runtime|hello"

        assert ir.eqns[0].tags == frozenset({Label("draft"), Label("cost")})

    def test_repr_includes_non_empty_tags(self):
        def program(x):
            head = string.concat(x, "!")
            with stage.tag(Label("draft"), CostTag()):
                return string.concat(head, "?")

        lines = repr(stage.trace(program)("seed")).splitlines()

        assert "tags=" not in lines[1]
        assert "tags={CostTag(), Label(name='draft')}" in lines[2]

    def test_calling_existing_ir_while_tracing_unions_runtime_and_equation_tags(self):
        def inner_program(x):
            with stage.tag(Label("inner")):
                return string.concat(x, "!")

        inner_ir = stage.trace(inner_program)("seed")

        def outer_program(x):
            with stage.tag(Label("outer")):
                return inner_ir.call(x)

        outer_ir = stage.trace(outer_program)("seed")

        assert outer_ir.eqns[0].tags == frozenset({Label("inner"), Label("outer")})


class TestRunIR:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_call_rejects_incompatible_injected_output(self, executor):
        ir = stage.trace(lambda x: intercept.checkpoint(x, key="value", collection="cache"))(
            "hello"
        )
        with intercept.inject(collection="cache", values={"value": [1.0]}):
            with pytest.raises(TypeError, match="Cannot join AValStr"):
                executor(ir, "world")

    @pytest.mark.parametrize(
        "program, output, error, message",
        [
            pytest.param(append_bang, 1.0, TypeError, "Cannot join AValStr", id="dynamic"),
            pytest.param(
                lambda x: (append_bang(x), x)[1],
                1.0,
                TypeError,
                "Cannot join AValStr",
                id="unused-output",
            ),
            pytest.param(
                lambda x: intercept.checkpoint((x, x), key="value"),
                ("world!",),
                ValueError,
                "arity mismatch",
                id="wrong-output-tree",
            ),
        ],
    )
    def test_walk_rejects_incompatible_supplied_output(self, program, output, error, message):
        gen = stage.trace(program)("hello").walk("world")
        next(gen)
        with pytest.raises(error, match=message):
            gen.send(output)

    @pytest.mark.parametrize(
        "use_in_equation", [False, True], ids=["final-output", "equation-input"]
    )
    def test_walk_rechecks_mutated_values(self, use_in_equation):
        class Value(Blob): ...

        extend.register_trace_type(Value, lambda value: AValBlob(value.size))

        def program(x):
            x = intercept.checkpoint(x, key="value")
            intercept.checkpoint("pause", key="pause")
            return intercept.checkpoint(x, key="next") if use_in_equation else x

        value = Value(3)
        gen = stage.trace(program)(value).walk(value)
        eqn, inputs = next(gen)
        eqn, inputs = gen.send(eqn.bind(inputs, **eqn.params))
        value.size = 4
        with pytest.raises(TypeError, match="Cannot join .*AValBlob"):
            gen.send(eqn.bind(inputs, **eqn.params))

    def test_walk_with_supplied_output(self):
        gen = stage.trace(append_bang)("hello").walk("world")
        eqn, in_values = next(gen)
        assert eqn.prim is string.p_concat
        assert in_values == ("world", "!")
        assert gen.send("world!") == (None, "world!")

    def test_walk_with_external_bind(self):
        gen = stage.trace(append_bang)("hello").walk("world")
        eqn, in_values = next(gen)
        assert eqn.prim is string.p_concat
        assert in_values == ("world", "!")
        output = eqn.bind(("there", "!"), **eqn.params)
        assert gen.send(output) == (None, "there!")

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "program, traced, runtime, expected, equations",
        [
            pytest.param(
                lambda x: string.concat(x, "!"),
                ("hello",),
                ("world",),
                "world!",
                1,
                id="single",
            ),
            pytest.param(
                lambda x: string.format("[{x}]", x=string.concat(x, x)),
                ("A",),
                ("B",),
                "[BB]",
                2,
                id="chain",
            ),
            pytest.param(
                lambda a, b: string.format("{a} + {b}", a=a, b=b),
                ("x", "y"),
                ("1", "2"),
                "1 + 2",
                1,
                id="multiple-inputs",
            ),
        ],
    )
    def test_execution(self, executor, program, traced, runtime, expected, equations):
        ir = stage.trace(program)(*traced)
        assert len(ir.in_tree) == len(traced)
        assert all((isinstance(v, stage.Var) for v in ir.in_tree))
        assert len(ir.eqns) == equations
        result = executor(ir, *runtime)
        assert result == expected


def test_nested_primitive_inputs():
    def impl(inputs):
        name, options = inputs
        return f"{options['greeting']}, {name}{options['punctuation']}"

    p_greet = core.Prim("greet")
    extend.r_impl.set(p_greet, impl)
    extend.r_abstract.set(p_greet, lambda inputs: string.AValStr())
    greeting_ir = stage.trace(
        lambda name: p_greet.bind((name, dict(greeting="Hi", punctuation="?")))
    )("World")

    (eqn,) = greeting_ir.eqns
    name, options = eqn.in_tree
    assert name is greeting_ir.in_tree[0]
    assert options == {"greeting": "Hi", "punctuation": "?"}
    assert greeting_ir.call("World") == "Hi, World?"


class TestKeywordArgumentBoundary:
    def test_trace_rejects_kwargs(self):
        def program(x, *, repeat=1):
            return string.concat("Hi", x, "!" * repeat)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            stage.trace(program)("A", repeat=3)

    @pytest.mark.parametrize(
        "executor",
        [
            pytest.param(stage.IR.call, id="sync"),
            pytest.param(stage.IR.acall, id="async"),
        ],
    )
    def test_call_rejects_kwargs(self, executor):

        def program(name, punctuation):
            return string.format(
                "Hello, {name}{punctuation}",
                name=name,
                punctuation=punctuation,
            )

        ir = stage.trace(program)("World", "!")
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            executor(ir, "World", punctuation="?")


def test_ir_and_equation_fields():
    def program(x):
        with stage.tag(Label("draft")):
            return intercept.checkpoint(x, key="x", collection="old")

    ir = stage.trace(program)("test")
    match ir:
        case stage.IR(eqns=[stage.Eqn(prim=prim, in_tree=inputs, out_tree=output, params=params)]):
            assert prim is intercept.p_checkpoint
            assert inputs is ir.in_tree[0]
            assert output is ir.out_tree
            assert params == {"key": "x", "collection": "old"}
        case _:
            pytest.fail("IR and Eqn fields must support keyword pattern matching")
    (eqn,) = ir.eqns
    updated = eqn.using(collection="new")
    assert eqn.params == {"key": "x", "collection": "old"}
    assert updated.params == {"key": "x", "collection": "new"}
    assert updated.prim is eqn.prim
    assert updated.in_tree is eqn.in_tree
    assert updated.out_tree is eqn.out_tree
    assert updated.tags == eqn.tags == frozenset({Label("draft")})
    rebuilt = stage.IR(eqns=[updated], in_tree=ir.in_tree, out_tree=ir.out_tree)
    assert rebuilt.call("hello") == "hello"


@pytest.mark.parametrize(
    "aval",
    [
        pytest.param(string.AValStr(), id="string"),
        pytest.param(numeric.AValFloat(), id="float"),
        pytest.param(AValBlob(3), id="custom-metadata"),
    ],
)
def test_variable_and_literal_boundary(aval):
    var = stage.Var(aval=aval)
    assert stage.is_var(var)
    assert var.aval is aval
    assert core.avalof(var) is var.aval
    assert not stage.is_traceable(var)
    assert not stage.is_var("hello")
    box = stage.TraceBox(owner=stage.TraceInterp(), var=var)
    assert box.aval is var.aval
    assert core.avalof(box) is aval
    assert {box: var}[box] is var


class TestToposortLevels:
    def test_empty_ir(self):
        def program(x):
            return x

        ir = stage.trace(program)("input")
        levels = stage.toposort_levels(ir)

        assert levels == []

    def test_single_equation(self):
        def program(x):
            return string.format("{x}", x=x)

        ir = stage.trace(program)("input")
        levels = stage.toposort_levels(ir)

        assert len(levels) == 1
        assert len(levels[0]) == 1

    def test_independent_equations(self):
        def program(a, b):
            x = string.format("hello {a}", a=a)
            y = string.format("world {b}", b=b)
            return x, y

        ir = stage.trace(program)("a", "b")
        levels = stage.toposort_levels(ir)

        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_dependent_equations(self):
        def program(a, b):
            x = string.format("hello {a}", a=a)
            y = string.format("world {b}", b=b)
            z = string.concat(x, y)
            return z

        ir = stage.trace(program)("a", "b")
        levels = stage.toposort_levels(ir)

        assert len(levels) == 2
        assert len(levels[0]) == 2
        assert len(levels[1]) == 1

    def test_chain_of_equations(self):
        def program(x):
            a = string.format("{x}", x=x)
            b = string.concat(a, "!")
            c = string.concat(b, "?")
            return c

        ir = stage.trace(program)("input")
        levels = stage.toposort_levels(ir)

        assert len(levels) == 3
        assert len(levels[0]) == 1
        assert len(levels[1]) == 1
        assert len(levels[2]) == 1


class TestToposortLevelsWithCheckpoints:
    def test_checkpoint_equations_can_parallelize(self):
        def program(a, b):
            x = intercept.checkpoint(string.format("hello {a}", a=a), key="x")
            y = intercept.checkpoint(string.format("world {b}", b=b), key="y")
            return x, y

        ir = stage.trace(program)("a", "b")
        levels = stage.toposort_levels(ir)

        checkpoint_levels = [
            index
            for index, level in enumerate(levels)
            for eqn in level
            if eqn.prim is intercept.p_checkpoint
        ]
        assert len(checkpoint_levels) == 2
        assert checkpoint_levels[0] == checkpoint_levels[1]

    def test_checkpoint_ordering_via_depends(self):
        def program(a, b):
            x = intercept.checkpoint(string.format("hello {a}", a=a), key="x")
            y = intercept.checkpoint(string.format("world {b}", b=b), key="y")
            return order.depends(y, x)

        ir = stage.trace(program)("a", "b")
        levels = stage.toposort_levels(ir)

        checkpoint_levels = [
            index
            for index, level in enumerate(levels)
            for eqn in level
            if eqn.prim is intercept.p_checkpoint
        ]
        assert checkpoint_levels[0] == checkpoint_levels[1]
        depends_level = next(
            index
            for index, level in enumerate(levels)
            if any(eqn.prim is order.p_depends for eqn in level)
        )
        assert depends_level > checkpoint_levels[0]

    def test_pure_equations_parallelize_around_checkpoints(self):
        def program(a, b, c):
            x = string.format("{a}", a=a)
            y = intercept.checkpoint(string.format("{b}", b=b), key="cp")
            z = string.format("{c}", c=c)
            return x, y, z

        ir = stage.trace(program)("a", "b", "c")
        levels = stage.toposort_levels(ir)

        has_parallel = any(len(lvl) > 1 for lvl in levels)

        assert len(levels) == 2
        assert has_parallel


class TestIrStructure:
    def test_ignores_variable_identity_and_body(self):
        lhs = stage.trace(lambda x: (x, "same"))("X")
        rhs = stage.trace(lambda x: (string.concat(x, "!"), "same"))("Y")

        assert stage.is_same_structure(lhs, rhs)

    @pytest.mark.parametrize(
        "left, right",
        [
            pytest.param("L", "R", id="different-values"),
            pytest.param(1, True, id="integer-boolean-types"),
        ],
    )
    def test_rejects_different_literal_outputs(self, left, right):
        lhs = stage.trace(lambda: left)()
        rhs = stage.trace(lambda: right)()

        assert not stage.is_same_structure(lhs, rhs)

    @pytest.mark.parametrize(
        "left, right",
        [
            pytest.param("X", 1, id="different-types"),
            pytest.param("X", ["X"], id="different-structures"),
        ],
    )
    def test_rejects_different_inputs(self, left, right):
        lhs = stage.trace(lambda x: "same")(left)
        rhs = stage.trace(lambda x: "same")(right)

        assert not stage.is_same_structure(lhs, rhs)

    def test_rejects_different_static_inputs(self):
        lhs = stage.trace(lambda x: "same", static=True)("L")
        rhs = stage.trace(lambda x: "same", static=True)("R")

        assert not stage.is_same_structure(lhs, rhs)

    def test_rejects_static_and_dynamic_inputs(self):
        lhs = stage.trace(lambda x: "same", static=True)("X")
        rhs = stage.trace(lambda x: "same")("X")

        assert not stage.is_same_structure(lhs, rhs)

    def test_rejects_variable_and_literal_outputs(self):
        lhs = stage.trace(lambda x: x)("X")
        rhs = stage.trace(lambda x: "X")("X")

        assert not stage.is_same_structure(lhs, rhs)
        assert not stage.is_same_structure(rhs, lhs)

    def test_rejects_different_output_structure(self):
        lhs = stage.trace(lambda x: x)("X")
        rhs = stage.trace(lambda x: [x])("X")

        assert not stage.is_same_structure(lhs, rhs)

    def test_rejects_different_aval_metadata(self):
        lhs = axis.batch(stage.trace(lambda text, count: text)("X", 1))
        rhs = axis.batch(stage.trace(lambda text, count: count)("X", 1))

        assert not stage.is_same_structure(lhs, rhs)


class TestIrVarLeaves:
    def test_returns_input_vars_in_leaf_order(self):
        def program(payload):
            head, (left, right) = payload
            return string.format("{head} {left} {right}", head=head, left=left, right=right)

        ir = stage.trace(program)(("head", ("left", "right")))
        (payload,) = ir.in_tree
        head, pair = payload

        assert stage.var_leaves(ir.in_tree) == [head, pair[0], pair[1]]

    def test_filters_static_input_literals(self):
        ir = stage.trace(prefix_name, static=(True, False))("Hello", "World")

        assert stage.var_leaves(ir.in_tree) == [ir.in_tree[1]]

    def test_returns_output_vars_in_leaf_order(self):
        def program(x):
            left = string.concat(x, "1")
            right = string.concat(x, "2")
            return ({"left": left}, (right, "const"))

        ir = stage.trace(program)("seed")
        left_tree, right_tree = ir.out_tree

        assert stage.var_leaves(ir.out_tree) == [left_tree["left"], right_tree[0]]

    def test_filters_literal_outputs(self):
        def program(x):
            return ("const", {"value": x})

        ir = stage.trace(program)("seed")

        assert stage.var_leaves(ir.out_tree) == [ir.out_tree[1]["value"]]


class TestIrVarProducers:
    def test_maps_each_output_var_to_its_producer(self):
        def program(x):
            left = string.concat(x, "1")
            right = string.concat(left, "2")
            return left, right

        ir = stage.trace(program)("seed")
        first_eqn, second_eqn = ir.eqns
        left, right = ir.out_tree

        assert stage.var_producers(ir) == {left: first_eqn, right: second_eqn}

    def test_includes_all_vars_from_tree_outputs(self):
        def program(x):
            pair = string.concat(x, "!")
            return {"value": pair, "original": x}

        ir = stage.trace(program)("seed")
        producers = stage.var_producers(ir)
        produced = ir.out_tree["value"]

        assert producers == {produced: ir.eqns[0]}

    def test_errors_if_same_var_is_produced_twice(self):
        shared = stage.Var.fresh(aval=string.AValStr())
        eqn_a = stage.Eqn(core.Prim("a"), (), shared, {})
        eqn_b = stage.Eqn(core.Prim("b"), (), shared, {})
        ir = stage.IR([eqn_a, eqn_b], in_tree=(), out_tree=shared)

        with pytest.raises(AssertionError):
            stage.var_producers(ir)


class TestIrEqnDependencyGraph:
    def test_returns_empty_graph_for_empty_ir(self):
        def program(x):
            return x

        ir = stage.trace(program)("seed")

        assert stage.eqn_graph(ir) == {}

    def test_includes_independent_equations_with_empty_children(self):
        def program(a, b):
            left = string.format("{a}", a=a)
            right = string.format("{b}", b=b)
            return left, right

        ir = stage.trace(program)("a", "b")
        left_eqn, right_eqn = ir.eqns

        assert stage.eqn_graph(ir) == {left_eqn: [], right_eqn: []}

    def test_maps_parent_equations_to_children(self):
        def program(x):
            a = string.format("{x}", x=x)
            b = string.concat(a, "!")
            c = string.concat(b, "?")
            return c

        ir = stage.trace(program)("seed")
        a_eqn, b_eqn, c_eqn = ir.eqns

        assert stage.eqn_graph(ir) == {a_eqn: [b_eqn], b_eqn: [c_eqn], c_eqn: []}

    def test_dedupes_repeated_input_dependencies(self):
        def program(x):
            a = string.format("{x}", x=x)
            b = string.concat(a, a)
            return b

        ir = stage.trace(program)("seed")
        a_eqn, b_eqn = ir.eqns

        assert stage.eqn_graph(ir) == {a_eqn: [b_eqn], b_eqn: []}


class TestIrLiveness:
    def test_empty_ir_returns_single_boundary(self):
        def program(x):
            return x

        ir = stage.trace(program)("seed")
        (x,) = ir.in_tree

        assert stage.liveness(ir) == [{x}]

    def test_empty_ir_respects_partial_output_mask(self):
        def program(x, y):
            return x, y

        ir = stage.trace(program)("x", "y")
        x, y = ir.in_tree

        assert stage.liveness(ir, out_used=(True, False)) == [{x}]
        assert stage.liveness(ir, out_used=(False, True)) == [{y}]
        assert stage.liveness(ir, out_used=(False, False)) == [set()]

    def test_chain_returns_boundary_liveness(self):
        def program(x):
            a = string.format("{x}", x=x)
            b = string.concat(a, "!")
            c = string.concat(b, "?")
            return c

        ir = stage.trace(program)("seed")
        (x,) = ir.in_tree
        a, b, c = (eqn.out_tree for eqn in ir.eqns)

        assert stage.liveness(ir) == [{x}, {a}, {b}, {c}]

    def test_parallel_equations_keep_suffix_live_ins(self):
        def program(a, b):
            left = string.format("{a}", a=a)
            right = string.format("{b}", b=b)
            return left, right

        ir = stage.trace(program)("left", "right")
        a, b = ir.in_tree
        left, right = (eqn.out_tree for eqn in ir.eqns)

        assert stage.liveness(ir) == [{a, b}, {b, left}, {left, right}]

    def test_partial_output_mask_reduces_output_boundary_liveness(self):
        def program(x):
            a = string.concat(x, "a")
            b = string.concat(x, "b")
            return a, b

        ir = stage.trace(program)("seed")
        (x,) = ir.in_tree
        a, b = (eqn.out_tree for eqn in ir.eqns)

        assert stage.liveness(ir, out_used=(True, False)) == [{x}, {x, a}, {a}]

    def test_static_inputs_do_not_become_live_vars(self):
        ir = stage.trace(prefix_name, static=(True, False))("Hello", "World")
        out_var = ir.out_tree

        assert stage.liveness(ir) == [{ir.in_tree[1]}, {out_var}]


class TestTraceValuePythonOps:
    @pytest.mark.parametrize(
        "dunder, program, operation, expected",
        [
            pytest.param(stage.Dunder.POS, operator.pos, operator.pos, 7, id="pos"),
            pytest.param(stage.Dunder.ABS, abs, abs, 7, id="abs"),
            pytest.param(stage.Dunder.INVERT, operator.invert, operator.invert, -8, id="invert"),
            pytest.param(
                stage.Dunder.FLOORDIV, lambda x: x // 3, operator.floordiv, 2, id="floordiv"
            ),
            pytest.param(
                stage.Dunder.FLOORDIV, lambda x: 20 // x, operator.floordiv, 2, id="rfloordiv"
            ),
            pytest.param(stage.Dunder.MOD, lambda x: x % 3, operator.mod, 1, id="mod"),
            pytest.param(stage.Dunder.MOD, lambda x: 20 % x, operator.mod, 6, id="rmod"),
            pytest.param(stage.Dunder.DIVMOD, lambda x: divmod(x, 3), divmod, (2, 1), id="divmod"),
            pytest.param(
                stage.Dunder.DIVMOD, lambda x: divmod(20, x), divmod, (2, 6), id="rdivmod"
            ),
            pytest.param(stage.Dunder.AND, lambda x: x & 2, operator.and_, 2, id="and"),
            pytest.param(stage.Dunder.AND, lambda x: 2 & x, operator.and_, 2, id="rand"),
            pytest.param(stage.Dunder.OR, lambda x: x | 8, operator.or_, 15, id="or"),
            pytest.param(stage.Dunder.OR, lambda x: 8 | x, operator.or_, 15, id="ror"),
            pytest.param(stage.Dunder.XOR, lambda x: x ^ 3, operator.xor, 4, id="xor"),
            pytest.param(stage.Dunder.XOR, lambda x: 3 ^ x, operator.xor, 4, id="rxor"),
            pytest.param(stage.Dunder.LSHIFT, lambda x: x << 2, operator.lshift, 28, id="lshift"),
            pytest.param(stage.Dunder.LSHIFT, lambda x: 2 << x, operator.lshift, 256, id="rlshift"),
            pytest.param(stage.Dunder.RSHIFT, lambda x: x >> 2, operator.rshift, 1, id="rshift"),
            pytest.param(stage.Dunder.RSHIFT, lambda x: 256 >> x, operator.rshift, 2, id="rrshift"),
            pytest.param(stage.Dunder.ROUND, round, round, 7, id="round"),
            pytest.param(stage.Dunder.ROUND, lambda x: round(x, -1), round, 10, id="round-digits"),
            pytest.param(stage.Dunder.CEIL, math.ceil, math.ceil, 7, id="ceil"),
            pytest.param(stage.Dunder.FLOOR, math.floor, math.floor, 7, id="floor"),
            pytest.param(stage.Dunder.TRUNC, math.trunc, math.trunc, 7, id="trunc"),
        ],
    )
    def test_numeric_operations_stage_registered_primitive(
        self, dunder, program, operation, expected
    ):
        class Value(Blob): ...

        class AValValue(core.AVal):
            def join(self, other):
                match other:
                    case core.Zero(aval=AValValue() as aval):
                        return self.join(aval)
                if type(other) in (Value, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        p_dunder = core.Prim(dunder.value)
        extend.register_trace_type(Value, lambda _: AValValue())
        core.aval_types[AValValue] = lambda aval: aval
        extend.r_impl.set(
            p_dunder,
            lambda inputs: operation(*(x.size if isinstance(x, Value) else x for x in inputs)),
        )
        extend.r_abstract.set(p_dunder, lambda _: utils.tree.map(core.avalof, expected))
        extend.register_dunder(dunder, AValValue, lambda *inputs: p_dunder.bind(inputs))

        ir = stage.trace(program)(Value(5))

        assert [eqn.prim for eqn in ir.eqns] == [p_dunder]
        assert ir.call(Value(7)) == expected

    def test_call_preserves_positional_and_keyword_inputs(self):
        class Value(Blob): ...

        class AValValue(core.AVal):
            def join(self, other):
                match other:
                    case core.Zero(aval=AValValue() as aval):
                        return self.join(aval)
                if type(other) in (Value, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        p_call_value = core.Prim("call_value")
        extend.register_trace_type(Value, lambda _: AValValue())
        core.aval_types[AValValue] = lambda aval: aval
        extend.r_impl.set(
            p_call_value, lambda inputs: inputs[0].size + sum(inputs[1]) + sum(inputs[2].values())
        )
        extend.r_abstract.set(p_call_value, lambda _: numeric.AValInt())
        extend.register_dunder(
            stage.Dunder.CALL,
            AValValue,
            lambda value, /, *args, **kwargs: p_call_value.bind((value, args, kwargs)),
        )

        ir = stage.trace(lambda x, y: x(y, self=y, dunder=2, box=3))(Value(5), 1)

        assert [eqn.prim for eqn in ir.eqns] == [p_call_value]
        assert ir.eqns[0].in_tree == (
            ir.in_tree[0],
            (ir.in_tree[1],),
            {"self": ir.in_tree[1], "dunder": 2, "box": 3},
        )
        assert ir.call(Value(7), 4) == 20

    def test_next_stages_iterator_advance(self):
        class Values:
            def __init__(self, values):
                self.values = iter(values)

        class AValValues(core.AVal):
            def join(self, other):
                match other:
                    case core.Zero(aval=AValValues() as aval):
                        return self.join(aval)
                if type(other) in (Values, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        p_next_value = core.Prim("next_value")
        extend.register_trace_type(Values, lambda _: AValValues())
        core.aval_types[AValValues] = lambda aval: aval
        extend.r_impl.set(p_next_value, lambda value: next(value.values))
        extend.r_abstract.set(p_next_value, lambda _: numeric.AValInt())
        extend.register_dunder(stage.Dunder.NEXT, AValValues, p_next_value.bind)

        ir = stage.trace(lambda x: (next(x), next(x)))(Values([1, 2]))

        assert [eqn.prim for eqn in ir.eqns] == [p_next_value, p_next_value]
        assert ir.call(Values([3, 4])) == (3, 4)

    def test_reversed_iterates_over_staged_elements(self):
        class Pair:
            def __init__(self, x, y):
                self.values = (x, y)

        class AValPair(core.AVal):
            def join(self, other):
                match other:
                    case core.Zero(aval=AValPair() as aval):
                        return self.join(aval)
                if type(other) in (Pair, type(self)):
                    return self
                raise TypeError(f"Cannot join {self!r} and {other!r}")

        p_pair_values = core.Prim("pair_values")
        extend.register_trace_type(Pair, lambda _: AValPair())
        core.aval_types[AValPair] = lambda aval: aval
        extend.r_impl.set(p_pair_values, lambda pair: pair.values)
        extend.r_abstract.set(p_pair_values, lambda _: (numeric.AValInt(), numeric.AValInt()))
        extend.register_dunder(
            stage.Dunder.REVERSED, AValPair, lambda pair: reversed(p_pair_values.bind(pair))
        )

        ir = stage.trace(lambda x: tuple(reversed(x)))(Pair(1, 2))

        assert [eqn.prim for eqn in ir.eqns] == [p_pair_values]
        assert ir.call(Pair(3, 4)) == (4, 3)

    @pytest.mark.parametrize(
        ("dunder", "program"),
        [
            pytest.param("bool", lambda x: "yes" if x else "no", id="bool"),
            pytest.param("str", lambda x: str(x), id="str"),
            pytest.param("format", lambda x: f"{x}", id="format"),
            pytest.param("iter", lambda x: list(x), id="iter"),
            pytest.param("index", lambda x: range(x), id="index"),
            pytest.param("int", lambda x: int(x), id="int"),
            pytest.param("float", lambda x: float(x), id="float"),
            pytest.param("complex", lambda x: complex(x), id="complex"),
            pytest.param("bytes", lambda x: bytes(x), id="bytes"),
            pytest.param("getitem", lambda x: x[0], id="getitem"),
            pytest.param("contains", lambda x: "a" in x, id="contains"),
            pytest.param("len", lambda x: len(x), id="len"),
        ],
    )
    def test_unregistered_python_operations_on_traced_values_error(self, dunder, program):
        with pytest.raises(
            TypeError,
            match=rf"No trace rule for {re.escape(dunder)} on values of type AValStr\(\)",
        ):
            stage.trace(program)("seed")


class TestFold:
    def test_fold_block_is_noop_outside_trace(self):
        counter = CountingInterp()

        with core.using_interp(counter):
            with stage.fold():
                result = string.concat("A", "B")

        assert result == "AB"
        assert counter.calls == 1

    def test_fold_block_evaluates_literals_during_trace(self):
        def program(x):
            with stage.fold():
                prefix = string.concat("A", "B")
            return string.concat(prefix, x)

        ir = stage.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.eqns[0].in_tree[0] == "AB"
        assert ir.call("C") == "ABC"

    def test_fold_block_allows_nested_interpreter_inside_trace(self):
        def program(x):
            with memo.memoize():
                with stage.fold():
                    prefix = string.concat("A", "B")
            return string.concat(prefix, x)

        ir = stage.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.call("C") == "ABC"

    def test_fold_block_rejects_dynamic_trace_values(self):
        def program(x):
            with stage.fold():
                return string.concat(x, "!")

        with pytest.raises(AssertionError, match="depends on traced value"):
            stage.trace(program)("seed")

    def test_fold_block_rejects_dynamic_trace_values_in_params(self):
        def param_probe(dynamic):
            return p_param_probe.bind("literal", dynamic=dynamic)

        def impl_param_probe(in_tree, *, dynamic):
            del in_tree
            return dynamic

        p_param_probe = core.Prim("fold_param_probe")
        extend.r_impl.set(p_param_probe, impl_param_probe)

        def program(x):
            with stage.fold():
                return param_probe(x)

        with pytest.raises(AssertionError, match="depends on traced value"):
            stage.trace(program)("seed")

    def test_fold_block_rejects_dynamic_trace_values_in_output(self):
        captured = {}

        def impl_output_probe(in_tree):
            del in_tree
            return captured["value"]

        p_output_probe = core.Prim("fold_output_probe")
        extend.r_impl.set(p_output_probe, impl_output_probe)

        def program(x):
            captured["value"] = x
            with stage.fold():
                return p_output_probe.bind("literal")

        with pytest.raises(AssertionError, match="depends on traced value"):
            stage.trace(program)("seed")

    def test_static_trace_args_are_available_in_fold_block(self):
        def program(prefix, x):
            with stage.fold():
                header = string.concat(prefix, ": ")
            return string.concat(header, x)

        ir = stage.trace(program, static=(True, False))("Q", "seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.eqns[0].in_tree[0] == "Q: "
        assert ir.call("Q", "hello") == "Q: hello"

    def test_tracing_resumes_after_static_block(self):
        def program(x):
            with stage.fold():
                prefix = string.concat("a", "b")
                prefix = string.format("[{prefix}]", prefix=prefix)
            value = string.concat(prefix, x)
            return string.concat(value, "!")

        ir = stage.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat", "concat"]
        assert ir.call("c") == "[ab]c!"

    def test_fold_block_evaluates_fill_during_trace(self):
        calls = []

        class FillClient:
            def responses(self, *, input, model, **kwargs):
                calls.append(input)
                return SimpleNamespace(output_text='{"output": "rubric"}')

            async def aresponses(self, **kwargs):
                return self.responses(**kwargs)

        def program(question):
            with stage.fold():
                rubric = lm.fill(
                    {"prompt": "make a rubric", "output": lm.Str()},
                    model="test-model",
                )["output"]
            return string.format("{rubric}: {question}", rubric=rubric, question=question)

        with lm.client(FillClient()):
            ir = stage.trace(program)("seed")

        assert len(calls) == 1
        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.call("question") == "rubric: question"

    def test_async_dynamic_trace_dispatch_stages_primitive(self):
        def abstract_async_probe(in_tree):
            del in_tree
            return string.AValStr()

        p_async_probe = core.Prim("async_dynamic_fold_probe")
        extend.r_abstract.set(p_async_probe, abstract_async_probe)

        with core.using_interp(stage.TraceInterp()) as tracer:
            result = asyncio.run(p_async_probe.abind("literal"))

        assert isinstance(result, stage.TraceBox)
        assert [eqn.prim.name for eqn in tracer.eqns] == ["async_dynamic_fold_probe"]

    def test_async_fold_trace_dispatch_evaluates_primitive(self):
        async def aimpl_async_probe(in_tree):
            return string.concat(in_tree, "!")

        p_async_probe = core.Prim("async_fold_probe")
        extend.r_aimpl.set(p_async_probe, aimpl_async_probe)

        with core.using_interp(stage.TraceInterp()) as tracer:
            with stage.fold():
                result = asyncio.run(p_async_probe.abind("literal"))

        assert result == "literal!"
        assert tracer.eqns == []


@pytest.mark.parametrize(
    "program, trace_values, runtime, expected, equations",
    [
        pytest.param(
            lambda x: string.format("Hello, {x}!", x=x),
            ["world", "test"],
            "x0",
            "Hello, x0!",
            1,
            id="single",
        ),
        pytest.param(
            lambda x: string.concat(string.format("[{x}]", x=x), "!"),
            ["x", "test"],
            "hello",
            "[hello]!",
            2,
            id="multiple",
        ),
        pytest.param(
            lambda x: string.format("({x})", x=x),
            ["x", "y", "z"],
            "hello",
            "(hello)",
            1,
            id="double",
        ),
        pytest.param(
            lambda x: string.concat(x, "!"),
            ["x", "y", "z", "w"],
            "test",
            "test!",
            1,
            id="triple",
        ),
    ],
)
def test_retracing_inlines_equations(program, trace_values, runtime, expected, equations):
    ir = stage.trace(program)(trace_values[0])
    for value in trace_values[1:]:
        ir = stage.trace(ir.call)(value)
    assert [eqn.prim for eqn in ir.eqns] == [string.p_concat] * equations
    assert ir.call(runtime) == expected


def test_inline_calls_preserve_dataflow():
    first = stage.trace(lambda x: string.concat(x, "1"))("X")
    second = stage.trace(lambda x: string.concat(x, "2"))("X")
    ir = stage.trace(lambda x: second.call(first.call(x)))("X")
    assert len(ir.eqns) == 2
    assert ir.eqns[1].in_tree[0] is ir.eqns[0].out_tree
    assert ir.call("start") == "start12"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "program, transform, args, expected, primitive",
    [
        pytest.param(
            bracket_text,
            ad.pushforward,
            (("primal",), ("tangent",)),
            ("[primal]", "tangent"),
            "pushforward_call",
            id="pushforward",
        ),
        pytest.param(
            append_bang,
            ad.pullback,
            (("primal",), "cotan"),
            ("primal!", ("cotan",)),
            "pullback_call",
            id="pullback",
        ),
        pytest.param(
            angle_text,
            ft.partial(axis.batch, in_axes=True),
            (["a", "b", "c"],),
            ["<a>", "<b>", "<c>"],
            "batch_call",
            id="batch",
        ),
    ],
)
@pytest.mark.parametrize(
    "build_ir",
    [
        pytest.param(
            lambda ir, transform, args: transform(stage.trace(ir.call)("test")),
            id="trace-transform",
        ),
        pytest.param(
            lambda ir, transform, args: stage.trace(transform(ir).call)(*args),
            id="transform-trace",
        ),
    ],
)
def test_trace_transform_boundary(
    executor,
    program,
    transform,
    args,
    expected,
    primitive,
    build_ir,
):
    inner = stage.trace(program)("x")
    ir = build_ir(inner, transform, args)
    assert len(ir.eqns) == 1
    assert ir.eqns[0].prim.name == primitive
    result = executor(ir, *args)
    assert result == expected
