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
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

import autoform as af
from tests import (
    CountingInterpreter,
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


class Blob:
    def __init__(self, size: int):
        self.size = size


class BlobAVal(af.core.AVal):
    __slots__ = ["size"]

    def __init__(self, size: int):
        self.size = size

    def __eq__(self, other):
        return type(self) is type(other) and self.size == other.size

    def __hash__(self):
        return hash((type(self), self.size))


class TestSpace:
    def test_registration_and_replacement(self):
        space = af.core.Space("blob")
        rule = lambda value: BlobAVal(value.size)
        replacement = lambda value: BlobAVal(value.size + 1)
        assert space.set(Blob, rule) is rule
        assert space.avalof(Blob(3)) == BlobAVal(3)
        with pytest.raises(AssertionError, match="already defined"):
            space.set(Blob, replacement)
        assert space.set(Blob, replacement, replace=True) is replacement
        assert space.avalof(Blob(3)) == BlobAVal(4)

    @pytest.mark.parametrize(
        "value_type, rule, replace, message",
        [
            pytest.param(Blob(3), lambda x: x, False, "Expected type", id="type"),
            pytest.param(Blob, BlobAVal(3), False, "Expected callable", id="callable"),
            pytest.param(Blob, lambda x: x, 1, "Expected bool for replace", id="replace"),
        ],
    )
    def test_invalid_registration(self, value_type, rule, replace, message):
        with pytest.raises(AssertionError, match=message):
            af.core.Space("blob").set(value_type, rule, replace=replace)

    def test_missing_rule(self):
        with pytest.raises(TypeError, match="No empty aval rule registered"):
            af.core.Space("empty").avalof(Blob(3))

    @pytest.mark.parametrize(
        ("space", "aval"),
        [
            (space, aval)
            for space in (af.core.tangent_s, af.core.cotangent_s)
            for aval in (
                af.core.StrAVal(),
                af.core.FloatAVal(),
                af.core.BoolAVal(),
            )
        ],
    )
    def test_builtin_ad_spaces_preserve_aval(self, space, aval):
        assert space.avalof(aval) is aval

    def test_custom_ad_spaces(self):
        class TextAVal(af.core.AVal): ...

        class TextEditAVal(af.core.AVal): ...

        class TextFeedbackAVal(af.core.AVal): ...

        tangent_s = af.core.Space("tangent")
        cotangent_s = af.core.Space("cotangent")
        tangent_s.set(TextAVal, lambda _: TextEditAVal())
        tangent_s.set(TextEditAVal, lambda aval: aval)
        cotangent_s.set(TextAVal, lambda _: TextFeedbackAVal())
        cotangent_s.set(TextFeedbackAVal, lambda aval: aval)

        tangent = tangent_s.avalof(TextAVal())
        cotangent = cotangent_s.avalof(TextAVal())

        assert isinstance(tangent, TextEditAVal)
        assert isinstance(cotangent, TextFeedbackAVal)
        assert tangent_s.avalof(tangent) is tangent
        assert cotangent_s.avalof(cotangent) is cotangent

    @pytest.mark.parametrize("space", [af.core.tangent_s, af.core.cotangent_s])
    def test_missing_ad_space_rule(self, space):
        class UnknownAVal(af.core.AVal): ...

        with pytest.raises(TypeError, match=f"No {space.name} aval rule registered"):
            space.avalof(UnknownAVal())


class TestBuildIR:
    @pytest.mark.parametrize(
        "traced, runtime, expected, aval",
        [
            pytest.param(1, 2, 2, af.core.IntAVal(), id="integer"),
            pytest.param(1.5, 2.5, 2.5, af.core.FloatAVal(), id="float"),
            pytest.param(True, False, False, af.core.BoolAVal(), id="boolean"),
        ],
    )
    def test_trace_scalar_input_is_dynamic(self, traced, runtime, expected, aval):
        def program(x):
            return x

        ir = af.trace(program)(traced)
        assert isinstance(ir.in_tree, tuple)
        assert len(ir.in_tree) == 1
        assert isinstance(ir.in_tree[0], af.core.Var)
        assert ir.in_tree[0].aval == aval
        assert ir.call(runtime) == expected

    def test_trace_dict_input_with_scalar_leaves(self):
        def program(payload):
            return payload["name"], payload["count"], payload["score"], payload["active"]

        ir = af.trace(program)({"name": "cats", "count": 1, "score": 1.5, "active": True})
        result = ir.call({"name": "dogs", "count": 2, "score": 2.5, "active": False})
        assert result == ("dogs", 2, 2.5, False)

    def test_trace_unsupported_input_leaf_errors(self):
        class Opaque: ...

        def program(x):
            return x

        with pytest.raises(AssertionError, match="Unsupported input leaf type"):
            af.trace(program)(Opaque())

    def test_trace_uses_registered_aval_rule(self):
        class TraceBlob(Blob): ...

        def program(x):
            return x

        af.core.primal_s.set(TraceBlob, lambda x: BlobAVal(x.size))
        af.core.trace_types.add(TraceBlob)

        ir = af.trace(program)(TraceBlob(3))

        assert ir.in_tree[0].aval == BlobAVal(3)

    def test_trace_static_unhashable_input_errors(self):
        class Unhashable:
            __hash__ = None

        def program(x):
            return x

        with pytest.raises(TypeError):
            af.trace(program, static=True)(Unhashable())

    def test_traces_literal_and_variable(self):
        def program(name):
            return af.string.concat("Hello, ", name)

        ir = af.trace(program)("x0")
        assert len(ir.eqns) == 1
        assert isinstance(ir.in_tree, tuple)
        assert len(ir.in_tree) == 1
        assert isinstance(ir.in_tree[0], af.core.Var)
        eqn = ir.eqns[0]
        assert len(eqn.in_tree) == 2
        lit_candidate = eqn.in_tree[0]
        assert lit_candidate == "Hello, "
        assert isinstance(eqn.in_tree[1], af.core.Var)

    def test_format_lowers_template_and_args_to_concat(self):
        def program(x):
            return af.string.format("Hello, {x}!", x=x)

        ir = af.trace(program)("World")
        assert len(ir.eqns) == 1
        eqn = ir.eqns[0]
        assert eqn.prim is af.string.concat_p
        prefix, value, suffix = eqn.in_tree
        assert prefix == "Hello, "
        assert suffix == "!"
        assert isinstance(value, af.core.Var)
        assert ir.call("x0") == "Hello, x0!"

    def test_tracing_unhashable_literal_leaf_errors(self):
        class Unhashable:
            __hash__ = None

        literal = Unhashable()

        def program(x):
            return af.checkpoint((literal, x), key="value")

        with pytest.raises(TypeError):
            af.trace(program)("x")

    def test_traced_literal_container_is_detached_from_source_mutation(self):
        parts = ["a", "b"]

        def program(x):
            return af.checkpoint((parts, x), key="parts")

        ir = af.trace(program)("x")
        eqn = ir.eqns[0]
        saved_parts, _ = eqn.in_tree

        assert saved_parts == ["a", "b"]
        assert saved_parts is not parts
        assert ir.call("z") == (["a", "b"], "z")

        parts.append("c")

        saved_parts, _ = eqn.in_tree
        assert saved_parts == ["a", "b"]
        assert ir.call("z") == (["a", "b"], "z")


class TestTraceStatic:
    def test_static_inputs_become_literals(self):
        ir = af.trace(prefix_name, static=(True, False))("Hello", "World")

        assert ir.in_tree[0] == "Hello"
        assert isinstance(ir.in_tree[1], af.core.Var)
        assert ir.call("Hello", "x0") == "Hello x0"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_static_input_mismatch_errors_before_execution(self, executor):
        ir = af.trace(prefix_name, static=(True, False))("Hello", "World")
        with pytest.raises(AssertionError, match="Static input mismatch"):
            executor(ir, "Hi", "x0")

    def test_static_input_check_is_separate_from_walk(self):
        ir = af.trace(prefix_name, static=(True, False))("Hello", "World")

        with pytest.raises(AssertionError, match="Static input mismatch"):
            af.core.check_static_inputs(ir.in_tree, ("Hi", "x0"))

        gen = ir.walk("Hi", "x0")

        eqn, in_values = next(gen)
        assert in_values == ("Hello", " ", "x0")
        done, out = gen.send(eqn.bind(in_values, **eqn.params))

        assert done is None
        assert out == "Hello x0"

    def test_static_spec_must_match_input_tree(self):
        with pytest.raises(ValueError):
            af.trace(prefix_name, static=(True, False, True))("Hello", "World")

    def test_static_bool_specializes_python_branch(self):
        def program(flag, name):
            if flag:
                return af.string.format("Hello {name}", name=name)
            return af.string.format("Bye {name}", name=name)

        ir = af.trace(program, static=(True, False))(True, "World")

        assert ir.in_tree[0] is True
        assert isinstance(ir.in_tree[1], af.core.Var)
        assert ir.call(True, "x0") == "Hello x0"


class TestTags:
    def test_trace_snapshots_tags_per_equation(self):
        def program(x):
            head = af.string.concat(x, "!")
            with af.tag(Label("planner")):
                mid = af.string.concat(head, "?")
                with af.tag(Label("draft"), CostTag()):
                    tail = af.string.concat(mid, ".")
            return tail

        ir = af.trace(program)("seed")

        assert ir.eqns[0].tags == frozenset()
        assert ir.eqns[1].tags == frozenset({Label("planner")})
        assert ir.eqns[2].tags == frozenset({
            Label("planner"),
            Label("draft"),
            CostTag(),
        })

    def test_tag_accepts_plain_hashable_values(self):
        with af.tag("draft", 1) as active:
            assert active == ("draft", 1)
            assert af.core.active_tags.get() == frozenset({"draft", 1})

        assert af.core.active_tags.get() == frozenset()

    def test_tag_rejects_unhashable_values(self):
        with pytest.raises(TypeError, match="Tags must be hashable"):
            with af.tag(["draft"]):
                ...

    def test_tag_unions_active_tags_and_restores_on_exit(self):
        assert af.core.active_tags.get() == frozenset()

        with af.tag(Label("outer")) as outer_tags:
            assert outer_tags == (Label("outer"),)
            assert af.core.active_tags.get() == frozenset({Label("outer")})

            with af.tag(Label("inner")) as inner_tags:
                assert inner_tags == (Label("inner"),)
                assert af.core.active_tags.get() == frozenset({Label("outer"), Label("inner")})

            assert af.core.active_tags.get() == frozenset({Label("outer")})

        assert af.core.active_tags.get() == frozenset()

    def test_ireqn_tags_input_is_frozenset(self):
        prim = af.core.Prim("tag_set")
        eqn = af.core.Eqn(prim, (), (), None, frozenset({Label("draft")}))

        assert eqn.tags == frozenset({Label("draft")})

        with pytest.raises(AssertionError):
            af.core.Eqn(prim, (), (), None, (Label("draft"),))

    def test_bind_reinstalls_equation_tags(self):
        def abstract_probe(x):
            del x
            return af.core.StrAVal()

        def impl_probe(x):
            names = sorted(tag.name for tag in af.core.active_tags.get() if isinstance(tag, Label))
            return f"{','.join(names)}|{x}"

        probe_p = af.core.Prim("tag_probe")
        af.core.impl_rules.set(probe_p, impl_probe)
        af.core.abstract_rules.set(probe_p, abstract_probe)

        def program(x):
            with af.tag(Label("draft"), Label("cost")):
                return probe_p.bind(x)

        ir = af.trace(program)("seed")

        assert ir.call("hello") == "cost,draft|hello"

        with af.tag(Label("runtime")):
            assert ir.call("hello") == "cost,draft,runtime|hello"

        assert ir.eqns[0].tags == frozenset({Label("draft"), Label("cost")})

    def test_repr_includes_non_empty_tags(self):
        def program(x):
            head = af.string.concat(x, "!")
            with af.tag(Label("draft"), CostTag()):
                return af.string.concat(head, "?")

        lines = repr(af.trace(program)("seed")).splitlines()

        assert "tags=" not in lines[1]
        assert "tags={CostTag(), Label(name='draft')}" in lines[2]

    def test_calling_existing_ir_while_tracing_unions_runtime_and_equation_tags(self):
        def inner_program(x):
            with af.tag(Label("inner")):
                return af.string.concat(x, "!")

        inner_ir = af.trace(inner_program)("seed")

        def outer_program(x):
            with af.tag(Label("outer")):
                return inner_ir.call(x)

        outer_ir = af.trace(outer_program)("seed")

        assert outer_ir.eqns[0].tags == frozenset({Label("inner"), Label("outer")})


class TestRunIR:
    def test_walk_with_supplied_output(self):
        gen = af.trace(append_bang)("hello").walk("world")
        eqn, in_values = next(gen)
        assert eqn.prim is af.string.concat_p
        assert in_values == ("world", "!")
        assert gen.send("world!") == (None, "world!")

    def test_walk_with_external_bind(self):
        gen = af.trace(append_bang)("hello").walk("world")
        eqn, in_values = next(gen)
        assert eqn.prim is af.string.concat_p
        assert in_values == ("world", "!")
        output = eqn.bind(("there", "!"), **eqn.params)
        assert gen.send(output) == (None, "there!")

    @pytest.mark.parametrize(
        "program, traced, runtime, expected, equations",
        [
            pytest.param(
                lambda x: af.string.concat(x, "!"),
                ("hello",),
                ("world",),
                "world!",
                1,
                id="single",
            ),
            pytest.param(
                lambda x: af.string.format("[{x}]", x=af.string.concat(x, x)),
                ("A",),
                ("B",),
                "[BB]",
                2,
                id="chain",
            ),
            pytest.param(
                lambda a, b: af.string.format("{a} + {b}", a=a, b=b),
                ("x", "y"),
                ("1", "2"),
                "1 + 2",
                1,
                id="multiple-inputs",
            ),
        ],
    )
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_execution(self, program, traced, runtime, expected, equations, executor):
        ir = af.trace(program)(*traced)
        assert len(ir.in_tree) == len(traced)
        assert all((isinstance(v, af.core.Var) for v in ir.in_tree))
        assert len(ir.eqns) == equations
        result = executor(ir, *runtime)
        assert result == expected


def test_nested_primitive_inputs():
    def impl(inputs):
        name, options = inputs
        return f"{options['greeting']}, {name}{options['punctuation']}"

    primitive = af.core.Prim("greet")
    af.core.impl_rules.set(primitive, impl)
    af.core.abstract_rules.set(primitive, lambda inputs: af.core.StrAVal())
    greeting_ir = af.trace(
        lambda name: primitive.bind((name, dict(greeting="Hi", punctuation="?")))
    )("World")

    (eqn,) = greeting_ir.eqns
    name, options = eqn.in_tree
    assert name is greeting_ir.in_tree[0]
    assert options == {"greeting": "Hi", "punctuation": "?"}
    assert greeting_ir.call("World") == "Hi, World?"


class TestKeywordArgumentBoundary:
    def test_trace_rejects_kwargs(self):
        def program(x, *, repeat=1):
            return af.string.concat("Hi", x, "!" * repeat)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            af.trace(program)("A", repeat=3)

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_call_rejects_kwargs(self, executor):

        def program(name, punctuation):
            return af.string.format(
                "Hello, {name}{punctuation}",
                name=name,
                punctuation=punctuation,
            )

        ir = af.trace(program)("World", "!")
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            executor(ir, "World", punctuation="?")

    def test_switch_rejects_kwargs(self):
        branches = {"a": af.trace(lambda x: af.string.concat("A:", x))("X")}

        with pytest.raises(AssertionError, match="switch.*keyword arguments"):
            af.switch("a", branches, x="test")


class TestInterpreterRuleMapping:
    def test_registration_and_replacement(self):
        mapping = af.core.InterpreterRuleMapping()
        primitive = af.core.Prim("test_replace")
        rule = lambda x: x
        replacement = lambda x: x + 1

        with pytest.raises(KeyError, match="rule defined for primitive"):
            mapping.get(primitive)
        assert mapping.set(primitive, rule) is rule
        assert mapping.get(primitive) is rule
        with pytest.raises(KeyError, match="rule defined for primitive"):
            mapping.get(af.core.Prim("test_replace"))
        with pytest.raises(AssertionError, match="already defined"):
            mapping.set(primitive, rule)
        assert mapping.set(primitive, replacement, replace=True) is replacement
        assert mapping.get(primitive) is replacement
        assert mapping.get(primitive)(1) == 2

    def test_async_registration_and_replacement(self):
        mapping = af.core.InterpreterRuleMapping()
        primitive = af.core.Prim("test_replace")

        async def rule(x):
            return x

        async def replacement(x):
            return x + 1

        with pytest.raises(KeyError, match="rule defined for primitive"):
            mapping.aget(primitive)
        assert mapping.aset(primitive, rule) is rule
        assert mapping.aget(primitive) is rule
        with pytest.raises(KeyError, match="rule defined for primitive"):
            mapping.aget(af.core.Prim("test_replace"))
        with pytest.raises(AssertionError, match="already defined"):
            mapping.aset(primitive, rule)
        assert mapping.aset(primitive, replacement, replace=True) is replacement
        assert mapping.aget(primitive) is replacement
        assert asyncio.run(mapping.aget(primitive)(1)) == 2

    def test_concurrent_registration(self):
        mapping = af.core.InterpreterRuleMapping()

        def register(index):
            primitive = af.core.Prim(f"concurrent_{index}")
            mapping.set(primitive, lambda x: x * index)
            return mapping.get(primitive)(1)

        with ThreadPoolExecutor() as pool:
            assert list(pool.map(register, range(50))) == list(range(50))


def test_ir_and_equation_fields():
    def program(x):
        with af.tag(Label("draft")):
            return af.checkpoint(x, key="x", collection="old")

    ir = af.trace(program)("test")
    match ir:
        case af.core.IR(
            eqns=[af.core.Eqn(prim=prim, in_tree=inputs, out_tree=output, params=params)]
        ):
            assert prim is af.extend.checkpoint_p
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
    rebuilt = af.core.IR(eqns=[updated], in_tree=ir.in_tree, out_tree=ir.out_tree)
    assert rebuilt.call("hello") == "hello"


class TestPrimitive:
    def test_creation(self):
        p = af.core.Prim("test_prim")
        assert p.name == "test_prim"
        assert repr(p) == "test_prim"

    @pytest.mark.parametrize(
        "name, registry, rule",
        [
            pytest.param("test_impl", af.core.impl_rules, lambda x: x, id="impl"),
            pytest.param(
                "test_abstract",
                af.core.abstract_rules,
                lambda x: af.core.StrAVal(),
                id="abstract",
            ),
            pytest.param(
                "test_batch",
                af.core.batch_rules,
                lambda inputs: (inputs[2], True),
                id="batch",
            ),
            pytest.param(
                "test_pushforward",
                af.core.push_rules,
                lambda inputs: inputs,
                id="pushforward",
            ),
            pytest.param(
                "test_pullback_fwd",
                af.core.pull_fwd_rules,
                lambda inputs: (inputs, inputs),
                id="pullback-forward",
            ),
            pytest.param(
                "test_pullback_bwd",
                af.core.pull_bwd_rules,
                lambda residuals, cotangent: cotangent,
                id="pullback-backward",
            ),
        ],
    )
    def test_register_rule(self, name, registry, rule):
        primitive = af.core.Prim(name)
        assert ft.partial(registry.set, primitive)(rule) is rule
        assert registry.get(primitive) is rule


def test_variable_and_literal_boundary():
    var = af.core.Var(aval=af.core.StrAVal())
    assert af.core.is_var(var)
    assert var.aval == af.core.StrAVal()
    assert af.core.primal_s.avalof(var) is var.aval
    assert not af.core.is_traceable(var)
    assert not af.core.is_var("hello")
    box = af.core.TraceBox(owner=af.core.TraceInterpreter(), var=var)
    assert box.aval is var.aval
    assert {box: var}[box] is var


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
        result = ir.call("B")
        assert result == "BBB"


def test_interpreter_context_restores_default():
    assert isinstance(af.core.active_interpreter.get(), af.core.EvalInterpreter)
    tracer = af.core.TraceInterpreter()
    with af.core.using_interpreter(tracer) as active:
        assert active is tracer
        af.string.format("Hello, {value}!", value=af.core.Var.fresh(aval=af.core.StrAVal()))
        assert len(tracer.eqns) == 1
    assert isinstance(af.core.active_interpreter.get(), af.core.EvalInterpreter)
    assert af.string.concat("a", "b") == "ab"


class TestTraceValuePythonOps:
    @pytest.mark.parametrize(
        ("dunder", "program"),
        [
            ("bool", lambda x: "yes" if x else "no"),
            ("str", lambda x: str(x)),
            ("format", lambda x: f"{x}"),
            ("iter", lambda x: list(x)),
            ("index", lambda x: range(x)),
            ("int", lambda x: int(x)),
            ("float", lambda x: float(x)),
            ("complex", lambda x: complex(x)),
            ("bytes", lambda x: bytes(x)),
            ("getitem", lambda x: x[0]),
            ("contains", lambda x: "a" in x),
            ("len", lambda x: len(x)),
        ],
    )
    def test_unregistered_python_operations_on_traced_values_error(self, dunder, program):
        with pytest.raises(
            TypeError,
            match=rf"No trace rule for {re.escape(dunder)} on values of type StrAVal\(\)",
        ):
            af.trace(program)("seed")


class TestFold:
    def test_fold_block_is_noop_outside_trace(self):
        counter = CountingInterpreter()

        with af.core.using_interpreter(counter):
            with af.fold():
                result = af.string.concat("A", "B")

        assert result == "AB"
        assert counter.calls == 1

    def test_fold_block_evaluates_literals_during_trace(self):
        def program(x):
            with af.fold():
                prefix = af.string.concat("A", "B")
            return af.string.concat(prefix, x)

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.eqns[0].in_tree[0] == "AB"
        assert ir.call("C") == "ABC"

    def test_fold_block_allows_nested_interpreter_inside_trace(self):
        def program(x):
            with af.memoize():
                with af.fold():
                    prefix = af.string.concat("A", "B")
            return af.string.concat(prefix, x)

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.call("C") == "ABC"

    def test_fold_block_rejects_dynamic_trace_values(self):
        def program(x):
            with af.fold():
                return af.string.concat(x, "!")

        with pytest.raises(AssertionError, match="depends on traced value"):
            af.trace(program)("seed")

    def test_fold_block_rejects_dynamic_trace_values_in_params(self):
        def param_probe(dynamic):
            return param_probe_p.bind("literal", dynamic=dynamic)

        def impl_param_probe(in_tree, *, dynamic):
            del in_tree
            return dynamic

        param_probe_p = af.core.Prim("fold_param_probe")
        af.core.impl_rules.set(param_probe_p, impl_param_probe)

        def program(x):
            with af.fold():
                return param_probe(x)

        with pytest.raises(AssertionError, match="depends on traced value"):
            af.trace(program)("seed")

    def test_fold_block_rejects_dynamic_trace_values_in_output(self):
        captured = {}

        def impl_output_probe(in_tree):
            del in_tree
            return captured["value"]

        output_probe_p = af.core.Prim("fold_output_probe")
        af.core.impl_rules.set(output_probe_p, impl_output_probe)

        def program(x):
            captured["value"] = x
            with af.fold():
                return output_probe_p.bind("literal")

        with pytest.raises(AssertionError, match="depends on traced value"):
            af.trace(program)("seed")

    def test_static_trace_args_are_available_in_fold_block(self):
        def program(prefix, x):
            with af.fold():
                header = af.string.concat(prefix, ": ")
            return af.string.concat(header, x)

        ir = af.trace(program, static=(True, False))("Q", "seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.eqns[0].in_tree[0] == "Q: "
        assert ir.call("Q", "hello") == "Q: hello"

    def test_tracing_resumes_after_static_block(self):
        def program(x):
            with af.fold():
                prefix = af.string.concat("a", "b")
                prefix = af.string.format("[{prefix}]", prefix=prefix)
            value = af.string.concat(prefix, x)
            return af.string.concat(value, "!")

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat", "concat"]
        assert ir.call("c") == "[ab]c!"

    def test_fold_block_evaluates_complete_during_trace(self):
        calls = []

        def render(messages):
            calls.append(messages)
            return "rubric"

        def program(question):
            with af.fold():
                rubric = af.lm.complete(
                    [{"role": "user", "content": "make a rubric"}],
                    model="test-model",
                )
            return af.string.format("{rubric}: {question}", rubric=rubric, question=question)

        with af.lm.client(af.lm.EchoClient(render)):
            ir = af.trace(program)("seed")

        assert len(calls) == 1
        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.call("question") == "rubric: question"

    def test_async_dynamic_trace_dispatch_stages_primitive(self):
        def abstract_async_probe(in_tree):
            del in_tree
            return af.core.StrAVal()

        async_probe_p = af.core.Prim("async_dynamic_fold_probe")
        af.core.abstract_rules.set(async_probe_p, abstract_async_probe)

        with af.core.using_interpreter(af.core.TraceInterpreter()) as tracer:
            result = asyncio.run(async_probe_p.abind("literal"))

        assert isinstance(result, af.core.TraceBox)
        assert [eqn.prim.name for eqn in tracer.eqns] == ["async_dynamic_fold_probe"]

    def test_async_fold_trace_dispatch_evaluates_primitive(self):
        async def aimpl_async_probe(in_tree):
            return af.string.concat(in_tree, "!")

        async_probe_p = af.core.Prim("async_fold_probe")
        af.core.impl_rules.aset(async_probe_p, aimpl_async_probe)

        with af.core.using_interpreter(af.core.TraceInterpreter()) as tracer:
            with af.fold():
                result = asyncio.run(async_probe_p.abind("literal"))

        assert result == "literal!"
        assert tracer.eqns == []


@pytest.mark.parametrize(
    "program, trace_values, runtime, expected, equations",
    [
        pytest.param(
            lambda x: af.string.format("Hello, {x}!", x=x),
            ["world", "test"],
            "x0",
            "Hello, x0!",
            1,
            id="single",
        ),
        pytest.param(
            lambda x: af.string.concat(af.string.format("[{x}]", x=x), "!"),
            ["x", "test"],
            "hello",
            "[hello]!",
            2,
            id="multiple",
        ),
        pytest.param(
            lambda x: af.string.format("({x})", x=x),
            ["x", "y", "z"],
            "hello",
            "(hello)",
            1,
            id="double",
        ),
        pytest.param(
            lambda x: af.string.concat(x, "!"),
            ["x", "y", "z", "w"],
            "test",
            "test!",
            1,
            id="triple",
        ),
    ],
)
def test_retracing_inlines_equations(program, trace_values, runtime, expected, equations):
    ir = af.trace(program)(trace_values[0])
    for value in trace_values[1:]:
        ir = af.trace(ir.call)(value)
    assert [eqn.prim for eqn in ir.eqns] == [af.string.concat_p] * equations
    assert ir.call(runtime) == expected


def test_inline_calls_preserve_dataflow():
    first = af.trace(lambda x: af.string.concat(x, "1"))("X")
    second = af.trace(lambda x: af.string.concat(x, "2"))("X")
    ir = af.trace(lambda x: second.call(first.call(x)))("X")
    assert len(ir.eqns) == 2
    assert ir.eqns[1].in_tree[0] is ir.eqns[0].out_tree
    assert ir.call("start") == "start12"


@pytest.mark.parametrize(
    "program, transform, args, expected, primitive",
    [
        pytest.param(
            bracket_text,
            af.pushforward,
            (("primal",), ("tangent",)),
            ("[primal]", "tangent"),
            "pushforward_call",
            id="pushforward",
        ),
        pytest.param(
            append_bang,
            af.pullback,
            (("primal",), "cotan"),
            ("primal!", ("cotan",)),
            "pullback_call",
            id="pullback",
        ),
        pytest.param(
            angle_text,
            ft.partial(af.batch, in_axes=True),
            (["a", "b", "c"],),
            ["<a>", "<b>", "<c>"],
            "batch_call",
            id="batch",
        ),
    ],
)
@pytest.mark.parametrize("order", ["trace-transform", "transform-trace"])
@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_trace_transform_boundary(program, transform, args, expected, primitive, order, executor):
    inner = af.trace(program)("x")
    match order:
        case "trace-transform":
            ir = transform(af.trace(inner.call)("test"))
        case "transform-trace":
            ir = af.trace(transform(inner).call)(*args)
    assert len(ir.eqns) == 1
    assert ir.eqns[0].prim.name == primitive
    result = executor(ir, *args)
    assert result == expected
