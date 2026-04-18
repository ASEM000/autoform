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

from dataclasses import dataclass

import pytest

import autoform as af


@dataclass(frozen=True)
class Label(af.Tag):
    name: str


@dataclass(frozen=True)
class CostTag(af.Tag):
    pass


class TestBuildIR:
    def test_trace_scalar_input_is_dynamic(self):
        def program(x):
            return af.format("{}", x)

        cases = [
            (1, 2, "2"),
            (1.5, 2.5, "2.5"),
            (True, False, "False"),
        ]

        for traced, runtime, expected in cases:
            ir = af.trace(program)(traced)
            assert isinstance(ir.in_ir_tree, tuple)
            assert len(ir.in_ir_tree) == 1
            assert isinstance(ir.in_ir_tree[0], af.core.IRVar)
            assert isinstance(ir.in_ir_tree[0].aval, af.core.TypedAVal)
            assert ir.in_ir_tree[0].aval.type is type(traced)
            assert ir.call(runtime) == expected

    def test_trace_dict_input_with_scalar_leaves(self):
        def program(payload):
            return af.format(
                "{} {} {} {}",
                payload["name"],
                payload["count"],
                payload["score"],
                payload["active"],
            )

        ir = af.trace(program)({"name": "cats", "count": 1, "score": 1.5, "active": True})
        result = ir.call({"name": "dogs", "count": 2, "score": 2.5, "active": False})
        assert result == "dogs 2 2.5 False"

    def test_trace_unsupported_input_leaf_errors(self):
        class Opaque:
            pass

        def program(x):
            return x

        with pytest.raises(AssertionError, match="Unsupported input leaf type"):
            af.trace(program)(Opaque())

    def test_trace_static_unhashable_input_errors(self):
        class Unhashable:
            __hash__ = None

        def program(x):
            return x

        with pytest.raises(TypeError):
            af.trace(program, static=True)(Unhashable())

    def test_traces_literal_and_variable(self):
        def program(name):
            return af.concat("Hello, ", name)

        ir = af.trace(program)("x0")
        assert len(ir.ir_eqns) == 1
        assert isinstance(ir.in_ir_tree, tuple)
        assert len(ir.in_ir_tree) == 1
        assert isinstance(ir.in_ir_tree[0], af.core.IRVar)
        eqn = ir.ir_eqns[0]
        assert len(eqn.in_ir_tree) == 2
        lit_candidate = eqn.in_ir_tree[0]
        assert lit_candidate == "Hello, "
        assert isinstance(eqn.in_ir_tree[1], af.core.IRVar)

    def test_format_traces_template_and_args(self):
        def program(x):
            return af.format("Hello, {}!", x)

        ir = af.trace(program)("World")
        assert len(ir.ir_eqns) == 1
        eqn = ir.ir_eqns[0]
        args, kwargs_values = eqn.in_ir_tree
        assert len(args) == 1
        assert len(kwargs_values) == 0
        assert eqn.params["template"] == "Hello, {}!"
        assert isinstance(args[0], af.core.IRVar)
        assert ir.call("x0") == "Hello, x0!"

    def test_tracing_unhashable_literal_leaf_errors(self):
        class Unhashable:
            __hash__ = None

        literal = Unhashable()

        def program(x):
            return af.format("{} {}", literal, x)

        with pytest.raises(TypeError):
            af.trace(program)("x")

    def test_traced_literal_container_is_detached_from_source_mutation(self):
        parts = ["a", "b"]

        def program(x):
            return af.format("{} {}", parts, x)

        ir = af.trace(program)("x")
        eqn = ir.ir_eqns[0]
        args, kwargs_values = eqn.in_ir_tree

        assert args[0] == ["a", "b"]
        assert args[0] is not parts
        assert len(kwargs_values) == 0
        assert ir.call("z") == "['a', 'b'] z"

        parts.append("c")

        args, _ = eqn.in_ir_tree
        assert args[0] == ["a", "b"]
        assert ir.call("z") == "['a', 'b'] z"

    def test_multiple_operations(self):
        def program(x, y):
            a = af.concat(x, y)
            b = af.format("[{}]", a)
            return b

        ir = af.trace(program)("A", "B")
        assert len(ir.ir_eqns) == 2

    def test_single_input_tree_structure(self):
        def program(x):
            return af.concat(x, x)

        ir = af.trace(program)("test")
        assert isinstance(ir.in_ir_tree, tuple)
        assert len(ir.in_ir_tree) == 1
        assert isinstance(ir.in_ir_tree[0], af.core.IRVar)

    def test_tuple_input_tree_structure(self):
        def program(a, b):
            return af.concat(a, b)

        ir = af.trace(program)("A", "B")
        assert isinstance(ir.in_ir_tree, tuple)
        assert len(ir.in_ir_tree) == 2


class TestTraceStatic:
    def test_static_inputs_become_literals(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        assert ir.in_ir_tree[0] == "Hello"
        assert isinstance(ir.in_ir_tree[1], af.core.IRVar)
        assert ir.call("Hello", "x0") == "Hello x0"

    def test_static_input_mismatch_errors_before_execution(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        with pytest.raises(AssertionError, match="Static input mismatch"):
            ir.call("Hi", "x0")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_static_input_mismatch_errors_before_async_execution(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        with pytest.raises(AssertionError, match="Static input mismatch"):
            await ir.acall("Hi", "x0")

    def test_static_spec_must_match_input_tree(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        with pytest.raises(ValueError):
            af.trace(program, static=(True, False, True))("Hello", "World")

    def test_static_bool_specializes_python_branch(self):
        def program(flag, name):
            if flag:
                return af.format("Hello {}", name)
            return af.format("Bye {}", name)

        ir = af.trace(program, static=(True, False))(True, "World")

        assert ir.in_ir_tree[0] is True
        assert isinstance(ir.in_ir_tree[1], af.core.IRVar)
        assert ir.call(True, "x0") == "Hello x0"


class TestTags:
    def test_trace_snapshots_tags_per_equation(self):
        def program(x):
            head = af.concat(x, "!")
            with af.tag(Label("planner")):
                mid = af.concat(head, "?")
                with af.tag(Label("draft"), CostTag()):
                    tail = af.concat(mid, ".")
            return tail

        ir = af.trace(program)("seed")

        assert ir.ir_eqns[0].tags == frozenset()
        assert ir.ir_eqns[1].tags == frozenset({Label("planner")})
        assert ir.ir_eqns[2].tags == frozenset({
            Label("planner"),
            Label("draft"),
            CostTag(),
        })

    def test_tag_rejects_non_tags(self):
        with pytest.raises(AssertionError, match="Expected Tag instances"):
            with af.tag("draft"):
                pass

    def test_tag_base_is_not_instantiable(self):
        with pytest.raises(AssertionError, match="Tag cannot be instantiated directly"):
            af.Tag()

    def test_tag_subclasses_must_be_hashable(self):
        with pytest.raises(AssertionError, match="Tag subclasses must be hashable"):

            class EqOnlyTag(af.Tag):
                def __eq__(self, other):
                    return isinstance(other, EqOnlyTag)

    def test_tag_extends_active_tags_and_restores_on_exit(self):
        assert af.core.active_tags.get() == ()

        with af.tag(Label("outer")) as outer_tags:
            assert outer_tags == (Label("outer"),)
            assert af.core.active_tags.get() == (Label("outer"),)

            with af.tag(Label("inner")) as inner_tags:
                assert inner_tags == (Label("inner"),)
                assert af.core.active_tags.get() == (Label("outer"), Label("inner"))

            assert af.core.active_tags.get() == (Label("outer"),)

        assert af.core.active_tags.get() == ()

    def test_ireqn_tags_input_is_tuple(self):
        prim = af.core.Prim("tag_tuple")
        eqn = af.core.IREqn(prim, (), (), tags=(Label("draft"),))

        assert eqn.tags == frozenset({Label("draft")})

        with pytest.raises(AssertionError):
            af.core.IREqn(prim, (), (), tags=[Label("draft")])

    def test_bind_reinstalls_equation_tags(self):
        probe_p = af.core.Prim("tag_probe")

        def abstract_probe(x):
            del x
            return af.core.TypedAVal(str)

        def impl_probe(x):
            names = sorted(tag.name for tag in af.core.active_tags.get() if isinstance(tag, Label))
            return f"{','.join(names)}|{x}"

        af.core.abstract_rules.set(probe_p, abstract_probe)
        af.core.impl_rules.set(probe_p, impl_probe)

        def program(x):
            with af.tag(Label("draft"), Label("cost")):
                return probe_p.bind(x)

        ir = af.trace(program)("seed")

        assert ir.call("hello") == "cost,draft|hello"

        with af.tag(Label("runtime")):
            assert ir.call("hello") == "cost,draft,runtime|hello"

        assert ir.ir_eqns[0].tags == frozenset({Label("draft"), Label("cost")})

    def test_using_preserves_tags(self):
        def program(x):
            with af.tag(Label("draft")):
                return af.concat(x, "!")

        ir = af.trace(program)("seed")
        eqn = ir.ir_eqns[0]

        new_eqn = eqn.using(collection="debug")

        assert new_eqn.params["collection"] == "debug"
        assert new_eqn.tags == frozenset({Label("draft")})

    def test_calling_existing_ir_while_tracing_unions_runtime_and_equation_tags(self):
        def inner_program(x):
            with af.tag(Label("inner")):
                return af.concat(x, "!")

        inner_ir = af.trace(inner_program)("seed")

        def outer_program(x):
            with af.tag(Label("outer")):
                return inner_ir.call(x)

        outer_ir = af.trace(outer_program)("seed")

        assert outer_ir.ir_eqns[0].tags == frozenset({Label("inner"), Label("outer")})


class TestRunIR:
    def test_walk_yields_eqn_inputs_and_return_final_output(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("hello")
        gen = ir.walk("world")

        ir_eqn, in_values = next(gen)
        assert ir_eqn.prim.name == "concat"
        assert in_values == ("world", "!")

        done, out = gen.send("world!")
        assert done is None
        assert out == "world!"

    def test_walk_allows_external_step_execution(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("hello")
        gen = ir.walk("world")

        ir_eqn, in_values = next(gen)
        assert ir_eqn.prim.name == "concat"
        assert in_values == ("world", "!")
        out_values = ir_eqn.bind(("there", "!"), **ir_eqn.params)

        done, out = gen.send(out_values)
        assert done is None
        assert out == "there!"

    def test_basic_execution(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("hello")
        result = ir.call("world")
        assert result == "world!"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_basic_execution_async(self):
        def program(x):
            return af.concat(x, "!")

        ir = af.trace(program)("hello")
        result = await ir.acall("world")
        assert result == "world!"

    def test_chained_operations(self):
        def program(x):
            step1 = af.concat(x, x)
            step2 = af.format("[{}]", step1)
            return step2

        ir = af.trace(program)("A")
        result = ir.call("B")
        assert result == "[BB]"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_chained_operations_async(self):
        def program(x):
            step1 = af.concat(x, x)
            step2 = af.format("[{}]", step1)
            return step2

        ir = af.trace(program)("A")
        result = await ir.acall("B")
        assert result == "[BB]"

    def test_multiple_args(self):
        def program(a, b):
            return af.format("{} + {}", a, b)

        ir = af.trace(program)("x", "y")
        result = ir.call("1", "2")
        assert result == "1 + 2"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_multiple_args_async(self):
        def program(a, b):
            return af.format("{} + {}", a, b)

        ir = af.trace(program)("x", "y")
        result = await ir.acall("1", "2")
        assert result == "1 + 2"
