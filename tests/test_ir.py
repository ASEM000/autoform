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
            assert ir.in_ir_tree[0].type is type(traced)
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

    def test_traces_literal_and_variable(self):
        def program(name):
            return af.concat("Hello, ", name)

        ir = af.trace(program)("Alice")
        assert len(ir.ir_eqns) == 1
        assert isinstance(ir.in_ir_tree, tuple)
        assert len(ir.in_ir_tree) == 1
        assert isinstance(ir.in_ir_tree[0], af.core.IRVar)
        eqn = ir.ir_eqns[0]
        assert len(eqn.in_ir_tree) == 2
        lit_candidate = eqn.in_ir_tree[0]
        assert (
            isinstance(lit_candidate, af.core.IRLit) and lit_candidate.value == "Hello, "
        ) or lit_candidate == "Hello, "
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
        assert ir.call("Alice") == "Hello, Alice!"

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

        assert isinstance(ir.in_ir_tree[0], af.core.IRLit)
        assert ir.in_ir_tree[0].value == "Hello"
        assert isinstance(ir.in_ir_tree[1], af.core.IRVar)
        assert ir.call("Hello", "Alice") == "Hello Alice"

    def test_static_input_mismatch_errors_before_execution(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        with pytest.raises(AssertionError, match="Static input mismatch"):
            ir.call("Hi", "Alice")

    @pytest.mark.asyncio(loop_scope="function")
    async def test_static_input_mismatch_errors_before_async_execution(self):
        def program(prefix, name):
            return af.format("{} {}", prefix, name)

        ir = af.trace(program, static=(True, False))("Hello", "World")

        with pytest.raises(AssertionError, match="Static input mismatch"):
            await ir.acall("Hi", "Alice")

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

        assert isinstance(ir.in_ir_tree[0], af.core.IRLit)
        assert ir.in_ir_tree[0].value is True
        assert isinstance(ir.in_ir_tree[1], af.core.IRVar)
        assert ir.call(True, "Alice") == "Hello Alice"


class TestRunIR:
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
