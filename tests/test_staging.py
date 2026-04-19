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

from types import SimpleNamespace

import pytest

import autoform as af


class CountingInterpreter(af.core.Interpreter):
    def __init__(self):
        self.parent = af.core.active_interpreter.get()
        self.calls = 0

    def interpret(self, prim, in_tree, /, **params):
        self.calls += 1
        return self.parent.interpret(prim, in_tree, **params)

    async def ainterpret(self, prim, in_tree, /, **params):
        self.calls += 1
        return await self.parent.ainterpret(prim, in_tree, **params)


class TestFold:
    def test_fold_block_is_noop_outside_trace(self):
        counter = CountingInterpreter()

        with af.core.using_interpreter(counter):
            with af.fold():
                result = af.concat("A", "B")

        assert result == "AB"
        assert counter.calls == 1

    def test_fold_block_evaluates_literals_during_trace(self):
        def program(x):
            with af.fold():
                prefix = af.concat("A", "B")
            return af.concat(prefix, x)

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.ir_eqns] == ["concat"]
        assert ir.ir_eqns[0].in_ir_tree[0] == "AB"
        assert ir.call("C") == "ABC"

    def test_fold_block_allows_nested_interpreter_inside_trace(self):
        def program(x):
            with af.memoize():
                with af.fold():
                    prefix = af.concat("A", "B")
            return af.concat(prefix, x)

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.ir_eqns] == ["concat"]
        assert ir.call("C") == "ABC"

    def test_fold_block_rejects_dynamic_trace_values(self):
        def program(x):
            with af.fold():
                return af.concat(x, "!")

        with pytest.raises(AssertionError, match="depends on traced value"):
            af.trace(program)("seed")

    def test_fold_block_rejects_dynamic_trace_values_in_params(self):
        param_probe_p = af.core.Prim("fold_param_probe")

        def param_probe(dynamic):
            return param_probe_p.bind("literal", dynamic=dynamic)

        def impl_param_probe(in_tree, *, dynamic):
            del in_tree
            return dynamic

        af.core.impl_rules.set(param_probe_p, impl_param_probe)

        def program(x):
            with af.fold():
                return param_probe(x)

        with pytest.raises(AssertionError, match="depends on traced value"):
            af.trace(program)("seed")

    def test_fold_block_rejects_dynamic_trace_values_in_output(self):
        output_probe_p = af.core.Prim("fold_output_probe")
        captured = {}

        def impl_output_probe(in_tree):
            del in_tree
            return captured["value"]

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
                header = af.concat(prefix, ": ")
            return af.concat(header, x)

        ir = af.trace(program, static=(True, False))("Q", "seed")

        assert [eqn.prim.name for eqn in ir.ir_eqns] == ["concat"]
        assert ir.ir_eqns[0].in_ir_tree[0] == "Q: "
        assert ir.call("Q", "hello") == "Q: hello"

    def test_tracing_resumes_after_static_block(self):
        def program(x):
            with af.fold():
                prefix = af.concat("a", "b")
                prefix = af.format("[{}]", prefix)
            value = af.concat(prefix, x)
            return af.concat(value, "!")

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.ir_eqns] == ["concat", "concat"]
        assert ir.call("c") == "[ab]c!"

    def test_fold_block_evaluates_lm_call_during_trace(self):
        class Response:
            def __init__(self):
                self.choices = [SimpleNamespace(message=SimpleNamespace(content="rubric"))]

        class Client:
            def __init__(self):
                self.calls = 0

            def completion(self, **kwargs):
                self.calls += 1
                return Response()

            async def acompletion(self, **kwargs):
                self.calls += 1
                return Response()

        def program(question):
            with af.fold():
                rubric = af.lm_call(
                    [{"role": "user", "content": "make a rubric"}],
                    model="test-model",
                )
            return af.format("{}: {}", rubric, question)

        client = Client()
        with af.using_client(client):
            ir = af.trace(program)("seed")

        assert client.calls == 1
        assert [eqn.prim.name for eqn in ir.ir_eqns] == ["format"]
        assert ir.call("question") == "rubric: question"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_dynamic_trace_dispatch_stages_primitive(self):
        async_probe_p = af.core.Prim("async_dynamic_fold_probe")

        def abstract_async_probe(in_tree):
            del in_tree
            return af.core.TypedAVal(str)

        af.core.abstract_rules.set(async_probe_p, abstract_async_probe)

        with af.core.using_interpreter(af.core.TracingInterpreter()) as tracer:
            result = await async_probe_p.abind("literal")

        assert isinstance(result, af.core.IRVar)
        assert [eqn.prim.name for eqn in tracer.ir_eqns] == ["async_dynamic_fold_probe"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_fold_trace_dispatch_evaluates_primitive(self):
        async_probe_p = af.core.Prim("async_fold_probe")

        async def aimpl_async_probe(in_tree):
            return af.concat(in_tree, "!")

        af.core.impl_rules.aset(async_probe_p, aimpl_async_probe)

        with af.core.using_interpreter(af.core.TracingInterpreter()) as tracer:
            with af.fold():
                result = await async_probe_p.abind("literal")

        assert result == "literal!"
        assert tracer.ir_eqns == []
