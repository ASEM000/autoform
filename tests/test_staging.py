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

import re
from types import SimpleNamespace

import pytest

import autoform as af
import autoform.extend as afe


class TestTraceValuePythonOps:
    @pytest.mark.parametrize(
        ("description", "program"),
        [
            ("truthiness", lambda x: "yes" if x else "no"),
            ("string coercion", lambda x: str(x)),
            ("string formatting", lambda x: f"{x}"),
            ("iteration", lambda x: list(x)),
            ("integer-index coercion", lambda x: range(x)),
            ("integer coercion", lambda x: int(x)),
            ("float coercion", lambda x: float(x)),
            ("bytes coercion", lambda x: bytes(x)),
            ("indexing", lambda x: x[0]),
            ("membership testing", lambda x: "a" in x),
        ],
    )
    def test_host_python_operations_on_traced_values_error(self, description, program):
        with pytest.raises(
            TypeError,
            match=rf"Cannot use {re.escape(description)} on a traced value\.",
        ):
            af.trace(program)("seed")

    def test_len_on_traced_value_has_informative_error(self):
        def program(x):
            return len(x)

        with pytest.raises(
            TypeError,
            match=r"Cannot use length on a traced value\..*"
            r"Python length needs a concrete runtime value.*"
            r"af\.trace\(\.\.\., static=\.\.\.\)",
        ):
            af.trace(program)("seed")


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


class TestConstfold:
    def test_evaluates_concrete_equations(self):
        def program(x):
            prefix = af.concat("A", "B")
            header = af.format("[{}]", prefix)
            return af.concat(header, x)

        ir = af.trace(program)("seed")
        folded = af.constfold(ir)

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat", "format", "concat"]
        assert [eqn.prim.name for eqn in folded.eqns] == ["concat"]
        assert folded.eqns[0].in_tree[0] == "[AB]"
        assert folded.call("C") == "[AB]C"

    def test_preserves_static_input_checks(self):
        def program(prefix, x):
            header = af.concat(prefix, ": ")
            return af.concat(header, x)

        ir = af.trace(program, static=(True, False))("Q", "seed")
        folded = af.constfold(ir)

        assert [eqn.prim.name for eqn in folded.eqns] == ["concat"]
        assert folded.eqns[0].in_tree[0] == "Q: "
        assert folded.call("Q", "hello") == "Q: hello"
        with pytest.raises(AssertionError, match="Static input mismatch"):
            folded.call("R", "hello")

    def test_cond_selects_concrete_equations(self):
        def program(x):
            prefix = af.concat("A", "B")
            header = af.format("[{}]", prefix)
            return af.concat(header, x)

        ir = af.trace(program)("seed")
        folded = af.constfold(ir, cond=lambda e: e.prim.name == "concat")

        assert [eqn.prim.name for eqn in folded.eqns] == ["format", "concat"]
        assert folded.eqns[0].in_tree[0][0] == "AB"
        assert folded.call("C") == "[AB]C"

    def test_cond_blocks_concrete_equation_evaluation(self):
        counter_p = af.core.Prim("constfold_counter_probe")
        calls = 0

        def counter(x):
            return counter_p.bind(x)

        def impl_counter(in_tree):
            nonlocal calls
            calls += 1
            return f"{in_tree}!"

        def abstract_counter(in_tree):
            del in_tree
            return af.core.StrAVal()

        af.core.impl_rules.set(counter_p, impl_counter)
        af.core.abstract_rules.set(counter_p, abstract_counter)

        def program(x):
            prefix = counter("A")
            return af.concat(prefix, x)

        ir = af.trace(program)("seed")
        folded = af.constfold(ir, cond=lambda e: e.prim is not counter_p)

        assert calls == 0
        assert [eqn.prim.name for eqn in folded.eqns] == [
            "constfold_counter_probe",
            "concat",
        ]
        assert folded.call("B") == "A!B"
        assert calls == 1

    def test_non_constfold_registration_cannot_be_overridden_by_cond(self):
        counter_p = afe.register_non_constfold(af.core.Prim("non_constfold_probe"))
        calls = 0

        def counter(x):
            return counter_p.bind(x)

        def impl_counter(in_tree):
            nonlocal calls
            calls += 1
            return f"{in_tree}!"

        af.core.impl_rules.set(counter_p, impl_counter)
        af.core.abstract_rules.set(counter_p, lambda _: af.core.StrAVal())

        def program(x):
            prefix = counter("A")
            return af.concat(prefix, x)

        ir = af.trace(program)("seed")
        folded = af.constfold(ir, cond=lambda _: True)

        assert calls == 0
        assert [eqn.prim.name for eqn in folded.eqns] == [
            "non_constfold_probe",
            "concat",
        ]
        assert folded.call("B") == "A!B"
        assert calls == 1

    def test_lm_call_remains_staged_when_inputs_are_concrete(self):
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
            rubric = af.lm_call(
                [{"role": "user", "content": "make a rubric"}],
                model="test-model",
            )
            return af.format("{}: {}", rubric, question)

        client = Client()
        with af.lm_client(client):
            ir = af.trace(program)("seed")
            folded = af.constfold(ir, cond=lambda _: True)

            assert client.calls == 0
            assert [eqn.prim.name for eqn in folded.eqns] == ["lm_call", "format"]
            assert folded.call("question") == "rubric: question"

        assert client.calls == 1


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

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.eqns[0].in_tree[0] == "AB"
        assert ir.call("C") == "ABC"

    def test_fold_block_allows_nested_interpreter_inside_trace(self):
        def program(x):
            with af.memoize():
                with af.fold():
                    prefix = af.concat("A", "B")
            return af.concat(prefix, x)

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
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

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat"]
        assert ir.eqns[0].in_tree[0] == "Q: "
        assert ir.call("Q", "hello") == "Q: hello"

    def test_tracing_resumes_after_static_block(self):
        def program(x):
            with af.fold():
                prefix = af.concat("a", "b")
                prefix = af.format("[{}]", prefix)
            value = af.concat(prefix, x)
            return af.concat(value, "!")

        ir = af.trace(program)("seed")

        assert [eqn.prim.name for eqn in ir.eqns] == ["concat", "concat"]
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
        with af.lm_client(client):
            ir = af.trace(program)("seed")

        assert client.calls == 1
        assert [eqn.prim.name for eqn in ir.eqns] == ["format"]
        assert ir.call("question") == "rubric: question"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_dynamic_trace_dispatch_stages_primitive(self):
        async_probe_p = af.core.Prim("async_dynamic_fold_probe")

        def abstract_async_probe(in_tree):
            del in_tree
            return af.core.StrAVal()

        af.core.abstract_rules.set(async_probe_p, abstract_async_probe)

        with af.core.using_interpreter(af.core.TraceInterpreter()) as tracer:
            result = await async_probe_p.abind("literal")

        assert isinstance(result, af.core.TraceBox)
        assert [eqn.prim.name for eqn in tracer.eqns] == ["async_dynamic_fold_probe"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_async_fold_trace_dispatch_evaluates_primitive(self):
        async_probe_p = af.core.Prim("async_fold_probe")

        async def aimpl_async_probe(in_tree):
            return af.concat(in_tree, "!")

        af.core.impl_rules.aset(async_probe_p, aimpl_async_probe)

        with af.core.using_interpreter(af.core.TraceInterpreter()) as tracer:
            with af.fold():
                result = await async_probe_p.abind("literal")

        assert result == "literal!"
        assert tracer.eqns == []
