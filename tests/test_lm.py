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
import json
from contextlib import nullcontext

import optree
import pytest
from litellm import ResponsesAPIResponse

import autoform as af
from autoform.lm import describe, parse
from autoform.utils import tree
from tests import aexecute, execute


class RenderClient:
    def __init__(self, render):
        self.render = render

    def responses(self, *, input, model, **kwargs):
        return fake_response(self.render(input))

    async def aresponses(self, **kwargs):
        return self.responses(**kwargs)


def fill_program(schema, *, model="m1"):
    def program(prompt, model=model):
        return af.lm.fill({"prompt": prompt, "output": schema}, model=model)["output"]

    return program


def fake_response(content):
    return ResponsesAPIResponse(
        id="test",
        created_at=0,
        model="test",
        object="response",
        output=[
            dict(
                type="message",
                id="test",
                role="assistant",
                status="completed",
                content=[dict(type="output_text", text=content, annotations=[])],
            ),
        ],
    )


def schema_response(value, text):
    properties = text["format"]["schema"].get("properties", {})
    if "output" in properties:
        value = {"output": value}
    return fake_response(json.dumps(value))


def parse_change_prompt(content):
    input_text, change_text = content.removeprefix("INPUT: ").split(" INPUT CHANGE: ")
    return json.loads(input_text), json.loads(change_text)


def fill_values(content):
    return json.loads(content)["values"]


def fill_request_text(content):
    return fill_values(content)["context"]["request"]


class EchoRouter:
    def responses(self, *, input, model, text=None, **kwargs):
        assert kwargs == {}
        content = f"{model}|{input}"
        if text is not None:
            return schema_response(content, text)
        return fake_response(content)

    async def aresponses(self, **kwargs):
        return self.responses(**kwargs)


class SchemaRouter(EchoRouter):
    __slots__ = ["text_formats"]

    def __init__(self):
        self.text_formats = []

    def responses(self, *, input: str, model: str, text, **kwargs):
        assert kwargs == {}
        self.text_formats.append(text)
        return schema_response(
            {
                "text": f"{model}|{fill_values(input)['prompt']}",
                "score": 0.5,
            },
            text,
        )


class SchemaGradientRouter(EchoRouter):
    def __init__(self):
        self.calls = []

    def responses(self, *, input, model, text, **kwargs):
        assert kwargs == {}
        self.calls.append(dict(input=input, model=model, text=text))
        values = fill_values(input)
        if "context" not in values:
            return schema_response(
                {"text": "Recursion calls itself.", "score": 0.92},
                text,
            )
        if values["context"]["instruction"] == af.lm.PUSH_SYSTEM_PROMPT:
            return schema_response(
                {"output": {"text": "output change", "score": 0.25}},
                text,
            )
        return schema_response(
            {"0": {"prompt": "input feedback"}, "1": "model feedback"},
            text,
        )


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("schema", "value"),
    [
        pytest.param(af.lm.Str(), 'Hello "world"!', id="string"),
        pytest.param(af.lm.Str(min=1, desc="Answer text."), "hello", id="described-string"),
        pytest.param(af.lm.Int(), 2, id="integer"),
        pytest.param(af.lm.Float(), 0.5, id="float"),
        pytest.param(af.lm.Bool(), True, id="boolean"),
        pytest.param(af.lm.Enum("yes", "no"), "yes", id="enum"),
    ],
)
def test_fill_passes_scalar_schemas_to_client(executor, schema, value):
    class ScalarClient(EchoRouter):
        def responses(self, *, text, **kwargs):
            assert text["format"]["schema"] == describe({"output": schema})
            return schema_response(value, text)

    program = fill_program(schema, model="echo")
    ir = af.trace(program)("seed")
    assert af.core.avalof(schema) == ir.out_tree.aval
    with af.lm.client(ScalarClient()):
        assert program("hello") == value
        assert executor(ir, "hello") == value


def test_fill_uses_runtime_model_with_structured_output():
    answer = {
        "text": af.lm.Str(desc="Short text."),
        "score": af.lm.Float(),
    }

    fill = fill_program(answer)

    def program(prompt, model):
        return af.string.format("{text}", text=fill(prompt, model)["text"])

    ir = af.trace(program)("test", "gpt-5.5")

    with af.lm.client(SchemaRouter()):
        assert ir.call("hello", "m1") == "m1|hello"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_fill_pushforward_preserves_non_string_literals(executor):
    class FillClient(EchoRouter):
        def responses(self, *, input, text, **kwargs):
            value = {"answer": "filled"}
            if "context" in fill_values(input):
                value = {"output": value}
            return schema_response(value, text)

    def program(x):
        return af.lm.fill(
            {"prompt": x, "output": {"answer": af.lm.Str(), "fixed": 1.0}},
            model="echo",
        )["output"]

    ir = af.pushforward(af.trace(program)("seed"))
    with af.lm.client(FillClient()):
        primal, tangent = executor(ir, ("q",), ("dq",))
    assert primal == {"answer": "filled", "fixed": 1.0}
    assert tangent["answer"] == "filled"
    assert af.core.materialize_zeros(tangent["fixed"]) == 0.0


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_fill_uses_values_schema_envelope(executor):
    class FillClient(EchoRouter):
        __slots__ = ["calls"]

        def __init__(self):
            self.calls = []

        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            self.calls.append(dict(input=input, model=model, text=text))
            return fake_response(json.dumps({"answer": "four", "score": 0.9}))

    def program(question):
        return af.lm.fill(
            {
                "question": question,
                "answer": af.lm.Str(desc="Answer text."),
                "score": af.lm.Float(min=0, max=1),
            },
            model="m1",
        )

    ir = af.trace(program)("seed")
    assert [eqn.prim for eqn in ir.eqns] == [
        af.control.stop_gradient_p,
        af.lm.fill_p,
    ]
    call = ir.eqns[-1]
    assert call.params == {
        "static_tree": {
            "question": None,
            "answer": af.lm.Str(),
            "score": af.lm.Float(min=0, max=1),
        }
    }
    assert call.in_tree == (
        {"question": ir.in_tree[0], "answer": None, "score": None},
        {
            "question": None,
            "answer": af.lm.Str(desc="Answer text."),
            "score": af.lm.Float(min=0, max=1),
        },
        ir.eqns[0].out_tree,
    )
    assert call.out_tree["question"] is None
    assert ir.out_tree["question"] is ir.in_tree[0]
    assert all(isinstance(x, af.stage.Var) for x in tree.leaves(call.out_tree))

    client = FillClient()
    with af.lm.client(client):
        assert executor(ir, "1+1?") == {"question": "1+1?", "answer": "four", "score": 0.9}

    call = client.calls[-1]
    assert call["model"] == "m1"
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["name"] == "autoform_schema"
    assert call["text"]["format"]["strict"] is True
    assert json.loads(call["input"]) == {
        "values": {"question": "1+1?"},
        "schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    }
    assert call["text"]["format"]["schema"] == {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Answer text."},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "score"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "template, response, expected",
    [
        pytest.param(af.lm.Str(), "filled", "filled", id="scalar"),
        pytest.param(
            ("fixed", [af.lm.Str(), None, {}]),
            {"1": {"0": "filled"}},
            ("fixed", ["filled", None, {}]),
            id="nested",
        ),
        pytest.param(("fixed", 0.5, 2, False), None, ("fixed", 0.5, 2, False), id="no-holes"),
        pytest.param((None, [], {}), None, (None, [], {}), id="empty-containers"),
    ],
)
def test_fill_preserves_tree_structure(executor, template, response, expected):
    calls = []

    def render(input):
        calls.append(input)
        return json.dumps(response)

    ir = af.trace(lambda: af.lm.fill(template, model="echo"))()
    with af.lm.client(RenderClient(render=render)):
        assert executor(ir) == expected
    assert len(calls) == (response is not None)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "x, json_type",
    [("context", "string"), (0.5, "number"), (2, "integer"), (False, "boolean")],
)
def test_fill_preserves_custom_pytree_and_json_property_names(executor, x, json_type):
    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        fields: object
        metadata: object

    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            content = json.loads(input)
            assert content["values"] == {"fields": {"0": x}}
            assert content["schema"]["properties"]["fields"]["properties"] == {
                "0": {"type": json_type},
            }
            schema = text["format"]["schema"]
            assert schema["properties"]["fields"]["properties"] == {"0_": {"type": "string"}}
            return fake_response(json.dumps({"fields": {"0_": "filled"}}))

    def program(x):
        return af.lm.fill(Answer({0: x, "0": af.lm.Str()}, (None, [], {})), model="echo")

    ir = af.trace(program)(x)
    with af.lm.client(FillClient()):
        assert program(x) == Answer({0: x, "0": "filled"}, (None, [], {}))
        assert executor(ir, x) == Answer({0: x, "0": "filled"}, (None, [], {}))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("values", [["a", "b"], [0.5, 1.0]], ids=["string", "float"])
def test_fill_batches_context(executor, values):
    class FillClient(EchoRouter):
        __slots__ = ["questions"]

        def __init__(self):
            self.questions = []

        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            assert model == "m1"
            assert text["format"]["strict"] is True
            question = json.loads(input)["values"]["question"]
            self.questions.append(question)
            return fake_response(json.dumps({"answer": f"filled:{question}"}))

    def program(question):
        return af.lm.fill({"question": question, "answer": af.lm.Str()}, model="m1")

    ir = af.batch(af.trace(program)(values[0]))
    client = FillClient()
    with af.lm.client(client):
        assert executor(ir, values) == {
            "question": values,
            "answer": [f"filled:{value}" for value in values],
        }
    assert client.questions == values


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("batch_order", [None, "before", "after"])
@pytest.mark.parametrize(
    "question, d_question, json_type",
    [("q", "dq", "string"), (0.5, 0.25, "number")],
)
def test_fill_pushforward_uses_tangent_context(
    executor,
    batch_order,
    question,
    d_question,
    json_type,
):
    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            values = fill_values(input)
            if "context" in values:
                original, change = parse_change_prompt(values["context"]["request"])
                for content in (original, change):
                    assert content["schema"]["properties"]["question"] == {"type": json_type}
                question = original["values"]["question"]
                d_question = change["values"]["question"]
                return schema_response(
                    {"answer": f"filled:{question}->{d_question}"},
                    text,
                )
            question = values["question"]
            return schema_response({"answer": f"filled:{question}"}, text)

    def program(question):
        return af.lm.fill({"question": question, "answer": af.lm.Str()}, model="m1")

    ir = af.trace(program)(question)
    ir = af.pushforward(af.batch(ir) if batch_order == "before" else ir)
    if batch_order == "after":
        ir = af.batch(ir)
    args = ((question,), (d_question,))
    expected = (
        {"question": question, "answer": f"filled:{question}"},
        {"question": d_question, "answer": f"filled:{question}->{d_question}"},
    )
    if batch_order is not None:
        args, expected = tree.map(lambda x: [x, x], (args, expected))
    client = FillClient()
    with af.lm.client(client):
        assert executor(ir, *args) == expected


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("description", ["Score clarity", "Estimate price"])
def test_fill_pushforward_preserves_task_schema(executor, description):
    schema = af.lm.Float(min=0, max=10, desc=description)
    calls = []

    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            calls.append(input)
            values = fill_values(input)
            response_schema = text["format"]["schema"]
            original_schema = describe({"y": schema})
            if "context" not in values:
                assert response_schema == original_schema
                return schema_response({"y": 8.0}, text)

            context = values["context"]
            assert json.loads(context["output_schema"]) == original_schema
            original, change = parse_change_prompt(context["request"])
            assert original["values"] == {"x": "x"}
            assert change["values"] == {"x": "dx"}
            assert response_schema["properties"]["output"] == describe({"y": af.lm.Float()})
            return schema_response({"y": -2.0}, text)

    def program(x):
        return af.lm.fill({"x": x, "y": schema}, model="m1")

    ir = af.pushforward(af.trace(program)("seed"))
    with af.lm.client(FillClient()):
        assert executor(ir, ("x",), ("dx",)) == (
            {"x": "x", "y": 8.0},
            {"x": "dx", "y": -2.0},
        )
    assert len(calls) == 2
    assert schema == af.lm.Float(min=0, max=10, desc=description)


@pytest.mark.parametrize(
    "t_tree",
    [
        pytest.param({"x": "dx", "y": af.lm.Float(min=0, desc="Score clarity")}, id="constraint"),
        pytest.param({"x": "dx", "y": af.lm.Str(desc="Score clarity")}, id="schema-type"),
        pytest.param({"x": "dx", "y": 0.0}, id="schema-to-leaf"),
        pytest.param({"x": ["dx"], "y": af.lm.Float(desc="Score clarity")}, id="structure"),
    ],
)
def test_fill_pushforward_rejects_mismatched_specs(t_tree):
    p_tree = {"x": "x", "y": af.lm.Float(desc="Score clarity")}
    ir = af.pushforward(af.trace(lambda x: af.lm.fill(x, model="m1"))(p_tree))
    with pytest.raises(ValueError):
        ir.call((p_tree,), (t_tree,))


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("description", [None, "description"], ids=["absent", "dynamic"])
def test_fill_pullback_retains_spec_constraints(executor, description):
    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            if "context" not in fill_values(input):
                return schema_response({"answer": "generated"}, text)
            value = {"1": ""}
            properties = text["format"]["schema"]["properties"]["output"]["properties"]
            assert ("2" in properties) == (description is not None)
            if description is not None:
                value["2"] = {"answer": "revised"}
            return schema_response(value, text)

    spec = af.lm.Str(min=1, max=100, desc=description)
    ir = af.pullback(af.trace(lambda x: af.lm.fill({"answer": x}, model="m1"))(spec))
    expected = af.lm.Str(min=1, max=100, desc="revised" if description is not None else None)
    with af.lm.client(FillClient()):
        assert executor(ir, (spec,), {"answer": "feedback"}) == (
            {"answer": "generated"},
            (expected,),
        )


def test_fill_pushforward_accepts_equal_static_metadata():
    class FillClient(EchoRouter):
        def responses(self, *, text, **kwargs):
            return schema_response({"y": 1.0}, text)

    p_tree = {"x": "x", "y": af.lm.Float(desc="Score clarity")}
    t_tree = {"x": "dx", "y": af.lm.Float(desc="Score clarity")}
    assert p_tree["y"] is not t_tree["y"]
    ir = af.pushforward(af.trace(lambda x: af.lm.fill(x, model="m1"))(p_tree))
    with af.lm.client(FillClient()):
        assert ir.call((p_tree,), (t_tree,)) == ({"x": "x", "y": 1.0}, {"x": "dx", "y": 1.0})


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("transform", [af.pushforward, af.pullback], ids=["pf", "pb"])
@pytest.mark.parametrize("batch_order", [None, "before", "after"])
def test_fill_transforms_traced_descriptions(executor, transform, batch_order):
    calls = []

    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            values = fill_values(input)
            schema = text["format"]["schema"]
            calls.append(schema)
            if "context" not in values:
                assert schema["properties"]["answer"] == {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "description",
                }
                return schema_response({"answer": "generated"}, text)
            if transform is af.pushforward:
                context = values["context"]
                assert json.loads(context["desc_change"]) == {"answer": "change"}
                assert (
                    json.loads(context["output_schema"])["properties"]["answer"]["description"]
                    == "description"
                )
                assert schema["properties"]["output"]["properties"]["answer"] == {"type": "string"}
                return schema_response({"answer": "changed"}, text)
            request = values["context"]["request"]
            original = json.loads(request.removeprefix("INPUT: ").split(" OUTPUT: ")[0])
            assert original["values"]["2"] == {"answer": "description"}
            return schema_response({"1": "", "2": {"answer": "revised"}}, text)

    def program(prompt):
        return af.lm.fill({"answer": af.lm.Str(min=1, max=100) @ prompt}, model="m1")["answer"]

    ir = af.trace(program)("description")
    ir = transform(af.batch(ir) if batch_order == "before" else ir)
    ir = af.batch(ir) if batch_order == "after" else ir
    change = ("change",) if transform is af.pushforward else "feedback"
    args = (("description",), change)
    expected = ("generated", "changed" if transform is af.pushforward else ("revised",))
    if batch_order is not None:
        args = tree.map(lambda x: [x, x], args)
        expected = tree.map(lambda x: [x, x], expected)
    with af.lm.client(FillClient()):
        assert executor(ir, *args) == expected
    expected_calls = 2 if batch_order is None else 4
    if batch_order == "before" and transform is af.pullback:
        expected_calls = 6
    assert len(calls) == expected_calls


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("generated", [False, True], ids=["copied-only", "copied-and-generated"])
@pytest.mark.parametrize("batch_order", [None, "before", "after"])
def test_fill_pullback_preserves_context_identity(executor, generated, batch_order):
    calls = []

    def render(input):
        calls.append(input)
        values = fill_values(input)
        if "context" in values:
            prompt = values["context"]["request"]
            feedback = json.loads(prompt.split(" OUTPUT FEEDBACK: ")[1])
            assert feedback["values"] == {"answer": "generated feedback"}
            return json.dumps({
                "output": {"0": {"question": "model feedback"}, "1": "ignored"},
            })
        return json.dumps({"answer": "filled"})

    def program(question):
        out = af.lm.fill({"question": question, "answer": af.lm.Str()}, model="m1")
        return (out["question"], out["answer"]) if generated else out["question"]

    ir = af.trace(program)("seed")
    ir = af.pullback(af.batch(ir) if batch_order == "before" else ir)
    if batch_order == "after":
        ir = af.batch(ir)
    cotangent = ("direct feedback", "generated feedback") if generated else "direct feedback"
    expected = (
        ("q", "filled") if generated else "q",
        ("direct feedbackmodel feedback" if generated else "direct feedback",),
    )
    args = (("q",), cotangent)
    if batch_order is not None:
        args, expected = tree.map(lambda x: [x, x], (args, expected))
    with af.lm.client(RenderClient(render=render)):
        assert executor(ir, *args) == expected
    # Pullback of batch_call replays the forward pass before transposing it.
    count = 1 + generated + (batch_order == "before")
    assert len(calls) == count * (1 if batch_order is None else 2)


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_fill_pullback_uses_original_output_schema_for_feedback(executor):
    class FillGradientClient(EchoRouter):
        __slots__ = ["calls"]

        def __init__(self):
            self.calls = []

        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            self.calls.append(dict(input=input, model=model, text=text))
            properties = text["format"]["schema"]["properties"]
            if "answer" in properties:
                return fake_response(
                    json.dumps({
                        "answer": "Recursion calls itself.",
                        "score": 0.92,
                    })
                )
            return schema_response(
                {
                    "0": {"question": "question feedback"},
                    "1": "model feedback",
                },
                text,
            )

    def program(question):
        filled = af.lm.fill(
            {
                "question": question,
                "answer": af.lm.Str(),
                "score": af.lm.Float(min=0, max=1),
            },
            model="m1",
        )
        return af.string.format("{answer}", answer=filled["answer"])

    ir = af.pullback(af.trace(program)("seed"))
    client = FillGradientClient()
    with af.lm.client(client):
        assert executor(ir, ("Explain recursion.",), "too terse") == (
            "Recursion calls itself.",
            ("question feedback",),
        )

    backward = client.calls[-1]
    prompt = fill_request_text(backward["input"])
    input_text, rest = prompt.removeprefix("INPUT: ").split(" OUTPUT: ")
    output_text, feedback_text = rest.split(" OUTPUT FEEDBACK: ")
    assert json.loads(input_text)["values"] == {
        "0": {"question": "Explain recursion."},
        "1": "m1",
    }
    assert json.loads(output_text) == {
        "values": {"answer": "Recursion calls itself.", "score": 0.92},
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["answer", "score"],
            "additionalProperties": False,
        },
    }
    assert json.loads(feedback_text) == {
        "values": {"answer": "too terse", "score": 0.0},
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["answer", "score"],
            "additionalProperties": False,
        },
    }
    schema = backward["text"]["format"]["schema"]["properties"]["output"]
    assert schema["properties"]["0"]["properties"]["question"] == {
        "type": "string",
        "description": "Input cotangent at (0, 'question'), original value 'Explain recursion.'.",
    }


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("generated", [False, True], ids=["copied-only", "copied-and-generated"])
def test_fill_pullback_generates_input_cotangent_types(executor, generated):
    calls = []

    class FeedbackClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            calls.append(input)
            values = fill_values(input)
            if "x" in values:
                return schema_response({"y": "filled"}, text)
            prompt = values["context"]["request"]
            content = json.loads(prompt.removeprefix("INPUT: ").split(" OUTPUT: ")[0])
            assert content["schema"]["properties"]["0"]["properties"]["x"] == {"type": "number"}
            schema = text["format"]["schema"]["properties"]["output"]
            assert schema["properties"]["0"]["properties"]["x"] == {
                "type": "number",
                "description": "Input cotangent at (0, 'x'), original value 0.5.",
            }
            return schema_response({"0": {"x": -0.5}, "1": "model feedback"}, text)

    def program(x):
        out = af.lm.fill({"x": x, "y": af.lm.Str()}, model="m1")
        return out if generated else out["x"]

    ir = af.pullback(af.trace(program)(0.5))
    cotangent = {"x": 0.25, "y": "feedback"} if generated else 0.25
    with af.lm.client(FeedbackClient()):
        out, (dx,) = executor(ir, (0.5,), cotangent)
    assert out == ({"x": 0.5, "y": "filled"} if generated else 0.5)
    assert dx == (-0.25 if generated else 0.25)
    assert len(calls) == 1 + generated


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("cotangent", [-1.0, 0.0, 1.0])
def test_fill_pullback_propagates_numeric_sensitivities(executor, cotangent):
    calls = []

    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            calls.append(input)
            values = fill_values(input)
            if "x" in values:
                return schema_response({"y": 2.0 * values["x"]}, text)

            context = values["context"]
            instruction = context["instruction"]
            assert instruction == af.lm.GRAD_SYSTEM_PROMPT
            input_text, rest = context["request"].removeprefix("INPUT: ").split(" OUTPUT: ")
            output_text, feedback_text = rest.split(" OUTPUT FEEDBACK: ")
            original, output, feedback = map(json.loads, (input_text, output_text, feedback_text))
            assert original["values"] == {"0": {"x": 1.0}, "1": "m1", "2": {"y": "Double x."}}
            assert original["schema"]["properties"]["0"]["properties"]["x"] == {"type": "number"}
            assert output["values"] == {"y": 2.0}
            assert output["schema"]["properties"]["y"] == {
                "type": "number",
                "minimum": 0,
                "maximum": 10,
                "description": "Double x.",
            }
            assert feedback["values"] == {"y": 3.0 * cotangent}
            assert feedback["schema"]["properties"]["y"] == {"type": "number"}
            schema = text["format"]["schema"]["properties"]["output"]
            assert schema["properties"]["0"]["properties"]["x"] == {
                "type": "number",
                "description": "Input cotangent at (0, 'x'), original value 1.0.",
            }
            dx = 2.0 * feedback["values"]["y"]
            return schema_response({"0": {"x": dx}, "1": "", "2": {"y": ""}}, text)

    def program(x):
        out = af.lm.fill({"x": x, "y": af.lm.Float(min=0, max=10, desc="Double x.")}, model="m1")
        return 3.0 * out["y"]

    ir = af.pullback(af.trace(program)(1.0))
    with af.lm.client(FillClient()):
        assert executor(ir, (1.0,), cotangent) == (6.0, (6.0 * cotangent,))
    assert len(calls) == 2


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "transform",
    [af.pushforward, af.pullback],
    ids=["pushforward", "pullback"],
)
def test_fill_composes_through_float_context(executor, transform):
    calls = []

    class FillClient(EchoRouter):
        def responses(self, *, input, model, text, **kwargs):
            assert kwargs == {}
            calls.append(input)
            content = json.loads(input)
            values = content["values"]
            if "x" in values:
                return schema_response({"y": 0.5}, text)
            if "y" in values:
                assert values["y"] == 0.5
                assert content["schema"]["properties"]["y"] == {"type": "number"}
                return schema_response({"z": "filled"}, text)

            prompt = values["context"]["request"]
            schema = text["format"]["schema"]["properties"]["output"]
            if transform is af.pushforward:
                original, change = parse_change_prompt(prompt)
                if "y" in schema["properties"]:
                    assert schema["properties"]["y"] == {"type": "number"}
                    return schema_response({"y": 0.25}, text)
                assert original["values"] == {"y": 0.5}
                assert change["values"] == {"y": 0.25}
                assert change["schema"]["properties"]["y"] == {"type": "number"}
                return schema_response({"z": "changed"}, text)

            feedback = json.loads(prompt.split(" OUTPUT FEEDBACK: ")[1])
            properties = schema["properties"]["0"]["properties"]
            if "y" in properties:
                assert properties["y"]["type"] == "number"
                assert feedback["values"] == {"z": "feedback"}
                return schema_response({"0": {"y": -0.5}, "1": ""}, text)
            assert properties["x"]["type"] == "string"
            assert feedback["values"] == {"y": -0.5}
            assert feedback["schema"]["properties"]["y"] == {"type": "number"}
            return schema_response({"0": {"x": "input feedback"}, "1": ""}, text)

    def program(x):
        out = af.lm.fill({"x": x, "y": af.lm.Float()}, model="m1")
        return af.lm.fill({"y": out["y"], "z": af.lm.Str()}, model="m1")["z"]

    ir = transform(af.trace(program)("seed"))
    change = ("dx",) if transform is af.pushforward else "feedback"
    expected = "changed" if transform is af.pushforward else ("input feedback",)
    with af.lm.client(FillClient()):
        assert executor(ir, ("x",), change) == ("filled", expected)
    assert len(calls) == 4


def test_fill_rejects_unsupported_context_and_non_string_model():
    with pytest.raises(TypeError, match="No aval rule registered"):
        af.lm.fill({"x": object(), "y": af.lm.Str()}, model="m1")
    with pytest.raises(AssertionError, match="Expected string model"):
        af.lm.fill({"x": 0.5, "y": af.lm.Str()}, model=1.0)
    with pytest.raises(AssertionError, match="Expected string model"):
        af.trace(lambda x: af.lm.fill_p.bind(x, static_tree={"y": af.lm.Str()}))(
            ({"x": 0.5}, {"y": af.lm.Str()}, 1.0),
        )


def test_describe_rejects_non_json_enum_values():
    enum = af.lm.Enum(object())

    with pytest.raises(TypeError, match="Enum values must be str, int, float, or bool"):
        describe({"kind": enum})


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_batch_supports_variable_models(executor):
    program = fill_program({"text": af.lm.Str(), "score": af.lm.Float()})
    ir = af.batch(af.trace(program)("test", "gpt-5.5"), in_axes=(True, True))
    args = (["hello", "goodbye"], ["m1", "m2"])
    with af.lm.client(SchemaRouter()):
        assert executor(ir, *args) == {
            "text": ["m1|hello", "m2|goodbye"],
            "score": [0.5, 0.5],
        }


@pytest.fixture
def gradient_client():
    with af.lm.client(SchemaGradientRouter()) as client:
        yield client


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "field, out_cotangent, expected, feedback",
    [
        pytest.param(
            "text",
            "too terse",
            "Recursion calls itself.",
            {"text": "too terse", "score": 0.0},
            id="unused-float",
        ),
        pytest.param(
            "score",
            -0.1,
            1.84,
            {"text": "", "score": -0.2},
            id="unused-string",
        ),
    ],
)
def test_fill_pullback_materializes_unused_fields(
    executor, field, out_cotangent, expected, feedback, gradient_client
):
    schema = {"text": af.lm.Str(min=1), "score": af.lm.Float(min=0, max=1)}
    fill = fill_program(schema)

    def program(x):
        y = fill(x)[field]
        return y * 2.0 if field == "score" else af.string.format("{text}", text=y)

    ir = af.sched(af.pullback(af.trace(program)("seed")))
    assert executor(ir, ("Explain recursion.",), out_cotangent) == (
        expected,
        ("input feedback",),
    )
    prompt = fill_request_text(gradient_client.calls[-1]["input"])
    assert json.loads(prompt.split(" OUTPUT FEEDBACK: ")[1])["values"]["output"] == feedback


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "out_cotangent, error",
    [
        pytest.param(
            {"text": "too terse", "score": "overconfident"}, "FloatAVal", id="text-for-float"
        ),
        pytest.param({"text": -0.2, "score": -0.2}, "StrAVal", id="float-for-text"),
    ],
)
def test_fill_pullback_rejects_wrong_schema_cotangent_type(
    executor, out_cotangent, error, gradient_client
):
    program = fill_program({"text": af.lm.Str(), "score": af.lm.Float()})
    ir = af.pullback(af.trace(program)("seed"))
    with pytest.raises(TypeError, match=f"Expected {error}"):
        executor(ir, ("Explain recursion.",), out_cotangent)
    assert len(gradient_client.calls) == 1


def test_project_value_preserves_generated_structure():
    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        fields: object
        metadata: object

    schema = Answer(
        [
            af.lm.Float(min=0, desc="Score"),
            (af.lm.Str(min=1), af.lm.Int(), af.lm.Bool(), af.lm.Enum("yes", "no")),
        ],
        {"source": "fixed", "nothing": None},
    )
    value = Answer([-0.2, ("", -1, False, "feedback")], {"source": "", "nothing": None})
    assert af.lm.project_value(schema, value) == {
        "fields": {"0": -0.2, "1": {"0": "", "1": -1, "2": False, "3": "feedback"}}
    }
    with pytest.raises(ValueError):
        af.lm.project_value(schema, Answer([], value.metadata))


@pytest.mark.parametrize(
    "schema, expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param({}, {}, id="empty-dict"),
        pytest.param("fixed", "fixed", id="literal-string"),
        pytest.param(
            {"source": "fixed", "nothing": None},
            {"source": "fixed", "nothing": None},
            id="literal-dict",
        ),
        pytest.param(
            ("fixed", None, []),
            ("fixed", None, []),
            id="literal-tuple",
        ),
    ],
)
def test_schema_without_generated_fields(schema, expected):
    assert describe(schema) is None
    assert parse(schema, None) == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="missing-object"),
        pytest.param(["x"], id="list"),
        pytest.param(("x",), id="tuple"),
        pytest.param({}, id="missing-field"),
        pytest.param({"0": "x", "extra": "y"}, id="extra-field"),
        pytest.param({0: "x"}, id="non-string-key"),
    ],
)
def test_parse_rejects_invalid_objects(value):
    with pytest.raises(ValueError):
        parse({"0": af.lm.Str()}, value)


def test_parse_rebuilds_nested_containers_from_objects():
    schema = (af.lm.Str(), {"score": af.lm.Float(), "source": "fixed"}, None)
    assert parse(schema, {"1": {"score": 2}, "0": "x"}) == (
        "x",
        {"score": 2.0, "source": "fixed"},
        None,
    )
    with pytest.raises(ValueError):
        parse(schema, {"0": "x", "1": [2]})


def test_schema_dsl_builds_described_schema():
    answer = {
        "name": af.lm.Str(desc="Subject name."),
        "kind": af.lm.Enum("summary", "definition", desc="Answer kind."),
        "score": af.lm.Float(desc="Confidence score."),
    }

    json_schema = describe(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["summary", "definition"],
                "description": "Answer kind.",
            },
            "name": {"type": "string", "description": "Subject name."},
            "score": {"type": "number", "description": "Confidence score."},
        },
        "required": ["kind", "name", "score"],
        "additionalProperties": False,
    }
    assert parse(
        answer,
        {"name": "subject", "kind": "summary", "score": 1},
    ) == {
        "name": "subject",
        "kind": "summary",
        "score": 1.0,
    }


def test_schema_dsl_reconstructs_literal_subtree():
    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        decision: object
        details: object

    details = {
        "literal": "fixed",
        "nothing": None,
    }
    answer = Answer(af.lm.Str(), details)

    json_schema = describe(answer)

    assert json_schema == {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
        "additionalProperties": False,
    }
    parsed = parse(answer, {"decision": "accept"})
    expected = Answer(
        "accept",
        {"literal": "fixed", "nothing": None},
    )
    assert parsed == expected

    ir = af.trace(fill_program(answer))("test")
    call = ir.eqns[-1]
    assert isinstance(call.out_tree["output"].decision, af.stage.Var)

    class FillClient(EchoRouter):
        def responses(self, *, text, **kwargs):
            return schema_response({"decision": "accept"}, text)

    with af.lm.client(FillClient()):
        assert ir.call("hello") == expected


def test_schema_dsl_reconstructs_untraceable_static_leaf():
    metadata = object()

    answer = {"decision": af.lm.Str(), "metadata": metadata}

    json_schema = describe(answer)

    assert json_schema == {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
        "additionalProperties": False,
    }
    parsed = parse(answer, {"decision": "accept"})
    assert parsed["decision"] == "accept"
    assert parsed["metadata"] is metadata


def test_lm_schema_trace_rejects_untraceable_static_leaf():
    metadata = object()

    program = fill_program({"decision": af.lm.Str(), "metadata": metadata})

    with pytest.raises(TypeError, match="No aval rule registered"):
        af.trace(program)("test")


def test_schema_dsl_builds_string_constraints():
    answer = {"name": af.lm.Str(min=2, max=4, pattern=r"^[a-z]+$")}

    json_schema = describe(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 2,
                "maxLength": 4,
                "pattern": r"^[a-z]+$",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert parse(answer, {"name": "okay"}) == {"name": "okay"}
    with pytest.raises(ValueError, match="Expected string with length >= 2"):
        parse(answer, {"name": "x"})
    with pytest.raises(ValueError, match="Expected string with length <= 4"):
        parse(answer, {"name": "hello"})
    with pytest.raises(ValueError, match="Expected string matching"):
        parse(answer, {"name": "OK"})


def test_schema_dsl_builds_number_constraints():
    answer = {
        "count": af.lm.Int(min=-2, max=2),
        "score": af.lm.Float(min=0, max=1),
    }

    json_schema = describe(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": -2, "maximum": 2},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["count", "score"],
        "additionalProperties": False,
    }
    assert parse(answer, {"count": 0, "score": 1}) == {"count": 0, "score": 1.0}
    with pytest.raises(ValueError, match="Expected integer >= -2"):
        parse(answer, {"count": -3, "score": 0.5})
    with pytest.raises(ValueError, match="Expected integer <= 2"):
        parse(answer, {"count": 3, "score": 0.5})
    with pytest.raises(ValueError, match="Expected number >= 0"):
        parse(answer, {"count": 0, "score": -0.1})
    with pytest.raises(ValueError, match="Expected number <= 1"):
        parse(answer, {"count": 0, "score": 1.1})


def test_schema_dsl_builds_custom_pytree_value():
    class Answer:
        __slots__ = ["text", "score"]

        def __init__(self, text, score):
            self.text = text
            self.score = score

        def __eq__(self, other):
            return (
                type(self) is type(other) and self.text == other.text and self.score == other.score
            )

    tree.register_node(
        Answer,
        lambda answer: ((answer.text, answer.score), None, ("text", "score")),
        lambda _, children: Answer(*children),
        path_entry_type=optree.GetAttrEntry,
    )

    answer = Answer(af.lm.Str(), af.lm.Float())

    json_schema = describe(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["text", "score"],
        "additionalProperties": False,
    }
    assert parse(answer, {"text": "hello", "score": 2}) == Answer("hello", 2.0)


def test_schema_dsl_reports_value_errors():
    count = {"count": af.lm.Int()}
    score = {"score": af.lm.Float()}

    with pytest.raises(ValueError, match="Expected integer"):
        parse(count, {"count": True})
    with pytest.raises(ValueError, match="Expected number"):
        parse(score, {"score": "bad"})


@pytest.mark.parametrize(
    "transform, args, width",
    [
        pytest.param(af.pushforward, (("hello",), ("change",)), 2, id="pushforward"),
        pytest.param(
            af.pullback, (("hello",), {"text": "feedback", "score": -0.2}), 1, id="pullback"
        ),
        pytest.param(af.batch, (["hello", "world"],), 2, id="batch"),
    ],
)
@pytest.mark.parametrize("serial", [False, True], ids=["concurrent", "serial"])
def test_async_transforms_respect_fanout(transform, args, width, serial):
    events = []

    class FillClient(SchemaGradientRouter):
        async def aresponses(self, **kwargs):
            events.append(1)
            await asyncio.sleep(0)
            response = self.responses(**kwargs)
            events.append(-1)
            return response

    program = fill_program({"text": af.lm.Str(), "score": af.lm.Float()})
    ir = transform(af.trace(program)("seed"))
    with af.lm.client(FillClient()) as client:
        expected = execute(ir, *args)
        client.calls.clear()
        with af.order.serial_fanout() if serial else nullcontext():
            assert aexecute(ir, *args) == expected
    active = peak = 0
    for event in events:
        active += event
        peak = max(peak, active)
    assert active == 0
    assert peak == (1 if serial else width)
    assert len(client.calls) == 2


def test_client_restores_outer_client_after_exception():
    outer, inner = EchoRouter(), EchoRouter()
    with af.lm.client(outer):
        with pytest.raises(ValueError, match="stop"):
            with af.lm.client(inner):
                assert af.extend.active_client.get() is inner
                raise ValueError("stop")
        assert af.extend.active_client.get() is outer


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(af.lm.LiteLLMClient.responses, id="sync"),
        pytest.param(
            lambda client, **kwargs: asyncio.run(client.aresponses(**kwargs)),
            id="async",
        ),
    ],
)
def test_litellm_client_forwards_request(executor, monkeypatch):
    calls = []
    response = fake_response("hello")

    def complete(**kwargs):
        calls.append(kwargs)
        return response

    async def acomplete(**kwargs):
        return complete(**kwargs)

    monkeypatch.setattr(af.lm, "responses", complete)
    monkeypatch.setattr(af.lm, "aresponses", acomplete)
    client = af.lm.LiteLLMClient()
    kwargs = dict(
        input="hello",
        model="m1",
        temperature=0.7,
        max_output_tokens=128,
    )
    result = executor(client, **kwargs)
    assert result is response
    assert calls == [kwargs]
