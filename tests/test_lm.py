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
from types import SimpleNamespace

import optree
import pytest

import autoform as af
from autoform.schemas import describe, parse
from autoform.utils import tree
from tests import aexecute, execute


@pytest.fixture
def echo_client():
    with af.lm.client(af.lm.EchoClient()) as client:
        yield client


def lm_program(primitive=af.lm.complete, *, model="m1", **params):
    def program(prompt, model=model):
        return primitive([dict(role="user", content=prompt)], model=model, **params)

    return program


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_complete_and_generate_use_distinct_primitives(executor):
    def program(prompt):
        messages = [dict(role="user", content=prompt)]
        return (
            af.lm.complete(messages, model="echo"),
            af.lm.generate(messages, model="echo", schema=None),
        )

    ir = af.trace(program)("seed")
    assert [eqn.prim for eqn in ir.eqns] == [
        af.control.stop_gradient_p,
        af.lm.complete_p,
        af.control.stop_gradient_p,
        af.lm.generate_p,
    ]
    with af.lm.client(af.lm.EchoClient()):
        assert program("hello") == ("<user> hello", None)
        assert executor(ir, "hello") == ("<user> hello", None)


def fake_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def schema_response(value, response_format):
    properties = response_format["json_schema"]["schema"].get("properties", {})
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


def echo_or_json_fill(messages, *, structured=False):
    content = af.lm.echo_messages(messages)
    try:
        payload = json.loads(messages[-1]["content"])
    except json.JSONDecodeError:
        return json.dumps(content) if structured else content
    if isinstance(payload, dict) and "values" in payload and "schema" in payload:
        return json.dumps({"output": content})
    return json.dumps(content) if structured else content


class EchoRouter:
    def completion(self, *, messages, model, response_format=None, **kwargs):
        assert kwargs == {}
        content = f"{model}|{messages[-1]['content']}"
        if response_format is not None:
            return schema_response(content, response_format)
        return fake_response(content)

    async def acompletion(self, **kwargs):
        return self.completion(**kwargs)


class SchemaRouter(EchoRouter):
    __slots__ = ["response_formats"]

    def __init__(self):
        self.response_formats = []

    def completion(self, *, messages: list[dict], model: str, response_format, **kwargs):
        assert kwargs == {}
        self.response_formats.append(response_format)
        return schema_response(
            {
                "text": f"{model}|{messages[-1]['content']}",
                "score": 0.5,
            },
            response_format,
        )


class SchemaGradientRouter(EchoRouter):
    __slots__ = ["calls"]

    def __init__(self):
        self.calls = []

    def completion(self, *, messages: list[dict], model: str, response_format=None, **kwargs):
        assert kwargs == {}
        self.calls.append(dict(messages=messages, model=model, response_format=response_format))
        if response_format is not None:
            schema = response_format["json_schema"]["schema"]
            schema = schema.get("properties", {}).get("output", schema)
            if schema.get("type") == "string":
                return schema_response("output change", response_format)
            properties = schema["properties"]
            if "1" in properties:
                feedback = {"1": "model feedback"}
                if "0" in properties:
                    feedback["0"] = {
                        key: dict(role=f"role feedback {key}", content=f"content feedback {key}")
                        for key in properties["0"]["properties"]
                    }
                return schema_response(feedback, response_format)
            return schema_response(
                {"text": "Recursion calls itself.", "score": 0.92},
                response_format,
            )
        return fake_response("input feedback")


def test_generate_executes_with_response_format():
    router = SchemaRouter()
    answer = {
        "text": af.Str(min=1, max=80),
        "metadata": {"source": "literal", "reasoning": None},
        "score": af.Float(min=0, max=1),
    }

    with af.lm.client(router):
        result = af.lm.generate(
            [dict(role="user", content="hello")],
            model="m1",
            schema=answer,
        )

    assert result == {
        "text": "m1|hello",
        "metadata": {"source": "literal", "reasoning": None},
        "score": 0.5,
    }
    assert router.response_formats == [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "autoform_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "text": {"type": "string", "minLength": 1, "maxLength": 80},
                    },
                    "required": ["score", "text"],
                    "additionalProperties": False,
                },
            },
        }
    ]


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("schema", "value"),
    [
        pytest.param(af.Str(), 'Hello "world"!', id="string"),
        pytest.param(af.Str(min=1, desc="Answer text."), "hello", id="described-string"),
        pytest.param(af.Int(), 2, id="integer"),
        pytest.param(af.Float(), 0.5, id="float"),
        pytest.param(af.Bool(), True, id="boolean"),
        pytest.param(af.Enum("yes", "no"), "yes", id="enum"),
    ],
)
def test_generate_passes_scalar_schemas_to_client(executor, schema, value):
    class ScalarClient(af.lm.EchoClient):
        def completion(self, *, response_format, **kwargs):
            assert response_format["json_schema"]["schema"] == describe(schema)
            return super().completion(**kwargs)

    program = lm_program(af.lm.generate, model="echo", schema=schema)
    ir = af.trace(program)("seed")
    assert af.core.avalof(schema) == ir.out_tree.aval
    with af.lm.client(ScalarClient(render=lambda _: json.dumps(value))):
        assert program("hello") == value
        assert executor(ir, "hello") == value


def test_generate_uses_runtime_model_with_structured_output():
    answer = {
        "text": af.Str(desc="Short text."),
        "score": af.Float(),
    }

    generate = lm_program(af.lm.generate, schema=answer)

    def program(prompt, model):
        return af.string.format("{text}", text=generate(prompt, model)["text"])

    ir = af.trace(program)("test", "gpt-5.5")

    with af.lm.client(SchemaRouter()):
        assert ir.call("hello", "m1") == "m1|hello"


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
def test_generate_pushforward_preserves_non_string_literals(executor):
    def program(x):
        return af.lm.generate(
            [dict(role="user", content=x)],
            model="echo",
            schema={"answer": af.Str(), "fixed": 1.0},
        )

    ir = af.pushforward(af.trace(program)("seed"))
    render = lambda _: json.dumps({"output": {"answer": "filled"}})
    with af.lm.client(af.lm.EchoClient(render=render)):
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

        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            self.calls.append(dict(messages=messages, model=model, response_format=response_format))
            return fake_response(json.dumps({"answer": "four", "score": 0.9}))

    def program(question):
        return af.lm.fill(
            {
                "question": question,
                "answer": af.Str(desc="Answer text."),
                "score": af.Float(min=0, max=1),
            },
            model="m1",
        )

    ir = af.trace(program)("seed")
    assert [eqn.prim for eqn in ir.eqns] == [
        af.json.encode_p,
        af.control.stop_gradient_p,
        af.lm.fill_p,
    ]
    call = ir.eqns[-1]
    assert call.params == {
        "schema": {
            "question": None,
            "answer": af.Str(desc="Answer text."),
            "score": af.Float(min=0, max=1),
        }
    }
    assert ir.eqns[0].in_tree == {"question": ir.in_tree[0], "answer": None, "score": None}
    assert call.in_tree == (ir.eqns[0].out_tree, ir.eqns[1].out_tree)
    assert call.out_tree["question"] is None
    assert ir.out_tree["question"] is ir.in_tree[0]
    assert all(isinstance(x, af.stage.Var) for x in tree.leaves(call.out_tree))

    client = FillClient()
    with af.lm.client(client):
        assert executor(ir, "1+1?") == {"question": "1+1?", "answer": "four", "score": 0.9}

    call = client.calls[-1]
    assert call["model"] == "m1"
    assert json.loads(call["messages"][-1]["content"]) == {
        "values": {"question": "1+1?"},
        "schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    }
    assert call["response_format"]["json_schema"]["schema"] == {
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
        pytest.param(af.Str(), "filled", "filled", id="scalar"),
        pytest.param(
            ("fixed", [af.Str(), None, {}]),
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

    def render(messages):
        calls.append(messages)
        return json.dumps(response)

    ir = af.trace(lambda: af.lm.fill(template, model="echo"))()
    with af.lm.client(af.lm.EchoClient(render=render)):
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
        def completion(self, *, messages, model, response_format, **kwargs):
            content = json.loads(messages[-1]["content"])
            assert content["values"] == {"fields": {"0": x}}
            assert content["schema"]["properties"]["fields"]["properties"] == {
                "0": {"type": json_type},
            }
            schema = response_format["json_schema"]["schema"]
            assert schema["properties"]["fields"]["properties"] == {"0_": {"type": "string"}}
            return fake_response(json.dumps({"fields": {"0_": "filled"}}))

    def program(x):
        return af.lm.fill(Answer({0: x, "0": af.Str()}, (None, [], {})), model="echo")

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

        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            assert model == "m1"
            assert response_format["json_schema"]["strict"] is True
            question = json.loads(messages[-1]["content"])["values"]["question"]
            self.questions.append(question)
            return fake_response(json.dumps({"answer": f"filled:{question}"}))

    def program(question):
        return af.lm.fill({"question": question, "answer": af.Str()}, model="m1")

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
        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            values = fill_values(messages[-1]["content"])
            if "context" in values:
                original, change = parse_change_prompt(values["context"]["request"])
                for content in (original, change):
                    assert content["schema"]["properties"]["question"] == {"type": json_type}
                question = original["values"]["question"]
                d_question = change["values"]["question"]
                return schema_response(
                    {"answer": f"filled:{question}->{d_question}"},
                    response_format,
                )
            question = values["question"]
            return schema_response({"answer": f"filled:{question}"}, response_format)

    def program(question):
        return af.lm.fill({"question": question, "answer": af.Str()}, model="m1")

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
    schema = af.Float(min=0, max=10, desc=description)
    calls = []

    class FillClient(EchoRouter):
        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            calls.append(messages)
            values = fill_values(messages[-1]["content"])
            response_schema = response_format["json_schema"]["schema"]
            original_schema = describe({"y": schema})
            if "context" not in values:
                assert response_schema == original_schema
                return schema_response({"y": 8.0}, response_format)

            context = values["context"]
            assert json.loads(context["output_schema"]) == original_schema
            original, change = parse_change_prompt(context["request"])
            assert original["values"] == {"x": "x"}
            assert change["values"] == {"x": "dx"}
            assert response_schema["properties"]["output"] == describe({"y": af.Float()})
            return schema_response({"y": -2.0}, response_format)

    def program(x):
        return af.lm.fill({"x": x, "y": schema}, model="m1")

    ir = af.pushforward(af.trace(program)("seed"))
    with af.lm.client(FillClient()):
        assert executor(ir, ("x",), ("dx",)) == (
            {"x": "x", "y": 8.0},
            {"x": "dx", "y": -2.0},
        )
    assert len(calls) == 2
    assert schema == af.Float(min=0, max=10, desc=description)


@pytest.mark.parametrize(
    "t_tree",
    [
        pytest.param({"x": "dx", "y": af.Float(desc="Estimate price")}, id="description"),
        pytest.param({"x": "dx", "y": af.Float(min=0, desc="Score clarity")}, id="constraint"),
        pytest.param({"x": "dx", "y": af.Str(desc="Score clarity")}, id="schema-type"),
        pytest.param({"x": "dx", "y": 0.0}, id="schema-to-leaf"),
        pytest.param({"x": ["dx"], "y": af.Float(desc="Score clarity")}, id="structure"),
    ],
)
def test_fill_pushforward_rejects_mismatched_specs(t_tree):
    p_tree = {"x": "x", "y": af.Float(desc="Score clarity")}
    ir = af.pushforward(af.trace(lambda x: af.lm.fill(x, model="m1"))(p_tree))
    with pytest.raises(ValueError):
        ir.call((p_tree,), (t_tree,))


def test_fill_pushforward_accepts_equal_static_metadata():
    class FillClient(EchoRouter):
        def completion(self, *, response_format, **kwargs):
            return schema_response({"y": 1.0}, response_format)

    p_tree = {"x": "x", "y": af.Float(desc="Score clarity")}
    t_tree = {"x": "dx", "y": af.Float(desc="Score clarity")}
    assert p_tree["y"] is not t_tree["y"]
    ir = af.pushforward(af.trace(lambda x: af.lm.fill(x, model="m1"))(p_tree))
    with af.lm.client(FillClient()):
        assert ir.call((p_tree,), (t_tree,)) == ({"x": "x", "y": 1.0}, {"x": "dx", "y": 1.0})


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("generated", [False, True], ids=["copied-only", "copied-and-generated"])
@pytest.mark.parametrize("batch_order", [None, "before", "after"])
def test_fill_pullback_preserves_context_identity(executor, generated, batch_order):
    calls = []

    def render(messages):
        calls.append(messages)
        values = fill_values(messages[-1]["content"])
        if "context" in values:
            prompt = values["context"]["request"]
            feedback = json.loads(prompt.split(" OUTPUT FEEDBACK: ")[1])
            assert feedback["values"] == {"answer": "generated feedback"}
            return json.dumps({
                "output": {"0": {"question": "model feedback"}, "1": "ignored"},
            })
        return json.dumps({"answer": "filled"})

    def program(question):
        out = af.lm.fill({"question": question, "answer": af.Str()}, model="m1")
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
    with af.lm.client(af.lm.EchoClient(render=render)):
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

        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            self.calls.append(dict(messages=messages, model=model, response_format=response_format))
            properties = response_format["json_schema"]["schema"]["properties"]
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
                response_format,
            )

    def program(question):
        filled = af.lm.fill(
            {
                "question": question,
                "answer": af.Str(),
                "score": af.Float(min=0, max=1),
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
    prompt = fill_request_text(backward["messages"][-1]["content"])
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
    schema = backward["response_format"]["json_schema"]["schema"]["properties"]["output"]
    assert schema["properties"]["0"]["properties"]["question"] == {
        "type": "string",
        "description": "Input cotangent at (0, 'question'), original value 'Explain recursion.'.",
    }


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize("generated", [False, True], ids=["copied-only", "copied-and-generated"])
def test_fill_pullback_generates_input_cotangent_types(executor, generated):
    calls = []

    class FeedbackClient(EchoRouter):
        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            calls.append(messages)
            values = fill_values(messages[-1]["content"])
            if "x" in values:
                return schema_response({"y": "filled"}, response_format)
            prompt = values["context"]["request"]
            content = json.loads(prompt.removeprefix("INPUT: ").split(" OUTPUT: ")[0])
            assert content["schema"]["properties"]["0"]["properties"]["x"] == {"type": "number"}
            schema = response_format["json_schema"]["schema"]["properties"]["output"]
            assert schema["properties"]["0"]["properties"]["x"] == {
                "type": "number",
                "description": "Input cotangent at (0, 'x'), original value 0.5.",
            }
            return schema_response({"0": {"x": -0.5}, "1": "model feedback"}, response_format)

    def program(x):
        out = af.lm.fill({"x": x, "y": af.Str()}, model="m1")
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
        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            calls.append(messages)
            values = fill_values(messages[-1]["content"])
            if "x" in values:
                return schema_response({"y": 2.0 * values["x"]}, response_format)

            context = values["context"]
            instruction = context["instruction"]
            assert instruction == af.lm.GRAD_SYSTEM_PROMPT
            input_text, rest = context["request"].removeprefix("INPUT: ").split(" OUTPUT: ")
            output_text, feedback_text = rest.split(" OUTPUT FEEDBACK: ")
            original, output, feedback = map(json.loads, (input_text, output_text, feedback_text))
            assert original["values"] == {"0": {"x": 1.0}, "1": "m1"}
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
            schema = response_format["json_schema"]["schema"]["properties"]["output"]
            assert schema["properties"]["0"]["properties"]["x"] == {
                "type": "number",
                "description": "Input cotangent at (0, 'x'), original value 1.0.",
            }
            dx = 2.0 * feedback["values"]["y"]
            return schema_response({"0": {"x": dx}, "1": ""}, response_format)

    def program(x):
        out = af.lm.fill({"x": x, "y": af.Float(min=0, max=10, desc="Double x.")}, model="m1")
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
        def completion(self, *, messages, model, response_format, **kwargs):
            assert kwargs == {}
            calls.append(messages)
            content = json.loads(messages[-1]["content"])
            values = content["values"]
            if "x" in values:
                return schema_response({"y": 0.5}, response_format)
            if "y" in values:
                assert values["y"] == 0.5
                assert content["schema"]["properties"]["y"] == {"type": "number"}
                return schema_response({"z": "filled"}, response_format)

            prompt = values["context"]["request"]
            schema = response_format["json_schema"]["schema"]["properties"]["output"]
            if transform is af.pushforward:
                original, change = parse_change_prompt(prompt)
                if "y" in schema["properties"]:
                    assert schema["properties"]["y"] == {"type": "number"}
                    return schema_response({"y": 0.25}, response_format)
                assert original["values"] == {"y": 0.5}
                assert change["values"] == {"y": 0.25}
                assert change["schema"]["properties"]["y"] == {"type": "number"}
                return schema_response({"z": "changed"}, response_format)

            feedback = json.loads(prompt.split(" OUTPUT FEEDBACK: ")[1])
            properties = schema["properties"]["0"]["properties"]
            if "y" in properties:
                assert properties["y"]["type"] == "number"
                assert feedback["values"] == {"z": "feedback"}
                return schema_response({"0": {"y": -0.5}, "1": ""}, response_format)
            assert properties["x"]["type"] == "string"
            assert feedback["values"] == {"y": -0.5}
            assert feedback["schema"]["properties"]["y"] == {"type": "number"}
            return schema_response({"0": {"x": "input feedback"}, "1": ""}, response_format)

    def program(x):
        out = af.lm.fill({"x": x, "y": af.Float()}, model="m1")
        return af.lm.fill({"y": out["y"], "z": af.Str()}, model="m1")["z"]

    ir = transform(af.trace(program)("seed"))
    change = ("dx",) if transform is af.pushforward else "feedback"
    expected = "changed" if transform is af.pushforward else ("input feedback",)
    with af.lm.client(FillClient()):
        assert executor(ir, ("x",), change) == ("filled", expected)
    assert len(calls) == 4


def test_fill_rejects_unsupported_context_and_non_string_model():
    with pytest.raises(TypeError, match="No aval rule registered"):
        af.lm.fill({"x": object(), "y": af.Str()}, model="m1")
    with pytest.raises(AssertionError, match="Expected string model"):
        af.lm.fill({"x": 0.5, "y": af.Str()}, model=1.0)
    with pytest.raises(AssertionError, match="Expected string model"):
        af.trace(lambda x: af.lm.fill_p.bind(x, schema={"y": af.Str()}))(
            (af.json.encode({"x": 0.5}), 1.0),
        )


def test_describe_rejects_non_json_enum_values():
    enum = af.Enum(object())

    with pytest.raises(TypeError, match="Enum values must be str, int, float, or bool"):
        describe({"kind": enum})


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "primitive, params, client_type, expected",
    [
        pytest.param(af.lm.complete, {}, EchoRouter, ["m1|hello", "m2|goodbye"], id="complete"),
        pytest.param(
            af.lm.generate,
            {"schema": {"text": af.Str(), "score": af.Float()}},
            SchemaRouter,
            {"text": ["m1|hello", "m2|goodbye"], "score": [0.5, 0.5]},
            id="generate",
        ),
    ],
)
def test_batch_supports_variable_models(executor, primitive, params, client_type, expected):
    ir = af.batch(
        af.trace(lm_program(primitive, **params))("test", "gpt-5.5"),
        in_axes=(True, True),
    )
    args = (["hello", "goodbye"], ["m1", "m2"])
    with af.lm.client(client_type()):
        actual = executor(ir, *args)
    assert actual == expected


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
def test_generate_pullback_materializes_unused_fields(
    executor, field, out_cotangent, expected, feedback, gradient_client
):
    schema = {"text": af.Str(min=1), "score": af.Float(min=0, max=1)}
    generate = lm_program(af.lm.generate, schema=schema)

    def program(x):
        y = generate(x)[field]
        return y * 2.0 if field == "score" else af.string.format("{text}", text=y)

    ir = af.sched(af.pullback(af.trace(program)("seed")))
    assert executor(ir, ("Explain recursion.",), out_cotangent) == (
        expected,
        ("content feedback 0",),
    )
    prompt = fill_request_text(gradient_client.calls[-1]["messages"][-1]["content"])
    assert json.loads(prompt.split(" OUTPUT FEEDBACK: ")[1]) == feedback


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
def test_generate_pullback_rejects_wrong_schema_cotangent_type(
    executor, out_cotangent, error, gradient_client
):
    program = lm_program(af.lm.generate, schema={"text": af.Str(), "score": af.Float()})
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
            af.Float(min=0, desc="Score"),
            (af.Str(min=1), af.Int(), af.Bool(), af.Enum("yes", "no")),
        ],
        {"source": "fixed", "nothing": None},
    )
    value = Answer([-0.2, ("", -1, False, "feedback")], {"source": "", "nothing": None})
    assert af.json.project_value(schema, value) == {
        "fields": {"0": -0.2, "1": {"0": "", "1": -1, "2": False, "3": "feedback"}}
    }
    with pytest.raises(ValueError):
        af.json.project_value(schema, Answer([], value.metadata))


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


@pytest.mark.parametrize("parse", [parse, af.lm.parse], ids=["schemas", "lm"])
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
def test_parse_rejects_invalid_objects(parse, value):
    with pytest.raises(ValueError):
        parse({"0": af.Str()}, value)


@pytest.mark.parametrize("parse", [parse, af.lm.parse], ids=["schemas", "lm"])
def test_parse_rebuilds_nested_containers_from_objects(parse):
    schema = (af.Str(), {"score": af.Float(), "source": "fixed"}, None)
    assert parse(schema, {"1": {"score": 2}, "0": "x"}) == (
        "x",
        {"score": 2.0, "source": "fixed"},
        None,
    )
    with pytest.raises(ValueError):
        parse(schema, {"0": "x", "1": [2]})


def test_schema_dsl_builds_described_schema():
    answer = {
        "name": af.Str(desc="Subject name."),
        "kind": af.Enum("summary", "definition", desc="Answer kind."),
        "score": af.Float(desc="Confidence score."),
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
    answer = Answer(af.Str(), details)

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

    ir = af.trace(lm_program(af.lm.generate, schema=answer))("test")
    assert isinstance(ir.eqns[1].out_tree.decision, af.stage.Var)
    assert ir.eqns[1].out_tree.details == expected.details

    walk = ir.walk("hello")
    equation, inputs = next(walk)
    equation, _ = walk.send(equation.bind(inputs))
    assert equation is ir.eqns[1]
    done, result = walk.send(parsed)
    assert done is None
    assert result == expected


def test_schema_dsl_reconstructs_untraceable_static_leaf():
    metadata = object()

    answer = {"decision": af.Str(), "metadata": metadata}

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

    program = lm_program(af.lm.generate, schema={"decision": af.Str(), "metadata": metadata})

    with pytest.raises(TypeError, match="Static schema leaf must be traceable"):
        af.trace(program)("test")


def test_schema_dsl_builds_string_constraints():
    answer = {"name": af.Str(min=2, max=4, pattern=r"^[a-z]+$")}

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
        "count": af.Int(min=-2, max=2),
        "score": af.Float(min=0, max=1),
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

    answer = Answer(af.Str(), af.Float())

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
    count = {"count": af.Int()}
    score = {"score": af.Float()}

    with pytest.raises(ValueError, match="Expected integer"):
        parse(count, {"count": True})
    with pytest.raises(ValueError, match="Expected number"):
        parse(score, {"score": "bad"})


class TestLMPrimitive:
    @pytest.mark.parametrize(
        "primitive, params, out_cotangent",
        [
            pytest.param(af.lm.complete, {}, "feedback", id="complete"),
            pytest.param(
                af.lm.generate,
                {"schema": {"text": af.Str(), "score": af.Float()}},
                {"text": "feedback", "score": -0.2},
                id="generate",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "transform, args, count, width",
        [
            pytest.param(af.pushforward, (("hello",), ("tangent",)), 2, 2, id="pushforward"),
            pytest.param(af.pullback, (("hello",), "feedback"), 2, 1, id="pullback"),
            pytest.param(af.batch, (["hello", "world"],), 2, 2, id="batch"),
        ],
    )
    @pytest.mark.parametrize("serial", [False, True], ids=["concurrent", "serial"])
    def test_async_transforms_respect_fanout(
        self, primitive, params, out_cotangent, transform, args, count, width, serial
    ):
        events = []

        class Client(SchemaGradientRouter):
            async def acompletion(self, **kwargs):
                events.append(1)
                await asyncio.sleep(0)
                response = self.completion(**kwargs)
                events.append(-1)
                return response

        with af.lm.client(Client()) as client:
            ir = af.trace(lm_program(primitive, **params))("seed")
            if transform is af.pullback:
                args = (args[0], out_cotangent)
            ir = transform(ir)
            assert client.calls == []
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
        assert len(client.calls) == count

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "primitive, params, render",
        [
            pytest.param(af.lm.complete, {}, echo_or_json_fill, id="complete"),
            pytest.param(
                af.lm.generate,
                {"schema": af.Str()},
                lambda messages: echo_or_json_fill(messages, structured=True),
                id="generate",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "roles",
        [
            pytest.param(["assistant", "user"], id="list"),
            pytest.param(("assistant", "user"), id="tuple"),
            pytest.param({"x": "assistant", "y": "user"}, id="dict"),
        ],
    )
    def test_traces_and_batches_message_roles(
        self, executor, primitive, params, render, roles, echo_client
    ):
        def program(messages):
            return primitive(messages, model="echo", **params)

        echo_client.render = render

        messages = [dict(role="user", content="hello"), dict(role="system", content="who is this")]
        ir = af.trace(program)(messages)
        stopped, call = ir.eqns
        assert stopped.prim is af.control.stop_gradient_p
        assert stopped.in_tree == ([m["role"] for m in ir.in_tree[0]], "echo")
        stopped_roles, stopped_model = stopped.out_tree
        assert call.in_tree == (
            [
                dict(role=r, content=m["content"])
                for r, m in zip(stopped_roles, ir.in_tree[0], strict=True)
            ],
            stopped_model,
        )
        assert executor(ir, messages) == "<user> hello\n<system> who is this"

        messages[0]["role"] = "assistant"
        assert executor(ir, messages) == "<assistant> hello\n<system> who is this"

        batched = af.batch(ir, in_axes=([dict(role=True, content=False), False],))
        messages[0]["role"] = roles
        assert executor(batched, messages) == tree.map(
            lambda role: f"<{role}> hello\n<system> who is this", roles
        )

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "primitive, params, render",
        [
            pytest.param(af.lm.complete, {}, echo_or_json_fill, id="complete"),
            pytest.param(
                af.lm.generate,
                {"schema": af.Str()},
                lambda messages: echo_or_json_fill(messages, structured=True),
                id="generate",
            ),
        ],
    )
    def test_pushforward_preserves_primal_roles(
        self, executor, primitive, params, render, echo_client
    ):
        def program(messages, model):
            return primitive(messages, model=model, **params)

        echo_client.render = render

        messages = [dict(role="system", content="hello"), dict(role="user", content="world")]
        ir = af.pushforward(af.trace(program)(messages, "echo"))
        zero = af.core.Zero(af.string.StrAVal())
        tangents = [dict(role="ignored", content=zero), dict(role="ignored", content="tangent")]
        primal, tangent = executor(ir, (messages, "echo"), (tangents, "ignored"))
        if params:
            assert primal.startswith("<user> ")
            assert fill_values(primal.removeprefix("<user> "))["context"] == {
                "messages": {
                    "0": {"content": "hello", "role": "system"},
                    "1": {"content": "world", "role": "user"},
                }
            }
        else:
            assert primal == "<system> hello\n<user> world"
        assert tangent.startswith("<user> ")
        original, change = parse_change_prompt(fill_request_text(tangent.removeprefix("<user> ")))
        assert original == {"messages": messages}
        assert change == {
            "messages": [
                {"role": "", "content": ""},
                {"role": "", "content": "tangent"},
            ]
        }

    def test_complete_leaves_litellm_params_to_active_client(self):
        class ConfiguredRouter(EchoRouter):
            def completion(self, *, messages, model, **kwargs):
                assert kwargs == {}
                params = {"m1": {"temperature": 0.7, "max_tokens": 128}}[model]
                return fake_response(
                    f"{model}|{params['temperature']}|{params['max_tokens']}|"
                    f"{messages[-1]['content']}"
                )

        ir = af.trace(lm_program())("test", "gpt-5.5")
        with af.lm.client(ConfiguredRouter()):
            assert ir.call("hello", "m1") == "m1|0.7|128|hello"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "primitive, raw, params, out_cotangent, expected, decode",
        [
            pytest.param(
                af.lm.complete,
                af.lm.complete_p,
                {},
                "feedback",
                "input feedback",
                str,
                id="complete",
            ),
            pytest.param(
                af.lm.generate,
                af.lm.generate_p,
                {"schema": {"text": af.Str(min=1), "score": af.Float(min=0, max=1)}},
                {"text": "feedback", "score": -0.2},
                {"text": "Recursion calls itself.", "score": 0.92},
                json.loads,
                id="generate",
            ),
        ],
    )
    @pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "raw"])
    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([], id="empty-messages"),
            pytest.param(
                [dict(role="system", content="hello"), dict(role="user", content="world")],
                id="multiple-messages",
            ),
        ],
    )
    def test_pullback_returns_input_tree_in_one_call(
        self,
        executor,
        primitive,
        raw,
        params,
        out_cotangent,
        expected,
        decode,
        stopped,
        messages,
        gradient_client,
    ):
        def program(messages, model):
            if stopped:
                return primitive(messages, model=model, **params)
            return raw.bind((messages, model), **params)

        ir = af.pullback(af.trace(program)(messages, "m1"))
        out, cotangent = executor(ir, (messages, "m2"), out_cotangent)
        zero = af.core.Zero(af.string.StrAVal())
        assert out == expected
        assert cotangent == (
            [
                dict(
                    role=zero if stopped else f"role feedback {i}", content=f"content feedback {i}"
                )
                for i in range(len(messages))
            ],
            zero if stopped else "model feedback",
        )
        forward, backward = gradient_client.calls
        assert forward["messages"] == messages
        assert all(call["model"] == "m2" for call in gradient_client.calls)
        *context, message = backward["messages"]
        assert message["role"] == "user"
        prompt = fill_request_text(message["content"])
        assert f"INPUT: {(messages, 'm2')}" in prompt
        response_format = backward["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]["properties"]["output"]
        properties = schema["properties"]
        assert properties["1"] == {
            "type": "string",
            "description": "Input cotangent at (1,), original value 'm2'.",
        }
        for i, m in enumerate(messages):
            fields = properties["0"]["properties"][str(i)]["properties"]
            assert fields == {
                key: {
                    "type": "string",
                    "description": f"Input cotangent at {(0, i, key)}, original value {value!r}.",
                }
                for key, value in m.items()
            }
        assert context == []
        output, feedback = prompt.split("OUTPUT: ", 1)[1].split(" OUTPUT FEEDBACK: ")
        assert decode(output) == out
        assert decode(feedback) == out_cotangent


class TestEchoLMClient:
    @pytest.mark.parametrize(
        ("entries", "expected"),
        [
            pytest.param([], "", id="empty-messages"),
            pytest.param([("user", "")], "<user> ", id="empty-content"),
            pytest.param([("user", "hello")], "<user> hello", id="single-message"),
            pytest.param(
                [("system", "Translate."), ("user", "Hello!"), ("assistant", "Hi!")],
                "<system> Translate.\n<user> Hello!\n<assistant> Hi!",
                id="multiple-roles",
            ),
            pytest.param(
                [("user", "first\nsecond")],
                "<user> first\nsecond",
                id="multiline-content",
            ),
        ],
    )
    def test_direct_call(self, entries, expected, echo_client):
        messages = [dict(role=role, content=content) for role, content in entries]
        assert af.lm.complete(messages, model="any-model") == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_traced_and_batched_execution(self, executor, echo_client):

        def program(text):
            prompt = af.string.format("Hello, {text}!", text=text)
            return af.lm.complete(
                [
                    dict(role="system", content="Translate to Korean."),
                    dict(role="user", content=prompt),
                ],
                model="echo",
            )

        ir = af.trace(program)("name")
        result = executor(ir, "World")
        assert result == "<system> Translate to Korean.\n<user> Hello, World!"
        batched = af.batch(ir)
        result = executor(batched, ["A", "B"])
        assert result == [
            "<system> Translate to Korean.\n<user> Hello, A!",
            "<system> Translate to Korean.\n<user> Hello, B!",
        ]

    def test_restores_outer_client_after_exception(self):
        messages = [dict(role="user", content="hello")]
        with af.lm.client(EchoRouter()):
            with pytest.raises(ValueError, match="stop"):
                with af.lm.client(af.lm.EchoClient()):
                    assert af.lm.complete(messages, model="m1") == "<user> hello"
                    raise ValueError("stop")
            assert af.lm.complete(messages, model="m1") == "m1|hello"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_custom_renderer_receives_all_messages(self, executor):

        def render(messages):
            return " | ".join((message["content"] for message in messages))

        def program(text):
            return af.lm.complete(
                [dict(role="system", content="Translate."), dict(role="user", content=text)],
                model="echo",
            )

        ir = af.trace(program)("text")
        with af.lm.client(af.lm.EchoClient(render=render)):
            result = executor(ir, "Hello!")
            assert result == "Translate. | Hello!"
            batched = af.batch(ir)
            result = executor(batched, ["A", "B"])
            assert result == ["Translate. | A", "Translate. | B"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize("text", ['Hello "world"!', "Hello!"], ids=["quoted", "plain"])
    def test_custom_renderer_supplies_schema_json(self, executor, text):
        def render(messages):
            return json.dumps({"text": messages[-1]["content"]})

        program = lm_program(af.lm.generate, model="echo", schema={"text": af.Str()})
        ir = af.trace(program)("text")
        with af.lm.client(af.lm.EchoClient(render=render)):
            assert executor(ir, text) == {"text": text}

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_schema_call_rejects_role_prefixed_text(self, executor):
        program = lm_program(af.lm.generate, model="echo", schema={"text": af.Str()})
        ir = af.trace(program)("json")
        with af.lm.client(af.lm.EchoClient()):
            with pytest.raises(json.JSONDecodeError):
                executor(ir, '{"text": "hello"}')


@pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
@pytest.mark.parametrize(
    "primitive, params, client_type, expected_primal",
    [
        pytest.param(af.lm.complete, {}, EchoRouter, "m1|hello", id="complete"),
        pytest.param(
            af.lm.generate,
            {"schema": {"text": af.Str(), "score": af.Float()}},
            SchemaRouter,
            {"text": "m1|hello", "score": 0.5},
            id="generate",
        ),
    ],
)
def test_pushforward(executor, primitive, params, client_type, expected_primal):
    ir = af.pushforward(af.trace(lm_program(primitive, **params))("test", "gpt-5.5"))
    args = (("hello", "m1"), ("tangent", "ignored model tangent"))
    with af.lm.client(client_type()):
        primal, tangent = executor(ir, *args)
    if params:
        assert primal["score"] == expected_primal["score"]
        primal_values = fill_values(primal["text"].removeprefix("m1|"))["context"]
        assert primal_values == {"messages": {"0": {"content": "hello", "role": "user"}}}
    else:
        assert primal == expected_primal
    tangent_text = tangent if type(tangent) is str else tangent["text"]
    original, change = parse_change_prompt(fill_request_text(tangent_text.removeprefix("m1|")))
    assert original == {"messages": [{"role": "user", "content": "hello"}]}
    assert change == {"messages": [{"role": "", "content": "tangent"}]}


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(af.lm.LiteLLMClient.completion, id="sync"),
        pytest.param(
            lambda client, **kwargs: asyncio.run(client.acompletion(**kwargs)),
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

    monkeypatch.setattr(af.lm, "completion", complete)
    monkeypatch.setattr(af.lm, "acompletion", acomplete)
    client = af.lm.LiteLLMClient()
    kwargs = dict(
        messages=[dict(role="user", content="hello")],
        model="m1",
        temperature=0.7,
        max_tokens=128,
    )
    result = executor(client, **kwargs)
    assert result is response
    assert calls == [kwargs]
