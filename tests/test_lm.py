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
from autoform.schemas import emit_json_schema, parse_json_value
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


class EchoRouter:
    def completion(self, *, messages, model, **kwargs):
        assert kwargs == {}
        return fake_response(f"{model}|{messages[-1]['content']}")

    async def acompletion(self, **kwargs):
        return self.completion(**kwargs)


class SchemaRouter(EchoRouter):
    __slots__ = ["response_formats"]

    def __init__(self):
        self.response_formats = []

    def completion(self, *, messages: list[dict], model: str, response_format, **kwargs):
        assert kwargs == {}
        self.response_formats.append(response_format)
        return fake_response(
            json.dumps({
                "text": f"{model}|{messages[-1]['content']}",
                "score": 0.5,
            })
        )


class SchemaGradientRouter(EchoRouter):
    __slots__ = ["calls"]

    def __init__(self):
        self.calls = []

    def completion(self, *, messages: list[dict], model: str, response_format=None, **kwargs):
        assert kwargs == {}
        self.calls.append(dict(messages=messages, model=model, response_format=response_format))
        if response_format is not None:
            properties = response_format["json_schema"]["schema"]["properties"]
            if "1" in properties:
                feedback = {"1": "model feedback"}
                if "0" in properties:
                    feedback["0"] = {
                        key: dict(role=f"role feedback {key}", content=f"content feedback {key}")
                        for key in properties["0"]["properties"]
                    }
                return fake_response(json.dumps(feedback))
        responses = {
            False: "input feedback",
            True: json.dumps({"text": "Recursion calls itself.", "score": 0.92}),
        }
        return fake_response(responses[response_format is not None])


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
            assert response_format["json_schema"]["schema"] == emit_json_schema(schema)
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


def test_emit_json_schema_rejects_non_json_enum_values():
    enum = af.Enum(object())

    with pytest.raises(TypeError, match="Enum values must be str, int, float, or bool"):
        emit_json_schema({"kind": enum})


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
    prompt = gradient_client.calls[-1]["messages"][-1]["content"]
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


def test_emit_json_schema_with_value_preserves_generated_structure():
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
    assert af.lm.emit_json_schema(schema, value=value) == {
        "fields": {"0": -0.2, "1": {"0": "", "1": -1, "2": False, "3": "feedback"}}
    }
    with pytest.raises(ValueError):
        af.lm.emit_json_schema(schema, value=Answer([], value.metadata))


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
    assert emit_json_schema(schema) is None
    assert parse_json_value(schema, None) == expected


@pytest.mark.parametrize("parse", [parse_json_value, af.lm.parse_json_value], ids=["schemas", "lm"])
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
def test_parse_json_value_rejects_invalid_objects(parse, value):
    with pytest.raises(ValueError):
        parse({"0": af.Str()}, value)


@pytest.mark.parametrize("parse", [parse_json_value, af.lm.parse_json_value], ids=["schemas", "lm"])
def test_parse_json_value_rebuilds_nested_containers_from_objects(parse):
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

    json_schema = emit_json_schema(answer)

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
    assert parse_json_value(
        answer,
        {"name": "subject", "kind": "summary", "score": 1},
    ) == {
        "name": "subject",
        "kind": "summary",
        "score": 1.0,
    }


def test_schema_dsl_reconstructs_unemitted_subtree():
    @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
    class Answer:
        decision: object
        details: object

    details = {
        "literal": "fixed",
        "nothing": None,
    }
    answer = Answer(af.Str(), details)

    json_schema = emit_json_schema(answer)

    assert json_schema == {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
        "additionalProperties": False,
    }
    parsed = parse_json_value(answer, {"decision": "accept"})
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

    json_schema = emit_json_schema(answer)

    assert json_schema == {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
        "additionalProperties": False,
    }
    parsed = parse_json_value(answer, {"decision": "accept"})
    assert parsed["decision"] == "accept"
    assert parsed["metadata"] is metadata


def test_lm_schema_trace_rejects_untraceable_static_leaf():
    metadata = object()

    program = lm_program(af.lm.generate, schema={"decision": af.Str(), "metadata": metadata})

    with pytest.raises(TypeError, match="Static schema leaf must be traceable"):
        af.trace(program)("test")


def test_schema_dsl_builds_string_constraints():
    answer = {"name": af.Str(min=2, max=4, pattern=r"^[a-z]+$")}

    json_schema = emit_json_schema(answer)

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
    assert parse_json_value(answer, {"name": "okay"}) == {"name": "okay"}
    with pytest.raises(ValueError, match="Expected string with length >= 2"):
        parse_json_value(answer, {"name": "x"})
    with pytest.raises(ValueError, match="Expected string with length <= 4"):
        parse_json_value(answer, {"name": "hello"})
    with pytest.raises(ValueError, match="Expected string matching"):
        parse_json_value(answer, {"name": "OK"})


def test_schema_dsl_builds_number_constraints():
    answer = {
        "count": af.Int(min=-2, max=2),
        "score": af.Float(min=0, max=1),
    }

    json_schema = emit_json_schema(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": -2, "maximum": 2},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["count", "score"],
        "additionalProperties": False,
    }
    assert parse_json_value(answer, {"count": 0, "score": 1}) == {"count": 0, "score": 1.0}
    with pytest.raises(ValueError, match="Expected integer >= -2"):
        parse_json_value(answer, {"count": -3, "score": 0.5})
    with pytest.raises(ValueError, match="Expected integer <= 2"):
        parse_json_value(answer, {"count": 3, "score": 0.5})
    with pytest.raises(ValueError, match="Expected number >= 0"):
        parse_json_value(answer, {"count": 0, "score": -0.1})
    with pytest.raises(ValueError, match="Expected number <= 1"):
        parse_json_value(answer, {"count": 0, "score": 1.1})


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

    json_schema = emit_json_schema(answer)

    assert json_schema == {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["text", "score"],
        "additionalProperties": False,
    }
    assert parse_json_value(answer, {"text": "hello", "score": 2}) == Answer("hello", 2.0)


def test_schema_dsl_reports_value_errors():
    count = {"count": af.Int()}
    score = {"score": af.Float()}

    with pytest.raises(ValueError, match="Expected integer"):
        parse_json_value(count, {"count": True})
    with pytest.raises(ValueError, match="Expected number"):
        parse_json_value(score, {"score": "bad"})


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
            pytest.param(af.lm.complete, {}, af.lm.echo_messages, id="complete"),
            pytest.param(
                af.lm.generate,
                {"schema": af.Str()},
                lambda messages: json.dumps(af.lm.echo_messages(messages)),
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
            pytest.param(af.lm.complete, {}, af.lm.echo_messages, id="complete"),
            pytest.param(
                af.lm.generate,
                {"schema": af.Str()},
                lambda messages: json.dumps(af.lm.echo_messages(messages)),
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
        assert executor(ir, (messages, "echo"), (tangents, "ignored")) == (
            "<system> hello\n<user> world",
            "<system> \n<user> tangent",
        )

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
        assert f"INPUT: {(messages, 'm2')}" in message["content"]
        response_format = backward["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        properties = response_format["json_schema"]["schema"]["properties"]
        assert properties["1"] == {
            "type": "string",
            "description": "Feedback for input at (1,): 'm2'.",
        }
        for i, m in enumerate(messages):
            fields = properties["0"]["properties"][str(i)]["properties"]
            assert fields == {
                key: {
                    "type": "string",
                    "description": f"Feedback for input at {(0, i, key)}: {value!r}.",
                }
                for key, value in m.items()
            }
        assert context == [
            dict(
                role="system",
                content="Translate output feedback into feedback on the corresponding input fields.",
            )
        ]
        output, feedback = message["content"].split("OUTPUT: ", 1)[1].split(" OUTPUT FEEDBACK: ")
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
    "primitive, params, client_type, expected",
    [
        pytest.param(af.lm.complete, {}, EchoRouter, ("m1|hello", "m1|tangent"), id="complete"),
        pytest.param(
            af.lm.generate,
            {"schema": {"text": af.Str(), "score": af.Float()}},
            SchemaRouter,
            ({"text": "m1|hello", "score": 0.5}, {"text": "m1|tangent", "score": 0.5}),
            id="generate",
        ),
    ],
)
def test_pushforward(executor, primitive, params, client_type, expected):
    ir = af.pushforward(af.trace(lm_program(primitive, **params))("test", "gpt-5.5"))
    args = (("hello", "m1"), ("tangent", "ignored model tangent"))
    with af.lm.client(client_type()):
        actual = executor(ir, *args)
    assert actual == expected


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
