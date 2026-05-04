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

import json

import pytest

import autoform as af
from autoform.schemas import make_json_schema_and_parser


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class SchemaRouter:
    __slots__ = ["response_formats"]

    def __init__(self):
        self.response_formats = []

    def completion(self, *, messages: list[dict], model: str, response_format, **kwargs):
        assert kwargs == {}
        self.response_formats.append(response_format)
        return FakeResponse(
            json.dumps({
                "text": f"{model}|{messages[-1]['content']}",
                "score": 0.5,
            })
        )

    async def acompletion(self, *, messages: list[dict], model: str, response_format, **kwargs):
        return self.completion(
            messages=messages,
            model=model,
            response_format=response_format,
            **kwargs,
        )


class SchemaGradientRouter:
    __slots__ = ["calls"]

    def __init__(self):
        self.calls = []

    def completion(self, *, messages: list[dict], model: str, response_format=None, **kwargs):
        assert kwargs == {}
        self.calls.append(dict(messages=messages, model=model, response_format=response_format))
        if response_format is None:
            return FakeResponse("input feedback")
        return FakeResponse(json.dumps({"text": "Recursion calls itself.", "score": 0.92}))

    async def acompletion(
        self, *, messages: list[dict], model: str, response_format=None, **kwargs
    ):
        return self.completion(
            messages=messages,
            model=model,
            response_format=response_format,
            **kwargs,
        )


def test_lm_schema_call_executes_with_response_format():
    router = SchemaRouter()
    answer = {
        "text": af.Str(min=1, max=80),
        "score": af.Float(min=0, max=1),
    }

    with af.lm_client(router):
        result = af.lm_schema_call(
            [dict(role="user", content="hello")],
            model="m1",
            schema=answer,
        )

    assert result == {"text": "m1|hello", "score": 0.5}
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


def test_lm_schema_call_traces_schema_as_static_param():
    answer = {
        "text": af.Str() @ af.Doc("Short text."),
        "score": af.Float(),
    } @ af.Doc("Answer object.")

    def program(prompt: str, model: str):
        result = af.lm_schema_call(
            [dict(role="user", content=prompt)],
            model=model,
            schema=answer,
        )
        return af.format("{}", result["text"])

    ir = af.trace(program)("test", "gpt-5.2")
    assert [eqn.prim.name for eqn in ir.ir_eqns] == ["lm_schema_call", "format"]
    assert make_json_schema_and_parser(ir.ir_eqns[0].params["schema"])[0] == make_json_schema_and_parser(answer)[0]
    assert "model" not in ir.ir_eqns[0].params
    assert isinstance(ir.ir_eqns[0].in_ir_tree[1], af.core.IRVar)
    assert isinstance(ir.ir_eqns[0].out_ir_tree["text"], af.core.IRVar)
    assert isinstance(ir.ir_eqns[0].out_ir_tree["score"], af.core.IRVar)

    with af.lm_client(SchemaRouter()):
        assert ir.call("hello", "m1") == "m1|hello"


def test_batch_lm_schema_call_supports_variable_models():
    answer = {
        "text": af.Str(),
        "score": af.Float(),
    }

    def program(prompt: str, model: str):
        return af.lm_schema_call(
            [dict(role="user", content=prompt)],
            model=model,
            schema=answer,
        )

    ir = af.trace(program)("test", "gpt-5.2")
    batched_ir = af.batch(ir, in_axes=(True, True))

    with af.lm_client(SchemaRouter()):
        result = batched_ir.call(["hello", "goodbye"], ["m1", "m2"])

    assert result == {
        "text": ["m1|hello", "m2|goodbye"],
        "score": [0.5, 0.5],
    }


def test_lm_schema_call_pullback_uses_schema_cotangent():
    router = SchemaGradientRouter()
    answer = {
        "text": af.Str(min=1, max=80),
        "score": af.Float(min=0, max=1),
    }

    def program(prompt: str):
        return af.lm_schema_call(
            [dict(role="user", content=prompt)],
            model="m1",
            schema=answer,
        )

    ir = af.trace(program)("seed")
    with af.lm_client(router):
        output, gradient = af.pullback(ir).call(
            ("Explain recursion.",),
            {"text": "too terse", "score": "overconfident"},
        )

    assert output == {"text": "Recursion calls itself.", "score": 0.92}
    assert gradient == ("input feedback",)

    prompt = router.calls[-1]["messages"][0]["content"]
    assert "STRUCTURED OUTPUT FEEDBACK:" in prompt
    assert "Each field is one leaf of the generated output." in prompt
    assert "Path locates the field from the root output object." in prompt
    assert "Value is the generated value." in prompt
    assert "empty feedback means no change" in prompt
    assert "$['text']" in prompt
    assert "value: 'Recursion calls itself.'" in prompt
    assert "feedback: 'too terse'" in prompt
    assert "$['score']" in prompt
    assert "value: 0.92" in prompt
    assert "feedback: 'overconfident'" in prompt


def test_lm_schema_call_pullback_treats_zero_as_no_feedback():
    router = SchemaGradientRouter()
    answer = {
        "text": af.Str(),
        "score": af.Float(min=0, max=1),
    }

    def program(prompt: str):
        result = af.lm_schema_call(
            [dict(role="user", content=prompt)],
            model="m1",
            schema=answer,
        )
        return af.format("{}", result["text"])

    ir = af.trace(program)("seed")
    with af.lm_client(router):
        output, gradient = af.pullback(ir).call(("Explain recursion.",), "too terse")

    assert output == "Recursion calls itself."
    assert gradient == ("input feedback",)

    prompt = router.calls[-1]["messages"][0]["content"]
    assert "$['text']" in prompt
    assert "feedback: 'too terse'" in prompt
    assert "$['score']" in prompt
    assert "feedback: 'No feedback'" in prompt


def test_lm_schema_call_pullback_rejects_non_text_schema_cotangent():
    router = SchemaGradientRouter()
    answer = {
        "text": af.Str(),
        "score": af.Float(min=0, max=1),
    }

    def program(prompt: str):
        return af.lm_schema_call(
            [dict(role="user", content=prompt)],
            model="m1",
            schema=answer,
        )

    ir = af.trace(program)("seed")
    with af.lm_client(router), pytest.raises(TypeError, match="must be text"):
        af.pullback(ir).call(("Explain recursion.",), {"text": "too terse", "score": -0.2})
