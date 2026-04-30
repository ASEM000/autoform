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

import autoform as af
from autoform.schemas import build


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
    assert build(ir.ir_eqns[0].params["schema"])[0] == build(answer)[0]
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
