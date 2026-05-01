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

"""LM (Language Model) primitives"""

from __future__ import annotations

import asyncio
import functools as ft
import json
from collections import defaultdict
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from litellm import acompletion, completion

from autoform.ad import Zero, is_zero, materialize
from autoform.core import (
    EvalType,
    Prim,
    TypedAVal,
    abstract_rules,
    batch_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    typeof,
)
from autoform.schemas import Bool, Documented, Enum, Float, Int, Str, build, is_schema_spec
from autoform.utils import (
    Struct,
    Tree,
    batch_index,
    batch_spec,
    batch_transpose,
    struct_type_tree,
    treelib,
)


@runtime_checkable
class LMClient(Protocol):
    def completion(self, *, messages: list[dict], model: str, **kwargs) -> Any: ...
    def acompletion(self, *, messages: list[dict], model: str, **kwargs) -> Awaitable[Any]: ...


class LiteLLMClient:
    __slots__ = []

    def completion(self, *, messages: list[dict], model: str, **kwargs) -> Any:
        return completion(messages=messages, model=model, **kwargs)

    async def acompletion(self, *, messages: list[dict], model: str, **kwargs) -> Any:
        return await acompletion(messages=messages, model=model, **kwargs)


active_client: ContextVar[LMClient] = ContextVar("active_client", default=LiteLLMClient())


@contextmanager
def lm_client(client: LMClient) -> Generator[LMClient, None, None]:
    """Set the LM client for all lm primitives.

    The client must expose ``.completion()`` and ``.acompletion()`` matching
    LiteLLM's chat completion signature.

    Acceptable clients include the default direct LiteLLM adapter, a configured
    ``litellm.Router``, or any wrapper object that forwards those two methods
    while preserving the LiteLLM request and response shapes.

    Example:
        >>> import autoform as af
        >>> from litellm import Router  # doctest: +SKIP
        >>> client = Router(  # doctest: +SKIP
        ...     model_list=[
        ...         dict(model_name="gpt-4", litellm_params=dict(model="gpt-5.2")),
        ...     ],
        ...     max_parallel_requests=10,
        ... )
        >>> with af.lm_client(client):  # doctest: +SKIP
        ...     ir.call(inputs)
    """
    assert isinstance(client, LMClient), f"Expected LMClient instance, got {type(client)}"
    token = active_client.set(client)
    try:
        yield client
    finally:
        active_client.reset(token)


# ==================================================================================================
# LM CALL
# ==================================================================================================

lm_call_p = Prim("lm_call")

# TODO(asem): take a look into this
GRAD_PROMPT = """Given this LLM interaction:

INPUT: {content}
OUTPUT: {out}
FEEDBACK ON OUTPUT: {out_cotangent}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""


def lm_call(messages: list[dict[str, str]], /, *, model: str) -> str:
    """Calls a language model with the given messages and model name using LiteLLM.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The model name or active client model alias to use (e.g., "gpt-5.2").

    Returns:
        The content of the model's response as a string.

    Use :func:`lm_client` to configure provider-specific settings like ``max_tokens``.

    Example:
        >>> import autoform as af
        >>> def program(name: str) -> str:
        ...     greeting = af.format("Hello, {}!", name)
        ...     sys = dict(role="system", content="translate the greeting to Korean")
        ...     usr = dict(role="user", content=greeting)
        ...     greeting = af.lm_call([sys, usr], model="gpt-5.2")
        ...     return greeting
        >>> ir = af.trace(program)("World") # doctest: +SKIP
        >>> result = ir.call("x0") # doctest: +SKIP

    Example with :func:`lm_client`:
        >>> import autoform as af
        >>> from litellm import Router  # doctest: +SKIP
        >>> params_1024 = dict(model="gpt-5.2", max_tokens=1024)
        >>> params_512 = dict(model="gpt-5.2", max_tokens=512)
        >>> model_list = [
        ...     dict(model_name="gpt-5.2-1024", litellm_params=params_1024),
        ...     dict(model_name="gpt-5.2-512", litellm_params=params_512),
        ... ]
        >>> router = Router(model_list=model_list)  # doctest: +SKIP
        >>> def program(text: str, model: str):
        ...     msg = [{"role": "user", "content": af.format("Explain {} in one line.", text)}]
        ...     answer = af.lm_call(msg, model=model)
        ...     return af.concat("Answer: ", answer)
        >>> ir = af.trace(program)("topic", "model")
        >>> model_names = ["gpt-5.2-1024", "gpt-5.2-512"]
        >>> with af.lm_client(router):  # doctest: +SKIP
        ...     result = af.batch(ir, in_axes=(False, True)).call("AI", model_names)
    """
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return lm_call_p.bind((contents, model), roles=roles)


def impl_lm_call(in_tree: Tree, /, *, roles: list[str]) -> str:
    contents, model = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    response = active_client.get().completion(
        messages=messages,
        model=model,
    )
    return response.choices[0].message.content


async def aimpl_lm_call(in_tree: Tree, /, *, roles: list[str]) -> str:
    contents, model = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    response = await active_client.get().acompletion(
        messages=messages,
        model=model,
    )
    return response.choices[0].message.content


def abstract_lm_call(in_tree: Tree, /, *, roles: list[str]) -> EvalType:
    contents, model = in_tree
    assert all(typeof(x) is str for x in contents), f"Expected string messages, got {contents!r}"
    assert typeof(model) is str, f"`lm_call` expects a string model, got {model!r}"
    return TypedAVal(str)


def pushforward_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model)
    p_resp = lm_call_p.bind(p_tree, roles=roles)
    t_tree = (materialize(tangent_contents), primal_model)
    t_resp = lm_call_p.bind(t_tree, roles=roles)
    return p_resp, t_resp


async def apush_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(lm_call_p.abind, roles=roles)
    p_tree = (primal_contents, primal_model)
    t_tree = (materialize(tangent_contents), primal_model)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    contents, model = in_tree
    out = lm_call_p.bind((contents, model), roles=roles)
    residuals = (contents, model, out)
    return out, residuals


async def apull_fwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    contents, model = in_tree
    out = await lm_call_p.abind((contents, model), roles=roles)
    residuals = (contents, model, out)
    return out, residuals


def pullback_bwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.bind(([grad_prompt], model), roles=["user"])
        grads.append(grad_out)
    return grads, Zero(str)


async def apull_bwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.abind(([prompt], model), roles=["user"])
        return await grad_out

    return (
        await asyncio.gather(*[grad(c) for c in contents]),
        Zero(str),
    )


def batch_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if (spec := batch_spec(in_values, in_batched)) is None:
        return lm_call_p.bind(in_values, roles=roles), False

    unbatch = ft.partial(batch_index, in_values, in_batched)
    results = [lm_call_p.bind(unbatch(b), roles=roles) for b in range(batch_size)]
    out_tree = spec.unflatten(results)
    return out_tree, True


async def abatch_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if (spec := batch_spec(in_values, in_batched)) is None:
        return await lm_call_p.abind(in_values, roles=roles), False

    unbatch = ft.partial(batch_index, in_values, in_batched)
    abind = ft.partial(lm_call_p.abind, roles=roles)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_tree = spec.unflatten(results)
    return out_tree, True


impl_rules.set(lm_call_p, impl_lm_call)
impl_rules.aset(lm_call_p, aimpl_lm_call)
abstract_rules.set(lm_call_p, abstract_lm_call)
push_rules.set(lm_call_p, pushforward_lm_call)
push_rules.aset(lm_call_p, apush_lm_call)
pull_fwd_rules.set(lm_call_p, pullback_fwd_lm_call)
pull_fwd_rules.aset(lm_call_p, apull_fwd_lm_call)
pull_bwd_rules.set(lm_call_p, pullback_bwd_lm_call)
pull_bwd_rules.aset(lm_call_p, apull_bwd_lm_call)
batch_rules.set(lm_call_p, batch_lm_call)
batch_rules.aset(lm_call_p, abatch_lm_call)

# ==================================================================================================
# STRUCT LM CALL
# ==================================================================================================

lm_struct_call_p = Prim("lm_struct_call")


def lm_struct_call[T: Struct](
    messages: list[dict[str, str]],
    /,
    *,
    model: str,
    struct: type[T],
) -> T:
    """Calls a language model with structured output using response_format.

    Uses LLM's built-in JSON mode with a schema to extract structured
    data. The model response is automatically parsed and validated.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use.
        struct: A ``Struct`` subclass defining the output schema.

    Returns:
        A validated instance of the struct type.

    Example:
        >>> import autoform as af
        >>> class Answer(af.Struct):
        ...     reasoning: str
        ...     answer: int
        >>> def solver(question):
        ...     messages = [{"role": "user", "content": question}]
        ...     return af.lm_struct_call(messages, model="gpt-5.2", struct=Answer)
        >>> ir = af.trace(solver)("What is 2+2?")  # doctest: +SKIP
        >>> result = ir.call("What is 2+2?")  # doctest: +SKIP
        >>> result.answer  # doctest: +SKIP
        4
    """
    assert issubclass(struct, Struct), "struct must be a subclass of ``Struct``"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    in_tree = (contents, model)
    return lm_struct_call_p.bind(in_tree, roles=roles, struct=struct)


def impl_lm_struct_call[T: Struct](
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[T],
) -> T:
    contents, model = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = active_client.get().completion(
        messages=messages,
        model=model,
        response_format=struct,
    )
    return struct.model_validate_json(resp.choices[0].message.content)


async def aimpl_lm_struct_call[T: Struct](
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[T],
) -> T:
    contents, model = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = await active_client.get().acompletion(
        messages=messages,
        model=model,
        response_format=struct,
    )
    return struct.model_validate_json(resp.choices[0].message.content)


def abstract_lm_struct_call[T: Struct](
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[T],
) -> Tree:
    contents, model = in_tree
    assert all(typeof(x) is str for x in contents), f"Expected string messages, got {contents!r}"
    assert typeof(model) is str, f"Expected string model, got {model!r}"
    return treelib.map(TypedAVal, struct_type_tree(struct))


def pushforward_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model)
    t_tree = (materialize(tangent_contents), primal_model)
    p_resp = lm_struct_call_p.bind(p_tree, roles=roles, struct=struct)
    t_resp = lm_struct_call_p.bind(t_tree, roles=roles, struct=struct)
    return p_resp, t_resp


async def apush_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(lm_struct_call_p.abind, roles=roles, struct=struct)
    p_tree = (primal_contents, primal_model)
    t_tree = (materialize(tangent_contents), primal_model)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    contents, model = in_tree
    out = lm_struct_call_p.bind(in_tree, roles=roles, struct=struct)
    residuals = (contents, model, out)
    return out, residuals


async def apull_fwd_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    contents, model = in_tree
    out = await lm_struct_call_p.abind(in_tree, roles=roles, struct=struct)
    residuals = (contents, model, out)
    return out, residuals


def pullback_bwd_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.bind(([grad_prompt], model), roles=["user"])
        grads.append(grad_out)
    return grads, Zero(str)


async def apull_bwd_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.abind(([prompt], model), roles=["user"])
        return await grad_out

    return (
        await asyncio.gather(*[grad(c) for c in contents]),
        Zero(str),
    )


def batch_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if batch_spec(in_values, in_batched) is None:
        result = lm_struct_call_p.bind(in_values, roles=roles, struct=struct)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    bind = ft.partial(lm_struct_call_p.bind, roles=roles, struct=struct)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, results)
    return out_ib, out_batched


async def abatch_lm_struct_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if batch_spec(in_values, in_batched) is None:
        result = await lm_struct_call_p.abind(in_values, roles=roles, struct=struct)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    abind = ft.partial(lm_struct_call_p.abind, roles=roles, struct=struct)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, list(results))
    return out_ib, out_batched


impl_rules.set(lm_struct_call_p, impl_lm_struct_call)
impl_rules.aset(lm_struct_call_p, aimpl_lm_struct_call)
abstract_rules.set(lm_struct_call_p, abstract_lm_struct_call)
push_rules.set(lm_struct_call_p, pushforward_lm_struct_call)
push_rules.aset(lm_struct_call_p, apush_lm_struct_call)
pull_fwd_rules.set(lm_struct_call_p, pullback_fwd_lm_struct_call)
pull_fwd_rules.aset(lm_struct_call_p, apull_fwd_lm_struct_call)
pull_bwd_rules.set(lm_struct_call_p, pullback_bwd_lm_struct_call)
pull_bwd_rules.aset(lm_struct_call_p, apull_bwd_lm_struct_call)
batch_rules.set(lm_struct_call_p, batch_lm_struct_call)
batch_rules.aset(lm_struct_call_p, abatch_lm_struct_call)

# ==================================================================================================
# LM SCHEMA CALL
# ==================================================================================================

lm_schema_call_p = Prim("lm_schema_call")


SCHEMA_GRAD_PROMPT = """Given this LLM interaction:

INPUT: {content}
STRUCTURED OUTPUT FEEDBACK:
{feedback}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise.

- Each field is one leaf of the generated output.
- Path locates the field from the root output object.
- Value is the generated value.
- Feedback is natural-language feedback for that field; empty feedback means no change.
"""


def lm_schema_call(
    messages: list[dict[str, str]],
    /,
    *,
    model: str,
    schema: Any,
) -> Any:
    """Calls a language model with an autoform schema response format.

    The schema tree is built from nodes such as :class:`autoform.Int`,
    :class:`autoform.Enum`, and the other schema nodes exported by autoform.

    Example:
        >>> import autoform as af
        >>> answer = {
        ...     "name": af.Str() @ af.Doc("Subject name."),
        ...     "kind": af.Enum("summary", "definition") @ af.Doc("Answer kind."),
        ...     "score": af.Float(min=0, max=1) @ af.Doc("Confidence score."),
        ... } @ af.Doc("Answer object.")

    Example with a registered pytree:
        >>> import optree
        >>> import autoform as af
        >>> @optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
        ... class Answer:
        ...     answer: float
        ...     reasoning: str
        >>> schema = Answer(
        ...     answer=af.Float() @ af.Doc("The numeric answer."),
        ...     reasoning=af.Str() @ af.Doc("The reasoning behind the answer."),
        ... )
        >>> msgs = [dict(role="user", content="1 + 1?")]
        >>> output = af.lm_schema_call(  # doctest: +SKIP
        ...     msgs,
        ...     model="openai/gpt-5.2",
        ...     schema=schema,
        ... )
        >>> output  # doctest: +SKIP
        Answer(answer=2.0, reasoning='Adding 1 and 1 gives 2.')
    """
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return lm_schema_call_p.bind((contents, model), roles=roles, schema=schema)


def schema_response_format(json_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "autoform_schema",
            "strict": True,
            "schema": json_schema,
        },
    }


def impl_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> Any:
    contents, model = in_tree
    json_schema, parse = build(schema)
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = active_client.get().completion(
        messages=messages,
        model=model,
        response_format=schema_response_format(json_schema),
    )
    return parse(json.loads(resp.choices[0].message.content))


async def aimpl_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> Any:
    contents, model = in_tree
    json_schema, parse = build(schema)
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = await active_client.get().acompletion(
        messages=messages,
        model=model,
        response_format=schema_response_format(json_schema),
    )
    return parse(json.loads(resp.choices[0].message.content))


schema_abstract_rules = defaultdict(lambda: lambda node: node)
schema_abstract_rules[Str] = lambda _: TypedAVal(str)
schema_abstract_rules[Int] = lambda _: TypedAVal(int)
schema_abstract_rules[Float] = lambda _: TypedAVal(float)
schema_abstract_rules[Bool] = lambda _: TypedAVal(bool)
schema_abstract_rules[Enum] = lambda _: TypedAVal(type(_.values[0]))
schema_abstract_rules[Documented] = lambda s: s.value


def schema_abstract_tree(schema: Any) -> Tree:
    build(schema)
    leaves, spec = treelib.flatten(schema, is_leaf=is_schema_spec, none_is_leaf=True)
    return spec.traverse(leaves, schema_abstract_rule, schema_abstract_rule)


def schema_abstract_rule(node: Any) -> Any:
    return schema_abstract_rules[type(node)](node)


def abstract_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> Tree:
    contents, model = in_tree
    assert all(typeof(x) is str for x in contents), f"Expected string messages, got {contents!r}"
    assert typeof(model) is str, f"Expected string model, got {model!r}"
    return schema_abstract_tree(schema)


def pushforward_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model)
    t_tree = (materialize(tangent_contents), primal_model)
    p_resp = lm_schema_call_p.bind(p_tree, roles=roles, schema=schema)
    t_resp = lm_schema_call_p.bind(t_tree, roles=roles, schema=schema)
    return p_resp, t_resp


async def apush_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(lm_schema_call_p.abind, roles=roles, schema=schema)
    p_tree = (primal_contents, primal_model)
    t_tree = (materialize(tangent_contents), primal_model)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> tuple[Tree, Tree]:
    contents, model = in_tree
    out = lm_schema_call_p.bind(in_tree, roles=roles, schema=schema)
    residuals = (contents, model, out)
    return out, residuals


async def apull_fwd_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> tuple[Tree, Tree]:
    contents, model = in_tree
    out = await lm_schema_call_p.abind(in_tree, roles=roles, schema=schema)
    residuals = (contents, model, out)
    return out, residuals


def build_cotangent_schema_summary(out: Tree, cotangent: Tree) -> str:

    def validate_schema_feedback(path: str, feedback: Any) -> str:
        if is_zero(feedback):
            return "No feedback"
        if type(feedback) is str:
            return feedback
        raise TypeError(f"{path}: schema output cotangent leaves must be text, got {feedback!r}")

    out_leaves, out_spec = treelib.flatten(out)
    cotangents = out_spec.flatten_up_to(cotangent)
    lines = ["Fields:"]

    for accessor, value, feedback in zip(out_spec.accessors(), out_leaves, cotangents, strict=True):
        feedback = validate_schema_feedback(accessor.codify("$"), feedback)
        lines.append(accessor.codify("$"))
        lines.append(f"\tvalue: {value!r}")
        lines.append(f"\tfeedback: {feedback!r}")
    return "\n".join(lines).expandtabs(2)


def pullback_bwd_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> Tree:
    residuals, out_cotangent = in_tree
    contents, model, out = residuals
    feedback = build_cotangent_schema_summary(out, out_cotangent)
    grads = []
    for content in contents:
        grad_prompt = SCHEMA_GRAD_PROMPT.format(content=content, feedback=feedback)
        grad_out = lm_call_p.bind(([grad_prompt], model), roles=["user"])
        grads.append(grad_out)
    return grads, Zero(str)


async def apull_bwd_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> Tree:
    residuals, out_cotangent = in_tree
    contents, model, out = residuals
    feedback = build_cotangent_schema_summary(out, out_cotangent)

    async def grad(c):
        prompt = SCHEMA_GRAD_PROMPT.format(content=c, feedback=feedback)
        grad_out = lm_call_p.abind(([prompt], model), roles=["user"])
        return await grad_out

    return (await asyncio.gather(*[grad(c) for c in contents]), Zero(str))


def batch_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Any,
) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if batch_spec(in_values, in_batched) is None:
        result = lm_schema_call_p.bind(in_values, roles=roles, schema=schema)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    bind = ft.partial(lm_schema_call_p.bind, roles=roles, schema=schema)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, results)
    return out_ib, out_batched


async def abatch_lm_schema_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    schema: Tree,
) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if batch_spec(in_values, in_batched) is None:
        result = await lm_schema_call_p.abind(in_values, roles=roles, schema=schema)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    abind = ft.partial(lm_schema_call_p.abind, roles=roles, schema=schema)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, list(results))
    return out_ib, out_batched


impl_rules.set(lm_schema_call_p, impl_lm_schema_call)
impl_rules.aset(lm_schema_call_p, aimpl_lm_schema_call)
abstract_rules.set(lm_schema_call_p, abstract_lm_schema_call)
push_rules.set(lm_schema_call_p, pushforward_lm_schema_call)
push_rules.aset(lm_schema_call_p, apush_lm_schema_call)
pull_fwd_rules.set(lm_schema_call_p, pullback_fwd_lm_schema_call)
pull_fwd_rules.aset(lm_schema_call_p, apull_fwd_lm_schema_call)
pull_bwd_rules.set(lm_schema_call_p, pullback_bwd_lm_schema_call)
pull_bwd_rules.aset(lm_schema_call_p, apull_bwd_lm_schema_call)
batch_rules.set(lm_schema_call_p, batch_lm_schema_call)
batch_rules.aset(lm_schema_call_p, abatch_lm_schema_call)
