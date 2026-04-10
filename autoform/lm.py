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
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from litellm import acompletion, completion

from autoform.ad import Zero, materialize
from autoform.core import (
    EvalType,
    Prim,
    PrimTag,
    TypedAVal,
    abstract_rules,
    batch_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    typeof,
)
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
class LMRouter(Protocol):
    def completion(self, *, messages: list[dict], model: str, **kwargs) -> Any: ...
    def acompletion(self, *, messages: list[dict], model: str, **kwargs) -> Awaitable[Any]: ...


active_router: ContextVar[LMRouter | None] = ContextVar("active_router", default=None)


@contextmanager
def using_router(router: LMRouter | None) -> Generator[LMRouter | None, None, None]:
    """Set the LM router for all lm primitives.

    The router must expose ``.completion()`` and ``.acompletion()`` matching
    litellm's call signature (e.g. ``litellm.Router``).
    Pass ``None`` to revert to direct litellm calls.

    See https://docs.litellm.ai/docs/routing for details.

    Example:
        >>> import autoform as af
        >>> from litellm import Router  # doctest: +SKIP
        >>> router = Router(  # doctest: +SKIP
        ...     model_list=[
        ...         dict(model_name="gpt-4", litellm_params=dict(model="gpt-4")),
        ...     ],
        ...     max_parallel_requests=10,
        ... )
        >>> with af.using_router(router):  # doctest: +SKIP
        ...     ir.call(inputs)
    """
    assert router is None or isinstance(router, LMRouter), f"Expected LMRouter or None."
    token = active_router.set(router)
    try:
        yield router
    finally:
        active_router.reset(token)


class LMTag(PrimTag): ...


# ==================================================================================================
# LM CALL
# ==================================================================================================

lm_call_p = Prim("lm_call", tag={LMTag})

# TODO(asem): take a look into this
GRAD_PROMPT = """Given this LLM interaction:

INPUT: {content}
OUTPUT: {out}
FEEDBACK ON OUTPUT: {out_cotangent}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""


def none_or_zero(x):
    return None if x is None else Zero(type(x))


def lm_call(
    messages: list[dict[str, str]],
    /,
    *,
    model: str,
    temperature: float | int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Calls a language model with the given messages and model name using Litellm.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use (e.g., "gpt-5.2").
        temperature:  The sampling temperature to be used, between 0 and 2.
        max_tokens: The maximum number of tokens to generate in the chat completion.

    Returns:
        The content of the model's response as a string.

    Example:
        >>> import autoform as af
        >>> def ir(name: str) -> str:
        ...     greeting = af.format("Hello, {}!", name)
        ...     system_message = dict(role="system", content="translate the greeting to Korean")
        ...     user_message = dict(role="user", content=greeting)
        ...     greeting = af.lm_call([system_message, user_message], model="gpt-5.2")
        ...     return greeting
        >>> ir = af.trace(ir)("World") # doctest: +SKIP
        >>> result = ir.call("x0") # doctest: +SKIP
    """
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return lm_call_p.bind((contents, model, temperature, max_tokens), roles=roles)


def impl_lm_call(in_tree: Tree, /, *, roles: list[str]) -> str:
    contents, model, temp, max_tokens = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    comp = client.completion if (client := active_router.get()) is not None else completion
    response = comp(messages=messages, model=model, temperature=temp, max_tokens=max_tokens)
    return response.choices[0].message.content


async def aimpl_lm_call(in_tree: Tree, /, *, roles: list[str]) -> str:
    contents, model, temp, max_tokens = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    acomp = client.acompletion if (client := active_router.get()) is not None else acompletion
    response = await acomp(messages=messages, model=model, temperature=temp, max_tokens=max_tokens)
    return response.choices[0].message.content


def abstract_lm_call(in_tree: Tree, /, *, roles: list[str]) -> EvalType:
    contents, model, temperature, max_tokens = in_tree
    assert all(typeof(x) is str for x in contents), f"Expected string messages, got {contents!r}"
    assert typeof(model) is str, f"`lm_call` expects a string model, got {model!r}"
    assert temperature is None or typeof(temperature) in (int, float)
    assert max_tokens is None or typeof(max_tokens) is int
    return TypedAVal(str)


def pushforward_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model, primal_temperature, primal_max_tokens = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model, primal_temperature, primal_max_tokens)
    p_resp = lm_call_p.bind(p_tree, roles=roles)
    t_tree = (materialize(tangent_contents), primal_model, primal_temperature, primal_max_tokens)
    t_resp = lm_call_p.bind(t_tree, roles=roles)
    return p_resp, t_resp


async def apush_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model, primal_temperature, primal_max_tokens = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(lm_call_p.abind, roles=roles)
    p_tree = (primal_contents, primal_model, primal_temperature, primal_max_tokens)
    t_tree = (materialize(tangent_contents), primal_model, primal_temperature, primal_max_tokens)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    contents, model, temperature, max_tokens = in_tree
    out = lm_call_p.bind((contents, model, temperature, max_tokens), roles=roles)
    residuals = (contents, model, temperature, max_tokens, out)
    return out, residuals


async def apull_fwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> tuple[Tree, Tree]:
    contents, model, temperature, max_tokens = in_tree
    out = await lm_call_p.abind((contents, model, temperature, max_tokens), roles=roles)
    residuals = (contents, model, temperature, max_tokens, out)
    return out, residuals


def pullback_bwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, temperature, max_tokens, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.bind(([grad_prompt], model, temperature, max_tokens), roles=["user"])
        grads.append(grad_out)
    return grads, Zero(str), none_or_zero(temperature), none_or_zero(max_tokens)


async def apull_bwd_lm_call(in_tree: Tree, /, *, roles: list[str]) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, temperature, max_tokens, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.abind(([prompt], model, temperature, max_tokens), roles=["user"])
        return await grad_out

    return (
        await asyncio.gather(*[grad(c) for c in contents]),
        Zero(str),
        none_or_zero(temperature),
        none_or_zero(max_tokens),
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

struct_lm_call_p = Prim("struct_lm_call", tag={LMTag})


def struct_lm_call(
    messages: list[dict[str, str]],
    *,
    model: str,
    struct: type[Struct],
    temperature: float | int | None = None,
    max_tokens: int | None = None,
) -> Struct:
    """Calls a language model with structured output using response_format.

    Uses LLM's built-in JSON mode with a schema to extract structured
    data. The model response is automatically parsed and validated.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use.
        struct: A ``Struct`` subclass defining the output schema.
        temperature:  The sampling temperature to be used, between 0 and 2.
        max_tokens: The maximum number of tokens to generate in the chat completion.

    Returns:
        A validated instance of the struct type.

    Example:
        >>> import autoform as af
        >>> class Answer(af.Struct):
        ...     reasoning: str
        ...     answer: int
        >>> def solver(question):
        ...     messages = [{"role": "user", "content": question}]
        ...     return af.struct_lm_call(messages, model="gpt-5.2", struct=Answer)
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
    in_tree = (contents, model, temperature, max_tokens)
    return struct_lm_call_p.bind(in_tree, roles=roles, struct=struct)


def impl_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Struct:
    contents, model, temperature, max_tokens = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    comp = client.completion if (client := active_router.get()) is not None else completion
    resp = comp(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=struct,
    )
    return struct.model_validate_json(resp.choices[0].message.content)


async def aimpl_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Struct:
    contents, model, temperature, max_tokens = in_tree
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    acomp = client.acompletion if (client := active_router.get()) is not None else acompletion
    resp = await acomp(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=struct,
    )
    return struct.model_validate_json(resp.choices[0].message.content)


def abstract_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Tree:
    contents, model, temperature, max_tokens = in_tree
    assert all(typeof(x) is str for x in contents), f"Expected string messages, got {contents!r}"
    assert typeof(model) is str, f"Expected string model, got {model!r}"
    assert temperature is None or typeof(temperature) in (int, float)
    assert max_tokens is None or typeof(max_tokens) is int
    return treelib.map(TypedAVal, struct_type_tree(struct))


def pushforward_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model, primal_temperature, primal_max_tokens = primals
    tangent_contents, *_ = tangents
    p_tree = (primal_contents, primal_model, primal_temperature, primal_max_tokens)
    t_tree = (materialize(tangent_contents), primal_model, primal_temperature, primal_max_tokens)
    p_resp = struct_lm_call_p.bind(p_tree, roles=roles, struct=struct)
    t_resp = struct_lm_call_p.bind(t_tree, roles=roles, struct=struct)
    return p_resp, t_resp


async def apush_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    primal_contents, primal_model, primal_temperature, primal_max_tokens = primals
    tangent_contents, *_ = tangents
    abind = ft.partial(struct_lm_call_p.abind, roles=roles, struct=struct)
    p_tree = (primal_contents, primal_model, primal_temperature, primal_max_tokens)
    t_tree = (materialize(tangent_contents), primal_model, primal_temperature, primal_max_tokens)
    p_resp, t_resp = await asyncio.gather(abind(p_tree), abind(t_tree))
    return p_resp, t_resp


def pullback_fwd_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    contents, model, temperature, max_tokens = in_tree
    out = struct_lm_call_p.bind(in_tree, roles=roles, struct=struct)
    residuals = (contents, model, temperature, max_tokens, out)
    return out, residuals


async def apull_fwd_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    contents, model, temperature, max_tokens = in_tree
    out = await struct_lm_call_p.abind(in_tree, roles=roles, struct=struct)
    residuals = (contents, model, temperature, max_tokens, out)
    return out, residuals


def pullback_bwd_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, temperature, max_tokens, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.bind(([grad_prompt], model, temperature, max_tokens), roles=["user"])
        grads.append(grad_out)
    return grads, Zero(str), none_or_zero(temperature), none_or_zero(max_tokens)


async def apull_bwd_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> Tree:
    residuals, out_cotangent = in_tree
    out_cotangent = materialize(out_cotangent)
    contents, model, temperature, max_tokens, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        grad_out = lm_call_p.abind(([prompt], model, temperature, max_tokens), roles=["user"])
        return await grad_out

    return (
        await asyncio.gather(*[grad(c) for c in contents]),
        Zero(str),
        none_or_zero(temperature),
        none_or_zero(max_tokens),
    )


def batch_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if batch_spec(in_values, in_batched) is None:
        result = struct_lm_call_p.bind(in_values, roles=roles, struct=struct)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    bind = ft.partial(struct_lm_call_p.bind, roles=roles, struct=struct)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, results)
    return out_ib, out_batched


async def abatch_struct_lm_call(
    in_tree: Tree,
    /,
    *,
    roles: list[str],
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    batch_size, in_batched, in_values = in_tree

    if batch_spec(in_values, in_batched) is None:
        result = await struct_lm_call_p.abind(in_values, roles=roles, struct=struct)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, in_values, in_batched)
    abind = ft.partial(struct_lm_call_p.abind, roles=roles, struct=struct)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, list(results))
    return out_ib, out_batched


impl_rules.set(struct_lm_call_p, impl_struct_lm_call)
impl_rules.aset(struct_lm_call_p, aimpl_struct_lm_call)
abstract_rules.set(struct_lm_call_p, abstract_struct_lm_call)
push_rules.set(struct_lm_call_p, pushforward_struct_lm_call)
push_rules.aset(struct_lm_call_p, apush_struct_lm_call)
pull_fwd_rules.set(struct_lm_call_p, pullback_fwd_struct_lm_call)
pull_fwd_rules.aset(struct_lm_call_p, apull_fwd_struct_lm_call)
pull_bwd_rules.set(struct_lm_call_p, pullback_bwd_struct_lm_call)
pull_bwd_rules.aset(struct_lm_call_p, apull_bwd_struct_lm_call)
batch_rules.set(struct_lm_call_p, batch_struct_lm_call)
batch_rules.aset(struct_lm_call_p, abatch_struct_lm_call)
