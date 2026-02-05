"""LM (Language Model) primitives"""

from __future__ import annotations

import asyncio
import functools as ft
from io import StringIO

from litellm import acompletion, completion, get_model_info

from autoform.core import (
    Effect,
    EvalType,
    Primitive,
    PrimitiveTag,
    Var,
    batch_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_effect,
)
from autoform.effects import effect_p
from autoform.utils import Struct, Tree, batch_index, batch_spec, batch_transpose, treelib


class LMTag(PrimitiveTag): ...


class StreamEffect(Effect):
    __slots__ = "text"

    def __init__(self, text: str):
        self.text = text


# ==================================================================================================
# LM CALL
# ==================================================================================================

lm_call_p = Primitive("lm_call", tag={LMTag})

# TODO(asem): take a look into this
GRAD_PROMPT = """Given this LLM interaction:

INPUT: {content}
OUTPUT: {out}
FEEDBACK ON OUTPUT: {out_cotangent}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""


def lm_call(messages: list[dict[str, str]], /, *, model: str) -> str:
    """Calls a language model with the given messages and model name using Litellm.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use (e.g., "gpt-5.2").

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
        >>> result = ir.call("Alice") # doctest: +SKIP
    """
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m=}"
        assert "content" in m, f"message must have a 'content' key, got {m=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return lm_call_p.bind(contents, roles=roles, model=model)


@ft.lru_cache(maxsize=256)
def can_lm_stream(model: str) -> bool:
    info = get_model_info(model)
    supports_streaming = "stream" in info.get("supported_openai_params", [])
    return supports_streaming


def impl_lm_call(contents: list[str], /, *, roles: list[str], model: str) -> str:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]

    if not can_lm_stream(model):
        response = completion(messages=messages, model=model)
        return response.choices[0].message.content

    # NOTE(asem): stream under effect handler context by default for all lm calls
    # effect is used here not over user code to avoid having lm call with `StreamEffect` in the ireqn.
    # as effectful equations makes any equation immovable.
    # downside of this approach is streaming is not possible if lm_call is transformed.
    buffer = StringIO()
    for chunk in completion(messages=messages, model=model, stream=True):
        text = chunk.choices[0].delta.content or ""
        buffer.write(text)
        with using_effect(StreamEffect(text)):
            # NOTE(asem): bind triggers the active interpreter to process the effect
            # if EffectInterpreter is active, otherwise pass through (rule-wise).
            effect_p.bind(text)
    return buffer.getvalue()


async def aimpl_lm_call(contents: list[str], /, *, roles: list[str], model: str) -> str:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    response = await acompletion(messages=messages, model=model)
    return response.choices[0].message.content


def eval_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> EvalType:
    return Var(str)


def pushforward_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_resp = lm_call_p.bind(primals, roles=roles, model=model)
    t_resp = lm_call_p.bind(tangents, roles=roles, model=model)
    return p_resp, t_resp


async def apush_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    abind = ft.partial(lm_call_p.abind, roles=roles, model=model)
    p_resp, t_resp = await asyncio.gather(abind(primals), abind(tangents))
    return p_resp, t_resp


def pullback_fwd_lm_call(contents: list, /, *, roles: list[str], model: str) -> tuple[Tree, Tree]:
    out = lm_call_p.bind(contents, roles=roles, model=model)
    residuals = (contents, out)
    return out, residuals


async def apull_fwd_lm_call(
    contents: list, /, *, roles: list[str], model: str
) -> tuple[Tree, Tree]:
    out = await lm_call_p.abind(contents, roles=roles, model=model)
    residuals = (contents, out)
    return out, residuals


def pullback_bwd_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> list:
    residuals, out_cotangent = in_tree
    contents, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grads.append(lm_call_p.bind([grad_prompt], roles=["user"], model=model))
    return grads


async def apull_bwd_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> list:
    residuals, out_cotangent = in_tree
    contents, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        return await lm_call_p.abind([prompt], roles=["user"], model=model)

    return list(await asyncio.gather(*[grad(c) for c in contents]))


def batch_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> tuple[Tree, Tree]:
    batch_size, in_batched, contents = in_tree

    if (spec := batch_spec(contents, in_batched)) is None:
        return lm_call_p.bind(contents, roles=roles, model=model), False

    unbatch = ft.partial(batch_index, contents, in_batched)
    results = [lm_call_p.bind(unbatch(b), roles=roles, model=model) for b in range(batch_size)]
    out_tree = spec.unflatten(results)
    return out_tree, True


async def abatch_lm_call(in_tree: Tree, /, *, roles: list[str], model: str) -> tuple[Tree, Tree]:
    batch_size, in_batched, contents = in_tree

    if (spec := batch_spec(contents, in_batched)) is None:
        return await lm_call_p.abind(contents, roles=roles, model=model), False

    unbatch = ft.partial(batch_index, contents, in_batched)
    abind = ft.partial(lm_call_p.abind, roles=roles, model=model)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_tree = spec.unflatten(results)
    return out_tree, True


impl_rules.set(lm_call_p, impl_lm_call)
impl_rules.aset(lm_call_p, aimpl_lm_call)
eval_rules.set(lm_call_p, eval_lm_call)
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

struct_lm_call_p = Primitive("struct_lm_call", tag={LMTag})


def struct_lm_call(messages: list[dict[str, str]], *, model: str, struct: type[Struct]) -> Struct:
    """Calls a language model with structured output using response_format.

    Uses LLM's built-in JSON mode with a Pydantic schema to extract structured
    data. The model response is automatically parsed and validated.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use.
        struct: A Pydantic model subclassing `Struct` for the output schema.

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
    return struct_lm_call_p.bind(contents, roles=roles, model=model, struct=struct)


def impl_struct_lm_call(
    contents: list, /, *, roles: list[str], model: str, struct: type[Struct]
) -> Struct:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct)
    return struct.model_validate_json(resp.choices[0].message.content)


async def aimpl_struct_lm_call(
    contents: list, /, *, roles: list[str], model: str, struct: type[Struct]
) -> Struct:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = await acompletion(messages=messages, model=model, response_format=struct)
    return struct.model_validate_json(resp.choices[0].message.content)


def eval_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> Tree:
    return struct.model_construct(**{k: Var(str) for k in struct.model_fields})


def pushforward_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    p_resp = struct_lm_call_p.bind(primals, roles=roles, model=model, struct=struct)
    t_resp = struct_lm_call_p.bind(tangents, roles=roles, model=model, struct=struct)
    return p_resp, t_resp


async def apush_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    abind = ft.partial(struct_lm_call_p.abind, roles=roles, model=model, struct=struct)
    p_resp, t_resp = await asyncio.gather(abind(primals), abind(tangents))
    return p_resp, t_resp


def pullback_fwd_struct_lm_call(
    contents: list, /, *, roles: list[str], model: str, struct: type[Struct]
) -> tuple[Tree, Tree]:
    out = struct_lm_call_p.bind(contents, roles=roles, model=model, struct=struct)
    residuals = (contents, out)
    return out, residuals


async def apull_fwd_struct_lm_call(
    contents: list, /, *, roles: list[str], model: str, struct: type[Struct]
) -> tuple[Tree, Tree]:
    out = await struct_lm_call_p.abind(contents, roles=roles, model=model, struct=struct)
    residuals = (contents, out)
    return out, residuals


def pullback_bwd_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> list:
    residuals, out_cotangent = in_tree
    contents, out = residuals
    grads = []
    for content in contents:
        grad_prompt = GRAD_PROMPT.format(content=content, out=out, out_cotangent=out_cotangent)
        grads.append(lm_call_p.bind([grad_prompt], roles=["user"], model=model))
    return grads


async def apull_bwd_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> list:
    residuals, out_cotangent = in_tree
    contents, out = residuals

    async def grad(c):
        prompt = GRAD_PROMPT.format(content=c, out=out, out_cotangent=out_cotangent)
        return await lm_call_p.abind([prompt], roles=["user"], model=model)

    return list(await asyncio.gather(*[grad(c) for c in contents]))


def batch_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> tuple[Tree, Tree]:
    batch_size, in_batched, contents = in_tree

    if batch_spec(contents, in_batched) is None:
        result = struct_lm_call_p.bind(contents, roles=roles, model=model, struct=struct)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, contents, in_batched)
    bind = ft.partial(struct_lm_call_p.bind, roles=roles, model=model, struct=struct)
    results = [bind(unbatch(b)) for b in range(batch_size)]
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, results)
    return out_ib, out_batched


async def abatch_struct_lm_call(
    in_tree: Tree, /, *, roles: list[str], model: str, struct: type[Struct]
) -> tuple[Tree, Tree]:
    batch_size, in_batched, contents = in_tree

    if batch_spec(contents, in_batched) is None:
        result = await struct_lm_call_p.abind(contents, roles=roles, model=model, struct=struct)
        out_batched = treelib.map(lambda _: False, result)
        return result, out_batched

    unbatch = ft.partial(batch_index, contents, in_batched)
    abind = ft.partial(struct_lm_call_p.abind, roles=roles, model=model, struct=struct)
    results = await asyncio.gather(*[abind(unbatch(b)) for b in range(batch_size)])
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = batch_transpose(batch_size, out_batched, list(results))
    return out_ib, out_batched


impl_rules.set(struct_lm_call_p, impl_struct_lm_call)
impl_rules.aset(struct_lm_call_p, aimpl_struct_lm_call)
eval_rules.set(struct_lm_call_p, eval_struct_lm_call)
push_rules.set(struct_lm_call_p, pushforward_struct_lm_call)
push_rules.aset(struct_lm_call_p, apush_struct_lm_call)
pull_fwd_rules.set(struct_lm_call_p, pullback_fwd_struct_lm_call)
pull_fwd_rules.aset(struct_lm_call_p, apull_fwd_struct_lm_call)
pull_bwd_rules.set(struct_lm_call_p, pullback_bwd_struct_lm_call)
pull_bwd_rules.aset(struct_lm_call_p, apull_bwd_struct_lm_call)
batch_rules.set(struct_lm_call_p, batch_struct_lm_call)
batch_rules.aset(struct_lm_call_p, abatch_struct_lm_call)
