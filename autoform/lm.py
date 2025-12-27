"""LM (Language Model) primitives"""

from __future__ import annotations

import functools as ft
import typing as tp

import pydantic
from litellm import acompletion, batch_completion, completion

from autoform.core import Var
from autoform.core import (
    Primitive,
    async_rules,
    batch_rules,
    eval_rules,
    impl_rules,
    iter_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import Tree, treelib, transpose_batch

# ==================================================================================================
# STRUCT
# ==================================================================================================


class Struct(pydantic.BaseModel):
    """Pydantic BaseModel that is also a PyTree.

    Auto-registers subclasses as pytrees.
    Uses ``model_construct`` in unflatten to skip validation.

    Example:
        >>> class Answer(Struct):
        ...     reasoning: str
        ...     answer: int
    """

    def __init_subclass__(k, **kwargs):
        super().__init_subclass__(**kwargs)

        treelib.register_node(
            k,
            lambda x: (tuple(getattr(x, k) for k in k.model_fields), tuple(k.model_fields)),
            lambda keys, children: k.model_construct(**dict(zip(keys, children, strict=True))),
        )

    def __hash__(self):
        return hash(tuple(getattr(self, k) for k in type(self).model_fields))


# ==================================================================================================
# LM CALL
# ==================================================================================================

lm_call_p = Primitive("lm_call", tag="lm")


def lm_call(messages: list[dict[str, str]], *, model: str) -> str:
    """Calls a language model with the given messages and model name using Litellm.

    Args:
        messages: A list of message dictionaries, each containing 'role' and 'content' keys.
        model: The name of the language model to use (e.g., "gpt-3.5-turbo").

    Returns:
        The content of the model's response as a string.

    Example:
        >>> import autoform as af
        >>> def ir(name: str) -> str:
        ...     greeting = af.format("Hello, {}!", name)
        ...     system_message = dict(role="system", content="translate the greeting to Korean")
        ...     user_message = dict(role="user", content=greeting)
        ...     greeting = af.lm_call([system_message, user_message], model="gpt-3.5-turbo")
        ...     return greeting
        >>> ir = af.build_ir(ir)("World") # doctest: +SKIP
        >>> result = ir.call("Alice") # doctest: +SKIP
    """
    assert isinstance(messages, list), f"messages must be a list, got {type(messages)=}"
    for m in messages:
        assert isinstance(m, dict), f"message must be a dict, got {type(m)=}"
        assert "role" in m, f"message must have a 'role' key, got {m.keys()=}"
        assert "content" in m, f"message must have a 'content' key, got {m.keys()=}"

    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return lm_call_p.bind(contents, roles=roles, model=model)


@ft.partial(impl_rules.def_rule, lm_call_p)
def impl_lm_call(contents: list, *, roles: tuple[str, ...], model: str) -> str:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents, strict=True)]
    resp = completion(messages=messages, model=model)
    return resp.choices[0].message.content


@ft.partial(eval_rules.def_rule, lm_call_p)
def eval_lm_call(in_tree: Tree, **params) -> Var:
    return Var()


@ft.partial(push_rules.def_rule, lm_call_p)
def pushforward_lm_call(
    primals: tuple, tangents: tuple, *, roles: tuple, model: str
) -> tuple[Tree, Tree]:
    p_messages = [dict(role=r, content=c) for r, c in zip(roles, primals, strict=True)]
    t_messages = [dict(role=r, content=c) for r, c in zip(roles, tangents, strict=True)]
    p_resp = completion(messages=p_messages, model=model)
    t_resp = completion(messages=t_messages, model=model)
    return p_resp.choices[0].message.content, t_resp.choices[0].message.content


@ft.partial(pull_fwd_rules.def_rule, lm_call_p)
def pullback_fwd_lm_call(contents: list, *, roles: tuple, model: str) -> tuple[Tree, Tree]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model)
    out = resp.choices[0].message.content
    residuals = (contents, out)
    return out, residuals


@ft.partial(pull_bwd_rules.def_rule, lm_call_p)
def pullback_bwd_lm_call(
    residuals: tuple, out_cotangent: Tree, *, roles: tuple, model: str
) -> list:
    contents, output = residuals
    grads = []
    for content in contents:
        grad_prompt = f"""Given this LLM interaction:

INPUT: {content}
OUTPUT: {output}
FEEDBACK ON OUTPUT: {out_cotangent}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""
        resp = completion(messages=[dict(role="user", content=grad_prompt)], model=model)
        grads.append(resp.choices[0].message.content)
    return grads


@ft.partial(batch_rules.def_rule, lm_call_p)
def batch_lm_call(
    batch_size: int, in_batched: Tree, contents: list, *, roles: tuple, model: str
) -> tuple[Tree, Tree]:
    def get_message(i: int, b: int) -> dict[str, str]:
        return dict(role=roles[i], content=contents[i][b] if in_batched[i] else contents[i])

    batched_messages = [[get_message(i, b) for i in range(len(roles))] for b in range(batch_size)]
    responses = batch_completion(messages=batched_messages, model=model)
    return [resp.choices[0].message.content for resp in responses], True


@ft.partial(iter_rules.def_rule, lm_call_p)
def iter_lm_call(contents: list, *, roles: tuple, model: str) -> tp.Iterator[str]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        yield delta


@ft.partial(async_rules.def_rule, lm_call_p)
async def async_lm_call(contents: list, *, roles: tuple, model: str) -> str:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = await acompletion(messages=messages, model=model)
    return resp.choices[0].message.content


# ==================================================================================================
# STRUCT LM CALL
# ==================================================================================================

struct_lm_call_p = Primitive("struct_lm_call", tag="lm")


def struct_lm_call(messages: list[dict[str, str]], *, model: str, struct: type[Struct]) -> str:
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
        ...     return af.struct_lm_call(messages, model="gpt-4o", struct=Answer)
        >>> ir = af.build_ir(solver)("What is 2+2?")  # doctest: +SKIP
        >>> result = ir.call("What is 2+2?")  # doctest: +SKIP
        >>> result.answer  # doctest: +SKIP
        4
    """
    assert issubclass(struct, Struct), "struct must be a subclass of ``Struct``"
    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    return struct_lm_call_p.bind(contents, roles=roles, model=model, struct=struct)


@ft.partial(impl_rules.def_rule, struct_lm_call_p)
def impl_struct_lm_call(
    contents: list,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> Struct:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct)
    return struct.model_validate_json(resp.choices[0].message.content)


@ft.partial(eval_rules.def_rule, struct_lm_call_p)
def eval_struct_lm_call(in_tree: Tree, *, struct: type[Struct], **params) -> Tree:
    return struct.model_construct(**{k: Var() for k in struct.model_fields})


@ft.partial(pull_fwd_rules.def_rule, struct_lm_call_p)
def pullback_fwd_struct_lm_call(
    contents: list,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct)
    out = struct.model_validate_json(resp.choices[0].message.content)
    residuals = (contents, out)
    return out, residuals


@ft.partial(pull_bwd_rules.def_rule, struct_lm_call_p)
def pullback_bwd_struct_lm_call(
    residuals: tuple,
    out_cotangent: Tree,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> list:
    contents, output = residuals
    grads = []
    for content in contents:
        grad_prompt = f"""Given this LLM interaction:

INPUT: {content}
OUTPUT: {output}
FEEDBACK ON OUTPUT: {out_cotangent}

Provide specific, actionable feedback on how to improve the INPUT to address the feedback. Be concise."""
        resp = completion(messages=[dict(role="user", content=grad_prompt)], model=model)
        grads.append(resp.choices[0].message.content)
    return grads


@ft.partial(batch_rules.def_rule, struct_lm_call_p)
def batch_struct_lm_call(
    batch_size: int,
    in_batched: Tree,
    contents: list,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> tuple[Tree, Tree]:
    def get_message(i: int, b: int) -> dict[str, str]:
        return dict(role=roles[i], content=contents[i][b] if in_batched[i] else contents[i])

    batched_messages = [[get_message(i, b) for i in range(len(roles))] for b in range(batch_size)]
    responses = batch_completion(messages=batched_messages, model=model, response_format=struct)
    results = [struct.model_validate_json(resp.choices[0].message.content) for resp in responses]
    out_batched = treelib.map(lambda _: True, results[0])
    out_ib = transpose_batch(batch_size, out_batched, results)
    return out_ib, out_batched


@ft.partial(iter_rules.def_rule, struct_lm_call_p)
def iter_struct_lm_call(
    contents: list,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> tp.Iterator[str]:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = completion(messages=messages, model=model, response_format=struct, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        yield delta


@ft.partial(async_rules.def_rule, struct_lm_call_p)
async def async_struct_lm_call(
    contents: list,
    *,
    roles: tuple,
    model: str,
    struct: type[Struct],
) -> Struct:
    messages = [dict(role=r, content=c) for r, c in zip(roles, contents)]
    resp = await acompletion(messages=messages, model=model, response_format=struct)
    return resp.choices[0].message.content
