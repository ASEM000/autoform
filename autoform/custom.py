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

"""Custom transform rules for composite Autoform functions."""

from __future__ import annotations

import functools as ft
from collections.abc import Callable
from typing import Any

from autoform.ad import pullback, pushforward
from autoform.batch import batch
from autoform.core import (
    IR,
    ABatchRule,
    APullbackBwdRule,
    APushforwardRule,
    BatchRule,
    IREqn,
    IRVar,
    Prim,
    PullbackBwdRule,
    PushforwardRule,
    TracingInterpreter,
    TypedAVal,
    abstract_rules,
    batch_rules,
    impl_rules,
    ir_aval,
    is_aval,
    is_irvar,
    is_val,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.dce import dce_rules, default_dce
from autoform.utils import Tree, batch_index, batch_spec, treelib

__all__ = ["custom"]


def trace_custom_func(func: Callable[..., Any], in_ir_tree: Tree, /) -> IR:
    def to_ir_input(x, /):
        if is_irvar(x):
            return x
        if is_aval(x):
            return IRVar.fresh(aval=x)
        assert is_val(x), f"Unsupported input leaf type for custom function: {type(x).__name__}"
        return IRVar.fresh(aval=TypedAVal(type(x)))

    in_ir_tree = treelib.map(to_ir_input, in_ir_tree)
    with using_interpreter(TracingInterpreter()) as tracer:
        out_ir_tree = func(*in_ir_tree)
    return IR(tracer.ir_eqns, in_ir_tree=in_ir_tree, out_ir_tree=out_ir_tree)


def call_custom_body(func: Callable[..., Any], in_tree: Tree, /) -> tuple[IR, Tree]:
    ir = trace_custom_func(func, in_tree)
    return ir, ir.call(*in_tree)


async def acall_custom_body(func: Callable[..., Any], in_tree: Tree, /) -> tuple[IR, Tree]:
    ir = trace_custom_func(func, in_tree)
    return ir, await ir.acall(*in_tree)


def tree_batched_like(tree: Tree, batched: bool, /) -> Tree:
    return treelib.map(lambda _: batched, tree)


def impl_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> Tree:
    return call(*in_tree)


async def aimpl_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> Tree:
    _, out = await acall_custom_body(call, in_tree)
    return out


def abstract_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> Tree:
    ir = trace_custom_func(call, in_tree)
    return treelib.map(ir_aval, ir.out_ir_tree)


def pushforward_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    ir, p_out = call_custom_body(call, primals)
    _, t_out = pushforward(ir).call(primals, tangents)
    return p_out, t_out


async def apushforward_custom_call(
    in_tree: Tree, /, *, call: Callable[..., Any]
) -> tuple[Tree, Tree]:
    primals, tangents = in_tree
    ir, p_out = await acall_custom_body(call, primals)
    _, t_out = await pushforward(ir).acall(primals, tangents)
    return p_out, t_out


def pullback_fwd_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> tuple[Tree, Tree]:
    primals = in_tree
    _, out = call_custom_body(call, primals)
    return out, (primals, out)


async def apullback_fwd_custom_call(
    in_tree: Tree, /, *, call: Callable[..., Any]
) -> tuple[Tree, Tree]:
    primals = in_tree
    _, out = await acall_custom_body(call, primals)
    return out, (primals, out)


def pullback_bwd_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> Tree:
    primals, out = in_tree[0]
    cotangent = in_tree[1]
    ir = trace_custom_func(call, primals)
    _, c_in = pullback(ir).call(primals, cotangent)
    return c_in


async def apullback_bwd_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> Tree:
    primals, out = in_tree[0]
    cotangent = in_tree[1]
    ir = trace_custom_func(call, primals)
    _, c_in = await pullback(ir).acall(primals, cotangent)
    return c_in


def batch_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> tuple[Tree, Tree]:
    _, axes, values = in_tree
    if batch_spec(values, axes) is None:
        _, out = call_custom_body(call, values)
        return out, tree_batched_like(out, False)
    example_values = batch_index(values, axes, 0)
    ir = trace_custom_func(call, example_values)
    batched_ir = batch(ir, in_axes=axes)
    out = batched_ir.call(*values)
    return out, tree_batched_like(ir.out_ir_tree, True)


async def abatch_custom_call(in_tree: Tree, /, *, call: Callable[..., Any]) -> tuple[Tree, Tree]:
    _, axes, values = in_tree
    if batch_spec(values, axes) is None:
        _, out = await acall_custom_body(call, values)
        return out, tree_batched_like(out, False)
    example_values = batch_index(values, axes, 0)
    ir = trace_custom_func(call, example_values)
    batched_ir = batch(ir, in_axes=axes)
    out = await batched_ir.acall(*values)
    return out, tree_batched_like(ir.out_ir_tree, True)


def dce_custom_call(ir_eqn: IREqn, out_used: Tree[bool], /) -> tuple[IREqn, Tree[bool]]:
    return default_dce(ir_eqn, out_used)


def install_custom_call_rules(prim: Prim, /) -> None:
    impl_rules.set(prim, impl_custom_call)
    impl_rules.aset(prim, aimpl_custom_call)
    abstract_rules.set(prim, abstract_custom_call)
    push_rules.set(prim, pushforward_custom_call)
    push_rules.aset(prim, apushforward_custom_call)
    pull_fwd_rules.set(prim, pullback_fwd_custom_call)
    pull_fwd_rules.aset(prim, apullback_fwd_custom_call)
    pull_bwd_rules.set(prim, pullback_bwd_custom_call)
    pull_bwd_rules.aset(prim, apullback_bwd_custom_call)
    batch_rules.set(prim, batch_custom_call)
    batch_rules.aset(prim, abatch_custom_call)
    dce_rules[prim] = dce_custom_call


def custom_prim_name(func: Callable[..., Any]) -> str:
    name = f"{func.__module__}.{func.__qualname__}"
    return f"custom_call:{name}"


class CustomFunc:
    def __init__(self, func: Callable[..., Any], /):
        self.func = func
        self.prim = Prim(custom_prim_name(func))
        install_custom_call_rules(self.prim)
        ft.update_wrapper(self, func)

    def __call__(self, *args):
        return self.prim.bind(args, call=self.func)

    def set_pushforward[R: PushforwardRule](self, rule: R, /) -> R:
        """Register ``rule(in_tree, *, call) -> (primal_output, tangent_output)``.

        Example:
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_push_example(x):
            ...     return af.format("[{}]", x)
            >>> @bracket_push_example.set_pushforward
            ... def bracket_push_rule(in_tree, /, *, call):
            ...     primals, tangents = in_tree
            ...     (dx,) = tangents
            ...     p_out = call(*primals)
            ...     t_out = af.format("delta {}", af.ad.materialize(dx))
            ...     return p_out, t_out
            >>> ir = af.trace(lambda x: bracket_push_example(x))("seed")
            >>> af.pushforward(ir).call(("hello",), ("change",))
            ('[hello]', 'delta change')
        """

        push_rules.set(self.prim, rule, replace=True)
        return rule

    def aset_pushforward[R: APushforwardRule](self, rule: R, /) -> R:
        """Register an async custom pushforward rule.

        Example:
            >>> import asyncio
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_apush_example(x):
            ...     return af.format("[{}]", x)
            >>> @bracket_apush_example.aset_pushforward
            ... async def bracket_apush_rule(in_tree, /, *, call):
            ...     primals, tangents = in_tree
            ...     (dx,) = tangents
            ...     p_out = call(*primals)
            ...     t_out = af.format("async delta {}", af.ad.materialize(dx))
            ...     return p_out, t_out
            >>> ir = af.trace(lambda x: bracket_apush_example(x))("seed")
            >>> asyncio.run(af.pushforward(ir).acall(("hello",), ("change",)))
            ('[hello]', 'async delta change')
        """

        push_rules.aset(self.prim, rule, replace=True)
        return rule

    def set_pullback[R: PullbackBwdRule](self, rule: R, /) -> R:
        """Register ``rule(in_tree, *, call) -> cotangents_in``.

        Example:
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_pull_example(x):
            ...     return af.format("[{}]", x)
            >>> @bracket_pull_example.set_pullback
            ... def bracket_pull_rule(in_tree, /, *, call):
            ...     del call
            ...     (primals, output), cotangent = in_tree
            ...     del primals
            ...     return (af.format("{} via {}", cotangent, output),)
            >>> ir = af.trace(lambda x: bracket_pull_example(x))("seed")
            >>> af.pullback(ir).call(("hello",), "feedback")
            ('[hello]', ('feedback via [hello]',))
        """

        pull_bwd_rules.set(self.prim, rule, replace=True)
        return rule

    def aset_pullback[R: APullbackBwdRule](self, rule: R, /) -> R:
        """Register an async custom pullback rule.

        Example:
            >>> import asyncio
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_apull_example(x):
            ...     return af.format("[{}]", x)
            >>> @bracket_apull_example.aset_pullback
            ... async def bracket_apull_rule(in_tree, /, *, call):
            ...     del call
            ...     (primals, output), cotangent = in_tree
            ...     del primals
            ...     return (af.format("async {} via {}", cotangent, output),)
            >>> ir = af.trace(lambda x: bracket_apull_example(x))("seed")
            >>> asyncio.run(af.pullback(ir).acall(("hello",), "feedback"))
            ('[hello]', ('async feedback via [hello]',))
        """

        pull_bwd_rules.aset(self.prim, rule, replace=True)
        return rule

    def set_batch[R: BatchRule](self, rule: R, /) -> R:
        """Register ``rule(in_tree, *, call) -> (outputs, output_axes)``.

        Example:
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_batch_example(x):
            ...     return af.format("[{}]", x)
            >>> @bracket_batch_example.set_batch
            ... def bracket_batch_rule(in_tree, /, *, call):
            ...     del call
            ...     batch_size, axes, values = in_tree
            ...     assert batch_size == 2
            ...     (xs,) = values
            ...     (x_axis,) = axes
            ...     assert x_axis is True
            ...     return [af.format("<{}>", x) for x in xs], True
            >>> ir = af.trace(lambda x: bracket_batch_example(x))("seed")
            >>> af.batch(ir).call(["a", "b"])
            ['<a>', '<b>']
        """

        batch_rules.set(self.prim, rule, replace=True)
        return rule

    def aset_batch[R: ABatchRule](self, rule: R, /) -> R:
        """Register an async custom batch rule.

        Example:
            >>> import asyncio
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_abatch_example(x):
            ...     return af.format("[{}]", x)
            >>> @bracket_abatch_example.aset_batch
            ... async def bracket_abatch_rule(in_tree, /, *, call):
            ...     del call
            ...     batch_size, axes, values = in_tree
            ...     assert batch_size == 2
            ...     (xs,) = values
            ...     (x_axis,) = axes
            ...     assert x_axis is True
            ...     return [af.format("async <{}>", x) for x in xs], True
            >>> ir = af.trace(lambda x: bracket_abatch_example(x))("seed")
            >>> asyncio.run(af.batch(ir).acall(["a", "b"]))
            ['async <a>', 'async <b>']
        """

        batch_rules.aset(self.prim, rule, replace=True)
        return rule


def custom(func: Callable[..., Any], /) -> CustomFunc:
    """Mark a Python function as a custom Autoform transform boundary.

    ``custom`` is a decorator for functions that should keep their ordinary call
    behavior while optionally overriding how Autoform transforms treat them.
    Without any registered rules, ``pushforward``, ``pullback``, and ``batch``
    produce the same results as transforming the function body directly.

    The returned wrapper supports these rule registration decorators:

    - ``set_pushforward(rule)`` for a synchronous pushforward rule.
    - ``aset_pushforward(rule)`` for an asynchronous pushforward rule.
    - ``set_pullback(rule)`` for a synchronous pullback backward rule.
    - ``aset_pullback(rule)`` for an asynchronous pullback backward rule.
    - ``set_batch(rule)`` for a synchronous batch rule.
    - ``aset_batch(rule)`` for an asynchronous batch rule.

    Rules receive one positional ``in_tree`` argument. The original behavior is
    available as the keyword-only ``call`` argument, so a rule can use
    ``call(*primals)`` when it wants to reuse the normal primal behavior.
    The rule signatures are:

    - Pushforward: ``rule((primals, tangents), /, *, call) -> (p_out, t_out)``.
    - Pullback backward:
      ``rule(((primals, output), cotangent), /, *, call) -> cotangents``.
    - Batch:
      ``rule((batch_size, axes, values), /, *, call) -> (outputs, output_axes)``.

    Synchronous and asynchronous registrations are independent. Use both
    ``set_*`` and ``aset_*`` when both execution modes need custom behavior.

    Args:
        func: Function to wrap. The function body may use Autoform primitives and
            any normal Python structure that is valid while tracing.

    Returns:
        A callable wrapper with the same call behavior as ``func`` and rule
        registration methods. The concrete wrapper class is an implementation
        detail; use only the returned callable and its ``set_*``/``aset_*``
        methods.

    Example:
        Direct calls behave like calls to the original function.

        >>> import autoform as af
        >>> @af.custom
        ... def bracket(x):
        ...     return af.format("[{}]", x)
        >>> bracket("hello")
        '[hello]'

        Without custom rules, transforms behave as if they were applied to the
        function body.

        >>> base = af.trace(lambda x: bracket(x))("seed")
        >>> af.pushforward(base).call(("hello",), ("change",))
        ('[hello]', '[change]')
        >>> af.pullback(base).call(("hello",), "feedback")
        ('[hello]', ('feedback',))
        >>> af.batch(base).call(["a", "b"])
        ['[a]', '[b]']

        A pushforward rule can replace only the pushforward behavior.

        >>> @bracket.set_pushforward
        ... def bracket_push(in_tree, /, *, call):
        ...     primals, tangents = in_tree
        ...     (dx,) = tangents
        ...     p_out = call(*primals)
        ...     t_out = af.format("delta: {}", af.ad.materialize(dx))
        ...     return p_out, t_out
        >>> af.pushforward(base).call(("hello",), ("change",))
        ('[hello]', 'delta: change')

        Pullback and batch rules use the same one-``in_tree`` convention.

        >>> @bracket.set_pullback
        ... def bracket_pull(in_tree, /, *, call):
        ...     del call
        ...     (primals, output), cotangent = in_tree
        ...     (x,) = primals
        ...     return (af.format("{} via {} from {}", cotangent, output, x),)
        >>> af.pullback(base).call(("hello",), "feedback")
        ('[hello]', ('feedback via [hello] from hello',))

        >>> @bracket.set_batch
        ... def bracket_batch(in_tree, /, *, call):
        ...     del call
        ...     batch_size, axes, values = in_tree
        ...     assert batch_size == 2
        ...     (xs,) = values
        ...     (x_axis,) = axes
        ...     assert x_axis is True
        ...     return [af.format("<{}>", x) for x in xs], True
        >>> af.batch(base).call(["a", "b"])
        ['<a>', '<b>']

        A common use is wrapping an LM call so the forward call remains normal,
        while pushforward and pullback use prompts written for that application.

        >>> from autoform.ad import materialize
        >>> @af.custom
        ... def summarize(text, model):
        ...     message = af.format("Summarize this in one sentence: {}", text)
        ...     return af.lm_call([{"role": "user", "content": message}], model=model)

        The custom pushforward rule can ask the model how the output should
        change under an input edit.

        >>> @summarize.set_pushforward
        ... def summarize_push(in_tree, /, *, call):
        ...     primals, tangents = in_tree
        ...     text, model = primals
        ...     text_tangent, _ = tangents
        ...     p_out = call(*primals)
        ...     prompt = af.format(
        ...         "Original input:\\n{}\\n\\nInput edit:\\n{}\\n\\n"
        ...         "Describe how the summary should change.",
        ...         text,
        ...         materialize(text_tangent),
        ...     )
        ...     t_out = af.lm_call([{"role": "user", "content": prompt}], model=model)
        ...     return p_out, t_out

        The custom pullback rule can replace the default backward LM prompt with
        a domain-specific feedback prompt.

        >>> @summarize.set_pullback
        ... def summarize_pull(in_tree, /, *, call):
        ...     del call
        ...     (primals, output), cotangent = in_tree
        ...     text, model = primals
        ...     prompt = af.format(
        ...         "Original input:\\n{}\\n\\nLM output:\\n{}\\n\\n"
        ...         "Downstream feedback:\\n{}\\n\\n"
        ...         "Return feedback for improving the original input.",
        ...         text,
        ...         output,
        ...         materialize(cotangent),
        ...     )
        ...     text_cotangent = af.lm_call([{"role": "user", "content": prompt}], model=model)
        ...     return text_cotangent, ""

        >>> lm_ir = af.trace(lambda text, model: summarize(text, model))("topic", "model")
        >>> af.pushforward(lm_ir).call(  # doctest: +SKIP
        ...     ("recursion", "ollama/llama3:8b"),
        ...     ("focus on the recursive step", ""),
        ... )
        >>> af.pullback(lm_ir).call(  # doctest: +SKIP
        ...     ("recursion", "ollama/llama3:8b"),
        ...     "make the answer more concrete",
        ... )
    """
    assert callable(func), f"Expected a callable function, got {type(func)}"
    return CustomFunc(func)
