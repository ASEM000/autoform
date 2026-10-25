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

"""Custom boundary primitives and transform rules."""

from __future__ import annotations

import functools as ft
from collections.abc import Awaitable, Callable
from typing import Any

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.core as core
import autoform.src.stage as stage
import autoform.src.tree as tree

type Tree[T] = Any
type TreePair = tuple[Tree, Tree]


def trace_custom_func(
    func: Callable[..., Any], in_tree: Tree, params: dict[str, Any], /
) -> stage.IR:
    in_tree = tree.map(lambda x: stage.Var.fresh(aval=core.avalof(x)), in_tree)
    with core.using_interp(stage.TraceInterp()) as trace_interpreter:
        out_trace_tree = func(*trace_interpreter.box(in_tree), **params)
    out_tree = trace_interpreter.unbox(out_trace_tree)
    return stage.IR(trace_interpreter.eqns, in_tree=in_tree, out_tree=out_tree)


def tree_batched_like(value: Tree, batched: bool, /) -> Tree:
    return tree.map(lambda _: batched, value)


def impl_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> Tree:
    return ir.call(*in_tree)


async def aimpl_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> Tree:
    return await ir.acall(*in_tree)


def abstract_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> Tree:
    return tree.map(stage.aval_if_var, ir.out_tree)


def pushforward_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> TreePair:
    return ad.pushforward(ir).call(*in_tree)


async def apushforward_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> TreePair:
    return await ad.pushforward(ir).acall(*in_tree)


def pullback_fwd_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> TreePair:
    fwd, _ = ad.fwd_bwd(ir)
    out, residual = fwd.call(*in_tree)
    return out, (in_tree, out, residual)


async def apullback_fwd_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> TreePair:
    fwd, _ = ad.fwd_bwd(ir)
    out, residual = await fwd.acall(*in_tree)
    return out, (in_tree, out, residual)


def pullback_bwd_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> Tree:
    (_, _, residual), c = in_tree
    _, bwd = ad.fwd_bwd(ir)
    return bwd.call(residual, c)


async def apullback_bwd_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> Tree:
    (_, _, residual), c = in_tree
    _, bwd = ad.fwd_bwd(ir)
    return await bwd.acall(residual, c)


def batch_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> TreePair:
    _, in_axes, values = in_tree
    out = axis.batch(ir, in_axes=in_axes).call(*values)
    return out, tree_batched_like(ir.out_tree, True)


async def abatch_custom_call(in_tree: Tree, /, *, ir: stage.IR, **params) -> TreePair:
    _, in_axes, values = in_tree
    out = await axis.batch(ir, in_axes=in_axes).acall(*values)
    return out, tree_batched_like(ir.out_tree, True)


def without_ir(rule: Callable[..., Any], /) -> Callable[..., Any]:
    def func(in_tree: Tree, /, *, ir: stage.IR, **params):
        return rule(in_tree, **params)

    return func


def custom_prim_name(func: Callable[..., Any]) -> str:
    return f"p_custom_call<{func.__name__}-{id(func):x}>"


class CustomFunc:
    def __init__(self, func: Callable[..., Any], /):
        self.func = func
        self.prim = core.Prim(custom_prim_name(func))
        core.r_impl.set(self.prim, impl_custom_call)
        core.r_aimpl.set(self.prim, aimpl_custom_call)
        core.r_abstract.set(self.prim, abstract_custom_call)
        core.r_batch.set(self.prim, batch_custom_call)
        core.r_abatch.set(self.prim, abatch_custom_call)
        core.r_push.set(self.prim, pushforward_custom_call)
        core.r_apush.set(self.prim, apushforward_custom_call)
        core.r_pull_fwd.set(self.prim, pullback_fwd_custom_call)
        core.r_apull_fwd.set(self.prim, apullback_fwd_custom_call)
        core.r_pull_bwd.set(self.prim, pullback_bwd_custom_call)
        core.r_apull_bwd.set(self.prim, apullback_bwd_custom_call)
        ft.update_wrapper(self, func)

    def __call__(self, *args, **params):
        ir = trace_custom_func(self.func, args, params)
        return self.prim.bind(args, ir=ir, **params)

    def set_pushforward[R: Callable[..., tuple[Tree, Tree]]](self, rule: R, /) -> R:
        """Register ``rule(in_tree, /, **params) -> (out_p, out_t)``.

        Example:
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_push_example(x):
            ...     return "[" + x + "]"
            >>> @bracket_push_example.set_pushforward
            ... def bracket_push_rule(in_tree, /):
            ...     p, t = in_tree
            ...     (dx,) = t
            ...     out_p = bracket_push_example(*p)
            ...     out_t = "delta " + af.extend.instantiate(dx)
            ...     return out_p, out_t
            >>> ir = af.trace(lambda x: bracket_push_example(x))("seed")
            >>> af.pushforward(ir).call(("hello",), ("change",))
            ('[hello]', 'delta change')
        """

        core.r_push.set(self.prim, without_ir(rule))
        return rule

    def aset_pushforward[R: Callable[..., Awaitable[tuple[Tree, Tree]]]](self, rule: R, /) -> R:
        """Register an async custom pushforward rule.

        Example:
            >>> import asyncio
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_apush_example(x):
            ...     return "[" + x + "]"
            >>> @bracket_apush_example.aset_pushforward
            ... async def bracket_apush_rule(in_tree, /):
            ...     p, t = in_tree
            ...     (dx,) = t
            ...     out_p = bracket_apush_example(*p)
            ...     out_t = "async delta " + af.extend.instantiate(dx)
            ...     return out_p, out_t
            >>> ir = af.trace(lambda x: bracket_apush_example(x))("seed")
            >>> asyncio.run(af.pushforward(ir).acall(("hello",), ("change",)))
            ('[hello]', 'async delta change')
        """

        core.r_apush.set(self.prim, without_ir(rule))
        return rule

    def set_pullback[R: Callable[..., Tree]](self, rule: R, /) -> R:
        """Register ``rule(in_tree, /, **params) -> in_c``.

        Example:
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_pull_example(x):
            ...     return "[" + x + "]"
            >>> @bracket_pull_example.set_pullback
            ... def bracket_pull_rule(in_tree, /):
            ...     (p, output), c = in_tree
            ...     del p
            ...     return (c + " via " + output),
            >>> ir = af.trace(lambda x: bracket_pull_example(x))("seed")
            >>> af.pullback(ir).call(("hello",), "feedback")
            ('[hello]', ('feedback via [hello]',))
        """

        def with_output(in_tree, /, *, ir, **params):
            (p, out, _), c = in_tree
            return rule(((p, out), c), **params)

        core.r_pull_bwd.set(self.prim, with_output)
        return rule

    def aset_pullback[R: Callable[..., Awaitable[Tree]]](self, rule: R, /) -> R:
        """Register an async custom pullback rule.

        Example:
            >>> import asyncio
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_apull_example(x):
            ...     return "[" + x + "]"
            >>> @bracket_apull_example.aset_pullback
            ... async def bracket_apull_rule(in_tree, /):
            ...     (p, output), c = in_tree
            ...     del p
            ...     return ("async " + c + " via " + output),
            >>> ir = af.trace(lambda x: bracket_apull_example(x))("seed")
            >>> asyncio.run(af.pullback(ir).acall(("hello",), "feedback"))
            ('[hello]', ('async feedback via [hello]',))
        """

        async def with_output(in_tree, /, *, ir, **params):
            (p, out, _), c = in_tree
            return await rule(((p, out), c), **params)

        core.r_apull_bwd.set(self.prim, with_output)
        return rule

    def set_batch[R: Callable[..., tuple[Tree, Tree[bool]]]](self, rule: R, /) -> R:
        """Register ``rule(in_tree, /, **params) -> (outputs, out_axes)``.

        Example:
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_batch_example(x):
            ...     return "[" + x + "]"
            >>> @bracket_batch_example.set_batch
            ... def bracket_batch_rule(in_tree, /):
            ...     spec, in_axes, values = in_tree
            ...     assert spec.size == 2
            ...     (xs,) = values
            ...     (x_axis,) = in_axes
            ...     assert x_axis is True
            ...     return [("<" + x + ">") for x in xs], True
            >>> ir = af.trace(lambda x: bracket_batch_example(x))("seed")
            >>> af.batch(ir).call(["a", "b"])
            ['<a>', '<b>']
        """

        core.r_batch.set(self.prim, without_ir(rule))
        return rule

    def aset_batch[R: Callable[..., Awaitable[tuple[Tree, Tree[bool]]]]](self, rule: R, /) -> R:
        """Register an async custom batch rule.

        Example:
            >>> import asyncio
            >>> import autoform as af
            >>> @af.custom
            ... def bracket_abatch_example(x):
            ...     return "[" + x + "]"
            >>> @bracket_abatch_example.aset_batch
            ... async def bracket_abatch_rule(in_tree, /):
            ...     spec, in_axes, values = in_tree
            ...     assert spec.size == 2
            ...     (xs,) = values
            ...     (x_axis,) = in_axes
            ...     assert x_axis is True
            ...     return [("async <" + x + ">") for x in xs], True
            >>> ir = af.trace(lambda x: bracket_abatch_example(x))("seed")
            >>> asyncio.run(af.batch(ir).acall(["a", "b"]))
            ['async <a>', 'async <b>']
        """

        core.r_abatch.set(self.prim, without_ir(rule))
        return rule


def custom(func: Callable[..., Any], /) -> CustomFunc:
    """Mark a traceable Python function as a custom Autoform transform boundary.

    ``custom`` is a decorator for traceable functions that should keep their
    ordinary call behavior while optionally overriding how Autoform transforms
    treat them.
    Without any registered rules, ``pushforward``, ``pullback``, and ``batch``
    produce the same results as transforming the function body directly.

    The returned wrapper supports these rule registration decorators:

    - ``set_pushforward(rule)`` for a synchronous pushforward rule.
    - ``aset_pushforward(rule)`` for an asynchronous pushforward rule.
    - ``set_pullback(rule)`` for a synchronous pullback backward rule.
    - ``aset_pullback(rule)`` for an asynchronous pullback backward rule.
    - ``set_batch(rule)`` for a synchronous batch rule.
    - ``aset_batch(rule)`` for an asynchronous batch rule.

    Positional arguments are traced inputs. Keyword arguments are static
    parameters: they are not traced or differentiated, and rules receive them
    as keyword arguments. Like core rules, a rule takes one positional
    ``in_tree`` and the static parameters. To reuse the normal primal
    behavior, a rule calls the custom function itself. The rule signatures are:

    - Pushforward: ``rule((primals, tangents), /, **params) -> (out_p, out_t)``.
    - Pullback backward:
      ``rule(((primals, output), cotangent), /, **params) -> cotangents``.
    - Batch:
      ``rule((spec, in_axes, values), /, **params) -> (outputs, out_axes)``.

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
        ...     return "[" + x + "]"
        >>> bracket("hello")
        '[hello]'

        Without custom rules, transforms behave as if they were applied to the
        function body.

        >>> base = af.trace(lambda x: bracket(x))("seed")
        >>> af.pushforward(base).call(("hello",), ("change",))
        ('[hello]', 'change')
        >>> af.pullback(base).call(("hello",), "feedback")
        ('[hello]', ('feedback',))
        >>> af.batch(base).call(["a", "b"])
        ['[a]', '[b]']

        A pushforward rule can replace only the pushforward behavior.

        >>> @bracket.set_pushforward
        ... def bracket_push(in_tree, /):
        ...     p, t = in_tree
        ...     (dx,) = t
        ...     out_p = bracket(*p)
        ...     out_t = "delta: " + af.extend.instantiate(dx)
        ...     return out_p, out_t
        >>> af.pushforward(base).call(("hello",), ("change",))
        ('[hello]', 'delta: change')

        Pullback and batch rules use the same one-``in_tree`` convention.

        >>> @bracket.set_pullback
        ... def bracket_pull(in_tree, /):
        ...     (p, output), c = in_tree
        ...     (x,) = p
        ...     return (c + " via " + output + " from " + x),
        >>> af.pullback(base).call(("hello",), "feedback")
        ('[hello]', ('feedback via [hello] from hello',))

        >>> @bracket.set_batch
        ... def bracket_batch(in_tree, /):
        ...     spec, in_axes, values = in_tree
        ...     assert spec.size == 2
        ...     (xs,) = values
        ...     (x_axis,) = in_axes
        ...     assert x_axis is True
        ...     return [("<" + x + ">") for x in xs], True
        >>> af.batch(base).call(["a", "b"])
        ['<a>', '<b>']

        A common use is wrapping an LM call so the forward call remains normal,
        while pushforward and pullback use prompts written for that application.

        >>> from autoform.extend import instantiate
        >>> @af.custom
        ... def summarize(text, *, model):
        ...     message = "Summarize this in one sentence: " + text
        ...     return af.lm.fill({"prompt": message, "output": af.lm.Str()}, model=model)["output"]

        The custom pushforward rule can ask the model how the output should
        change under an input edit.

        >>> @summarize.set_pushforward
        ... def summarize_push(in_tree, /, *, model):
        ...     (text,), (t_text,) = in_tree
        ...     out_p = summarize(text, model=model)
        ...     prompt = (
        ...         "Original input:\\n" + text
        ...         + "\\n\\nInput edit:\\n" + instantiate(t_text)
        ...         + "\\n\\nDescribe how the summary should change."
        ...     )
        ...     out_t = af.lm.fill({"prompt": prompt, "output": af.lm.Str()}, model=model)["output"]
        ...     return out_p, out_t

        The custom pullback rule can replace the default backward LM prompt with
        a domain-specific feedback prompt.

        >>> @summarize.set_pullback
        ... def summarize_pull(in_tree, /, *, model):
        ...     ((text,), output), c = in_tree
        ...     prompt = (
        ...         "Original input:\\n" + text
        ...         + "\\n\\nLM output:\\n" + output
        ...         + "\\n\\nDownstream feedback:\\n" + instantiate(c)
        ...         + "\\n\\nReturn feedback for improving the original input."
        ...     )
        ...     text_cotangent = af.lm.fill({"prompt": prompt, "output": af.lm.Str()}, model=model)["output"]
        ...     return (text_cotangent,)

        >>> lm_ir = af.trace(lambda text: summarize(text, model="model-name"))("topic")
        >>> af.pushforward(lm_ir).call(("recursion",), ("focus on the recursive step",))  # doctest: +SKIP
        >>> af.pullback(lm_ir).call(("recursion",), "make the answer more concrete")  # doctest: +SKIP
    """
    assert callable(func), f"Expected a callable function, got {type(func)}"
    return CustomFunc(func)
