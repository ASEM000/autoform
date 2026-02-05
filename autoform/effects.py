"""Effects"""

# NOTE(asem): effects provide a way to intercept primitive evaluation
# with custom behavior, without defining new primitives or rules. the handler
# intercepts during execution, not during transformations (pushforward,
# pullback, batch) - those still use standard rules.
#
# a handler communicates with the interpreter using a generator pattern:
#   yield in_tree  -> invoke the primitive, receive result
#   return result  -> return final value to caller
#
# this enables: logging, caching, multi-shot continuations value injection, and other
# runtime behaviors hard to achieve with standard rules.

from __future__ import annotations

from autoform.core import (
    Primitive,
    PrimitiveTag,
    batch_rules,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)
from autoform.utils import asyncify

# ==================================================================================================
# EFFECT
# ==================================================================================================


class EffectTag(PrimitiveTag): ...


effect_p = Primitive("effect", tag={EffectTag})


def effect(x, /, **p):
    return effect_p.bind(x, **p)


def impl_effect(x, /, **_):
    return x


def eval_effect(x, /, **_):
    return x


def push_effect(in_tree, /, **params):
    primal, tangent = in_tree
    return effect_p.bind(primal, **params), effect_p.bind(tangent, **params)


def pull_fwd_effect(x, /, **params):
    return effect_p.bind(x, **params), None


def pull_bwd_effect(in_tree, /, **params):
    _, cotangent = in_tree
    return effect_p.bind(cotangent, **params)


def batch_effect(in_tree, /, **params):
    _, in_batched, x = in_tree
    return effect_p.bind(x, **params), in_batched


impl_rules.set(effect_p, impl_effect)
impl_rules.aset(effect_p, asyncify(impl_effect))
eval_rules.set(effect_p, eval_effect)
push_rules.set(effect_p, push_effect)
push_rules.aset(effect_p, asyncify(push_effect))
pull_fwd_rules.set(effect_p, pull_fwd_effect)
pull_fwd_rules.aset(effect_p, asyncify(pull_fwd_effect))
pull_bwd_rules.set(effect_p, pull_bwd_effect)
pull_bwd_rules.aset(effect_p, asyncify(pull_bwd_effect))
batch_rules.set(effect_p, batch_effect)
batch_rules.aset(effect_p, asyncify(batch_effect))
