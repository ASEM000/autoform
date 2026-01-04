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

import functools as ft

from autoform.core import (
    Primitive,
    PrimitiveTag,
    batch_rules,
    dce_rules,
    default_batch,
    default_dce,
    default_eval,
    default_impl,
    default_pull_bwd,
    default_pull_fwd,
    default_push,
    eval_rules,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
)


class EffectTag(PrimitiveTag): ...


effect_p = Primitive("effect", tag={EffectTag})


def effect(x, **p):
    return effect_p.bind(x, **p)


impl_rules.def_rule(effect_p, default_impl)
eval_rules.def_rule(effect_p, default_eval)
push_rules.def_rule(effect_p, ft.partial(default_push, effect))
pull_fwd_rules.def_rule(effect_p, ft.partial(default_pull_fwd, effect))
pull_bwd_rules.def_rule(effect_p, ft.partial(default_pull_bwd, effect))
batch_rules.def_rule(effect_p, ft.partial(default_batch, effect))
dce_rules.def_rule(effect_p, default_dce)
