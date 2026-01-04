"""Effects system"""

from __future__ import annotations

import functools as ft
import typing as tp
from contextlib import contextmanager

from autoform.core import (
    Effect,
    EffectInterpreter,
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
    using_interpreter,
)
from autoform.utils import Tree


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


# ==================================================================================================
# EFFECT HANDLER MAPPING
# ==================================================================================================


Handler = tp.Callable[[Effect, Tree], tp.Generator[tp.Any, tp.Any, tp.Any]]


@contextmanager
def using_effect_handler(handlers: dict[type[Effect], Handler]):
    """Context manager for activating effect handlers.

    Args:
        handlers: Mapping from Effect type to handler callable.

    Example:
        >>> def my_handler(effect, in_tree):
        ...     result = yield in_tree
        ...     return result
        ...     yield
        >>> with using_effect_handler({MyEffect: my_handler}):  # doctest: +SKIP
        ...     result = af.call(ir)(x)
    """
    assert isinstance(handlers, dict), f"Expected dict, got {type(handlers)}"
    for effect_key in handlers:
        assert issubclass(effect_key, Effect), f"Expected Effect, got {effect_key}"

    with using_interpreter(EffectInterpreter(handlers)):
        yield
