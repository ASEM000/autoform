"""Effects system"""

from __future__ import annotations

import functools as ft
import typing as tp
from abc import ABC, abstractmethod
from contextlib import contextmanager

from autoform.core import (
    Interpreter,
    Primitive,
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
    get_interpreter,
    impl_rules,
    pull_bwd_rules,
    pull_fwd_rules,
    push_rules,
    using_interpreter,
)
from autoform.utils import Tree

# ==================================================================================================
# EFFECTS
# ==================================================================================================


class Effect:
    __slots__ = "key"

    def __init__(self, *, key: tp.Hashable):
        self.key = key


effect_p = Primitive("effect", tag="effect")


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
# HANDLERS
# ==================================================================================================


class EffectHandler(Interpreter, ABC):
    handles: tp.ClassVar[tuple[type[Effect], ...]] = ()

    def __init__(self):
        self.parent = get_interpreter()

    def interpret(self, prim: Primitive, in_tree: Tree, **params) -> Tree:
        if prim == effect_p:
            if (effect := params.get("effect")) is not None and isinstance(effect, self.handles):
                result = self.handle(effect, in_tree)
                return self.parent.interpret(prim, result, **params)
        return self.parent.interpret(prim, in_tree, **params)

    @abstractmethod
    def handle(self, effect: Effect, value: tp.Any) -> tp.Any: ...


@contextmanager
def using_handler(handler: EffectHandler) -> tp.Generator[EffectHandler, None, None]:
    assert isinstance(handler, EffectHandler), "Handler must be an instance of EffectHandler"
    with using_interpreter(handler):
        yield handler
