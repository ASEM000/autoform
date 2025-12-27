"""Autoform: Composable function transformations for LLM programs."""

from autoform.core import build_ir

from autoform.ad import pushforward, pullback
from autoform.batch import batch
from autoform.harvest import collect, inject, checkpoint
from autoform.optims import dce, fold
from autoform.core import call, icall, acall

from autoform.string import format, concat
from autoform.lm import lm_call, struct_lm_call, Struct
from autoform.control import stop_gradient, switch

from autoform.utils import PYTREE_NAMESPACE

__all__ = [
    # core
    "build_ir",
    # execution
    "call",
    "icall",
    "acall",
    # transformations
    "pushforward",
    "pullback",
    "batch",
    "collect",
    "inject",
    "dce",
    "fold",
    # primitives (user-facing functions)
    "format",
    "concat",
    "lm_call",
    "struct_lm_call",
    "stop_gradient",
    "checkpoint",
    "switch",
    # types
    "Struct",
    "PYTREE_NAMESPACE",
]
