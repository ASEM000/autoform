"""Autoform: Composable function transformations for LLM programs."""

from autoform.core import build_ir
from autoform.evaluation import run_ir, iter_ir, arun_ir

from autoform.ad import pushforward_ir, pullback_ir
from autoform.batch import batch_ir
from autoform.harvest import reap_ir, plant_ir, split_ir, merge_ir, sow
from autoform.optims import dce_ir, fold_ir

from autoform.string import format, concat
from autoform.lm import lm_call, struct_lm_call, Struct
from autoform.control import stop_gradient, ir_call, switch

__all__ = [
    # core
    "build_ir",
    "run_ir",
    "iter_ir",
    "arun_ir",
    # transformations
    "pushforward_ir",
    "pullback_ir",
    "batch_ir",
    "reap_ir",
    "plant_ir",
    "split_ir",
    "merge_ir",
    "dce_ir",
    "fold_ir",
    # primitives (user-facing functions)
    "format",
    "concat",
    "lm_call",
    "struct_lm_call",
    "stop_gradient",
    "sow",
    "ir_call",
    "switch",
    # types
    "Struct",
]
