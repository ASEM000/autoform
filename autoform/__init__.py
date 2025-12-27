"""Autoform: Composable function transformations for LLM programs."""

from autoform.core import build_ir
from autoform.evaluation import run_ir, iter_ir, arun_ir

from autoform.ad import pushforward_ir, pullback_ir
from autoform.batch import batch_ir
from autoform.harvest import run_and_reap, run_and_plant, split_ir, merge_ir, sow
from autoform.optims import dce_ir, fold_ir

from autoform.string import format, concat
from autoform.lm import lm_call, struct_lm_call, Struct
from autoform.control import stop_gradient, switch

from autoform.utils import PYTREE_NAMESPACE

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
    "run_and_reap",
    "run_and_plant",
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
    "switch",
    # types
    "Struct",
    "PYTREE_NAMESPACE",
]
