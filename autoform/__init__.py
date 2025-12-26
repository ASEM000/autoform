"""Autoform: Composable function transformations for LLM programs."""

# Subpackages
from autoform import core
from autoform import evaluation
from autoform import transforms

# Core API
from autoform.core import build_ir
from autoform.evaluation import run_ir, iter_ir, arun_ir

# Transformations
from autoform.transforms import (
    pushforward_ir,
    pullback_ir,
    batch_ir,
    reap_ir,
    plant_ir,
    split_ir,
    merge_ir,
    sow,
    dce_ir,
    fold_ir,
)

# User-facing primitives (functions only)
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
    # subpackages
    "core",
    "evaluation",
    "transforms",
]
