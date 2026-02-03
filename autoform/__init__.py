"""Autoform: Composable function transformations for LLM programs."""

from autoform.ad import pullback, pushforward
from autoform.batch import batch
from autoform.control import stop_gradient, switch, while_loop
from autoform.core import acall, atrace, call, trace
from autoform.dce import dce
from autoform.intercept import checkpoint, collect, inject, memoize
from autoform.lm import lm_call, struct_lm_call
from autoform.scheduling import depends, gather, sched
from autoform.string import concat, format, match
from autoform.surgery import split, splitpoint
from autoform.utils import PYTREE_NAMESPACE, Struct

__all__ = [
    # core
    "trace",
    "atrace",
    # execution
    "call",
    "acall",
    # transformations
    "pushforward",
    "pullback",
    "batch",
    "collect",
    "inject",
    "dce",
    "sched",
    "memoize",
    # primitives
    "format",
    "concat",
    "match",
    "lm_call",
    "struct_lm_call",
    "stop_gradient",
    "checkpoint",
    "split",
    "splitpoint",
    "switch",
    "while_loop",
    "gather",
    "depends",
    # types
    "Struct",
    "PYTREE_NAMESPACE",
]
