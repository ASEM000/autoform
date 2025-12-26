"""IR evaluation functions"""

from __future__ import annotations

import functools as ft
import itertools as it
import typing as tp
from operator import setitem

from autoform.core import IR, IRLit, IRVar, Value, is_irvar
from autoform.core import async_rules, iter_rules
from autoform.utils import Tree, pack_user_input, treelib

# ==================================================================================================
# RUN IR
# ==================================================================================================


def run_ir(ir: IR, *args, **kwargs) -> Tree:
    """Run the given intermediate representation (IR) with the provided inputs.

    Args:
        ir: The intermediate representation to run.
        *args: Positional arguments to provide as inputs to the IR.
        **kwargs: Keyword arguments to provide as inputs to the IR.

    Returns:
        The output tree produced by running the IR.

    Example:
        >>> import autoform as af
        >>> def ir(x, y):
        ...     return af.concat(x, y)
        >>> ir = af.build_ir(ir, "Hello, ", "World!")
        >>> af.run_ir(ir, "Hello, ", "World!")
        'Hello, World!'
    """
    assert isinstance(ir, IR), f"{type(ir)=} is not an IR instance."

    in_tree = pack_user_input(*args, **kwargs)
    env: dict[IRVar, Value] = {}

    def write(atom: IRVar, value):
        is_irvar(atom) and setitem(env, atom, value)

    def read(atom) -> Value:
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    treelib.map(write, ir.in_irtree, in_tree)

    for ireqn in ir.ireqns:
        in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
        out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
        treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
    return treelib.map(read, ir.out_irtree)


# ==================================================================================================
# ITER IR
# ==================================================================================================


def accumulate_chunks(chunks: list[tp.Any]) -> tp.Any:
    if not chunks:
        return None
    head = chunks[0]
    if isinstance(head, str):
        return "".join(chunks)
    if isinstance(head, list):
        return list(it.chain.from_iterable(chunks))
    try:
        return ft.reduce(lambda a, b: a + b, chunks)
    except TypeError:
        return chunks


def iter_ir(ir: IR, *args, **kwargs):
    """Iterate through IR execution, yielding intermediate results.

    Enables streaming execution for primitives with `iter_rules` (e.g., `lm_call`).
    Yields chunks as they become available, with the final yield being the complete result.

    Args:
        ir: The intermediate representation to execute.
        *args: Positional arguments to provide as inputs.
        **kwargs: Keyword arguments to provide as inputs.

    Yields:
        Intermediate chunks from streaming primitives, then the final result.

    Example:
        >>> import autoform as af
        >>> def program(x):
        ...     return af.concat("Result: ", x)
        >>> ir = af.build_ir(program, "test")
        >>> chunks = list(af.iter_ir(ir, "hello"))
        >>> chunks[-1]  # Final result is always last
        'Result: hello'
    """
    in_tree = pack_user_input(*args, **kwargs)
    env: dict[IRVar, Value] = {}

    def write(atom: IRVar, value: Value):
        is_irvar(atom) and setitem(env, atom, value)

    treelib.map(write, ir.in_irtree, in_tree)

    def read(atom):
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    for ireqn in ir.ireqns:
        in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
        if ireqn.prim in iter_rules:
            iter_rule = iter_rules[ireqn.prim]
            out_treespec = treelib.structure(ireqn.out_irtree)
            acc = [[] for _ in range(out_treespec.num_leaves)]
            for chunk in iter_rule(in_ireqn_tree, **ireqn.params):
                for i, leaf in enumerate(out_treespec.flatten_up_to(chunk)):
                    acc[i].append(leaf)
                yield chunk
            out_ireqn_tree = out_treespec.unflatten(map(accumulate_chunks, acc))
        else:
            out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)

        treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
    yield treelib.map(read, ir.out_irtree)


# ==================================================================================================
# ASYNC RUN IR
# ==================================================================================================


async def arun_ir(ir: IR, *args, **kwargs):
    """Asynchronously execute the IR.

    Enables concurrent execution for primitives with `async_rules` (e.g., `lm_call`).
    Useful for running multiple IR executions in parallel with `asyncio.gather`.

    Args:
        ir: The intermediate representation to execute.
        *args: Positional arguments to provide as inputs.
        **kwargs: Keyword arguments to provide as inputs.

    Returns:
        The output tree produced by running the IR.

    Example:
        >>> import autoform as af
        >>> import asyncio
        >>> def program(x):
        ...     return af.concat("Hello, ", x)
        >>> ir = af.build_ir(program, "test")
        >>> asyncio.run(af.arun_ir(ir, "World"))
        'Hello, World'
    """
    in_tree = pack_user_input(*args, **kwargs)
    env: dict[IRVar, Value] = {}

    def write(atom: IRVar, value):
        is_irvar(atom) and setitem(env, atom, value)

    treelib.map(write, ir.in_irtree, in_tree)

    def read(atom) -> Value:
        return env[atom] if is_irvar(atom) else tp.cast(IRLit, atom).value

    for ireqn in ir.ireqns:
        in_ireqn_tree = treelib.map(read, ireqn.in_irtree)
        if ireqn.prim in async_rules:
            async_rule = async_rules[ireqn.prim]
            out_ireqn_tree = await async_rule(in_ireqn_tree, **ireqn.params)
        else:
            out_ireqn_tree = ireqn.prim.bind(in_ireqn_tree, **ireqn.params)
        treelib.map(write, ireqn.out_irtree, out_ireqn_tree)
    return treelib.map(read, ir.out_irtree)
