# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import autoform as af

execute = af.core.IR.call


def aexecute(ir, *args, **kwargs):
    return asyncio.run(ir.acall(*args, **kwargs))


def prefix_name(prefix, name):
    return af.string.format("{prefix} {name}", prefix=prefix, name=name)


def append_bang(x):
    return af.string.concat(x, "!")


def bracket_text(x):
    return af.string.format("[{x}]", x=x)


def angle_text(x):
    return af.string.format("<{x}>", x=x)


def dependent_formats(x):
    a = af.string.format("A: {x}", x=x)
    b = af.string.format("B: {x}", x=x)
    return af.depends(b, a)


def always_true(x):
    return True


def switch_program(branches):
    def program(key, x):
        return af.switch(key, branches, x)

    return program


def while_program(cond_ir, body_ir, *, max_iters):
    def program(init):
        return af.while_loop(cond_ir, body_ir, init, max_iters=max_iters)

    return program


def fixpoint_program(step_ir, **kwargs):
    def program(init, parameter):
        return af.fixpoint(step_ir, init, parameter, **kwargs)

    return program


class CountingInterpreter(af.core.Interpreter):
    def __init__(self):
        self.parent = af.core.active_interpreter.get()
        self.calls = 0

    def interpret(self, prim, in_tree, /, **params):
        self.calls += 1
        return self.parent.interpret(prim, in_tree, **params)

    async def ainterpret(self, prim, in_tree, /, **params):
        self.calls += 1
        return await self.parent.ainterpret(prim, in_tree, **params)
