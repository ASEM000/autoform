# Define a Custom Interpreter

A custom interpreter is an execution-time layer around primitive dispatch. Use
one when a program should keep the same traced function and IR, but execution
needs an extra policy such as recording, routing, blocking, or modifying
primitive calls.

```{admonition} Advanced
:class: info

Custom interpreters use `autoform.extend`. Most programs should use ordinary
execution, transforms, and public context managers first.
```

```{admonition} Concept
[Trace, IR, Execute](../../concepts/trace-ir-execute.md) ·
[Primitives](../../concepts/primitives.md) · [Tags](../../concepts/tags.md) ·
[Walk](../../concepts/walk.md)
```

## Trace a Program Once

Trace the program once. The interpreter is installed later, around execution.

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import autoform as af
import autoform.extend as afe


def program(topic: str) -> str:
    with af.tag("draft"):
        draft = af.format("draft for {}", topic)
    return af.concat(draft, ".")


ir = af.trace(program)("topic x")
```

## Define the Interpreter

Store the current interpreter as `parent`, then delegate to it from
`interpret(...)` and `ainterpret(...)`. The example below records primitive
outputs, but the same shape can route, block, or modify primitive dispatch.

An interpreter method receives one primitive call:

| Name | Meaning |
| --- | --- |
| `prim` | The primitive key being executed. |
| `in_tree` | The concrete input pytree for that primitive call. |
| `params` | Static primitive parameters recorded in the IR equation. |

The method must return the primitive output. Calling `self.parent.interpret(...)`
runs the next interpreter in the stack. If there is no custom parent, the default
interpreter reaches the registered implementation rule.

The parent is captured before the custom interpreter is installed. This gives
the custom interpreter a place to delegate the actual primitive call. Calling
`prim.bind(...)` from inside `interpret(...)` would dispatch to the active
interpreter again, which is the same custom interpreter, causing recursive
dispatch instead of execution.

If another interpreter is already active, `parent` points to that interpreter.
This preserves interpreter stacking: the new policy runs first, then delegates
to the policy that was active before it.

```python
@dataclass(frozen=True)
class CallRecord:
    prim_name: str
    tags: frozenset[Any]
    output: Any


class RecordingInterpreter(afe.Interpreter):
    def __init__(self):
        self.parent = afe.active_interpreter.get()
        self.records: list[CallRecord] = []

    def interpret(self, prim: afe.Prim, in_tree: Any, /, **params):
        output = self.parent.interpret(prim, in_tree, **params)
        self.records.append(
            CallRecord(
                prim_name=prim.name,
                tags=afe.active_tags.get(),
                output=output,
            )
        )
        return output

    async def ainterpret(self, prim: afe.Prim, in_tree: Any, /, **params):
        output = await self.parent.ainterpret(prim, in_tree, **params)
        self.records.append(
            CallRecord(
                prim_name=prim.name,
                tags=afe.active_tags.get(),
                output=output,
            )
        )
        return output
```

The sync and async methods are separate because `.call(...)` uses
`interpret(...)`, while `.acall(...)` uses `ainterpret(...)`.

Record before delegation when the policy should inspect or reject a call before
it runs. Record after delegation when the policy needs the produced output.

## Install the Interpreter

Use {py:func}`using_interpreter <autoform.extend.using_interpreter>` as a
temporary execution context:

```python
@contextmanager
def record_calls():
    with afe.using_interpreter(RecordingInterpreter()) as interpreter:
        yield interpreter


with record_calls() as recorder:
    result = ir.call("topic y")

print(result)
print(recorder.records)
```

Expected result:

```text
draft for topic y.
[CallRecord(prim_name='format', tags=frozenset({'draft'}), output='draft for topic y'), CallRecord(prim_name='concat', tags=frozenset(), output='draft for topic y.')]
```

The traced function does not change. The IR does not change. Only the execution
context around `ir.call(...)` changes.

## Choose the Boundary

| Need | Use |
| --- | --- |
| Add a runtime operation to the IR | [Write a Primitive](writing-primitives.md) |
| Customize transform behavior at a traceable helper boundary | {py:func}`custom <autoform.custom>` |
| Add execution-time policy around primitive dispatch | interpreter |
| Pause, yield, stream, or replace top-level equation outputs | `ir.walk(...)` |

`ir.walk(...)` is usually the right boundary for custom execution loops. An
interpreter is lower level: it runs inside ordinary `.call(...)` or `.acall(...)`
and changes how primitive dispatch is handled, including dispatch that happens
while binding a top-level equation.
