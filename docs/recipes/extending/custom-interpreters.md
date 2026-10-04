# Custom Interpreters

Custom interpreters provide a way to add an execution-time layer in between dispatching a primitive. A custom interpreter adds an execution policy while preserving the traced function and IR.

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

## Program

Trace a program whose draft operations are tagged:

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import autoform as af
import autoform.extend as afe


def program(topic: str) -> str:
    with af.tag("draft"):
        draft = "draft for " + topic
    return draft + "."


ir = af.trace(program)("topic x")
```

## Dispatch

Store the current interpreter as `parent`, then delegate to it from
`interpret(...)` and `ainterpret(...)`. The example below records primitive
outputs, but the same shape can route, block, or modify primitive dispatch.

Each method of an interpreter will be passed a single primitive call.

| Name | Meaning |
| --- | --- |
| `prim` | The primitive key being executed. |
| `in_tree` | The concrete input pytree for that primitive call. |
| `params` | Static primitive parameters recorded in the IR equation. |

The method must return the primitive output. Calling `self.parent.interpret(...)`
runs the next interpreter in the stack. If there is no custom parent, the default
interpreter reaches the registered implementation rule.

Capture the parent before installing the custom interpreter. Calling `prim.bind(...)` from `interpret(...)` would dispatch to the same active interpreter again and recurse.

If another interpreter is already active, `parent` points to that interpreter.
Delegation keeps that policy in the dispatch chain:

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

Note that the method for each of the sync and async cases must be implemented separately, because `.call(...)` will use `interpret(...)` whereas `.acall(...)` will use `ainterpret(...)`.

Record before delegation when the policy should inspect or reject a call before
it runs. Record after delegation when the policy needs the produced output.

## Context

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

This will give the following result:

```text
draft for topic y.
[CallRecord(prim_name='concat', tags=frozenset({'draft'}), output='draft for topic y'), CallRecord(prim_name='concat', tags=frozenset(), output='draft for topic y.')]
```

The records contain the primitive name, tags, and produced value for each executed call. The context restores the previous interpreter when it exits.

## Boundary Choice

Choose the API according to the part of execution that needs control:

| Need | Use |
| --- | --- |
| Add a runtime operation to the IR | [Primitive Definitions](writing-primitives.md) |
| Customize transform behavior at a traceable helper boundary | {py:func}`custom <autoform.custom>` |
| Add execution-time policy around primitive dispatch | interpreter |
| Pause, yield, stream, or replace top-level equation outputs | `ir.walk(...)` |

`ir.walk(...)` is usually the right boundary for custom execution loops. An
interpreter is lower level: it runs inside ordinary `.call(...)` or `.acall(...)`
and changes how primitive dispatch is handled, including dispatch that happens
while binding a top-level equation.
