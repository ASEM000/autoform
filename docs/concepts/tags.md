# Tags

Tags are simply metadata that is traced along with the IR equations of a program. Tags are used to select a certain part of the program for schedulers/runners etc. but have no inherent effect on the execution.

Pass any hashable value to {py:func}`tag <autoform.tag>`:

```python
import autoform as af


draft = "draft"


def program(text: str) -> str:
    with af.tag(draft):
        text = text + "!"
    return "[" + text + "]"


ir = af.trace(program)("seed")
assert draft in ir.eqns[0].tags
assert draft not in ir.eqns[1].tags
```

Nested tag blocks accumulate tags. Code outside the block does not receive the tags from the block.

## Scheduling

{py:func}`sched <autoform.sched>` accepts a `cond` callback that receives each IR equation. Use a tag to select equations for scheduling:

```python
scheduled = af.sched(ir, cond=lambda eqn: draft in eqn.tags)
assert scheduled.call("world") == "[world!]"
```

## Manual Execution

[Walk](walk.md) steps through an IR equation by equation. Tags are available on each yielded equation, so a debugger or custom runner can act on tagged regions using the metadata recorded on each equation:

```python
def run_and_record_tagged_prims(ir, text: str):
    tagged_prims = []
    gen = ir.walk(text)
    eqn, in_values = next(gen)

    while eqn is not None:
        if draft in eqn.tags:
            tagged_prims.append(eqn.prim.name)
        out_values = eqn.bind(in_values, **eqn.params)
        eqn, in_values = gen.send(out_values)

    return in_values, tagged_prims


output, tagged_prims = run_and_record_tagged_prims(ir, "world")

assert output == "[world!]"
assert tagged_prims == ["concat"]
```

## Hashable Values

The {py:func}`tag <autoform.tag>` function asserts that the tag is hashable, because it is stored with each equation in a `frozenset`. Often, simple strings, integers, tuples of other hashable things etc. are enough to tag parts of a program.

However, if more complex tag information is needed, any hashable object can be used. Usually, a frozen dataclass is the easiest way to create such a payload:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    stage: int


draft_region = Region("draft", 1)
```
