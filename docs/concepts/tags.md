# Tags

Tags attach metadata to IR equations while tracing. They do not change execution by themselves; they give later code a way to recognize equations that belong to a logical region.

Pass any hashable value to {py:func}`tag <autoform.tag>`:

```python
import autoform as af


draft = "draft"


def program(text: str) -> str:
    with af.tag(draft):
        text = af.concat(text, "!")
    return af.format("[{}]", text)


ir = af.trace(program)("seed")
assert draft in ir.eqns[0].tags
assert draft not in ir.eqns[1].tags
```

Nested tag blocks accumulate tags. Code outside the block does not receive the tags from the block.

## Scheduling Tags

{py:func}`sched <autoform.sched>` accepts a `cond` callback that receives each IR equation. Tags give that callback a stable way to select only part of a traced program.

```python
scheduled = af.sched(ir, cond=lambda eqn: draft in eqn.tags)
assert scheduled.call("world") == "[world!]"
```

## Walk Tags

[Walk](walk.md) steps through an IR equation by equation. Tags are available on each yielded equation, so a debugger or custom runner can act on tagged regions without guessing from primitive names or source order.

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

{py:func}`tag <autoform.tag>` asserts that every tag value is hashable because equation tags are stored in a `frozenset`. Strings, integers, tuples of hashable values, and other immutable identifiers work.

For structured tags, use a hashable object directly. A frozen dataclass is usually the most convenient payload:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    stage: int


draft_region = Region("draft", 1)
```
