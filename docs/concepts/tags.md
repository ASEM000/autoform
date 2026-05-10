# Tags

Tags attach metadata to IR equations while tracing. They do not change execution by themselves; they give later code a way to recognize equations that belong to a logical region.

Create a tag from any hashable value, then activate it with `af.tag(...)`:

```python
import autoform as af


draft = af.Tag("draft")


def program(text: str) -> str:
    with af.tag(draft):
        text = af.concat(text, "!")
    return af.format("[{}]", text)


ir = af.trace(program)("seed")
assert draft in ir.ir_eqns[0].tags
assert draft not in ir.ir_eqns[1].tags
```

Nested tag blocks accumulate tags. Code outside the block does not receive the tags from the block.

## Tags in `sched`

`sched` accepts a `cond` callback that receives each IR equation. Tags give that callback a stable way to select only part of a traced program.

```python
scheduled = af.sched(ir, cond=lambda ir_eqn: draft in ir_eqn.tags)
assert scheduled.call("world") == "[world!]"
```

## Tags in `walk`

[Walk](walk.md) steps through an IR equation by equation. Tags are available on each yielded equation, so a debugger or custom runner can act on tagged regions without guessing from primitive names or source order.

```python
def run_and_record_tagged_prims(ir, text: str):
    tagged_prims = []
    gen = ir.walk(text)
    ir_eqn, in_values = next(gen)

    while ir_eqn is not None:
        if draft in ir_eqn.tags:
            tagged_prims.append(ir_eqn.prim.name)
        out_values = ir_eqn.bind(in_values, **ir_eqn.params)
        ir_eqn, in_values = gen.send(out_values)

    return in_values, tagged_prims


output, tagged_prims = run_and_record_tagged_prims(ir, "world")

assert output == "[world!]"
assert tagged_prims == ["concat"]
```

## `Tag` Values

`Tag(value)` asserts that `value` is hashable because equation tags are stored in a `frozenset`. Strings, integers, tuples of hashable values, and other immutable identifiers work.

Two plain `Tag` instances compare by their wrapped value: `af.Tag("draft") == af.Tag("draft")`.

For structured tags, wrap a hashable object. A frozen dataclass is usually the most convenient payload:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    stage: int


draft_region = af.Tag(Region("draft", 1))
```

Do not subclass `Tag`; keep `Tag` as the wrapper around a hashable value.
