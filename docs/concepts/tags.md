# Tags

Tags attach structured metadata to IR equations while tracing. They do not change execution by themselves; they give later code a way to recognize equations that belong to a logical region.

Define a hashable `Tag` subclass, then activate tag instances with `af.tag(...)`:

```python
from dataclasses import dataclass
import autoform as af


@dataclass(frozen=True)
class Region(af.Tag):
    name: str


def program(text: str) -> str:
    with af.tag(Region("draft")):
        text = af.concat(text, "!")
    return af.format("[{}]", text)


ir = af.trace(program)("seed")
assert Region("draft") in ir.ir_eqns[0].tags
assert Region("draft") not in ir.ir_eqns[1].tags
```

Nested tag blocks accumulate tags. Code outside the block does not receive the tags from the block.

## Use Tags With `sched`

`sched` accepts a `cond` callback that receives each IR equation. Tags give that callback a stable way to select only part of a traced program.

```python
scheduled = af.sched(ir, cond=lambda ir_eqn: Region("draft") in ir_eqn.tags)
assert scheduled.call("world") == "[world!]"
```

Tags are also visible when manually stepping through an IR. See [Walk](walk.md) for the advanced execution interface.

## Tag Class Rules

`Tag` itself cannot be instantiated directly. Subclasses must be hashable because equation tags are stored in a `frozenset`. A frozen dataclass is usually the most convenient shape.

Equality is whatever your subclass defines. A frozen dataclass compares by fields; a plain class compares by identity unless you implement equality.
