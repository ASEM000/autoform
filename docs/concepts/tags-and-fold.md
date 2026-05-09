# Tags And Fold

`Tag`, `tag`, and `fold` are trace-time tools. Most users can ignore them, but they are useful when you are inspecting or shaping IR construction.

## Tags

A tag marks equations with structured metadata while tracing. Define a hashable `Tag` subclass, then activate instances with `af.tag(...)`.

```python
from dataclasses import dataclass
import autoform as af


@dataclass(frozen=True)
class Label(af.Tag):
    name: str


def program(text: str) -> str:
    with af.tag(Label("draft")):
        return af.concat(text, "!")


ir = af.trace(program)("seed")
scheduled = af.sched(ir, cond=lambda ir_eqn: Label("draft") in ir_eqn.tags)
assert scheduled.call("world") == "world!"
```

The scheduling predicate receives each recorded equation. Nested tag blocks
accumulate tags. Code outside the block does not receive them.

## Tag Class Rules

`Tag` itself cannot be instantiated directly. Subclasses must be hashable because equation tags are stored in a `frozenset`. A frozen dataclass is usually the most convenient shape.

The equality behavior is whatever your subclass defines. A frozen dataclass compares by fields; a plain class compares by identity unless you implement equality.

## Fold

`fold()` changes tracing inside its block. Normally, primitives inside `af.trace(...)` become IR equations. Inside `af.fold()`, primitives are evaluated immediately and their concrete result is embedded as a literal in the surrounding trace.

```python
def program(text: str) -> str:
    with af.fold():
        prefix = af.concat("hello", " ")
    return af.concat(prefix, text)


ir = af.trace(program)("seed")

assert ir.call("world") == "hello world"
```

Without `fold`, the prefix concat would also be an equation. With `fold`, it is computed at trace time.

Folded work must not depend on dynamic traced values:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = af.concat(text, " ")  # depends on traced text
    return prefix
```

That raises during tracing because `text` is not concrete. Mark the dependency static or move the computation out of the fold block.

Outside tracing, `fold()` is a no-op context manager.
