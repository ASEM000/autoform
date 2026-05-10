# Fold

`fold()` changes tracing inside its block. Normally, primitives inside `af.trace(...)` become IR equations. Inside `af.fold()`, primitives are evaluated immediately and their concrete result is embedded as a literal in the surrounding trace.

```python
import autoform as af


def program(text: str) -> str:
    with af.fold():
        prefix = af.concat("hello", " ")
    return af.concat(prefix, text)


ir = af.trace(program)("seed")

assert len(ir.ir_eqns) == 1
assert ir.call("world") == "hello world"
```

Without `fold`, the prefix concat would also be an equation. With `fold`, it is computed at trace time.

(trace-time-decisions)=
## Trace-Time Decisions

Folded work can choose ordinary Python control flow because it runs while tracing:

```python
def route(text: str) -> str:
    with af.fold():
        label = af.concat("priority", ": high")
    if label == "priority: high":
        return af.concat("yes: ", text)
    return af.concat("no: ", text)


ir = af.trace(route)("seed")
assert ir.call("answer") == "yes: answer"
```

The branch is chosen during tracing. The resulting IR contains only the path that was taken.

## Dynamic Values Cannot Be Folded

Folded work must not depend on dynamic traced values:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = af.concat(text, " ")
    return prefix
```

That raises during tracing because `text` is not concrete. Mark the dependency static or move the computation out of the `fold` block.

Outside tracing, `fold()` is a no-op context manager.
