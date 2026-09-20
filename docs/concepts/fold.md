# Fold

{py:func}`fold <autoform.fold>` changes tracing inside its block. Normally, primitives inside {py:func}`trace <autoform.trace>` become IR equations. Inside the {py:func}`fold <autoform.fold>` block, primitives are evaluated immediately and their concrete result is embedded as a literal in the surrounding trace.

```python
import autoform as af


def program(text: str) -> str:
    with af.fold():
        prefix = f"v{af.numeric.add(1, 1)}: "
    return prefix + text


ir = af.trace(program)("seed")

assert len(ir.eqns) == 1
assert ir.call("world") == "v2.0: world"
```

Without {py:func}`fold <autoform.fold>`, the numeric addition would also be an equation. With {py:func}`fold <autoform.fold>`, it is computed at trace time.

(trace-time-decisions)=
## Trace-Time Decisions

Folded work can choose ordinary Python control flow because it runs while tracing:

```python
def route(text: str) -> str:
    with af.fold():
        label = af.string.match("priority", "priority")
    if label:
        return "yes: " + text
    return "no: " + text


ir = af.trace(route)("seed")
assert ir.call("answer") == "yes: answer"
```

The branch is chosen during tracing. The resulting IR contains only the path that was taken.

## Dynamic Value Limits

Folded work must not depend on dynamic traced values:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = text + " "
    return prefix
```

That raises during tracing because `text` is not concrete. Mark the dependency static or move the computation out of the {py:func}`fold <autoform.fold>` block.

Outside tracing, {py:func}`fold <autoform.fold>` is a no-op context manager.
