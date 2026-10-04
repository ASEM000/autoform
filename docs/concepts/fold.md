# Fold

The {py:func}`fold <autoform.fold>` block changes how Primitives are handled within the block. Outside of the {py:func}`fold <autoform.fold>` block, when a Primitive is called within a {py:func}`trace <autoform.trace>` block, the Primitive is recorded as an equation in the returned IR. Within the {py:func}`fold <autoform.fold>` block, the Primitive is run and the resulting value is recorded as a literal in the surrounding IR.

Evaluate a fixed prefix while tracing:

```python
import autoform as af


increment = af.trace(lambda value: value + 1)(1.0)


def program(text: str) -> str:
    with af.fold():
        prefix = f"v{increment.call(1.0)}: "
    return prefix + text


ir = af.trace(program)("seed")

assert len(ir.eqns) == 1
assert ir.call("world") == "v2.0: world"
```

Without {py:func}`fold <autoform.fold>`, the trace would record the call to `increment` in the surrounding IR. With {py:func}`fold <autoform.fold>`, the call runs at trace time.

(trace-time-decisions)=
## Trace-Time Decisions

Folded work can choose ordinary Python control flow because it runs while tracing:

```python
def route(text: str) -> str:
    with af.fold():
        label = increment.call(1.0)
    if label == 2:
        return "yes: " + text
    return "no: " + text


ir = af.trace(route)("seed")
assert ir.call("answer") == "yes: answer"
```

Tracing chooses the branch. The resulting IR contains only the chosen path.

## Dynamic Value Limits

Folded work needs concrete values. This example fails because `text` is dynamic:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = text + " "
    return prefix
```

That raises during tracing because `text` is not concrete. Mark the dependency static or move the computation out of the {py:func}`fold <autoform.fold>` block.

Outside tracing, {py:func}`fold <autoform.fold>` is a no-op context manager.
