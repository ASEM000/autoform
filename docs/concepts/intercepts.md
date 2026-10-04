# Intercepts

Checkpoints are a way to capture or replace intermediate values in an [IR](the-ir.md) during a run.

The three public pieces are:

- {py:func}`checkpoint <autoform.checkpoint>`: mark a value.
- {py:func}`collect <autoform.collect>`: capture marked values during execution.
- {py:func}`inject <autoform.inject>`: substitute marked values during execution.

## {py:func}`checkpoint <autoform.checkpoint>`

{py:func}`checkpoint <autoform.checkpoint>` is transparent by default:

```python
import autoform as af

step = "draft"
step = af.checkpoint(step, key="step", collection="debug")
```

Without {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>`, it returns `step`. An active context captures or replaces the value at that checkpoint.

## Capture and Replacement

Here, first, the normalized text is captured, and then it is replaced with a new value for a subsequent run of the same IR.

```python
import autoform as af


def pipeline(text: str) -> str:
    normalized = "item: " + text
    normalized = af.checkpoint(normalized, key="normalized", collection="debug")
    return normalized + "!"


ir = af.trace(pipeline)("seed")

with af.collect(collection="debug") as captured:
    result = ir.call("alpha")

assert result == "item: alpha!"
assert captured["normalized"] == ["item: alpha"]

with af.inject(collection="debug", values={"normalized": ["cached item"]}):
    result = ir.call("alpha")

assert result == "cached item!"
```

The replacement changes what downstream equations receive. Earlier equations still run.

The values are stored as lists, because there could be multiple encounters of a given key. In the case of {py:func}`inject <autoform.inject>`, replacement values are consumed in encounter order.

## Trace-Time Printing

`print` inside the traced function runs while tracing. It sees placeholders or trace-time constants, not the concrete values from every later execution.

{py:func}`collect <autoform.collect>` runs around `ir.call(...)` or `ir.acall(...)`. It sees the runtime values produced by the IR.

## Runtime Contexts

{py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` do not produce new IRs. Both contexts wrap execution when used around an IR call:

```python
with af.collect(collection="debug") as captured:
    af.batch(ir).call(["alpha", "beta"])
```

Transformed IR execution is still execution, so checkpoints work after {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>`, or {py:func}`sched <autoform.sched>`.
