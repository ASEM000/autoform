# Intercepts

Interceptors are the runtime side-channel for intermediate values. Use them to inspect or replace a value inside an [IR](the-ir.md) without rewriting the original function.

The three public pieces are:

- {py:func}`checkpoint <autoform.checkpoint>`: mark a value.
- {py:func}`collect <autoform.collect>`: capture marked values during execution.
- {py:func}`inject <autoform.inject>`: substitute marked values during execution.

## {py:func}`checkpoint <autoform.checkpoint>`

{py:func}`checkpoint <autoform.checkpoint>` is transparent by default:

```python
step = af.checkpoint(step, key="step", collection="debug")
```

Without {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>`, it returns `step`. With a context active, the same checkpoint becomes a hook.

## {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>`

```python
import autoform as af


def pipeline(text: str) -> str:
    normalized = af.format("item: {}", text)
    normalized = af.checkpoint(normalized, key="normalized", collection="debug")
    return af.concat(normalized, "!")


ir = af.trace(pipeline)("seed")

with af.collect(collection="debug") as captured:
    result = ir.call("alpha")

assert result == "item: alpha!"
assert captured["normalized"] == ["item: alpha"]

with af.inject(collection="debug", values={"normalized": ["cached item"]}):
    result = ir.call("alpha")

assert result == "cached item!"
```

Values are stored in lists because the same key may be encountered more than once. {py:func}`inject <autoform.inject>` consumes values in encounter order.

## Trace-Time Printing

`print` inside the traced function runs while tracing. It sees placeholders or trace-time constants, not the concrete values from every later execution.

{py:func}`collect <autoform.collect>` runs around `ir.call(...)` or `ir.acall(...)`. It sees the runtime values produced by the IR.

## Runtime Contexts

{py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` do not produce new IRs. They wrap execution:

```python
with af.collect(collection="debug") as captured:
    af.batch(ir).call(["alpha", "beta"])
```

Transformed IR execution is still execution, so checkpoints work after {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>`, or {py:func}`sched <autoform.sched>`.
