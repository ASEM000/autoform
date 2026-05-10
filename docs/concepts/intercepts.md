# Intercepts

Interceptors are the runtime side-channel for intermediate values. Use them when you want to inspect or replace a value inside an [IR](the-ir.md) without rewriting the original function.

The three public pieces are:

- `checkpoint(value, key=..., collection=...)`: mark a value.
- `collect(collection=...)`: capture marked values during execution.
- `inject(collection=..., values=...)`: substitute marked values during execution.

## The Primitive

`checkpoint` is transparent by default:

```python
step = af.checkpoint(step, key="step", collection="debug")
```

Without `collect` or `inject`, it returns `step`. With a context active, the same checkpoint becomes a hook.

## Capture And Inject

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

Values are stored in lists because the same key may be encountered more than once. `inject` consumes values in encounter order.

## Why Not Just Print?

`print` inside the traced function runs while tracing. It sees placeholders or trace-time constants, not the concrete values from every later execution.

`collect` runs around `ir.call(...)` or `ir.acall(...)`. It sees the runtime values produced by the IR.

## Runtime Context, Not Transform

`collect` and `inject` do not produce new IRs. They wrap execution:

```python
with af.collect(collection="debug") as captured:
    af.batch(ir).call(["alpha", "beta"])
```

Transformed IR execution is still execution, so checkpoints work after [`batch`,
`pullback`, or `sched`](transforms.md).
