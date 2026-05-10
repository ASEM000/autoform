# Transforms

An [IR](the-ir.md) transform is a function with this shape:

```{mermaid}
flowchart LR
    input[/"IR"/] --> transform["IR transform"] --> output[/"IR"/]
```

The output is another executable IR. That one property is what makes composition ordinary Python function composition.

| Transform | Returned IR expects | Returned IR produces | Use when |
| --- | --- | --- | --- |
| `batch(ir)` | Batched leaves where `in_axes=True`; broadcast leaves where `in_axes=False`. | Batched outputs. | Run the same program over many examples. |
| `pushforward(ir)` | Original inputs plus input tangents. | Original output plus output tangent. | Push a proposed input change forward. |
| `pullback(ir)` | Original inputs plus feedback on the output. | Original output plus input feedback. | Turn output critique into prompt/input critique. |
| `sched(ir)` | The same inputs as `ir`. | The same output as `ir`. | Overlap independent equations during async execution. |
| `dce(ir)` | The same inputs as `ir`. | The selected output shape, with unused leaves removed or replaced. | Drop work that cannot affect the needed outputs. |

## IR Transforms

``````{tab-set}

`````{tab-item} batch

```python
batch(ir, /, *, in_axes=True) -> IR
```

`batch` vectorizes an IR over one or more input leaves. `in_axes` is a bool [pytree](pytrees.md) matching the input structure: `True` means batched, `False` means broadcast.

```python
batched = af.batch(ir)
outputs = batched.call(["DNA", "gravity", "recursion"])
```

`````

`````{tab-item} pushforward

```python
pushforward(ir, /) -> IR
```

`pushforward` builds a forward-mode-style IR. The transformed IR takes primals and tangents, then returns output primals and output tangents.

```python
pf = af.pushforward(ir)
output, tangent = pf.call(("topic",), ("make it concrete",))
```

`````

`````{tab-item} pullback

```python
pullback(ir, /) -> IR
```

`pullback` builds a reverse-mode-style IR. In `autoform`, cotangents are text feedback. The transformed IR takes the original inputs plus an output cotangent, then returns the output and input cotangents.

```python
pb = af.pullback(ir)
output, input_feedback = pb.call(("topic",), "too abstract")
```

`````

`````{tab-item} sched

```python
sched(ir, /, *, cond=None) -> IR
```

`sched` groups independent equations into parallel stages. The resulting IR can
still run with `.call(...)`, but `.acall(...)` is where concurrent stages become
useful.

```python
import asyncio

scheduled = af.sched(ir)
result = asyncio.run(scheduled.acall("topic"))
```

`````

`````{tab-item} dce

```python
dce(ir, /, *, out_used=None) -> IR
```

`dce` removes equations that do not contribute to the selected output leaves.

```python
trimmed = af.dce(ir)
result = trimmed.call("topic")
```

`````

``````

## Composition

Composition works because every transform returns an IR:

```python
transformed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = transformed.call((topics,), critiques)
```

There is no special combined mode. `pullback(ir)` returns an IR; `batch(...)` consumes that IR.

Order still matters:

| Expression | Meaning |
| --- | --- |
| `batch(pullback(ir))` | Run many independent pullback calls at once. Each input pairs with its own output feedback. |
| `pullback(batch(ir))` | Treat the whole batched function as the program receiving feedback. The cotangent matches the batched output. |

## Non-Transforms

Some nearby public APIs are intentionally not `IR -> IR`:

- [`custom`](custom-rules.md) is a decorator on user functions. It marks a function boundary and lets transforms consult custom rules at that boundary.
- [`memoize`](../recipes/memoize.md) is a context manager. It caches primitive results within a `with` block.
- [`lm_client`](../recipes/litellm-config.md) is a context manager. It changes provider routing during execution.
- [`collect` and `inject`](intercepts.md) are context managers. They capture or replace checkpointed values during execution.
- [`tag`](tags.md) and [`fold`](fold.md) are context managers. They alter trace-time annotation or trace-time evaluation.

The IR transforms reshape the IR. `custom` changes rule lookup at a boundary. Contexts wrap trace-time or execute-time behavior. They are complementary axes.

## Transform and Execution Axes

The transform axis and execution axis compose independently:

```python
import asyncio

transformed = af.batch(af.pullback(ir))

sync_result = transformed.call((topics,), critiques)
async_result = asyncio.run(transformed.acall((topics,), critiques))
```

The original function was not written as `async def`. Async execution is chosen when running the transformed IR. See [Trace, IR, Execute](trace-ir-execute.md) for the execution split.
