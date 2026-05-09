# Transforms

An IR transform is a function with this shape:

```text
IR -> IR
```

The output is another executable IR. That one property is what makes composition ordinary Python function composition.

## The Five IR Transforms

**`batch(ir, /, *, in_axes=True) -> IR`**

`batch` vectorizes an IR over one or more input leaves. `in_axes` is a bool pytree matching the input structure: `True` means batched, `False` means broadcast.

```python
batched = af.batch(ir)
outputs = batched.call(["DNA", "gravity", "recursion"])
```

**`pushforward(ir, /) -> IR`**

`pushforward` builds a forward-mode-style IR. The transformed IR takes primals and tangents, then returns output primals and output tangents.

```python
pf = af.pushforward(ir)
output, tangent = pf.call(("topic",), ("make it concrete",))
```

**`pullback(ir, /) -> IR`**

`pullback` builds a reverse-mode-style IR. In `autoform`, cotangents are text feedback. The transformed IR takes the original inputs plus an output cotangent, then returns the output and input cotangents.

```python
pb = af.pullback(ir)
output, input_feedback = pb.call(("topic",), "too abstract")
```

**`sched(ir, /, *, cond=None) -> IR`**

`sched` groups independent equations into parallel stages. The resulting IR can still run with `.call(...)`, but `await .acall(...)` is where concurrent stages become useful.

```python
scheduled = af.sched(ir)
result = await scheduled.acall("topic")
```

**`dce(ir, /, *, out_used=None) -> IR`**

`dce` removes equations that do not contribute to the selected output leaves.

```python
trimmed = af.dce(ir)
result = trimmed.call("topic")
```

## Composition

Composition works because every transform returns an IR:

```python
transformed = af.batch(af.pullback(ir))
outputs, input_feedback = transformed.call((topics, critiques))
```

There is no special combined mode. `pullback(ir)` returns an IR; `batch(...)` consumes that IR.

Order still matters:

| Expression | Meaning |
| --- | --- |
| `batch(pullback(ir))` | Run many independent pullback calls at once. Each input pairs with its own output feedback. |
| `pullback(batch(ir))` | Treat the whole batched function as the differentiated program. The cotangent matches the batched output. |

## What Is Not A Transform

Some nearby public APIs are intentionally not `IR -> IR`:

- `custom` is a decorator on user functions. It marks a function boundary and lets transforms consult your custom rules at that boundary.
- `memoize` is a context manager. It caches primitive results within a `with` block.
- `lm_client` is a context manager. It changes provider routing during execution.
- `collect` and `inject` are context managers. They capture or replace checkpointed values during execution.
- `tag` and `fold` are context managers. They alter trace-time annotation or trace-time evaluation.

The IR transforms reshape the IR. `custom` changes rule lookup at a boundary. Contexts wrap trace-time or execute-time behavior. They are complementary axes.

## Axes

The transform axis and execution axis compose independently:

```python
transformed = af.batch(af.pullback(ir))

sync_result = transformed.call((topics, critiques))
async_result = await transformed.acall((topics, critiques))
```

You did not write the original function as `async def`. You chose async execution when running the transformed IR.
