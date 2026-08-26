# Transforms

An [IR](the-ir.md) transform is a function with this shape:

```{mermaid}
flowchart TD
    input_ir["IR"] --> transform["IR transform"]
    transform --> output_ir["IR"]
```

The output is another executable IR. That one property is what makes composition ordinary Python function composition.

| Transform | Returned IR expects | Returned IR produces | Use when |
| --- | --- | --- | --- |
| {py:func}`batch <autoform.batch>` | Batched leaves where `in_axes=True`; broadcast leaves where `in_axes=False`. | Batched outputs. | Run the same program over many examples. |
| {py:func}`pushforward <autoform.pushforward>` | Original inputs plus input tangents. | Original output plus output tangent. | Push a proposed input change forward. |
| {py:func}`pullback <autoform.pullback>` | Original inputs plus feedback on the output. | Original output plus input feedback. | Turn output critique into prompt/input critique. |
| {py:func}`sched <autoform.sched>` | The same inputs as `ir`. | The same output as `ir`. | Overlap independent equations during async execution. |
| {py:func}`dce <autoform.dce>` | The same inputs as `ir`. | The selected output shape, with unused leaves removed or replaced. | Drop work that cannot affect the needed outputs. |
| {py:func}`weight <autoform.weight>` | The same inputs as `ir`. | `(output, path_weight)`. | Score one concrete path with reached `factor` calls. |

## IR Transforms

``````{tab-set}

`````{tab-item} batch

```python
batch(ir, /, *, in_axes=True) -> IR
```

{py:func}`batch <autoform.batch>` vectorizes an IR over one or more input leaves. `in_axes` is a bool [pytree](pytrees.md) matching the input structure: `True` means batched, `False` means broadcast.
Higher-order primitives can have their own batch rules too, including {py:func}`while_loop <autoform.while_loop>` and {py:func}`fixpoint <autoform.fixpoint>`.[^batched-while-loop]

```python
batched = af.batch(ir)
outputs = batched.call(["DNA", "gravity", "recursion"])
```

`````

`````{tab-item} pushforward

```python
pushforward(ir, /) -> IR
```

{py:func}`pushforward <autoform.pushforward>` builds a forward-mode-style IR. The transformed IR takes primals and tangents, then returns output primals and output tangents.

```python
pf = af.pushforward(ir)
output, tangent = pf.call(("topic",), ("make it concrete",))
```

`````

`````{tab-item} pullback

```python
pullback(ir, /) -> IR
```

{py:func}`pullback <autoform.pullback>` builds a reverse-mode-style IR. In `autoform`, cotangents are text feedback. The transformed IR takes the original inputs plus an output cotangent, then returns the output and input cotangents.

```python
pb = af.pullback(ir)
output, input_feedback = pb.call(("topic",), "too abstract")
```

`````

`````{tab-item} sched

```python
sched(ir, /, *, cond=None) -> IR
```

{py:func}`sched <autoform.sched>` groups independent equations into parallel stages. The resulting IR can
still run with `.call(...)`, but `.acall(...)` is where concurrent stages become
useful.

```python
import asyncio

scheduled = af.sched(ir)
result = asyncio.run(scheduled.acall("topic"))
```

`````

`````{tab-item} weight

```python
weight(ir, /) -> IR
```

{py:func}`weight <autoform.weight>` turns an IR into a path scorer. The returned IR runs one concrete path and returns the original output plus the product of reached {py:func}`factor <autoform.factor>` weights.

```python
scored = af.weight(ir)
output, path_weight = scored.call("topic", 0.8)
```

`````

`````{tab-item} dce

```python
dce(ir, /, *, out_used=None) -> IR
```

{py:func}`dce <autoform.dce>` removes equations that do not contribute to the selected output leaves.

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

There is no special combined mode. {py:func}`pullback <autoform.pullback>` returns an IR; {py:func}`batch <autoform.batch>` consumes that IR.

Order still matters:

| Expression | Meaning |
| --- | --- |
| `batch(pullback(ir))` | Run many independent pullback calls at once. Each input pairs with its own output feedback. |
| `pullback(batch(ir))` | Treat the whole batched function as the program receiving feedback. The cotangent matches the batched output. |
| `batch(weight(ir))` | Score many candidate paths separately. The result contains one weight per candidate. |
| `weight(batch(ir))` | Score one batched path. Reached factors across the batched execution multiply into one weight. |

## Non-Transforms

Some nearby [public APIs](../api/index.md) are intentionally not `IR -> IR`:

- {py:func}`custom <autoform.custom>` is a decorator on traceable user functions. It marks a function boundary and lets transforms consult custom rules at that boundary. See [Custom Rules](custom-rules.md).
- {py:func}`memoize <autoform.memoize>` is a context manager. It caches primitive results within a `with` block. See [Cache Repeated Computations with `memoize`](../recipes/execution/memoize.md).
- {py:func}`lm_client <autoform.lm_client>` is a context manager. It changes provider routing during execution. See [Configure LiteLLM Routing](../recipes/llm/litellm-config.md).
- {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` are context managers. They capture or replace checkpointed values during execution. See [Intercepts](intercepts.md).
- {py:func}`tag <autoform.tag>` and {py:func}`fold <autoform.fold>` are context managers. They alter trace-time annotation or trace-time evaluation. See [Tags](tags.md) and [Fold](fold.md).

The IR transforms reshape the IR. {py:func}`custom <autoform.custom>` changes rule lookup at a boundary. Contexts wrap trace-time or execute-time behavior. They are complementary axes.

## Transform and Execution Axes

The transform axis and execution axis compose independently:

```python
import asyncio

transformed = af.batch(af.pullback(ir))

sync_result = transformed.call((topics,), critiques)
async_result = asyncio.run(transformed.acall((topics,), critiques))
```

The original function was not written as `async def`. Async execution is chosen when running the transformed IR. See [Trace, IR, Execute](trace-ir-execute.md) for the execution split.

[^batched-while-loop]: The batched {py:func}`while_loop <autoform.while_loop>` implementation keeps an independent state for each batch item. Each iteration checks the condition for live items, runs the body only for items still active, and transposes between a batched pytree and per-item states internally. This lets different batch items exit on different iterations while the whole loop remains bounded by `max_iters`.

    The batched {py:func}`fixpoint <autoform.fixpoint>` implementation follows the same live-item pattern, but the liveness check is the fixed-point equivalence between the previous state and the newly produced state.
