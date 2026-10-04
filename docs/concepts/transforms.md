# Transforms

An [IR](the-ir.md) transform can be thought of as a function that looks like this:

```{mermaid}
flowchart TD
    input_ir["IR"] --> transform["IR transform"]
    transform --> output_ir["IR"]
```

The result of the transform is itself an IR that can be executed. The result can also serve as input to another transform, allowing composition. This composition happens through regular Python function calls, but is constrained by the rules of the different operations.

The transforms have the following call shapes:

| Transform | Returned IR expects | Returned IR produces | Use when |
| --- | --- | --- | --- |
| {py:func}`batch <autoform.batch>` | Batched leaves where `in_axes=True`; broadcast leaves where `in_axes=False`. | Batched outputs. | Run the same program over many examples. |
| {py:func}`pushforward <autoform.pushforward>` | Original inputs plus input tangents. | Original output plus output tangent. | Push a proposed input change forward. |
| {py:func}`pullback <autoform.pullback>` | Original inputs plus feedback on the output. | Original output plus input feedback. | Turn output critique into prompt/input critique. |
| {py:func}`sched <autoform.sched>` | The same inputs as `ir`. | The same output as `ir`. | Overlap independent equations during async execution. |
| {py:func}`dce <autoform.dce>` | The same inputs as `ir`. | The selected output shape, with unused leaves removed or replaced. | Drop work that cannot affect the needed outputs. |
| {py:func}`weight <autoform.weight>` | The same inputs as `ir`. | `(output, path_weight)`. | Score one concrete path with reached `factor` calls. |


The code fragments use this one-input program:

```python
import autoform as af


def label(topic: str) -> str:
    return "Topic: " + topic


ir = af.trace(label)("recursion")
```

## IR Transforms

``````{tab-set}

`````{tab-item} batch

```text
batch(ir, /, *, in_axes=True) -> IR
```

{py:func}`batch <autoform.batch>` vectorizes an IR over one or more input leaves. `in_axes` is a bool [pytree](pytrees.md) matching the input structure: `True` means batched, `False` means broadcast.
Higher-order primitives can have separate batch rules too, including {py:func}`while_loop <autoform.while_loop>` and {py:func}`fixpoint <autoform.fixpoint>`.[^batched-while-loop]

Run one call for each topic:

```python
batched = af.batch(ir)
outputs = batched.call(["DNA", "gravity", "recursion"])
```

`````

`````{tab-item} pushforward

```text
pushforward(ir, /) -> IR
```

{py:func}`pushforward <autoform.pushforward>` builds a forward-mode-style IR. The transformed IR takes primals and tangents, then returns output primals and output tangents. For a string input:

```python
pf = af.pushforward(ir)
output, tangent = pf.call(("topic",), ("input change",))
```

`````

`````{tab-item} pullback

```text
pullback(ir, /) -> IR
```

{py:func}`pullback <autoform.pullback>` builds a reverse-mode-style IR. Cotangent types depend on the output types: string outputs receive text feedback, while floating-point outputs receive numerical cotangents. The transformed IR takes the original inputs plus an output cotangent, then returns the output and input cotangents. For a string output:

```python
pb = af.pullback(ir)
output, input_feedback = pb.call(("topic",), "output feedback")
```

`````

`````{tab-item} sched

```text
sched(ir, /, *, cond=None) -> IR
```

{py:func}`sched <autoform.sched>` groups independent equations into parallel stages. The resulting IR can
still run with `.call(...)`, but `.acall(...)` is where concurrent stages become
useful:

```python
import asyncio

scheduled = af.sched(ir)
result = asyncio.run(scheduled.acall("topic"))
```

`````

`````{tab-item} weight

```text
weight(ir, /) -> IR
```

{py:func}`weight <autoform.weight>` turns an IR into a path scorer. The returned IR runs one concrete path and returns the original output plus the product of reached {py:func}`factor <autoform.factor>` weights. Score a labeled path:

```python
def scored_label(topic: str, score: float) -> str:
    af.factor(score)
    return label(topic)


score_ir = af.trace(scored_label)("recursion", 1.0)
scored = af.weight(score_ir)
output, path_weight = scored.call("topic", 0.8)
```

`````

`````{tab-item} dce

```text
dce(ir, /, *, out_used=None) -> IR
```

{py:func}`dce <autoform.dce>` removes equations that do not contribute to the selected output leaves:

```python
trimmed = af.dce(ir)
result = trimmed.call("topic")
```

`````

``````

## Composition

One can compose the different transforms because each one returns an IR.

```python
topics = ["DNA", "gravity", "recursion"]
critiques = ["output feedback 1", "output feedback 2", "output feedback 3"]
transformed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = transformed.call((topics,), critiques)
```

{py:func}`pullback <autoform.pullback>` returns an IR, and {py:func}`batch <autoform.batch>` operates on that.

However, the order matters, because now the program that receives the feedback or that is scored along its path is different.

| Expression | Meaning |
| --- | --- |
| `batch(pullback(ir))` | Run many independent pullback calls at once. Each input pairs with its own output feedback. |
| `pullback(batch(ir))` | Treat the whole batched function as the program receiving feedback. The cotangent matches the batched output. |
| `batch(weight(ir))` | Score many candidate paths separately. The result contains one weight per candidate. |
| `weight(batch(ir))` | Score one batched path. Reached factors across the batched execution multiply into one weight. |

Trying to perform AD around the function `weight(ir)` with `pullback(weight(ir))` or `pushforward(weight(ir))` will result in an error. If this is the semantics that one wants, then the path scoring should be applied after the AD.

## Rules and Contexts

Other [public APIs](../api/index.md) work at different boundaries:

- {py:func}`custom <autoform.custom>` is a decorator on traceable user functions. It marks a function boundary and lets transforms consult custom rules at that boundary. See [Custom Rules](custom-rules.md).
- {py:func}`memoize <autoform.memoize>` is a context manager. It caches primitive results within a `with` block. See [Memoization](memoize.md).
- {py:func}`client <autoform.lm.client>` is a context manager. It changes provider routing during execution. See [Model Routing](../recipes/llm/litellm-config.md).
- {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` are context managers. These contexts capture or replace checkpointed values during execution. See [Intercepts](intercepts.md).
- {py:func}`tag <autoform.tag>` and {py:func}`fold <autoform.fold>` are context managers. These contexts alter trace-time annotation or trace-time evaluation. See [Tags](tags.md) and [Fold](fold.md).

Use a transform to produce another IR, a custom rule to change behavior at a function boundary, or a context to control behavior within a block.

## Execution Modes

The transformed IR supports synchronous and asynchronous execution:

```python
import asyncio

transformed = af.batch(af.pullback(ir))

sync_result = transformed.call((topics,), critiques)
async_result = asyncio.run(transformed.acall((topics,), critiques))
```

The original function was not written as `async def`. Async execution is chosen when running the transformed IR. See [Trace, IR, Execute](trace-ir-execute.md) for the execution split.

[^batched-while-loop]: The batched {py:func}`while_loop <autoform.while_loop>` implementation keeps an independent state for each batch item. Each iteration checks the condition for live items, runs the body only for items still active, and transposes between a batched pytree and per-item states internally. This lets different batch items exit on different iterations while the whole loop remains bounded by `max_iters`.

    The batched {py:func}`fixpoint <autoform.fixpoint>` implementation follows the same live-item pattern, but the liveness check is the fixed-point equivalence between the previous state and the newly produced state.
