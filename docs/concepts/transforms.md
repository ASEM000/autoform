# Transforms

An [IR](the-ir.md) transform can be thought of as a function that looks like this:

```{raw} html
:file: ../assets/ir-transform.svg
```

The result of the transform is itself an IR that can be executed. The result can also serve as input to another transform, allowing composition. This composition happens through regular Python function calls, but is constrained by the rules of the different operations.

The code fragments use this one-input program:

```python
import autoform as af


def label(topic: str) -> str:
    return "Topic: " + topic


ir = af.trace(label)("topic text")
```

## IR Transforms

``````{tab-set}

`````{tab-item} batch

```{raw} html
:file: ../assets/transform-batch.svg
```

```text
batch(ir, /, *, in_axes=True) -> IR
```

{py:func}`batch <autoform.batch>` vectorizes an IR over one or more input leaves. `in_axes` is a bool [pytree](pytrees.md) matching the input structure: `True` means batched, `False` means broadcast.
Higher-order primitives can have separate batch rules too, including {py:func}`while_loop <autoform.while_loop>` and {py:func}`fixpoint <autoform.fixpoint>`.[^batched-while-loop]

Run one call for each topic:

```python
batched = af.batch(ir)
outputs = batched.call(["topic text 1", "topic text 2", "topic text 3"])
```

`````

`````{tab-item} pushforward

```{raw} html
:file: ../assets/transform-pushforward.svg
```

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

```{raw} html
:file: ../assets/transform-pullback.svg
```

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

```{raw} html
:file: ../assets/transform-sched.svg
```

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

```{raw} html
:file: ../assets/transform-weight.svg
```

```text
weight(ir, /) -> IR
```

{py:func}`weight <autoform.weight>` turns an IR into a path scorer. The returned IR runs one concrete path and returns the original output plus the product of reached {py:func}`factor <autoform.factor>` weights. Score a labeled path:

```python
def scored_label(topic: str, score: float) -> str:
    af.factor(score)
    return label(topic)


score_ir = af.trace(scored_label)("topic text", 1.0)
scored = af.weight(score_ir)
output, path_weight = scored.call("topic", 0.8)
```

`````

`````{tab-item} dce

```{raw} html
:file: ../assets/transform-dce.svg
```

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

```{raw} html
:file: ../assets/pullback-batch.svg
```

One can compose the different transforms because each one returns an IR.

```python
topics = ["topic text 1", "topic text 2", "topic text 3"]
critiques = ["output feedback 1", "output feedback 2", "output feedback 3"]
transformed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = transformed.call((topics,), critiques)
```

{py:func}`pullback <autoform.pullback>` returns an IR, and {py:func}`batch <autoform.batch>` operates on that.

However, the order matters, because now the program that receives the feedback or that is scored along its path is different.

| Expression | Meaning |
| --- | --- |
| `batch(pullback(ir))` | Runs a separate pullback for each batch item, with its own output feedback. |
| `pullback(batch(ir))` | Takes the pullback of the whole batched program. Feedback has the same structure as the batched output. |
| `batch(weight(ir))` | Runs a weighted program with a separate path weight for each batch item. |
| `weight(batch(ir))` | Runs a batched program with one path weight: the product of all reached factors in the batch. |

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
