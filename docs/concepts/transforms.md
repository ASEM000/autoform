# Transforms

An [IR](programs-and-ir.md#the-ir) transform can be thought of as a function that looks like this:

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

{py:func}`batch <autoform.batch>` vectorizes an IR over one or more input leaves. `in_axes` is a bool [pytree](pytrees.md#pytrees) matching the input structure: `True` means batched, `False` means broadcast.
Higher-order primitives can have separate batch rules too, including {py:func}`while_loop <autoform.while_loop>` and {py:func}`fixpoint <autoform.fixpoint>`.[^batched-while-loop]

Run one call for each topic:

```python
batched = af.batch(ir)
outputs = batched.call(["topic text 1", "topic text 2", "topic text 3"])
```

### Input Axes

To share an input, include `False` in `in_axes`. Here, of two inputs, one varies but the other is shared across the batch:

```{raw} html
:file: ../assets/batch-input-axes.svg
```

```python
def prefix_text(text: str, prefix: str) -> str:
    return prefix + ": " + text


axes_ir = af.trace(prefix_text)("text", "prefix")
shared = af.batch(axes_ir, in_axes=(True, False))
outputs = shared.call(["text 1", "text 2"], "prefix")
assert outputs == ["prefix: text 1", "prefix: text 2"]

paired = af.batch(axes_ir, in_axes=(True, True))
outputs = paired.call(["text 1", "text 2"], ["prefix 1", "prefix 2"])
assert outputs == ["prefix 1: text 1", "prefix 2: text 2"]
```

Batched leaves must have the same length and are paired by position. `in_axes=True` batches all leaves. For nested inputs, `in_axes` must be a tree of booleans in the corresponding position of the arguments tuple.

```python
def render(request: dict[str, str]) -> str:
    return request["instruction"] + ": " + request["topic"]


request = dict(instruction="answer instructions", topic="topic text")
request_ir = af.trace(render)(request)
axes = (dict(instruction=False, topic=True),)
requests = dict(instruction="answer instructions", topic=["topic 1", "topic 2"])
outputs = af.batch(request_ir, in_axes=axes).call(requests)
assert outputs == [
    "answer instructions: topic 1",
    "answer instructions: topic 2",
]
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

See [Pullback](#pullback) for feedback boundaries and an input update loop.

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

### Dependencies

Use {py:func}`depends <autoform.depends>` to delay a result until another traced value is available:

```python
import autoform as af


def ordered(topic: str) -> str:
    audit = "audit " + topic
    answer = "answer " + topic
    # return answer through a barrier that also waits for audit
    return af.depends(answer, audit)


ordered_ir = af.trace(ordered)("topic text")
scheduled = af.sched(ordered_ir)
print(scheduled.call("topic text"))
```

Use {py:func}`depends <autoform.depends>` when a result should not become available until another traced
value has also been evaluated, even though the returned value does not consume it directly. It does not
force the computation that produces the returned value to start after the dependencies; a scheduler may still
run independent producers concurrently and place the `depends` barrier after those producers.

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

### Selected Outputs

{py:func}`dce <autoform.dce>` removes equations that cannot affect the required outputs. The `out_used` argument selects which output leaves to keep:

```python
def pair(text: str) -> tuple[str, str]:
    return "left: " + text, "right: " + text


pair_ir = af.trace(pair)("text")
left_only = af.dce(pair_ir, out_used=(True, False))
assert left_only.call("text") == ("left: text", None)
```

The output has the same shape, but with `None` for any output that was removed. Here, the concatenation in the right output was removed. Without `out_used`, all output leaves are kept. Primitives registered to survive dead code elimination, such as checkpoints and factors, will remain in the IR.

`````

``````

## Pullback

A pullback computes feedback for the inputs of a program. Applying that feedback requires a separate step. In this example, the instruction is updated while the topic stays fixed.

### Feedback Boundaries

{py:func}`stop_gradient <autoform.stop_gradient>` blocks feedback to the topic. The forward value is unchanged, and its cotangent is a symbolic zero:

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

```python
model = "model-name"


def answer(instruction: str, topic: str) -> str:
    topic = af.stop_gradient(topic)
    content = dict(topic=topic, answer=af.lm.Str() @ instruction)
    return af.lm.fill(content, model=model)["answer"]
```

### Updating Inputs

The answer critique passes through the pullback to produce instruction feedback. A separate model call uses that feedback to generate the next instruction:

```python
instruction = "answer instructions"
topic = "topic text"
feedback_program = af.pullback(af.trace(answer)(instruction, topic))
critique = "answer feedback"

for step in range(3):
    inputs = (instruction, topic)
    output, (feedback, _) = feedback_program.call(inputs, critique)
    print(step, output)
    content = dict(
        instruction=instruction,
        feedback=feedback,
        updated=af.lm.Str(desc="revision instructions"),
    )
    instruction = af.lm.fill(content, model=model)["updated"]

print(instruction)
```

The fixed critique demonstrates the update loop. It does not measure whether each revision improves the answer. For evaluation, compute critiques or losses from reference examples and compare the revised instructions on held-out inputs.

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

- {py:func}`custom <autoform.custom>` is a decorator on traceable user functions. It marks a function boundary and lets transforms consult custom rules at that boundary. See [Custom Rules](primitives-and-rules.md#custom-rules).
- {py:func}`memoize <autoform.memoize>` is a context manager. It caches primitive results within a `with` block. See [Memoization](execution.md#memoization).
- {py:func}`client <autoform.lm.client>` is a context manager. It changes provider routing during execution. See [Model Clients](../language-models.md#model-clients).
- {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` are context managers. These contexts capture or replace checkpointed values during execution. See [Checkpoints](execution.md#checkpoints).
- {py:func}`tag <autoform.tag>` and {py:func}`fold <autoform.fold>` are context managers. These contexts alter trace-time annotation or trace-time evaluation. See [Tags](programs-and-ir.md#tags) and [Fold](tracing.md#fold).

Use a transform to produce another IR, a custom rule to change behavior at a function boundary, or a context to control behavior within a block.

[^batched-while-loop]: The batched {py:func}`while_loop <autoform.while_loop>` implementation keeps an independent state for each batch item. Each iteration checks the condition for live items, runs the body only for items still active, and transposes between a batched pytree and per-item states internally. This lets different batch items exit on different iterations while the whole loop remains bounded by `max_iters`.

    The batched {py:func}`fixpoint <autoform.fixpoint>` implementation follows the same live-item pattern, but the liveness check is the fixed-point equivalence between the previous state and the newly produced state.
