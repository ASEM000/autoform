# Transforms

A transformation derives another program from an existing [IR](programs-and-ir.md#the-ir). Depending on the transform, the new program can process a batch, propagate feedback, score an execution, or change which operations run. The shared interface is a function from one IR to another:

```{raw} html
:file: ../assets/ir-transform.svg
```

Because the result is an IR, it can be executed or passed to another transform. Composition uses ordinary Python calls, while the registered operation rules determine which combinations are supported. Each transformation below changes a different aspect of the program.

The examples start from a short program that labels one topic. Its simple input and output make the changes introduced by each transform easier to see:

```python
import autoform as af


def label(topic: str) -> str:
    return "Topic: " + topic


ir = af.trace(label)("topic text")
```

## IR Transforms

``````{tab-set}
:sync-group: transform

`````{tab-item} batch
:name: batch
:sync: batch

```{raw} html
:file: ../assets/transform-batch.svg
```

```text
batch(ir, /, *, in_axes=True) -> IR
```

{py:func}`batch <autoform.batch>` vectorizes an IR over one or more input leaves. `in_axes` is a bool [pytree](pytrees.md#pytrees) matching the input structure: `True` means batched, `False` means broadcast. Higher-order primitives can have separate batch rules too, including {py:func}`while_loop <autoform.while_loop>` and {py:func}`fixpoint <autoform.fixpoint>`.[^batched-while-loop]

The result is a new IR that can be called with a collection of topics and produces the corresponding outputs.

```python
batched = af.batch(ir)
outputs = batched.call(["topic text 1", "topic text 2", "topic text 3"])
```

### Input Axes

Batching often varies the data while keeping configuration fixed. In the example below, the text varies across the batch and the prefix can either be shared or paired with each text. A `False` entry in `in_axes` marks the shared input:

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
:name: pushforward
:sync: pushforward

```{raw} html
:file: ../assets/transform-pushforward.svg
```

```text
pushforward(ir, /) -> IR
```

{py:func}`pushforward <autoform.pushforward>` propagates changes from inputs toward outputs. The transformed IR receives a pair of input trees: original values, called primals, and corresponding changes, called tangents. It returns the original output and its tangent. The registered spaces and rules determine what a change means for a string or another custom type:

```python
pf = af.pushforward(ir)
output, t = pf.call(("topic",), ("input change",))
```

`````

`````{tab-item} pullback
:name: pullback
:sync: pullback

```{raw} html
:file: ../assets/transform-pullback.svg
```

```text
pullback(ir, /) -> IR
```

{py:func}`pullback <autoform.pullback>` propagates output feedback back to the inputs. Its call receives the original input tree and an output cotangent, then returns the output and input cotangents. A string output receives text feedback under the built-in rules; a floating-point output receives a numerical cotangent. The example supplies a critique for the labeled topic:

```python
pb = af.pullback(ir)
output, input_feedback = pb.call(("topic",), "output feedback")
```

Feedback propagation and parameter updates are separate computations. A pullback produces feedback for the program inputs, while an update method decides how to revise those inputs. The following example updates an instruction and keeps the topic fixed.

### Feedback Boundaries

{py:func}`stop_gradient <autoform.stop_gradient>` blocks feedback to the topic. The forward value is unchanged, and its cotangent is a symbolic zero:

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

```python
model = "model-name"


def answer(instruction: str, topic: str) -> str:
    topic = af.stop_gradient(topic)
    content = dict(topic=topic, answer=af.lm.Str() @ instruction)
    return af.lm.fill(content, model=model)["answer"]
```

### Updating Inputs

An answer critique enters the pullback and produces feedback for the instruction. A second model call uses the current instruction and that feedback to propose a revision. This explicit update step allows a method to choose its own revision policy:

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

The fixed critique illustrates how an update loop is assembled. It does not establish that successive instructions improve the answer. Assessing improvement requires critiques or losses derived from evaluation examples and comparison on held-out inputs.

`````

`````{tab-item} sched
:name: sched
:sync: sched

```{raw} html
:file: ../assets/transform-sched.svg
```

```text
sched(ir, /, *, cond=None) -> IR
```

{py:func}`sched <autoform.sched>` groups equations that can run independently into concurrent stages. The resulting IR still supports `.call(...)`; asynchronous execution with `.acall(...)` allows those stages to overlap. Dependencies between operations are preserved:

```python
import asyncio

scheduled = af.sched(ir)
result = asyncio.run(scheduled.acall("topic"))
```

### Dependencies

A result may need to wait for another computation even when that computation does not supply its value. The {py:func}`depends <autoform.depends>` primitive records such a dependency. Here, the returned answer waits for the audit value:

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

{py:func}`depends <autoform.depends>` is a barrier on the returned result. It does not require the answer-producing computation to start after the audit-producing computation. A scheduler can run both independent producers concurrently and place the barrier after both have completed.

`````

`````{tab-item} weight
:name: weight
:sync: weight

```{raw} html
:file: ../assets/transform-weight.svg
```

```text
weight(ir, /) -> IR
```

A program can score an execution while still returning its ordinary output. A *path* is one concrete execution, including the branches and loop iterations reached for the supplied inputs. The {py:func}`weight <autoform.weight>` transform produces a program that returns `(output, path_weight)` for that path.

The following program returns a label and records two scoring contributions through {py:func}`factor <autoform.factor>`. Ordinary execution returns only the label. The weighted program also returns the product of the factors:

```python
def scored_label(topic: str, quality: float, relevance: float) -> str:
    af.factor(quality, name="quality")
    af.factor(relevance, name="relevance")
    return label(topic)


score_ir = af.trace(scored_label)("topic text", 1.0, 1.0)
scored = af.weight(score_ir)
output, path_weight = scored.call("topic text", 0.5, 0.25)
assert output == "Topic: topic text"
assert round(path_weight, 3) == 0.125
```

```{raw} html
:file: ../assets/path-weight-channels.svg
```

Candidates can be compared by scoring each execution separately. Applying {py:func}`batch <autoform.batch>` to the weighted program pairs the factor values by position and returns one output and one path weight per candidate. Caller code then decides how to select or rank the results.[^probability-reading]

```python
topics = ["topic text 1", "topic text 2"]
quality = [0.5, 0.75]
relevance = [0.25, 0.5]
outputs, path_weights = af.batch(scored).call(topics, quality, relevance)
assert [round(w, 3) for w in path_weights] == [0.125, 0.375]
```

Factors must be finite, non-negative numbers. The path weight starts at one, and each reached factor multiplies it. A zero factor makes the score zero but does not stop later computation. Transform order matters: `weight(batch(ir))` returns a single product across the batch, as shown under [Composition](#composition).

`````

`````{tab-item} dce
:name: dce
:sync: dce

```{raw} html
:file: ../assets/transform-dce.svg
```

```text
dce(ir, /, *, out_used=None) -> IR
```

### Selected Outputs

A program may compute several outputs when only some are needed. The {py:func}`dce <autoform.dce>` transform removes equations that cannot affect the selected outputs, subject to primitive retention rules. `out_used` identifies the output leaves that remain required:

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

## Composition

```{raw} html
:file: ../assets/pullback-batch.svg
```

Composition combines these behaviors without rewriting the original function. In `batch(pullback(ir))`, the inner transform constructs a feedback program and the outer transform runs it across paired topics and critiques:

```python
topics = ["topic text 1", "topic text 2", "topic text 3"]
critiques = ["output feedback 1", "output feedback 2", "output feedback 3"]
transformed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = transformed.call((topics,), critiques)
```

The IR returned by {py:func}`pullback <autoform.pullback>` is the program consumed by {py:func}`batch <autoform.batch>`. The batch therefore contains separate feedback computations.

Reversing the order can change which program receives feedback or contributes a path weight. The expressions below make that distinction explicit:

| Expression | Meaning |
| --- | --- |
| `batch(pullback(ir))` | Runs a separate pullback for each batch item, with its own output feedback. |
| `pullback(batch(ir))` | Takes the pullback of the whole batched program. Feedback has the same structure as the batched output. |
| `batch(weight(ir))` | Runs a weighted program with a separate path weight for each batch item. |
| `weight(batch(ir))` | Runs a batched program with one path weight: the product of all reached factors in the batch. |

`pullback(weight(ir))` and `pushforward(weight(ir))` are unsupported and raise an error. The supported order places path scoring outside the differentiation transform when the required operation rules are available.

## Rules and Contexts

Other [public APIs](../api/index.md) work at different boundaries:

- {py:func}`custom <autoform.custom>` is a decorator on traceable user functions. It marks a function boundary and lets transforms consult custom rules at that boundary. [Custom Rules](primitives-and-rules.md#custom-rules) describes these hooks.
- {py:func}`memoize <autoform.memoize>` is a context manager. It caches primitive results within a `with` block. [Memoization](execution.md#memoization) describes the scope of reuse.
- {py:func}`client <autoform.lm.client>` is a context manager. It changes provider routing during execution. [Model Clients](../language-models.md#model-clients) covers client configuration.
- {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` are context managers. These contexts capture or replace checkpointed values during execution. [Checkpoints](execution.md#checkpoints) describes both contexts.
- {py:func}`tag <autoform.tag>` and {py:func}`fold <autoform.fold>` are context managers. These contexts alter trace-time annotation or trace-time evaluation. [Tags](programs-and-ir.md#tags) and [Fold](tracing.md#fold) describe these behaviors.

These interfaces act at different levels: transforms produce new IRs, custom rules change behavior at a function boundary, and contexts select behavior within a block. Choosing the appropriate level keeps the program definition separate from its execution policy.

[^batched-while-loop]: The batched {py:func}`while_loop <autoform.while_loop>` implementation keeps an independent state for each batch item. Each iteration checks the condition for live items, runs the body only for items still active, and transposes between a batched pytree and per-item states internally. This lets different batch items exit on different iterations while the whole loop remains bounded by `max_iters`.

    The batched {py:func}`fixpoint <autoform.fixpoint>` implementation follows the same live-item pattern, but the liveness check is the fixed-point equivalence between the previous state and the newly produced state.

[^probability-reading]: **Probability Reading.** A likelihood interpretation depends on the application. For candidates enumerated once, multiplying prior mass by path weight and normalizing a positive total gives posterior masses if the weights are likelihoods. For samples drawn from the prior, sample frequencies already represent that prior; multiplying by it again would count it twice. Normalized heuristic ratings remain decision scores rather than calibrated probabilities.
