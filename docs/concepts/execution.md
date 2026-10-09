# Execution

Once a program has been traced, the same IR can run under different execution policies. An ordinary call returns a result, while execution contexts and runners can inspect intermediate values, reuse earlier computations, or control individual steps.

The following example generates a summary and an analogy of a topic, then combines both into an answer. A checkpoint and a tag mark the summary. Each section uses this same program, so the effect of an execution policy can be compared against the same computation.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

```python
import asyncio
import autoform as af


def explain(topic: str) -> str:
    content = dict(topic=topic, summary=af.lm.Str(desc="summary instructions"))
    summary = af.lm.fill(content, model="model-name")["summary"]
    with af.tag("summary"):
        summary = af.checkpoint(summary, key="summary", collection="review")

    content = dict(topic=topic, analogy=af.lm.Str(desc="analogy instructions"))
    analogy = af.lm.fill(content, model="model-name")["analogy"]
    content = dict(
        summary=summary,
        analogy=analogy,
        answer=af.lm.Str(desc="answer instructions"),
    )
    return af.lm.fill(content, model="model-name")["answer"]


ir = af.trace(explain)("topic text")
```

## Execution Modes

Both `.call(...)` and `.acall(...)` execute the recorded equations. The first uses synchronous primitive rules; the second uses asynchronous rules. Each call below makes new model requests, so the answers may differ even though both calls supply the same topic to the same program.

```python
sync_result = ir.call("topic text")
async_result = asyncio.run(ir.acall("topic text"))
```

### Concurrent Execution

The summary and analogy can be computed independently. The {py:func}`sched <autoform.sched>` transform groups these calls so execution with `.acall(...)` can overlap them. The final model call waits for both results. The dependencies are preserved, but the independent calls no longer need to run one after the other.[^concurrent-runtime]

```{raw} html
:file: ../assets/concurrent-calls.svg
```

```python
scheduled = af.sched(ir)
result = asyncio.run(scheduled.acall("topic text"))
```

## Checkpoints

An unexpected answer may originate in the summary, the analogy, or the final model call. A {py:func}`checkpoint <autoform.checkpoint>` marks an intermediate value for inspection and returns it unchanged during ordinary execution. Here, {py:func}`collect <autoform.collect>` captures the summary so the input to the final call can be inspected. The rest of the program still runs normally.

```{raw} html
:file: ../assets/checkpoint-flow.svg
```

### Capture and Replacement

```python
with af.collect(collection="review") as captured:
    result = ir.call("topic text")

print(captured["summary"])
```

Each checkpoint key maps to a list because a checkpoint may be encountered more than once, for example inside a loop or batch. The `"summary"` list contains one value for this run.

A later run can substitute a chosen summary through {py:func}`inject <autoform.inject>`. Replacements are consumed in encounter order. The final model call receives the replacement summary and the analogy generated during that run.

```python
replacements = {"summary": ["replacement summary"]}
with af.inject(collection="review", values=replacements):
    result = ir.call("topic text")
```

(runtime-contexts)=
Replacement takes effect when execution reaches the checkpoint. The call that produces the summary still runs before its result is replaced. Both the analogy and final model call can also vary between executions, which matters when comparing answers.

The {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` contexts do not create a new IR. These contexts can also wrap execution of a transformed program, such as a batch of explanations:

```python
with af.collect(collection="review") as captured:
    af.batch(ir).call(["topic text 1", "topic text 2"])
```

## Memoization

Within a {py:func}`memoize <autoform.memoize>` context, matching primitive calls reuse recorded results. In this example, the second execution with the same topic reuses all three model responses without sending new requests. The cache lasts only for the enclosing block. The second answer is therefore a reused result, rather than an independent model sample.[^memoized-checkpoints]

```{raw} html
:file: ../assets/memo-cache.svg
```

(runtime-reuse)=
```python
with af.memoize():
    first = ir.call("topic text")
    second = ir.call("topic text")

assert first == second
```

## Manual Execution

Some execution policies need control over individual equations. The `ir.walk(...)` method returns a generator that yields each equation and its input values, then waits for the equation output. Sending that output back advances to the next equation. After the last equation, the generator yields `(None, result)`.

The following runner records primitive names at the summary tag while executing the same explanation program:

```python
def run_and_record_tags(ir, *inputs):
    tagged_prims = []
    gen = ir.walk(*inputs)
    eqn, values = next(gen)

    while eqn is not None:
        if "summary" in eqn.tags:
            tagged_prims.append(eqn.prim.name)
        output = eqn.bind(values, **eqn.params)
        eqn, values = gen.send(output)

    return values, tagged_prims


output, tagged_prims = run_and_record_tags(ir, "topic text")
assert tagged_prims == ["checkpoint"]
```

The summary tag marks the checkpoint in this program, so `tagged_prims` contains `"checkpoint"`. Each equation runs through `eqn.bind(...)`; an asynchronous runner can await `eqn.abind(...)` instead. [Human Review](../recipes/execution/human-review.md) extends this pattern with a runner that pauses at a tagged result and continues with a reviewed value.

## Custom Interpreters

An interpreter controls primitive dispatch, including calls made inside the equations exposed by a walk. This is useful for recording results across nested program calls or applying a common execution policy. The interpreter below delegates to the previously active interpreter, then records the primitive name, active tags, and output. Both synchronous and asynchronous execution follow this pattern.

```python
import autoform.extend as afe


class RecordingInterpreter(afe.Interpreter):
    def __init__(self):
        self.parent = afe.active_interpreter.get()
        self.records = []

    def interpret(self, prim, in_tree, /, **params):
        output = self.parent.interpret(prim, in_tree, **params)
        self.records.append((prim.name, afe.active_tags.get(), output))
        return output

    async def ainterpret(self, prim, in_tree, /, **params):
        output = await self.parent.ainterpret(prim, in_tree, **params)
        self.records.append((prim.name, afe.active_tags.get(), output))
        return output
```

Delegating to the parent preserves the dispatch behavior that was active before recording began. Calling `prim.bind(...)` from inside `interpret(...)` would instead dispatch back to the recording interpreter and recurse. The `using_interpreter` context temporarily installs the recorder and restores the previous interpreter when the block exits.

```python
with afe.using_interpreter(RecordingInterpreter()) as recorder:
    result = ir.call("topic text")

print(recorder.records)
```

[^concurrent-runtime]: Inside an existing async function, the call is `await scheduled.acall("topic text")` rather than `asyncio.run(...)`. Latency also depends on the provider and its rate limits.

(trace-time-reuse)=
[^memoized-checkpoints]: Memoization can also be active during tracing. Matching primitive calls then share a recorded equation, rather than a model response: ordinary tracing makes no model request. Primitives excluded from memoization remain active during both tracing and execution. Checkpoints are excluded so collection and replacement still observe each encounter.
