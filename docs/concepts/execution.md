# Execution

IRs can be run synchronously or asynchronously, with execution contexts to capture or replace checkpointed intermediate values and reuse results. A custom runner can step through equations.

## Execution Modes

The execution mode is chosen at the call site: a function defined with `def` can be called with `.call(...)` or `.acall(...)` after tracing.

```python
import asyncio
import autoform as af


def label(topic: str) -> str:
    prompt = "Explain " + topic
    return "Prompt: " + prompt


ir = af.trace(label)("topic text")
sync_result = ir.call("another topic text")
async_result = asyncio.run(ir.acall("another topic text"))
assert sync_result == async_result == "Prompt: Explain another topic text"
```

IRs transformed with {py:func}`sched <autoform.sched>` group together independent equations to permit overlapping execution when run asynchronously. IRs can also be executed asynchronously without scheduling, and scheduled IRs can be executed synchronously.

### Concurrent Execution

A question may be answered by creating a summary and an analogy of the topic, then combining those in a final pass. The parts can be generated concurrently, then passed together to a final model call.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

```python
model = "model-name"


def explain(topic: str) -> str:
    content = dict(topic=topic, summary=af.lm.Str(desc="summary instructions"))
    summary = af.lm.fill(content, model=model)["summary"]
    content = dict(topic=topic, analogy=af.lm.Str(desc="analogy instructions"))
    analogy = af.lm.fill(content, model=model)["analogy"]
    content = dict(
        summary=summary,
        analogy=analogy,
        answer=af.lm.Str(desc="response instructions"),
    )
    return af.lm.fill(content, model=model)["answer"]


ir = af.trace(explain)("topic text")
scheduled = af.sched(ir)
output = asyncio.run(scheduled.acall("topic text"))
print(output)
```

The result is an answer about the supplied topic that uses both generated pieces.
The final call waits for both inputs:[^concurrent-runtime]

```{raw} html
:file: ../assets/concurrent-calls.svg
```

## Checkpoints

Checkpointing captures or replaces intermediate values in an [IR](programs-and-ir.md#the-ir) at runtime:

There are three public functions:

* {py:func}`checkpoint <autoform.checkpoint>` to mark a value for checkpointing,
* {py:func}`collect <autoform.collect>` to capture checkpointed values during execution, and
* {py:func}`inject <autoform.inject>` to replace checkpointed values during execution.

```{raw} html
:file: ../assets/checkpoint-flow.svg
```

Outside a {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>` context, a checkpoint returns its input.

### Capture and Replacement

If the final answer is unsatisfactory, it may be due to a poor outline or a poor draft. To debug, the parts can be captured, then the final rewrite can be executed again with a known-good replacement draft.

```python
import autoform as af

model = "model-name"


def draft_answer(topic: str) -> str:
    content = dict(topic=topic, outline=af.lm.Str(desc="outline instructions"))
    outline = af.lm.fill(content, model=model)["outline"]
    outline = af.checkpoint(outline, key="outline", collection="debug")
    content = dict(outline=outline, draft=af.lm.Str(desc="draft instructions"))
    draft = af.lm.fill(content, model=model)["draft"]
    draft = af.checkpoint(draft, key="draft", collection="debug")
    content = dict(draft=draft, answer=af.lm.Str(desc="revision instructions"))
    return af.lm.fill(content, model=model)["answer"]


ir = af.trace(draft_answer)("topic text")
with af.collect(collection="debug") as captured:
    result = ir.call("topic text")

print(result)
print(captured["outline"])
print(captured["draft"])
```

Values are stored in a list for each checkpoint key because a key may occur more than once. Inspect the captured values, then supply a replacement draft:

```python
replacements = {"draft": ["replacement draft"]}
with af.inject(collection="debug", values=replacements):
    result = ir.call("topic text")
print(result)
```

Replacing the value of a checkpoint changes the input to downstream equations, but earlier equations are still executed. The model and rewrite instructions are fixed, but because the model is stochastic, repeated executions may produce different results.

For {py:func}`inject <autoform.inject>`, the replacement values will be consumed in encounter order.

### Runtime Contexts

{py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` do not return new IRs. Both contexts are used as execution contexts wrapping a call to an IR:

```python
with af.collect(collection="debug") as captured:
    af.batch(ir).call(["topic text 1", "topic text 2"])
```

Checkpoint contexts also work on transformed IRs, for example after calling {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>` or {py:func}`sched <autoform.sched>`.

## Memoization

{py:func}`memoize <autoform.memoize>` creates a runtime context in which results from [primitives](primitives-and-rules.md) will be cached for the duration of the `with` block.

```{raw} html
:file: ../assets/memo-cache.svg
```

### Runtime Reuse

Here, repeated operations will be executed in the same cache context:

```python
import autoform as af


def program(text: str) -> str:
    left = "<" + text + ">"
    right = "<" + text + ">"
    return left + right


ir = af.trace(program)("seed")

# building right repeats the two concat operations used for left
with af.memoize():
    result = ir.call("alpha")

print(result)
```

Here, each addition leads to recording an equation calling {py:func}`concat <autoform.string.concat>`. The two equations for `right` use the same inputs as the two for `left`, so both read from the cache, and the output is `<alpha><alpha>`. The cache is discarded at the end of the context.

### Trace-Time Reuse

{py:func}`memoize <autoform.memoize>` can also be used during tracing, in which case repeated identical calls to a primitive will only record one equation:

```python
import autoform as af


def duplicated(text: str) -> tuple[str, str]:
    with af.memoize():
        first = text + "!"
        second = text + "!"
        return first, second


ir = af.trace(duplicated)("seed")
print(ir.call("alpha"))
```

The program outputs `("alpha!", "alpha!")`, but only one concatenation is recorded. Use {py:func}`memoize <autoform.memoize>` when repeated calls with identical inputs should reuse a result.[^memoized-checkpoints] Memoized model calls do not produce independent samples.

## Manual Execution

The `ir.walk(...)` method can be used to step through equations and input values with a custom runner. This step-by-step execution is performed under the hood when calling `ir.call(...)` or `await ir.acall(...)`, but it is exposed here for custom runtimes.

The generator performs three steps:

1. Calling `next(gen)` yields the first equation and input values.
2. Calling `gen.send(output_values)` takes the output values from the previous equation, and yields the next equation and input values.
3. Calling `gen.send(output_values)` with the last output values yields `None` and the output values of the program.

An example of a custom runner which inspects equation tags before dispatching is below:

```python
def run_and_record_tags(ir, *inputs):
    tagged_prims = []
    gen = ir.walk(*inputs)
    eqn, values = next(gen)

    while eqn is not None:
        if "draft" in eqn.tags:
            tagged_prims.append(eqn.prim.name)
        output = eqn.bind(values, **eqn.params)
        eqn, values = gen.send(output)

    return values, tagged_prims


output, tagged_prims = run_and_record_tags(ir, "topic text")
```

Here, `eqn.bind(...)` calls the synchronous implementation of the equation.[^async-runner] Note that results must match the expected type and container structure of the equation. See [Tags](programs-and-ir.md#tags) for how to tag equations during tracing.

## Custom Interpreters

Interpreters control primitive dispatch during `.call(...)` or `.acall(...)`. A [custom runner](#manual-execution) uses `ir.walk(...)` to access top-level equations; an interpreter can also control primitive dispatch inside those equations. This example records each primitive name, its active tags, and its output:

```{admonition} Advanced
:class: info

This section uses `autoform.extend`.
```

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import autoform as af
import autoform.extend as afe


def program(topic: str) -> str:
    with af.tag("draft"):
        draft = "draft for " + topic
    return draft + "."


ir = af.trace(program)("topic x")
```

Capture the previous interpreter before installing the new one, then delegate calls to that parent. Calling `prim.bind(...)` inside `interpret(...)` would dispatch to the active interpreter again and recurse. Recording after delegation captures the outputs; recording before delegation can inspect a call before it runs.

```python
@dataclass(frozen=True)
class CallRecord:
    prim_name: str
    tags: frozenset[Any]
    output: Any


class RecordingInterpreter(afe.Interpreter):
    def __init__(self):
        self.parent = afe.active_interpreter.get()
        self.records: list[CallRecord] = []

    def interpret(self, prim: afe.Prim, in_tree: Any, /, **params):
        output = self.parent.interpret(prim, in_tree, **params)
        self.records.append(
            CallRecord(
                prim_name=prim.name,
                tags=afe.active_tags.get(),
                output=output,
            )
        )
        return output

    async def ainterpret(self, prim: afe.Prim, in_tree: Any, /, **params):
        output = await self.parent.ainterpret(prim, in_tree, **params)
        self.records.append(
            CallRecord(
                prim_name=prim.name,
                tags=afe.active_tags.get(),
                output=output,
            )
        )
        return output
```

Each dispatch to `interpret(...)` (or `ainterpret(...)` for async calls) is provided the primitive being dispatched to, a pytree of inputs to that primitive, and any static parameters for the primitive. The function should return the output of the primitive. On exit, the context manager will restore the previous interpreter dispatch policy.

```python
@contextmanager
def record_calls():
    with afe.using_interpreter(RecordingInterpreter()) as interpreter:
        yield interpreter


with record_calls() as recorder:
    result = ir.call("topic y")

print(result)
print(recorder.records)
```

The result is `draft for topic y.` and a list of records for two concatenations. The `draft` tag is active during the first. Parent delegation preserves any earlier interpreter in the dispatch chain. Use [checkpoints](#checkpoints) to inspect named values and a [manual runner](#manual-execution) to pause or resume top-level execution.

[^concurrent-runtime]: To run within an existing async function, replace `asyncio.run(...)` with `await scheduled.acall("topic text")`. The actual latency of this program will depend on the model provider and its rate limits.

[^memoized-checkpoints]: Note that {py:func}`checkpoint <autoform.checkpoint>` is not memoized, since repeated checkpoints are likely intended for {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>`.

[^async-runner]: An asynchronous runner could instead call `await eqn.abind(...)`.
