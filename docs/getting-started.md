# Getting Started

This is the first path through `autoform`: install it, trace one ordinary Python function, run the resulting IR, then apply the core transforms and execution contexts.

## Install and Smoke Test

`autoform` requires Python 3.12 or newer.

Install from GitHub:

```bash
# install from github
pip install git+https://github.com/ASEM000/autoform.git
```

`autoform` uses [LiteLLM](https://docs.litellm.ai/) for provider calls. For OpenAI, set `OPENAI_API_KEY` and use an OpenAI model name:

```bash
# set the provider key
export OPENAI_API_KEY="..."
```

Any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) works if the provider credentials and model name are configured for that provider.[^litellm-model-names]

Run one direct LM call before starting with [tracing](concepts/trace-ir-execute.md):

```python
import autoform as af

# smoke test the provider before tracing
messages = [dict(role="user", content="Say hello in five words.")]
response = af.lm_call(messages, model="gpt-5.5")
print(response)
```

This is only a provider smoke test. It does not use tracing.

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: autoform` | Install into the same environment that runs Python. |
| Python version error | Use Python 3.12 or newer. |
| Provider authentication error | Set the provider key expected by LiteLLM, such as `OPENAI_API_KEY` for OpenAI. |
| Provider model error | Use a model name supported by the configured provider or route through [`litellm.Router`](https://docs.litellm.ai/docs/routing). |

## Trace a Function

Start with an ordinary Python function. This one formats a prompt and makes one LM call:

```python
import autoform as af


def explain(topic: str) -> str:
    # use traceable primitives for values that should enter the ir
    prompt = af.format("Explain {} in one paragraph.", topic)
    msg = dict(role="user", content=prompt)
    return af.lm_call([msg], model="gpt-5.5")
```

Trace it with an example argument:

```python
# trace with an example value; this does not call the provider
ir = af.trace(explain)("placeholder topic")
```

The string `"placeholder topic"` is a shape/type witness. It tells {py:func}`trace <autoform.trace>` that `topic` is a string. It is not sent to the model. See [Tracing Semantics](concepts/tracing-semantics.md) for the static/dynamic input rules.

Tracing runs the function once with placeholder values. Calls to [`autoform` primitives](concepts/primitives.md) are recorded as [IR equations](concepts/the-ir.md). The {py:func}`lm_call <autoform.lm_call>` is recorded, not executed.

The resulting IR contains:

- one runtime input, `topic`;
- one {py:func}`format <autoform.format>` equation that builds the prompt;
- one {py:func}`lm_call <autoform.lm_call>` equation that records a future provider call with role `user` and model `gpt-5.5`;
- one string output.

The result, `ir`, is the object every [transform](concepts/transforms.md) consumes.

## Run the IR

The [IR](concepts/the-ir.md) is the recipe. [Execution](concepts/trace-ir-execute.md) runs that recipe with real inputs.

Using the `explain` function from the previous step:

```python
# run the traced ir with a real input
output = ir.call("quantum entanglement")
print(output)
```

This does hit the active LM provider. The runtime input replaces the placeholder string used during tracing, and the recorded {py:func}`lm_call <autoform.lm_call>` equation executes.

Calling the IR again calls the provider again:

```python
# each call executes the recorded lm call again
first = ir.call("quantum entanglement")
second = ir.call("quantum entanglement")
```

`ir` is not a cached response. It is an executable program representation.

Every IR also has an [async execution](concepts/trace-ir-execute.md) method:

```python
import asyncio

# use asyncio.run in a normal script
output = asyncio.run(ir.acall("quantum entanglement"))
```

Use `.call(...)` from synchronous code. Use `.acall(...)` when the caller is already async, or when a transformed IR such as {py:func}`sched <autoform.sched>` can overlap independent work.

## Batch Inputs

The `explain` function has been traced once. The same IR can now run over several topics.

The direct Python version is a loop:

```python
# this is just a python loop around the ir
outputs = [ir.call(topic) for topic in ["DNA", "gravity", "recursion"]]
```

That works, but the loop is not itself an IR. {py:func}`batch <autoform.batch>` returns an IR that accepts batched inputs:

```python
import autoform as af


topics = ["DNA", "gravity", "recursion"]

# transform the ir so the input can be a list
batched_ir = af.batch(ir)
outputs = batched_ir.call(topics)

for topic, output in zip(topics, outputs):
    print(topic, "->", output)
```

`batched_ir` is a new IR, not a list. Calling it still executes the provider calls, and the result is a list of strings with the same length as `topics`.

The result is equivalent to the list comprehension, but the representation is different:

- the list comprehension returns a Python list of results;
- {py:func}`batch <autoform.batch>` returns a transformed IR that can be transformed again.

That second point is the reason to use {py:func}`batch <autoform.batch>` in `autoform`. The batched form can be composed with {py:func}`pullback <autoform.pullback>`, {py:func}`sched <autoform.sched>`, or other IR transforms.

By default, every [input leaf](concepts/pytrees.md) is batched. For functions with multiple inputs, `in_axes` controls which leaves are batched and which are broadcast:

```python
# batch the first input and broadcast the second
batched = af.batch(two_arg_ir, in_axes=(True, False))
```

That means: batch over the first positional input, reuse the second input for every item.

## Send Feedback Backward

{py:func}`pullback <autoform.pullback>` sends feedback backward through an IR.

The situation is:

- there is an output;
- there is a critique of that output;
- the goal is feedback for the inputs that contributed to it.

In autodiff terms, that critique is a [cotangent](reference/glossary.md). In `autoform`, cotangents are usually text.

Start from the same `ir`:

```python
# build an ir that returns the output and input feedback
pb_ir = af.pullback(ir)

inputs = ("quantum entanglement",)
output, grad = pb_ir.call(inputs, "too technical")

print(output)
print(grad)
```

The call shape is:

```text
original inputs + output feedback -> output + input feedback
```

For `explain(topic)`, the original input tree has one positional input, so the first argument is a one-item tuple: `("quantum entanglement",)`. The second argument is feedback on the output: `"too technical"`.

The result has the same structure:

- `output` is the model output from the forward run;
- `grad` is a one-item tuple containing text feedback for `topic`.

For example, `grad` might suggest narrowing the topic, asking for less jargon, or adding an audience constraint. The exact text depends on the active model, because the backward pass through {py:func}`lm_call <autoform.lm_call>` is itself an LM call.

Pullback becomes more useful as the program grows. If the IR has several LM calls, the output feedback flows backward through every recorded step. Each [primitive rule](concepts/primitives.md) decides how to translate feedback for its output into feedback for its inputs.

Cotangent shapes must match output shapes. If the function returns a tuple, pass tuple-shaped feedback. If it returns a [schema](concepts/schemas.md)-shaped object, pass feedback with the same [pytree](concepts/pytrees.md) shape.

## Compose {py:func}`batch <autoform.batch>` and {py:func}`pullback <autoform.pullback>`

The next task combines {py:func}`batch <autoform.batch>` and {py:func}`pullback <autoform.pullback>`: run prompt-feedback gradients across many inputs.

Without composition, the loop is manual:

```python
# manual version: run one pullback per input
pb_ir = af.pullback(ir)

outputs = []
hints = []

for topic, critique in zip(topics, critiques):
    output, (topic_hint,) = pb_ir.call((topic,), critique)
    outputs.append(output)
    hints.append(topic_hint)
```

That glue pairs every input with its critique, runs the backward pass, unpacks the one input cotangent, and keeps the result aligned with the original batch.

```python
# compose the transforms instead
composed = af.batch(af.pullback(ir))
```

*{py:func}`pullback <autoform.pullback>` returned an IR. {py:func}`batch <autoform.batch>` accepts an IR. So `batch(pullback(ir))` is just function composition, and no special combined mode was added for the composition.*

Run it:

```python
topics = ["DNA", "gravity", "recursion"]
critiques = ["too terse", "too abstract", "too long"]

# the original inputs are one positional tree: (topics,)
composed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = composed.call((topics,), critiques)

for topic, hint in zip(topics, topic_hints):
    print(topic, "->", hint)
```

The call shape follows from {py:func}`pullback <autoform.pullback>`:

- the original input tree is one positional input, so batched topics are passed as `(topics,)`;
- output feedback is batched as `critiques`;
- input feedback has the same structure as the original input tree, so it returns `(topic_hints,)`.

Both [transforms](concepts/transforms.md) are `IR -> IR`. {py:func}`pullback <autoform.pullback>` does not know {py:func}`batch <autoform.batch>` will be applied next. {py:func}`batch <autoform.batch>` does not know the IR came from {py:func}`pullback <autoform.pullback>`. The type does the work.

Order matters:

- `batch(pullback(ir))` means many independent pullback calls at once.
- `pullback(batch(ir))` means feedback for the batched function as a whole.

Every other IR transform composes the same way. `sched(batch(pullback(ir)))` is a real expression. So is `batch(sched(ir))`.

## Return Structured Results

Plain {py:func}`lm_call <autoform.lm_call>` returns text. That is fine for prose, but many LM programs need a value the rest of Python can use without another parsing step: a label, a score, a route decision, or a short extracted field.

Use {py:func}`lm_schema_call <autoform.lm_schema_call>` when the LM output should have a known shape.

```python
import optree
import autoform as af


# register the dataclass in autoform's pytree namespace
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    kind: str
    confidence: float
```

The `namespace=af.PYTREE_NAMESPACE` argument uses [Optree's dataclass integration](https://optree.readthedocs.io/en/latest/dataclasses.html) to register `Summary` as an `autoform` [pytree](concepts/pytrees.md). See {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>` for the namespace constant. The schema is an instance of that class:

```python
# build the schema directly as a value-shaped instance
summary_schema = Summary(
    title=af.Str(max=80),
    kind=af.Enum("definition", "analogy", "warning"),
    confidence=af.Float(min=0, max=1),
)
```

Now write the function normally:

```python
def summarize(topic: str) -> Summary:
    prompt = af.format("Summarize {} for a technical audience.", topic)
    msg = dict(role="user", content=prompt)
    # return a summary value, not a raw string
    return af.lm_schema_call([msg], model="gpt-5.5", schema=summary_schema)
```

Trace it:

```python
# trace once with a placeholder topic
ir = af.trace(summarize)("placeholder topic")
```

The schema is a static parameter of the recorded {py:func}`lm_schema_call <autoform.lm_schema_call>`. The returned fields are still part of the [IR](concepts/the-ir.md) output tree, so transforms can walk them like ordinary Python structure.

Execute it with a configured provider:

```python
# execute with a real topic and use the returned fields
result = ir.call("recursion")
print(result.title)
print(result.kind)
print(result.confidence)
```

The returned value is a `Summary`, not a string blob.

Schemas also work with transforms:

- {py:func}`batch <autoform.batch>` returns a batched `Summary` tree.
- {py:func}`pullback <autoform.pullback>` accepts feedback with the same schema shape.
- {py:func}`sched <autoform.sched>` can schedule schema calls like any other primitive.

Schemas are not tied to dataclasses. Any [pytree](concepts/pytrees.md) shape works, and the schema can be built inline for one-off calls or transformed with `optree.tree_map` before execution. See [Schemas](concepts/schemas.md) for those patterns.

For {py:func}`pullback <autoform.pullback>`, feedback lands on fields:

```python
# feedback has the same shape as the structured output
feedback = Summary(title="too vague", kind="classification is wrong", confidence="overconfident")
output, (topic_hint,) = af.pullback(ir).call(("recursion",), feedback)
```

The field feedback is summarized into a prompt-feedback request for the input message. For the full schema model, see [Schemas](concepts/schemas.md).

## Inspect Intermediates

Tracing produces an IR, but debugging often starts with a smaller question: what did the middle of the program produce?

Put a {py:func}`checkpoint <autoform.checkpoint>` at the value to inspect:

```python
import autoform as af


def explain_then_rewrite(topic: str) -> str:
    draft_prompt = af.format("Draft a one-sentence explanation of {}.", topic)
    draft_msg = dict(role="user", content=draft_prompt)
    step1 = af.lm_call([draft_msg], model="gpt-5.5")
    # mark the intermediate value for runtime inspection
    step1 = af.checkpoint(step1, key="step1", collection="debug")

    rewrite_prompt = af.format("Rewrite for a beginner: {}", step1)
    rewrite_msg = dict(role="user", content=rewrite_prompt)
    return af.lm_call([rewrite_msg], model="gpt-5.5")
```

Trace once:

```python
# trace once, then choose the execution context later
ir = af.trace(explain_then_rewrite)("placeholder topic")
```

{py:func}`checkpoint <autoform.checkpoint>` is transparent during ordinary execution. Without a context, this just runs both LM calls:

```python
# no collection context means checkpoint is transparent
result = ir.call("recursion")
```

Wrap execution with {py:func}`collect <autoform.collect>` to capture checkpointed values:

```python
# collect captured intermediates during execution
with af.collect(collection="debug") as captured:
    result = ir.call("recursion")

print(captured["step1"])
```

The captured value is a list because the same key can be reached more than once. In this function there is one `step1` value per run.

Use {py:func}`inject <autoform.inject>` to replace an intermediate and keep the rest of the IR unchanged:

```python
# replace step1 only for this execution
with af.inject(collection="debug", values={"step1": ["Recursion is a function calling itself."]}):
    result = ir.call("recursion")
```

This execution still enters the IR at the same input and still runs the downstream rewrite call. The checkpointed `step1` value is replaced before the downstream prompt is built.

That makes {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` useful for tight debugging loops:

- capture the intermediate value from a failing run;
- edit or replace that value;
- rerun the downstream part of the same IR;
- keep the original Python function intact.

{py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` are [runtime context managers](concepts/intercepts.md). They do not modify the IR object. The same `ir` can run normally, run under {py:func}`collect <autoform.collect>`, or run under {py:func}`inject <autoform.inject>` depending on the execution context around `ir.call(...)` or `ir.acall(...)`.

For more detail, see [Intercepts](concepts/intercepts.md).

## Schedule Independent Calls

Independent LM calls do not need to wait for each other. The Python function can stay ordinary and sequential:

```python
import asyncio
import time
import autoform as af


# write the function sequentially; scheduling happens after tracing
def compare(topic: str) -> str:
    explain_prompt = af.format("Explain {} in one sentence.", topic)
    example_prompt = af.format("Give one concrete example of {}.", topic)
    explain_msg = dict(role="user", content=explain_prompt)
    example_msg = dict(role="user", content=example_prompt)

    explanation = af.lm_call([explain_msg], model="gpt-5.5")
    example = af.lm_call([example_msg], model="gpt-5.5")

    combine_prompt = af.format("Combine these into a concise answer:\n{}\n{}", explanation, example)
    combine_msg = dict(role="user", content=combine_prompt)
    return af.lm_call([combine_msg], model="gpt-5.5")
```

There is no `async def` in `compare`. The first two calls read only `topic`, so they are independent. The final call depends on both results.

```{mermaid}
flowchart TD
    topic["topic"] --> explain["LM: explain"]
    topic --> example["LM: example"]
    explain --> combine["LM: combine"]
    example --> combine
    combine --> answer["answer"]
```

Trace the function:

```python
# trace once
ir = af.trace(compare)("placeholder topic")
```

Run the original IR synchronously:

```python
# measure the original ir
start = time.perf_counter()
sequential = ir.call("recursion")
sequential_s = time.perf_counter() - start
```

Schedule the IR with {py:func}`sched <autoform.sched>` and run it asynchronously:

```python
# schedule independent equations and run asynchronously
scheduled = af.sched(ir)

start = time.perf_counter()
parallel = asyncio.run(scheduled.acall("recursion"))
parallel_s = time.perf_counter() - start

print(f"sequential: {sequential_s:.2f}s")
print(f"scheduled:  {parallel_s:.2f}s")
```

The scheduled form groups independent [equations](concepts/the-ir.md) into `gather` steps. With `scheduled.acall(...)`, those groups use `asyncio.gather`, so the two first LM calls can overlap. The final LM call still waits for both inputs.

The measured speedup depends on provider latency, provider-side rate limits, and the active [LiteLLM client](recipes/litellm-config.md). The invariant is the dependency structure: independent equations can share a scheduling level; dependent equations cannot.

Compare the two pieces of code:

```python
# compare stays a normal function
def compare(topic: str) -> str: ...


scheduled = af.sched(ir)
parallel = asyncio.run(scheduled.acall("recursion"))
```

Execution mode is chosen at the call site: `.call(...)` for sync execution, `.acall(...)` for async execution. {py:func}`sched <autoform.sched>` changes the IR that executes, so the original function stays sequential.

{py:func}`sched <autoform.sched>` is another `IR -> IR` transform, so it composes with the earlier transforms:

```python
# transforms still compose after scheduling
fast_batch = af.sched(af.batch(ir))
fast_feedback = af.sched(af.batch(af.pullback(ir)))
```

Custom boundaries need matching async behavior when they should run under `acall`. When custom rules are added, define the async rule alongside the synchronous rule so scheduled async execution does the same work. See [Custom Rules](concepts/custom-rules.md).

## Next Steps

The core loop is now visible:

- write an LM program as ordinary Python;
- trace it into an IR;
- execute the IR with real inputs;
- transform the IR with {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>`, and {py:func}`sched <autoform.sched>`;
- inspect or replace intermediates with {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>`;
- return structured values with {py:func}`lm_schema_call <autoform.lm_schema_call>`.

[^litellm-model-names]: Use the model name expected by LiteLLM, not necessarily the provider's direct API model string. For example, an OpenRouter model uses an `openrouter/...` prefix when called through LiteLLM.

The main conceptual pages are useful when building larger programs:

| Need | Go to |
| --- | --- |
| Understand the trace/execution split | [Trace, IR, Execute](concepts/trace-ir-execute.md) |
| Understand the recorded operations | [Primitives](concepts/primitives.md) |
| Compose {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>`, and {py:func}`sched <autoform.sched>` | [Transforms](concepts/transforms.md) |
| Inspect or replace intermediate values | [Intercepts](concepts/intercepts.md) |
| Use structured LM outputs | [Schemas](concepts/schemas.md) |
| Build a Tool-Use Agent | [Build a Tool-Use Agent](recipes/tool-use-agent.md) |
| Read glossary terms quickly | [Glossary](reference/glossary.md) |

For exact call signatures, use the [API Reference](api/index.md).

For bugs, design questions, or examples that do not behave as expected, open a [GitHub issue](https://github.com/ASEM000/autoform/issues).
