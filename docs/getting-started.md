# Getting Started

The examples start with a function that asks a language model to explain a topic.
The examples show how to run the function on different topics, batch its inputs, and compute feedback.
The examples assume familiarity with Python. Run the code blocks in order in the same Python session.

`autoform` has three main phases: tracing, transforming, and executing. First, trace a program defined as a Python function to get an intermediate representation (IR). Then, transform the IR to produce new IR. Finally, execute the IR.

```{raw} html
:file: assets/program-lifecycle.svg
```

```{admonition} Concept
[Trace, IR, Execute](concepts/trace-ir-execute.md) · [Transforms](concepts/transforms.md)
```

(getting-started-installation)=
## Installation

Install `autoform`, which requires Python 3.12+:

```bash
pip install git+https://github.com/ASEM000/autoform.git
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"explanation instructions"` with text for the task.
```

(getting-started-tracing)=
## Tracing

Consider a function that asks an LLM for a one-paragraph explanation of a topic. Trace it to get IR:

```python
import autoform as af

model = "model-name"


def explain(topic: str) -> str:
    content = dict(
        topic=topic,
        answer=af.lm.Str(desc="explanation instructions"),
    )
    return af.lm.fill(content, model=model)["answer"]


ir = af.trace(explain)("recursion")
print(ir)
```

The class {py:class}`Str <autoform.lm.Str>` defines a specification for how to generate a string. The function {py:func}`fill <autoform.lm.fill>` takes as context the topic of the explanation and replaces the `Str` with the generated string, returning the resulting filled in dictionary.

During tracing, {py:func}`fill <autoform.lm.fill>` becomes a recorded operation; it doesn't call the model.
The argument `"recursion"` tells {py:func}`trace <autoform.trace>` that the topic is a string.
Later runs can use another topic. The model and generation instructions stay fixed in this example.
The printed IR shows the recorded model call.
See [Tracing Semantics](concepts/tracing-semantics.md) for static inputs and Python control flow.

## Execution

Run the IR to generate an explanation for a different topic:

```python
output = ir.call("gravity")
print(output)
```

This call prints an explanation of gravity.
Each call reuses the recorded equations and makes a fresh request to the model.
The IR doesn't cache model responses.

## Batching

To run it on multiple topics, transform the IR with {py:func}`batch <autoform.batch>`:

```python
topics = ["DNA", "gravity", "recursion"]
batched = af.batch(ir)
outputs = batched.call(topics)
print(outputs)
```

The call returns a list of explanations in the same order as the topics.
{py:func}`batch <autoform.batch>` returns an IR, so other transforms can act on the result.

By default, {py:func}`batch <autoform.batch>` tries to batch all inputs. See [how to keep parts of inputs the same across a batch using the `in_axes` argument](recipes/core/batch-in-axes.md).

## Feedback

Transform the IR with {py:func}`pullback <autoform.pullback>` to compute input feedback for a given output feedback:

```python
feedback_program = af.pullback(ir)
output, (topic_feedback,) = feedback_program.call(
    ("gravity",),
    "answer feedback",
)
print(output)
print(topic_feedback)
```

The variable `topic_feedback` contains suggestions for how to change the value of the only runtime input to the program, the topic. To update the topic based on the feedback, add a separate step for that.

Each operation in the IR has a corresponding registered rule for computing input feedback. For {py:func}`fill <autoform.lm.fill>`, the rule makes a model call.

`autoform` uses the term cotangent for this feedback, as in reverse-mode differentiation.
Registered rules determine its type: text for string inputs, numerical gradients for numerical arithmetic.
The feedback supplied to the pullback must match the output's container structure.
See [Pytrees](concepts/pytrees.md) for supported containers.

## Composition

Batch the pullback to compute input feedback for different critiques of the output, for each topic:

```python
critiques = ["answer feedback 1", "answer feedback 2", "answer feedback 3"]
composed = af.batch(af.pullback(ir))
outputs, (topic_feedback,) = composed.call((topics,), critiques)
print(topic_feedback)
```

The printed result is a list of topic suggestions, one for each topic.
Both transforms take an IR and return an IR.
That shared interface lets {py:func}`batch <autoform.batch>` act on the result of {py:func}`pullback <autoform.pullback>`.
The composed program can be transformed again, for example with {py:func}`sched <autoform.sched>`.

In this case, `batch(pullback(ir))` performs one pullback for each example. However, `pullback(batch(ir))` would produce the pullback for the entire batched program. For the shapes expected by each transform's `.call`, see [Transforms](concepts/transforms.md).

## Structured Output

For a more complicated output type, use a dataclass and register it as a [pytree](concepts/pytrees.md):

```python
import optree


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    kind: str


def summarize(topic: str) -> Summary:
    content = dict(
        topic=topic,
        summary=Summary(
            title=af.lm.Str(max=80, desc="title instructions"),
            kind=af.lm.Enum("definition", "analogy", "warning"),
        ),
    )
    return af.lm.fill(content, model=model)["summary"]


summary_ir = af.trace(summarize)("recursion")
summary = summary_ir.call("recursion")
print(summary.title, summary.kind)
```

This produces an instance of `Summary` with the generated field values. Instead of a dataclass, a dict, list, or tuple can be used to construct nested output. For more on output types, see [Schemas](concepts/schemas.md).

## Checkpoints

To inspect or replace intermediate values, use {py:func}`checkpoint <autoform.checkpoint>`:

```python
def explain_then_rewrite(topic: str) -> str:
    draft = af.checkpoint(explain(topic), key="draft", collection="debug")
    content = dict(draft=draft, answer=af.lm.Str(desc="rewrite instructions"))
    return af.lm.fill(content, model=model)["answer"]


rewrite_ir = af.trace(explain_then_rewrite)("recursion")

with af.collect(collection="debug") as captured:
    output = rewrite_ir.call("recursion")
print(captured["draft"])

with af.inject(collection="debug", values={"draft": ["draft text"]}):
    output = rewrite_ir.call("recursion")
print(output)
```

The call within the {py:func}`collect <autoform.collect>` context manager captures the value passed to {py:func}`checkpoint <autoform.checkpoint>`, in this case the value of `draft`. The value in `captured` is actually a list, since it's possible for a checkpoint to be called multiple times.

The {py:func}`inject <autoform.inject>` context manager replaces the checkpointed value with the supplied text. The first model call is still made, but the rewrite is performed on the provided draft text. Note that both context managers are applied to the same IR. For more on checkpoint and the related intercepts, see [Intercepts](concepts/intercepts.md).

## Scheduling

To run multiple parts of a program concurrently, use {py:func}`sched <autoform.sched>`:

```python
def compare(topic: str) -> str:
    explanation = explain(topic)
    content = dict(topic=topic, example=af.lm.Str(desc="example instructions"))
    example = af.lm.fill(content, model=model)["example"]
    content = dict(
        explanation=explanation,
        example=example,
        answer=af.lm.Str(desc="response instructions"),
    )
    return af.lm.fill(content, model=model)["answer"]


import asyncio

compare_ir = af.trace(compare)("recursion")
scheduled = af.sched(compare_ir)
output = asyncio.run(scheduled.acall("recursion"))
print(output)
```

This makes the final model call after obtaining both the explanation and the example, and produces a short answer based on both.

The same Python function can be run either synchronously, with `ir.call(...)`, or asynchronously, with `ir.acall(...)` (or `await ir.acall(...)` within an async function). For more on scheduling and running parts of an LM pipeline concurrently, see [Run an LM Pipeline Concurrently](recipes/core/concurrent-pipeline.md).

## More

For more on the types, operations, feedback rules, and transforms, see the [concepts guide](concepts/index.md). For more examples of how these pieces can be combined, see the [recipes](recipes/index.md).

If looking for something specific, some useful pages include:

| Task | Page |
| --- | --- |
| Update prompts from feedback | [Prompt Optimization](recipes/llm/prompt-optimization.md) |
| Add custom types and operations | [Extending `autoform`](recipes/extending/index.md) |
| Inspect or replace intermediate values | [Intercepts](concepts/intercepts.md) |
| Find call signatures | [API Reference](api/index.md) |
