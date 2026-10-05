# A First Program

This page follows a simple program that calls a language model to explain a topic. The examples run in order in one Python session, and assume familiarity with Python.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"explanation instructions"` with text for the task.
```

(getting-started-tracing)=
## Tracing

The function uses a string specification to describe the answer the model should generate. Tracing records the model call without sending a request:

```python
import autoform as af

model = "model-name"


def explain(topic: str) -> str:
    content = dict(
        topic=topic,
        answer=af.lm.Str(desc="explanation instructions"),
    )
    return af.lm.fill(content, model=model)["answer"]


ir = af.trace(explain)("topic text")
print(ir)
```

The IR can run for another topic. The model name and answer instructions remain fixed; see [Tracing](concepts/tracing.md) for details.

## Execution

Run the IR to generate an explanation for a different topic:

```python
output = ir.call("another topic text")
print(output)
```

Each run reuses the recorded equations but makes a new model request. Tracing does not cache model responses.

## Batching

The {py:func}`batch <autoform.batch>` transform runs the program for multiple topics:

```python
topics = ["topic text 1", "topic text 2", "topic text 3"]
batched = af.batch(ir)
outputs = batched.call(topics)
print(outputs)
```

The result contains one explanation per topic, in input order. `batch` returns an IR.

## Feedback

The {py:func}`pullback <autoform.pullback>` transform produces a program that propagates output feedback to the inputs:

```python
feedback_program = af.pullback(ir)
output, (topic_feedback,) = feedback_program.call(
    ("another topic text",),
    "answer feedback",
)
print(output)
print(topic_feedback)
```

The registered LM rule returns text feedback for `topic`, the only runtime input. The feedback suggests a change to that input; applying the change requires a separate update step.

## Composition

Batching the pullback computes input feedback for each topic from its corresponding answer critique:

```python
critiques = ["answer feedback 1", "answer feedback 2", "answer feedback 3"]
composed = af.batch(af.pullback(ir))
outputs, (topic_feedback,) = composed.call((topics,), critiques)
print(topic_feedback)
```

Both transforms take an IR and return an IR. This lets `batch` act on the result of `pullback`, and the composed program can be transformed again.

See [Language Models](language-models.md) for structured output and client configuration, [Concepts](concepts/index.md) for the framework, and [Recipes](recipes/index.md) for examples that combine several features.
