# A First Program

A program can be written once and then used for several related computations. This example starts with a language model that explains a topic. The same program is then run on several topics and transformed to return feedback on its input. The code blocks build on one another in a single Python session and assume familiarity with Python.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"explanation instructions"` stand for task-specific instructions.
```

(getting-started-tracing)=
## Tracing

The function describes what the model should generate with a string specification. The topic provides context, and the specification marks the answer to fill. Tracing records this model call as an equation in an intermediate representation (IR), without sending a request to the provider.

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

The recorded program accepts a different topic on each call. The model name and answer instructions belong to the function configuration and remain fixed in this IR. [Tracing](concepts/tracing.md) explains how this distinction determines which values can change after a program is traced.

## Execution

Execution supplies a concrete topic to the recorded program. The `.call(...)` method sends the model request and returns the generated explanation:

```python
output = ir.call("another topic text")
print(output)
```

Each run reuses the recorded equations but makes a new model request. Tracing does not cache model responses.

## Batching

A collection of topics can use the same explanation program. The {py:func}`batch <autoform.batch>` transform creates an IR that accepts the topics together and runs the original computation for each input:

```python
topics = ["topic text 1", "topic text 2", "topic text 3"]
batched = af.batch(ir)
outputs = batched.call(topics)
print(outputs)
```

The result contains one explanation per topic in input order. The batched program is another IR, so it can be called again or passed to a compatible transform.

## Feedback

An explanation may need to change after receiving a critique. The {py:func}`pullback <autoform.pullback>` transform builds a program that propagates that output feedback back to the original inputs. The call below supplies both a topic and feedback on the generated answer:

```python
feedback_program = af.pullback(ir)
output, (topic_feedback,) = feedback_program.call(
    ("another topic text",),
    "answer feedback",
)
print(output)
print(topic_feedback)
```

The registered LM rule returns text feedback for `topic`, the only runtime input. The model name and fixed instructions receive no input feedback. The returned text suggests a change to the topic; a separate update step would apply that suggestion.

## Composition

Feedback may be needed for several explanations at once. Batching the pullback pairs each topic with its corresponding answer critique and returns input feedback for every pair:

```python
critiques = ["answer feedback 1", "answer feedback 2", "answer feedback 3"]
composed = af.batch(af.pullback(ir))
outputs, (topic_feedback,) = composed.call((topics,), critiques)
print(topic_feedback)
```

Both transforms take an IR and return an IR. As a result, `batch` can act on the feedback program produced by `pullback`, without another implementation of the explanation function. Further composition depends on the registered rules for the operations in that program.

[Language Models](language-models.md) covers structured output and client configuration. [Concepts](concepts/index.md) explains the underlying interfaces, while [Recipes](recipes/index.md) combines those interfaces to solve larger problems.
