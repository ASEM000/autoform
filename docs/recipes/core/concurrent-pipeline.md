# Concurrent Execution

An explanation and an analogy can be generated independently from the same topic.
{py:func}`sched <autoform.sched>` groups independent calls for concurrent asynchronous execution.
The Python function stays sequential.

```{admonition} Concept
[Trace, IR, Execute](../../concepts/trace-ir-execute.md) · [Transforms](../../concepts/transforms.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Program

Generate the two pieces, then combine both results in a final call:

```python
import asyncio
import autoform as af

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
The final call waits for both inputs:

```{raw} html
:file: ../../assets/concurrent-calls.svg
```

## Execution

Use `.call(...)` for synchronous execution and `.acall(...)` for asynchronous execution.
In async code, use `await scheduled.acall(...)` instead of `asyncio.run(...)`.
Scheduling preserves dependencies; it does not make dependent calls independent.

The running time will depend on the latency and rate limits of the model provider. The above example is merely to illustrate how `autoform` can be used to overlap LLM calls. It is not intended to be a benchmark for speedup. Please refer to [Model Routing](../llm/litellm-config.md) to see how to configure the client.
