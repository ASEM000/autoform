# Language Models

A model call often combines known context with values that still need to be generated. `autoform.lm.fill` represents both in one structure. Ordinary values provide the context, and specifications mark the slots for generated text, numbers, or choices. The returned structure keeps the context and replaces each specification with a parsed value.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

```python
import autoform as af

content = dict(
    topic="topic text",
    answer=af.lm.Str(desc="answer instructions"),
    score=af.lm.Float(min=0, max=1, desc="scoring instructions"),
)
result = af.lm.fill(content, model="model-name")
print(result["answer"], result["score"])
assert result["topic"] == "topic text"
```

```{raw} html
:file: assets/schema-fill.svg
```

The model route must support the JSON Schema response format used by the [LiteLLM Responses API](https://docs.litellm.ai/docs/response_api).

## Specifications

`Str`, `Int`, `Float`, `Bool`, and `Enum` describe the values to generate. A specification can also provide instructions through `desc` and constraints such as numerical bounds. Parsing checks those constraints against the returned value; the description supplies generation guidance. The [specification reference](api/schemas.md) lists the available options.

## Structured Output

The same pattern extends to structured results. Specifications can appear inside dictionaries, tuples, lists, and [registered dataclasses](concepts/pytrees.md#pytrees). The container shape is fixed during tracing, while the model fills its specified leaves during execution. This example requests a title and score in a dataclass:

```python
import optree


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    score: float


schema = Summary(
    title=af.lm.Str(max=80, desc="title instructions"),
    score=af.lm.Float(min=0, max=1, desc="scoring instructions"),
)
content = dict(article="article text", summary=schema)
summary = af.lm.fill(content, model="model-name")["summary"]
print(summary.title, summary.score)
```

The returned `summary` is a `Summary` instance. [A First Program](a-first-program.md#feedback) shows how feedback passes through model calls.

## Model Clients

Model routing can be configured separately from the traced program. The {py:func}`client <autoform.lm.client>` context selects the client used during execution, and a [LiteLLM Router](https://docs.litellm.ai/docs/routing) can supply aliases, retries, and fallbacks. The context below applies that configuration to one call and restores the previous client on exit:

```python
from litellm import Router

models = [
    dict(model_name="docs-model", litellm_params=dict(model="model-name")),
]
router = Router(model_list=models, num_retries=2)
with af.lm.client(router):
    result = af.lm.fill(content, model="docs-model")
```

This context works for both synchronous and asynchronous execution.[^custom-clients]

[^custom-clients]: Custom clients provide `.responses(...)` and `.aresponses(...)` with the [LiteLLM Responses interface](https://docs.litellm.ai/docs/response_api). The [client reference](api/context-managers.md#lm-clients) defines this contract.
