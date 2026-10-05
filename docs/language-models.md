# Language Models

`autoform.lm.fill` calls a language model with context values and specifications for the output. Specifications mark the values to generate, such as text or a bounded score. The result follows the input structure, with specifications replaced by parsed values.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
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

The model route must support the JSON Schema response format used by the
[LiteLLM Responses API](https://docs.litellm.ai/docs/response_api).

## Specifications

`Str`, `Int`, `Float`, `Bool`, and `Enum` specify the value types to generate. The `desc` field supplies generation instructions. Bounds and other constraints are checked when parsing the response. See the [specification reference](api/schemas.md) for all options.

## Structured Output

Specifications can be nested in dictionaries, tuples, lists, and [registered dataclasses](concepts/pytrees.md#pytrees). Container structure is fixed when the function is traced. For example, a dataclass can hold a generated title and score:

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

The returned `summary` is a `Summary` instance. For feedback through model calls, see [A First Program](a-first-program.md#feedback).

## Model Clients

The {py:func}`client <autoform.lm.client>` context selects the model client at execution time. A [LiteLLM Router](https://docs.litellm.ai/docs/routing) can configure model aliases, retries, and fallbacks. The context restores the previous client when it exits:

```python
from litellm import Router

models = [
    dict(model_name="docs-model", litellm_params=dict(model="model-name")),
]
router = Router(model_list=models, num_retries=2)
with af.lm.client(router):
    result = af.lm.fill(content, model="docs-model")
```

The context works with synchronous and asynchronous execution.[^custom-clients]

[^custom-clients]: Custom clients provide `.responses(...)` and `.aresponses(...)` with the [LiteLLM Responses interface](https://docs.litellm.ai/docs/response_api). See the [client reference](api/context-managers.md#lm-clients) for details.
