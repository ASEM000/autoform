# Model Routing

The same IR can run with different model clients.
{py:func}`client <autoform.lm.client>` selects the client during execution.
Use a [LiteLLM Router](https://docs.litellm.ai/docs/routing) to configure model aliases, retries, and fallback policy.

```{admonition} Concept
[Trace, IR, Execute](../../concepts/trace-ir-execute.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Router

Map the program's `docs-model` alias to a provider model:

```python
from litellm import Router
import autoform as af

model_list = [
    dict(model_name="docs-model", litellm_params=dict(model="model-name")),
]
router = Router(model_list=model_list, num_retries=2)


def explain(topic: str) -> str:
    content = dict(topic=topic, answer=af.lm.Str(desc="answer instructions"))
    return af.lm.fill(content, model="docs-model")["answer"]


ir = af.trace(explain)("recursion")
with af.lm.client(router):
    output = ir.call("recursion")
print(output)
```

Replace `"model-name"` with the provider route. Keep `"docs-model"` as the alias used by the program.
The router resolves the alias when the IR runs; ordinary tracing makes no model request.
The output is an explanation of recursion from the configured route.

## Client Interface

A client must provide `.responses(...)` and `.aresponses(...)` with LiteLLM's
[Responses request and response shapes](https://docs.litellm.ai/docs/response_api).
The default client forwards calls directly to LiteLLM.
A wrapper can add routing policy while preserving this interface.

This sets the client for the context of the block, but restores the previous client when the block is exited. This can be used to set the client for a single `.call(...)` or `.acall(...)` without changing the traced function or IR.
