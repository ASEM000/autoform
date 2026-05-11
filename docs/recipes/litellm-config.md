# Configure LiteLLM Routing

`autoform` uses the active LM client at execution time. By default that client
calls [LiteLLM](https://docs.litellm.ai/) directly. Use `lm_client` for a
configured [`litellm.Router`](https://docs.litellm.ai/docs/routing) with retries, aliases, or provider fallback.

```{admonition} Concept
[Trace, IR, Execute](../concepts/trace-ir-execute.md)
```

```python
from litellm import Router
import autoform as af


model_list = [dict(model_name="docs-model", litellm_params=dict(model="gpt-5.2"))]
router = Router(model_list=model_list, num_retries=2)


def explain(topic: str) -> str:
    prompt = af.format("Explain {} in one paragraph.", topic)
    msg = dict(role="user", content=prompt)
    # docs-model is resolved by the active router
    return af.lm_call([msg], model="docs-model")


ir = af.trace(explain)("recursion")

# credentials are still provider credentials, such as openai_api_key or env vars
with af.lm_client(router):
    print(ir.call("recursion"))
```

The context applies when the IR executes, not when it is traced. That means the
same IR can run with different routers:

```python
# run the same ir with a different execution context
with af.lm_client(router):
    answer = ir.call("memoization")

print(answer)
```

Keep provider-specific routing policy in [LiteLLM](https://docs.litellm.ai/docs/routing). Keep program structure in
`autoform`: [trace](../concepts/trace-ir-execute.md) the Python function, transform the [IR](../concepts/the-ir.md), and choose the LM
client around execution.
