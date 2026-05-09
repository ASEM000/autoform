# Trace Your First Function

Start with an ordinary Python function. This one formats a prompt and makes one LM call:

```python
import autoform as af


def explain(topic: str) -> str:
    # use traceable primitives for values that should enter the ir
    prompt = af.format("Explain {} in one paragraph.", topic)
    msg = dict(role="user", content=prompt)
    return af.lm_call([msg], model="gpt-5.2")
```

Trace it with an example argument:

```python
# trace with an example value; this does not call the provider
ir = af.trace(explain)("placeholder topic")
```

The string `"placeholder topic"` is a shape/type witness. It tells `trace` that `topic` is a string. It is not sent to the model. See [Tracing Semantics](../concepts/tracing-semantics.md) for the static/dynamic input rules.

Tracing runs the function once with placeholder values. Calls to [`autoform` primitives](../concepts/primitives.md) are recorded as [IR equations](../concepts/the-ir.md). The `lm_call` is recorded, not executed.

The resulting IR contains:

- one runtime input, `topic`;
- one `format` equation that builds the prompt;
- one `lm_call` equation that records a future provider call with role `user` and model `gpt-5.2`;
- one string output.

The result, `ir`, is the object every [transform](../concepts/transforms.md) consumes.
