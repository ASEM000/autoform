# Intermediate Values

A poor final answer can come from an earlier outline or draft.
Add {py:func}`checkpoint <autoform.checkpoint>` at those values, then capture those values during execution or run with a replacement.

```{admonition} Concept
[Intercepts](../../concepts/intercepts.md) · [Trace, IR, Execute](../../concepts/trace-ir-execute.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Capture

Generate an outline and a draft, marking both for inspection:

```python
import autoform as af

model = "model-name"


def draft_answer(topic: str) -> str:
    content = dict(topic=topic, outline=af.lm.Str(desc="outline instructions"))
    outline = af.lm.fill(content, model=model)["outline"]
    outline = af.checkpoint(outline, key="outline", collection="debug")
    content = dict(outline=outline, draft=af.lm.Str(desc="draft instructions"))
    draft = af.lm.fill(content, model=model)["draft"]
    draft = af.checkpoint(draft, key="draft", collection="debug")
    content = dict(draft=draft, answer=af.lm.Str(desc="revision instructions"))
    return af.lm.fill(content, model=model)["answer"]


ir = af.trace(draft_answer)("topic text")
with af.collect(collection="debug") as captured:
    result = ir.call("topic text")

print(result)
print(captured["outline"])
print(captured["draft"])
```

The result is the final answer. Each captured key holds a list, with one entry for each time that checkpoint ran.

## Replacement

To test the final rewrite with a known draft, supply a replacement:

```python
replacements = {
    "draft": ["replacement draft"],
}
with af.inject(collection="debug", values=replacements):
    result = ir.call("topic text")
print(result)
```

The outline and draft calls still run. At the checkpoint, {py:func}`inject <autoform.inject>` substitutes the supplied draft, which the final model call receives.
Replacement lists are consumed in encounter order.

These contexts wrap the execution of the same IR, and can be removed to revert to normal execution.
