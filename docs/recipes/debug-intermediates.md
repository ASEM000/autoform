# Debug Intermediate Values

Use {py:func}`checkpoint <autoform.checkpoint>` for values that may need inspection
or replace while running the same IR.

```{admonition} Concept
[Intercepts](../concepts/intercepts.md) · [Trace, IR, Execute](../concepts/trace-ir-execute.md)
```

```python
import autoform as af


def draft_answer(topic: str) -> str:
    outline_msg = dict(role="user", content=af.format("Outline {}.", topic))
    outline = af.lm_call([outline_msg], model="gpt-5.5")
    outline = af.checkpoint(outline, key="outline", collection="debug")

    draft_prompt = af.format("Write a short answer from this outline:\n{}", outline)
    draft_msg = dict(role="user", content=draft_prompt)
    draft = af.lm_call([draft_msg], model="gpt-5.5")
    draft = af.checkpoint(draft, key="draft", collection="debug")

    final_prompt = af.format("Tighten this answer:\n{}", draft)
    final_msg = dict(role="user", content=final_prompt)
    return af.lm_call([final_msg], model="gpt-5.5")


ir = af.trace(draft_answer)("recursion")

# collect stores checkpointed values during execution
with af.collect(collection="debug") as collected:
    result = ir.call("recursion")

print(result)
print(collected["outline"])
print(collected["draft"])
```

To rerun from a known intermediate value, inject a replacement:

```python
# inject replaces the named checkpoint during execution
replacements = {"draft": "Recursion is a function calling itself to solve a smaller case."}

with af.inject(collection="debug", values=replacements):
    result = ir.call("recursion")

print(result)
```

{py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>` are runtime contexts. They do not change the original
function or the traced [IR](../concepts/the-ir.md). That makes them useful for inspecting a bad value,
trying a corrected intermediate, and then going back to normal execution.
