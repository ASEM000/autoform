# Prompt Optimization

Feedback on an answer can guide changes to the instruction that produced it.
{py:func}`pullback <autoform.pullback>` computes input feedback from an output critique.
An update step then uses that feedback to revise the instruction.

```{admonition} Concept
[Trace, IR, Execute](../../concepts/trace-ir-execute.md) · [Transforms](../../concepts/transforms.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Program and Update

To make the instruction an input at runtime, keep the topic fixed:

```python
import autoform as af

model = "model-name"
topic = "recursion"


def answer(instruction: str) -> str:
    content = dict(topic=topic, answer=af.lm.Str() @ instruction)
    return af.lm.fill(content, model=model)["answer"]


instruction = "answer instructions"
feedback_program = af.pullback(af.trace(answer)(instruction))
critique = "answer feedback"

for step in range(3):
    output, (feedback,) = feedback_program.call((instruction,), critique)
    print(step, output)
    content = dict(
        instruction=instruction,
        feedback=feedback,
        updated=af.lm.Str(desc="revision instructions"),
    )
    instruction = af.lm.fill(content, model=model)["updated"]

print(instruction)
```

The pullback returns an answer and feedback for `instruction`. The separate model call uses that feedback to produce the next instruction.
The topic is captured by the function, so the pullback does not return feedback for it.

The fixed critique demonstrates the update loop. It does not measure whether each revision improves the answer.
For evaluation, compute critiques or losses from reference examples and compare the revised instructions on held-out inputs.
See [Transforms](../../concepts/transforms.md) to batch feedback calls.
