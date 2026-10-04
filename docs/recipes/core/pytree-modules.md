# Pytree Modules

Module-style code can package prompt parameters and methods without hiding the
parameters from transforms. Register the class as an Optree dataclass in the
`autoform` pytree namespace, then pass the module instance as an input to the
traced function.

```{admonition} Concept
[Pytrees](../../concepts/pytrees.md) · [Trace, IR, Execute](../../concepts/trace-ir-execute.md) · [Transforms](../../concepts/transforms.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Module

Keep the prompt fields as leaves and the model as static metadata:

```python
import optree
import autoform as af


MODEL = "model-name"


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Explainer:
    instruction: str
    style: str
    model: str = optree.dataclasses.field(pytree_node=False)

    def prompt(self, topic: str) -> str:
        return self.instruction + "\nStyle: " + self.style + "\nTopic: " + topic

    def __call__(self, topic: str) -> str:
        content = dict(
            prompt=self.prompt(topic),
            answer=af.lm.Str(desc="answer instructions"),
        )
        return af.lm.fill(content, model=self.model)["answer"]
```

The methods can call traceable primitives such as {py:func}`concat <autoform.string.concat>`
and {py:func}`fill <autoform.lm.fill>`. The fields remain visible as pytree
leaves because the class is registered under
{py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`.

`model` is static metadata because it uses
`optree.dataclasses.field(pytree_node=False)`. It travels with the module but
does not become a transform leaf. `instruction` and `style` remain the
transform-visible leaves.

## Tracing

Trace a function that accepts the module instance:

```python
def run(module: Explainer, topic: str) -> str:
    return module(topic)


module = Explainer(
    instruction="answer instructions",
    style="style instructions",
    model=MODEL,
)
ir = af.trace(run)(module, "topic text")

print(ir.call(module, "another topic text"))
```

Note that it’s important to pass the `module` as an input: if `run` instead closed over module, then the fields of module would get fixed at trace time, and transforms wouldn’t receive any inputs shaped like module.

## Batching

Because the module is a pytree, {py:func}`batch <autoform.batch>` can vary its
fields the same way it varies a tuple or dictionary:

```python
batched = af.batch(ir)
modules = Explainer(
    instruction=["answer instructions 1", "answer instructions 2"],
    style=["style instructions 1", "style instructions 2"],
    model=MODEL,
)
topics = ["topic text 1", "topic text 2"]

print(batched.call(modules, topics))
```

The transform-visible module fields are batched in this call. `model` remains
static metadata. To reuse one module across many topics, broadcast the module
and batch only the topic input:

```python
batched_topics = af.batch(ir, in_axes=(False, True))
topics = ["topic text 1", "topic text 2"]

print(batched_topics.call(module, topics))
```

## Feedback

Similarly, {py:func}`pullback <autoform.pullback>` will return feedback with the same shape as the inputs. Since the first input is an `Explainer`, the first feedback will also be an Explainer:

```python
pb_ir = af.pullback(ir)
feedback = "answer feedback"
output, (module_feedback, topic_feedback) = pb_ir.call(
    (module, "topic text"),
    feedback,
)

print(output)
print(module_feedback)
print(topic_feedback)
```

This feedback can be used with an update policy like:

```python
new_instruction = (
    module.instruction + "\nFeedback: " + module_feedback.instruction
)
next_module = Explainer(
    instruction=new_instruction,
    style=module.style,
    model=module.model,
)
```

This is still ordinary Python. The transform only supplies module-shaped
feedback; the update rule decides how to use it.

## Runtime Operations

The methods of the module should perform traceable `autoform` work. If the module needs to perform work at runtime in the Python environment (e.g. making an HTTP request, querying a database, interfacing with a retrieval system), then that work should be done in a primitive. See [Primitive Definitions](../extending/writing-primitives.md) for more information.
