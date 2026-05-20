# Use Object-Oriented Modules

Module-style code can package prompt parameters and methods without hiding the
parameters from transforms. Register the class as an Optree dataclass in the
`autoform` pytree namespace, then pass the module instance as an input to the
traced function.

This recipe uses live provider calls. It requires `OPENAI_API_KEY`. Change the
`MODEL` constant to use a different LiteLLM model.

```{admonition} Concept
[Pytrees](../../concepts/pytrees.md) · [Trace, IR, Execute](../../concepts/trace-ir-execute.md) · [Transforms](../../concepts/transforms.md)
```

## Define the Module

```python
import optree
import autoform as af


MODEL = "gpt-5.5"


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Explainer:
    instruction: str
    style: str
    model: str = optree.dataclasses.field(pytree_node=False)

    def prompt(self, topic: str) -> str:
        return af.format("{}\nstyle: {}\ntopic: {}", self.instruction, self.style, topic)

    def __call__(self, topic: str) -> str:
        msg = dict(role="user", content=self.prompt(topic))
        return af.lm_call([msg], model=self.model)
```

The methods can call traceable primitives such as {py:func}`format <autoform.format>`
and {py:func}`lm_call <autoform.lm_call>`. The fields remain visible as pytree
leaves because the class is registered under
{py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`.

`model` is static metadata because it uses
`optree.dataclasses.field(pytree_node=False)`. It travels with the module but
does not become a transform leaf. `instruction` and `style` remain the
transform-visible leaves.

## Trace the Module Call

```python
def run(module: Explainer, topic: str) -> str:
    return module(topic)


module = Explainer(instruction="Explain in one paragraph.", style="plain", model=MODEL)
ir = af.trace(run)(module, "recursion")

print(ir.call(module, "memoization"))
```

The module is an explicit input. That is the important part. If `module` were
closed over by `run`, its fields would be fixed at trace time and transforms
would not receive module-shaped inputs.

## Batch Module Configurations

Because the module is a pytree, {py:func}`batch <autoform.batch>` can vary its
fields the same way it varies a tuple or dictionary.

```python
batched = af.batch(ir)
modules = Explainer(
    instruction=["Explain briefly.", "Give one concrete example."],
    style=["plain", "technical"],
    model=MODEL,
)
topics = ["recursion", "memoization"]

print(batched.call(modules, topics))
```

The transform-visible module fields are batched in this call. `model` remains
static metadata. To reuse one module across many topics, broadcast the module
and batch only the topic input:

```python
batched_topics = af.batch(ir, in_axes=(False, True))
topics = ["recursion", "memoization"]

print(batched_topics.call(module, topics))
```

## Receive Module-Shaped Feedback

{py:func}`pullback <autoform.pullback>` returns feedback with the same input
shape. Since the first input is an `Explainer`, the first feedback value is also
an `Explainer`.

```python
pb_ir = af.pullback(ir)
feedback = "too abstract; ask for an example"
output, (module_feedback, topic_feedback) = pb_ir.call((module, "recursion"), feedback)

print(output)
print(module_feedback)
print(topic_feedback)
```

The feedback can drive an update policy:

```python
new_instruction = module.instruction + "\nrevision guidance: " + module_feedback.instruction
next_module = Explainer(instruction=new_instruction, style=module.style, model=module.model)
```

This is still ordinary Python. The transform only supplies module-shaped
feedback; the update rule decides how to use it.

## Keep Runtime Boundaries Separate

Module methods should contain traceable `autoform` work when they are meant to
be transformed. External runtime work that needs concrete Python values, such as
HTTP calls, databases, or retrieval systems, belongs behind a primitive. See
[Write a Primitive](../extending/writing-primitives.md) for that boundary.
