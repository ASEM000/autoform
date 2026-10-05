# Pytrees

A pytree is a nested structure of containers and leaves. A string is a leaf; a tuple or registered dataclass is a container. `autoform` uses this structure to batch selected fields and return feedback in the same container shape.[^pytrees-and-schemas]

`autoform` uses [Optree's pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html) for traversal and registration.

## Registration

Register a dataclass in {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>` so `autoform` can work with its fields.[^manual-registration] For example, a review can contain a written comment, a numerical score, and a source label. The comment and score are leaves. The source label is fixed metadata, marked with `pytree_node=False`:

```python
import optree

import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Review:
    comment: str
    score: float
    source: str = optree.dataclasses.field(pytree_node=False)


review = Review(comment="review text", score=2.0, source="source label")
```

Static metadata stays in the tree structure. It is not mapped or batched and does not receive feedback.

## Structure

Flattening separates the comment and score from the container structure and source label. Unflattening reconstructs the review from that structure and its leaves:

```python
leaves, structure = optree.tree_flatten(
    review,
    namespace=af.PYTREE_NAMESPACE,
)
assert leaves == ["review text", 2.0]
assert optree.tree_unflatten(structure, leaves) == review
```

```{raw} html
:file: ../assets/pytree-structure.svg
```

## Method-Bearing Pytrees

Once a class is registered, it can be used as a module with parameters and methods that use those parameters. The pytree leaves can be batched or receive feedback, while static metadata stays fixed.

```python
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


module = Explainer("answer instructions", "style instructions", "model-name")
```

When the module is an explicit input, transforms can act on its fields. If the module is captured in a closure, its field values are fixed during tracing. This example traces the `prompt` method and runs without a model call:

```python
def build_prompt(module: Explainer, topic: str) -> str:
    return module.prompt(topic)


module_ir = af.trace(build_prompt)(module, "topic text")
print(module_ir.call(module, "topic text"))
```

## Transform Behavior

The module can be broadcast to use the same parameter values for each topic, or its parameter leaves can be batched with the topics. Either way, the `model` metadata stays the same:

```python
topics = ["topic 1", "topic 2"]
shared_module = af.batch(module_ir, in_axes=(False, True))
print(shared_module.call(module, topics))

modules = Explainer(
    instruction=["instructions 1", "instructions 2"],
    style=["style instructions 1", "style instructions 2"],
    model="model-name",
)
print(af.batch(module_ir).call(modules, topics))
```

The feedback for the module is another `Explainer`, with feedback for the text fields and the same static metadata. The feedback and input have the same container structure:

```python
output, (module_feedback, topic_feedback) = af.pullback(module_ir).call(
    (module, "topic text"),
    "prompt feedback",
)
print(module_feedback.instruction, module_feedback.style)
assert module_feedback.model == module.model
```

Registered pytrees can carry state through [control flow](control-flow.md), as long as the input and output structures match. Registration exposes the fields to `autoform`. Methods that need concrete runtime values require a [primitive](primitives-and-rules.md#primitive-definitions).

## Leaves

The fields exposed as leaves must contain values that `autoform` can trace, or containers of such values. Open files, sockets, and closures do not become valid traced leaves through pytree registration. Registration also does not make an object serializable or safe to mutate.

[^pytrees-and-schemas]: Schemas describe structured LM output. Pytrees describe how `autoform` walks data; see [Language Models](../language-models.md) for schema output.

[^manual-registration]: For classes without Optree's dataclass decorator, register flatten and unflatten functions manually. Put transformable values among the children and fixed, hashable data in the metadata. Use the same namespace for registration and tree traversal.
