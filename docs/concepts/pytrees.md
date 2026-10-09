# Pytrees

Programs often pass related values together in a dictionary, tuple, or object. A *pytree* describes such data as a structure of containers and leaves. This gives `autoform` a consistent way to find the values inside a container, batch selected fields, and return feedback in the original shape. A string is a leaf; a tuple or registered dataclass is a container.[^pytrees-and-schemas]

`autoform` uses [Optree's pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html) for traversal and registration.

## Registration

A custom dataclass becomes a pytree when its fields are registered in {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`.[^manual-registration] The review below contains a written comment and a numerical score as leaves. Its source label is fixed metadata, marked with `pytree_node=False`, because the label identifies the review rather than a value to transform.

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

Flattening separates the values from the structure that holds them. In this review, the comment and score become the leaves, while the dataclass type and source label remain in the tree structure. Unflattening combines that structure with the leaves to reconstruct the review:

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

A registered class can also group parameters with methods that use them. The `Explainer` below keeps instructions and style as transformable leaves, with the model name as fixed metadata. Its methods remain ordinary Python methods: tracing records the supported operations performed when a method is called.

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

Passing the module as an explicit program input exposes its leaves to transforms. Capturing the module in a closure instead fixes its field values during tracing. The following wrapper makes the module an input and traces only the `prompt` method, so the example requires no model request:

```python
def build_prompt(module: Explainer, topic: str) -> str:
    return module.prompt(topic)


module_ir = af.trace(build_prompt)(module, "topic text")
print(module_ir.call(module, "topic text"))
```

## Transform Behavior

Batching can share one module across many topics or pair each topic with a different set of parameter leaves. The first call below shares the module. The second provides a batch for each transformable field. The `model` metadata stays fixed in both cases:

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

[^pytrees-and-schemas]: Schemas describe structured LM output. Pytrees describe how `autoform` walks data. [Language Models](../language-models.md) covers schema output.

[^manual-registration]: Classes without Optree's dataclass decorator can supply flatten and unflatten functions directly. Transformable values belong among the children, and fixed, hashable values belong in the metadata. Registration and traversal use the same namespace.
