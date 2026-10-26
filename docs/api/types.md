# Types

The package exposes its version and `af.tree.NAMESPACE`, the OpTree namespace used for pytree registration. A shared namespace lets registration and traversal agree on which fields are leaves and which fields belong to fixed metadata.

```{eval-rst}
.. py:data:: autoform.__version__
   :type: str

   The package version.

.. py:data:: autoform.tree.NAMESPACE
   :type: str

   The Optree namespace reserved by ``autoform``.
```

In the dataclass below, `topic` is a leaf and `model` is fixed metadata. `af.tree.field(pytree_node=False)` excludes the model name from tree traversal, so mapping a string operation changes only the topic:

```python
import autoform as af


@af.tree.dataclass
class State:
    topic: str
    model: str = af.tree.field(pytree_node=False)


state = State(topic="topic text", model="model-name")
upper = af.tree.map(str.upper, state)
```

As a result, only the `topic` is changed to `"TOPIC TEXT"` while the `model` remains equal to `"model-name"`.

[Pytrees](../concepts/pytrees.md#pytrees) explains how registration affects tracing, batching, and feedback.
