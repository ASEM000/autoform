# Types

The package version and pytree namespace are public constants. Use the namespace for both Optree registration and traversal.

```{eval-rst}
.. py:data:: autoform.__version__
   :type: str

   The package version.

.. py:data:: autoform.PYTREE_NAMESPACE
   :type: str

   The Optree namespace reserved by ``autoform``.
```

Register a dataclass in the namespace, then use it for traversal:
Use `optree.dataclasses.field(pytree_node=False)` for fields that remain fixed, such as the model name:

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    topic: str
    model: str = optree.dataclasses.field(pytree_node=False)


state = State(topic="topic text", model="model-name")
upper = optree.tree_map(str.upper, state, namespace=af.PYTREE_NAMESPACE)
```

As a result, only the `topic` is changed to `"TOPIC TEXT"` while the `model` remains equal to `"model-name"`.

See [Pytrees](../concepts/pytrees.md) for the full pattern.
