# Types

Public type-level constants.

```{eval-rst}
.. py:data:: autoform.PYTREE_NAMESPACE
   :type: str

   The Optree namespace reserved by ``autoform``.
```

Use it anywhere Optree needs the same tree rules as `autoform`: registration and traversal functions such as `tree_map`.

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    topic: str


state = State(topic="recursion")
upper = optree.tree_map(str.upper, state, namespace=af.PYTREE_NAMESPACE)
```

See [Pytrees](../concepts/pytrees.md) for the full pattern.
