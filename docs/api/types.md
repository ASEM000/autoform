# Types

Public type-level constants.

```{eval-rst}
.. py:data:: autoform.PYTREE_NAMESPACE
   :type: str

   The Optree namespace reserved by ``autoform``.
```

Use it when registering project dataclasses as pytrees:

```python
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    topic: str
```

See [Pytrees](../concepts/pytrees.md) for the full pattern.
