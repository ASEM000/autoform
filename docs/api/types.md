# Types

Public type-level constants.

## `PYTREE_NAMESPACE`

```python
PYTREE_NAMESPACE: str = "OPTREE_AUTOFORM_NAMESPACE"
```

The [Optree](https://optree.readthedocs.io/) namespace reserved by `autoform`.
Use it when registering project dataclasses as pytrees:

```python
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    topic: str
```

See [Pytrees](../concepts/pytrees.md) for the full pattern.
