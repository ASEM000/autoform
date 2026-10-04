# Schemas

Specifications describe generated values and validate parsed results. The specifications belong to `autoform.lm` and can appear inside a pytree passed to {py:func}`fill <autoform.lm.fill>`.

```{eval-rst}
.. autoclass:: autoform.lm.Str
.. autoclass:: autoform.lm.Int
.. autoclass:: autoform.lm.Float
.. autoclass:: autoform.lm.Bool
.. autoclass:: autoform.lm.Enum
```

## Schema Utilities

Functions for turning specification trees into a JSON Schema, and for parsing the returned value.

```{eval-rst}
.. autofunction:: autoform.lm.describe
.. autofunction:: autoform.lm.parse
```
