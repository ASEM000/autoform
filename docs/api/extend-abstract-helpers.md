# Abstract Helpers

Spaces map a primal abstract value to its primal, tangent, or cotangent type. Register each mapping for a user-defined abstract value. `materialize_zeros` replaces symbolic zeros with the concrete zeros defined by those types.

```{eval-rst}
.. autoclass:: autoform.extend.Space
    :members: map, set

.. autodata:: autoform.extend.primal_s
.. autodata:: autoform.extend.tangent_s
.. autodata:: autoform.extend.cotangent_s
.. autofunction:: autoform.extend.materialize_zeros
```
