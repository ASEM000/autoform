# Abstract Helpers

Spaces map an abstract value to its representation for a particular role, such as primal values, tangents, or cotangents. An extension registers the mappings needed by its types. `materialize_zeros` converts symbolic zeros into the concrete zeros defined by the corresponding abstract values.

```{eval-rst}
.. autoclass:: autoform.extend.Space
    :members: map, set

.. autodata:: autoform.extend.primal_s
.. autodata:: autoform.extend.tangent_s
.. autodata:: autoform.extend.cotangent_s
.. autofunction:: autoform.extend.materialize_zeros
```
