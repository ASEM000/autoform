# Abstract Helpers

The method `AVal.join(other)` should either return an `AVal` describing abstract value covering
both operands, including a symbolic `Zero` carrying that abstract value.

The method `AVal.zero()` should construct the concrete value for the additive
identity of that abstract type, and `AVal.add(left, right)` should be able to add
two concrete contributions of that abstract type. These are used by AutoForm to
combine many contributions of the same abstract type with
`ft.reduce(aval.add, values, aval.zero())`.

Each space describes how each abstract value should be represented in that space,
e.g. for primals, tangents, cotangents, etc. When writing an extension, these are
the mappings that need to be registered in order to tell AutoForm how to work
with the new types. Transform rules build symbolic zeros with `Space.zeros_like(value)`,
which returns one `Zero` per leaf of `value` in that space, and materialize them into concrete
values with `autoform.extend.instantiate(tree)`.

```{eval-rst}
.. autoclass:: autoform.extend.Space
    :members: map, set, zeros_like

.. autodata:: autoform.extend.s_primal
.. autodata:: autoform.extend.s_tangent
.. autodata:: autoform.extend.s_cotangent
```
