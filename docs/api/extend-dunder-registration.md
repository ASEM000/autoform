# Dunder Registration

Dunder rules define how Python syntax behaves for a traced abstract value.
Arithmetic and indexing rules may stage primitives. Python coercion protocols
must return their required concrete type—for example, `Dunder.BOOL` must return
`bool` and `Dunder.LEN` must return `int`.

```{eval-rst}
.. autofunction:: autoform.extend.register_dunder
```
