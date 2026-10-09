# Batch Helpers

Batch rules work with values and boolean axes that identify which leaves vary across a batch. These helpers index batched leaves, determine batch structure, and transpose between a pytree of batches and a batch of pytrees. Shared leaves remain available to each per-item computation.

```{eval-rst}
.. autofunction:: autoform.extend.batch_index
.. autofunction:: autoform.extend.batch_spec
.. autofunction:: autoform.extend.batch_transpose
```
