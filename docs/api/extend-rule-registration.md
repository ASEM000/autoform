# Rule Registration

Rule registration defines how a primitive executes, describes its outputs during tracing, and participates in transforms. The interfaces below cover batching, differentiation, and dead code elimination. Synchronous and asynchronous rules are registered separately so both execution paths can provide the required behavior.

Use `table.set(prim, rule)` to register a primitive rule. Each `Rule` table also supports `get` and scoped `override` contexts.

```{eval-rst}
.. autodata:: autoform.extend.r_impl
.. autodata:: autoform.extend.r_aimpl
.. autodata:: autoform.extend.r_abstract
.. autodata:: autoform.extend.r_batch
.. autodata:: autoform.extend.r_abatch
.. autodata:: autoform.extend.r_push
.. autodata:: autoform.extend.r_apush
.. autodata:: autoform.extend.r_pull_fwd
.. autodata:: autoform.extend.r_apull_fwd
.. autodata:: autoform.extend.r_pull_bwd
.. autodata:: autoform.extend.r_apull_bwd
.. autofunction:: autoform.extend.register_dce
```
