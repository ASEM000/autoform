# Rule Registration

Rule registration defines how a primitive executes, describes its outputs during tracing, and participates in transforms. The interfaces below cover batching, differentiation, and dead code elimination. Synchronous and asynchronous rules are registered separately so both execution paths can provide the required behavior.

```{eval-rst}
.. autofunction:: autoform.extend.register_impl
.. autofunction:: autoform.extend.register_aimpl
.. autofunction:: autoform.extend.register_abstract
.. autofunction:: autoform.extend.register_batch
.. autofunction:: autoform.extend.register_abatch
.. autofunction:: autoform.extend.register_pushforward
.. autofunction:: autoform.extend.register_apushforward
.. autofunction:: autoform.extend.register_pullback_fwd
.. autofunction:: autoform.extend.register_apullback_fwd
.. autofunction:: autoform.extend.register_pullback_bwd
.. autofunction:: autoform.extend.register_apullback_bwd
.. autofunction:: autoform.extend.register_dce
```
