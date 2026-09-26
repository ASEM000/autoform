# Rule Registration

Register each rule with its corresponding function:

```python
import autoform.extend as afe

afe.register_impl(primitive, implementation)
afe.register_abstract(primitive, abstract_rule)
```

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
