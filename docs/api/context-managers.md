# Context Managers

These context managers can be used to set up the tracing or execution within a `with` block. When exiting the context manager, the previous context is restored again. For an explanation of the difference between contexts and IR transforms see [Transforms](../concepts/transforms.md).

```{eval-rst}
.. autofunction:: autoform.fold
.. autofunction:: autoform.tag
.. autofunction:: autoform.memoize
.. autofunction:: autoform.collect
.. autofunction:: autoform.inject
.. autofunction:: autoform.lm.client
```

## LM Clients

A custom client supplies the synchronous and asynchronous Responses methods:

```{eval-rst}
.. autoclass:: autoform.lm.Client
   :members:
.. autoclass:: autoform.lm.LiteLLMClient
   :members:
```
