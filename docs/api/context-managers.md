# Context Managers

Context managers select tracing or execution behavior within a `with` block and restore the previous context on exit. Some capture values or cache results; others select an interpreter or model client. The [Transforms](../concepts/transforms.md) page explains how these scoped policies differ from transformations that return a new IR.

```{eval-rst}
.. autofunction:: autoform.fold
.. autofunction:: autoform.tag
.. autofunction:: autoform.memoize
.. autofunction:: autoform.collect
.. autofunction:: autoform.inject
.. autofunction:: autoform.lm.client
```

## LM Clients

A custom model client supplies both synchronous and asynchronous Responses methods, allowing the same client context to support either execution mode:

```{eval-rst}
.. autoclass:: autoform.lm.Client
   :members:
.. autoclass:: autoform.lm.LiteLLMClient
   :members:
```
