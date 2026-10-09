# Core

{py:func}`trace <autoform.trace>` records a Python function as an executable IR. Static inputs provide fixed configuration during tracing, while dynamic inputs become placeholders supplied at execution time. [Programs and IR](../concepts/programs-and-ir.md) introduces the workflow, and [Tracing](../concepts/tracing.md) describes its boundaries. The resulting IR can be transformed.

```{eval-rst}
.. autofunction:: autoform.trace
```
