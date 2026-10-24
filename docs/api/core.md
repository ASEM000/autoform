# Core

{py:func}`trace <autoform.trace>` records a Python function as an executable IR. Every input leaf becomes a placeholder supplied at execution time. Capture fixed configuration in the function's enclosing scope. [Programs and IR](../concepts/programs-and-ir.md) introduces the workflow, and [Tracing](../concepts/tracing.md) describes its boundaries. The resulting IR can be transformed.

```{eval-rst}
.. autofunction:: autoform.trace
```
