# Custom Boundaries

{py:func}`custom <autoform.custom>` preserves a boundary around a traceable function so batching, pushforward, and pullback can use dedicated rules. Ordinary calls retain the function body. [Custom Rules](../concepts/primitives-and-rules.md#custom-rules) explains how these hooks receive inputs, call the original function, and return results.

```{eval-rst}
.. autofunction:: autoform.custom
```
