# Batch Helpers

Batch rules receive a native PyTreeSpec, boolean axes, and values that identify which leaves vary across a batch. These helpers separate values into per-item trees and collect results into the supplied container structure. Shared leaves remain available to each per-item computation.

```{eval-rst}
.. autofunction:: autoform.tree.unstack
.. autofunction:: autoform.tree.stack
```
