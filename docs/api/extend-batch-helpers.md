# Batch Helpers

Batch rules receive a container spec, values, and boolean axes that identify which leaves vary across a batch. These helpers separate values into per-item trees and collect results into the supplied container structure. Shared leaves remain available to each per-item computation.

```{eval-rst}
.. automethod:: autoform.src.axis.BatchSpec.unstack
.. automethod:: autoform.src.axis.BatchSpec.stack
```
