# IR Transforms

Functions which consume an IR and return a new IR. The rules which describe an operation will determine which combinations of transforms may be applied to an IR. See the [Transforms](../concepts/transforms.md) concept page for information about the shape and composition of transform calls.

```{eval-rst}
.. autofunction:: autoform.batch
.. autofunction:: autoform.pullback
.. autofunction:: autoform.pushforward
.. autofunction:: autoform.sched
.. autofunction:: autoform.dce
.. autofunction:: autoform.weight
```
