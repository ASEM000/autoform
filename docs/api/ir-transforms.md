# IR Transforms

Each function below consumes an IR and returns another IR with different computation or execution behavior. Supported combinations depend on the operation rules and transform order. The [Transforms](../concepts/transforms.md) guide explains the call shapes, input axes, and composition patterns through examples.

```{eval-rst}
.. autofunction:: autoform.batch
.. autofunction:: autoform.pullback
.. autofunction:: autoform.pushforward
.. autofunction:: autoform.sched
.. autofunction:: autoform.dce
.. autofunction:: autoform.weight
```
