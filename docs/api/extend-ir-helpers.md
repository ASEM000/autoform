# IR Helpers

These helpers support interpreter contexts and inspection of IR values. `serial_fanout` selects serial execution for fanout operations within its context. The interfaces are useful when an extension needs to control dispatch or examine recorded program structure without defining another public transform.

```{eval-rst}
.. autofunction:: autoform.extend.using_interpreter
.. autofunction:: autoform.order.serial_fanout
.. autofunction:: autoform.extend.is_var
.. autofunction:: autoform.extend.aval_if_var
```
