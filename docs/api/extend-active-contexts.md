# Active Contexts

Extension code can inspect the active interpreter, equation tags, and model client through these context variables. The public context managers install temporary behavior and restore the previous setting on exit. These variables provide access to that active state rather than creating a new program.

```{eval-rst}
.. autodata:: autoform.extend.active_interpreter
.. autodata:: autoform.extend.active_tags
.. autodata:: autoform.lm.active_client
```
