# Registration

These registrations connect concrete Python types to tracing and identify primitives that need special handling. A non-DCE registration preserves a primitive even when its output is unused. A non-memoizable registration prevents result reuse for that primitive. Execution and transform rules have separate registration interfaces.

```{eval-rst}
.. autofunction:: autoform.extend.register_trace_type
.. autofunction:: autoform.extend.register_non_dce
.. autofunction:: autoform.extend.register_non_memoizable
```
