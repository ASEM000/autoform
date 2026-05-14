# Extend

## Types

```{eval-rst}
.. autoclass:: autoform.extend.AVal
.. autoclass:: autoform.extend.StrAVal
.. autoclass:: autoform.extend.IntAVal
.. autoclass:: autoform.extend.FloatAVal
.. autoclass:: autoform.extend.BoolAVal
.. autoclass:: autoform.extend.Prim
.. autoclass:: autoform.extend.Zero
```

## IR

```{eval-rst}
.. autoclass:: autoform.extend.IR
.. autoclass:: autoform.extend.IREqn
.. autoclass:: autoform.extend.IRVar
```

## Interpreters

```{eval-rst}
.. autoclass:: autoform.extend.Interpreter
.. autofunction:: autoform.extend.using_interpreter
.. autodata:: autoform.extend.active_interpreter
.. autodata:: autoform.extend.active_tags
```

## Registration

```{eval-rst}
.. autofunction:: autoform.extend.register_trace_type
.. autofunction:: autoform.extend.register_zero
.. autofunction:: autoform.extend.register_cotangent_accumulator
.. autofunction:: autoform.extend.register_add
.. autofunction:: autoform.extend.register_sub
.. autofunction:: autoform.extend.register_mul
.. autofunction:: autoform.extend.register_div
.. autofunction:: autoform.extend.register_matmul
.. autofunction:: autoform.extend.register_eq
```

## Helpers

```{eval-rst}
.. autofunction:: autoform.extend.avalof
.. autofunction:: autoform.extend.zeroof
.. autofunction:: autoform.extend.materialize
.. autofunction:: autoform.extend.is_zero
.. autofunction:: autoform.extend.batch_index
.. autofunction:: autoform.extend.batch_spec
.. autofunction:: autoform.extend.is_irvar
.. autofunction:: autoform.extend.ir_aval
```
