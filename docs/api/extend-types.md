# Types

These classes describe runtime values, primitive keys, equations, and interpreters. Extension authors use these classes to define behavior for new types and operations.

```{eval-rst}
.. autoclass:: autoform.extend.AVal
   :members: check, zero, accum
.. autoclass:: autoform.string.StrAVal
.. autoclass:: autoform.numeric.IntAVal
.. autoclass:: autoform.numeric.FloatAVal
.. autoclass:: autoform.numeric.BoolAVal
```

## Primitives

```{eval-rst}
.. autoclass:: autoform.extend.Rule
   :members: set, get
.. autoclass:: autoform.extend.Prim
.. autoclass:: autoform.extend.Zero
```

## Dunders

```{eval-rst}
.. autoclass:: autoform.extend.Dunder
   :members:
```

## IR

```{eval-rst}
.. autoclass:: autoform.extend.IR
.. autoclass:: autoform.extend.Eqn
.. autoclass:: autoform.extend.Var
```

## Interpreters

```{eval-rst}
.. autoclass:: autoform.extend.Interpreter
```
