# Types

These classes represent the values and program structure used by extension code: abstract values, primitive keys, equations, and interpreters. A domain extension combines these descriptions with registered rules to make new types and operations available to tracing and the transforms it supports.

```{eval-rst}
.. autoclass:: autoform.extend.AVal
   :members: check, zero, accum
.. autoclass:: autoform.extend.AValStr
.. autoclass:: autoform.extend.AValInt
.. autoclass:: autoform.extend.AValFloat
.. autoclass:: autoform.extend.AValBool
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
