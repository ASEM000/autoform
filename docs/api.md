# API Reference

## Core

Tracing and execution.

```{eval-rst}
.. autofunction:: autoform.trace
.. autofunction:: autoform.fold
.. autofunction:: autoform.tag
.. autoclass:: autoform.Tag
```

## Transforms

IR-to-IR transformations.

```{eval-rst}
.. autofunction:: autoform.batch
.. autofunction:: autoform.pullback
.. autofunction:: autoform.pushforward
.. autofunction:: autoform.custom
.. autofunction:: autoform.sched
.. autofunction:: autoform.dce
.. autofunction:: autoform.memoize
```

## Checkpointing

Capture and replay intermediate values.

```{eval-rst}
.. autofunction:: autoform.checkpoint
.. autofunction:: autoform.collect
.. autofunction:: autoform.inject
```

## LM Primitives

Language model calls.

```{eval-rst}
.. autofunction:: autoform.lm_call
.. autofunction:: autoform.lm_schema_call
.. autofunction:: autoform.lm_struct_call
.. autofunction:: autoform.lm_client
```

## Schemas

Schema nodes for structured LM output.

```{eval-rst}
.. autoclass:: autoform.Str
.. autoclass:: autoform.Int
.. autoclass:: autoform.Float
.. autoclass:: autoform.Bool
.. autoclass:: autoform.Enum
.. autoclass:: autoform.Doc
```

## String Primitives

String operations with autodiff support.

```{eval-rst}
.. autofunction:: autoform.format
.. autofunction:: autoform.concat
.. autofunction:: autoform.match
```

## Control Flow

Control flow primitives.

```{eval-rst}
.. autofunction:: autoform.stop_gradient
.. autofunction:: autoform.switch
.. autofunction:: autoform.while_loop
```

## Scheduling

Concurrency and dependency management.

```{eval-rst}
.. autofunction:: autoform.depends
```

## Types

Data structures.

```{eval-rst}
.. autoclass:: autoform.Struct
   :members:
   :undoc-members:

.. autodata:: autoform.PYTREE_NAMESPACE
```
