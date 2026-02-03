# API Reference

## Core

Tracing and execution.

```{eval-rst}
.. autofunction:: autoform.trace
.. autofunction:: autoform.atrace
.. autofunction:: autoform.call
.. autofunction:: autoform.acall
```

## Transforms

IR-to-IR transformations.

```{eval-rst}
.. autofunction:: autoform.batch
.. autofunction:: autoform.pullback
.. autofunction:: autoform.pushforward
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
.. autofunction:: autoform.struct_lm_call
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
.. autofunction:: autoform.gather
.. autofunction:: autoform.depends
```

## Surgery

IR manipulation.

```{eval-rst}
.. autofunction:: autoform.split
.. autofunction:: autoform.splitpoint
```

## Types

Data structures.

```{eval-rst}
.. autoclass:: autoform.Struct
   :members:
   :undoc-members:

.. autodata:: autoform.PYTREE_NAMESPACE
```
