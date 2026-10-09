# Primitives

Primitives have registered behavior for execution and supported transforms. Tracing records calls to these operations as equations in the IR. The groups below cover strings, numbers, language models, control flow, and execution effects. The string formatting helper composes concatenations rather than defining another primitive.

```{eval-rst}
.. autofunction:: autoform.lm.fill
```

## String

```{eval-rst}
.. autofunction:: autoform.string.format
.. autofunction:: autoform.string.concat
.. autofunction:: autoform.string.match
```

## Numeric

```{eval-rst}
.. autofunction:: autoform.numeric.neg
.. autofunction:: autoform.numeric.add
.. autofunction:: autoform.numeric.sub
.. autofunction:: autoform.numeric.mul
.. autofunction:: autoform.numeric.div
.. autofunction:: autoform.numeric.eq
.. autofunction:: autoform.numeric.ne
.. autofunction:: autoform.numeric.lt
.. autofunction:: autoform.numeric.le
.. autofunction:: autoform.numeric.gt
.. autofunction:: autoform.numeric.ge
```

## Control Flow

```{eval-rst}
.. autofunction:: autoform.stop_gradient
.. autofunction:: autoform.switch
.. autofunction:: autoform.while_loop
.. autofunction:: autoform.fixpoint
```

## Scheduling

```{eval-rst}
.. autofunction:: autoform.depends
```

## Intercepts

```{eval-rst}
.. autofunction:: autoform.checkpoint
```

## Trace Weight

```{eval-rst}
.. autofunction:: autoform.factor
```
