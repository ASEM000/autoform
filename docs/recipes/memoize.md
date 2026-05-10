# Cache Repeated Computations with `memoize`

[`memoize`](../reference/glossary.md) is a runtime context that caches [primitive](../concepts/primitives.md)
results inside the `with` block. Use it when the same primitive call is repeated
with the same inputs.

## Cache During Execution

```python
import autoform as af


def program(text: str) -> str:
    left = af.format("<{}>", text)
    right = af.format("<{}>", text)
    return af.concat(left, right)


ir = af.trace(program)("seed")

# both format calls have the same primitive and inputs
with af.memoize():
    result = ir.call("alpha")

print(result)
```

The two `af.format("<{}>", "alpha")` calls are the same primitive call, so the
second one reads the cached result.

## Deduplicate During Tracing

`memoize` can also be used while tracing. In that case, identical primitive
calls become one recorded equation.

```python
import autoform as af


def duplicated(text: str) -> tuple[str, str]:
    with af.memoize():
        first = af.concat(text, "!")
        second = af.concat(text, "!")
        return first, second


ir = af.trace(duplicated)("seed")
print(ir.call("alpha"))
```

The repeated `concat` call is cached while the trace is being built. `checkpoint`
is not memoized because repeated checkpoints are meant to remain visible to
`collect` and `inject`. For other primitives, use `memoize` when the same inputs
really should mean the same result.
