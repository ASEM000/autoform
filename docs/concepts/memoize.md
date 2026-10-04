# Memoization

{py:func}`memoize <autoform.memoize>` is a runtime context that caches [primitive](primitives.md)
results inside the `with` block. Use it when the same primitive call is repeated
with the same inputs.

## Execution

Run repeated operations within one cache context:

```python
import autoform as af


def program(text: str) -> str:
    left = "<" + text + ">"
    right = "<" + text + ">"
    return left + right


ir = af.trace(program)("seed")

# building right repeats the two concat operations used for left
with af.memoize():
    result = ir.call("alpha")

print(result)
```

Each `+` records a {py:func}`concat <autoform.string.concat>` equation. The two equations that build `right` repeat the inputs used to build `left`, so both read cached results. The output is `<alpha><alpha>`. The cache is discarded when the context exits.

## Tracing

{py:func}`memoize <autoform.memoize>` can also be used while tracing. In that case, identical primitive
calls become one recorded equation:

```python
import autoform as af


def duplicated(text: str) -> tuple[str, str]:
    with af.memoize():
        first = text + "!"
        second = text + "!"
        return first, second


ir = af.trace(duplicated)("seed")
print(ir.call("alpha"))
```

The result is `("alpha!", "alpha!")`, produced from one recorded concatenation. {py:func}`checkpoint <autoform.checkpoint>`
is not memoized because repeated checkpoints are meant to remain visible to
{py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>`. For other primitives, use {py:func}`memoize <autoform.memoize>` when the same inputs
really should mean the same result. This also applies to model calls: memoization reuses a response rather than requesting an independent sample.
