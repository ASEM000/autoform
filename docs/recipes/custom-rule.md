# Define A Custom Transform Rule

Use [`custom`](../concepts/custom-rules.md) when a block of ordinary Python
should appear as one primitive boundary in the IR. Add transform rules for the
transforms you need.

```python
import autoform as af


calls = []


@af.custom
def bracket(text: str) -> str:
    return af.format("[{}]", text)


@bracket.set_batch
def batch_bracket(in_tree, /, *, call):
    batch_size, axes, values = in_tree
    del batch_size
    (texts,) = values
    (text_axis,) = axes

    # broadcast inputs call the original function once
    if not text_axis:
        calls.append("broadcast")
        return call(texts), False

    # batched inputs can use a domain-specific vectorized rule
    calls.append("batch")
    return [af.format("<{}>", text) for text in texts], True


def clean(text: str) -> str:
    return bracket(text)


ir = af.trace(clean)("  Hello  ")
batched = af.batch(ir)

print(batched.call(["a", "b"]))
print(calls)
```

The batch rule receives three pieces:

| Value | Meaning |
| --- | --- |
| `batch_size` | the inferred batch length |
| `axes` | booleans matching the input leaves |
| `values` | the actual input values |

Return `(output, output_axes)`, where `output_axes` has the same pytree shape as
the output and marks which output leaves are batched.

Add only the rules your program needs. If a custom boundary should participate
in prompt feedback, add `set_pullback`. If it should run under scheduled async
execution, add the matching async rule.
