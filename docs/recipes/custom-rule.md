# Define a `custom` Rule

Use [`custom`](../concepts/custom-rules.md) when a block of ordinary Python
should appear as one primitive boundary in the [IR](../concepts/the-ir.md). Add transform rules for the
[transforms](../concepts/transforms.md) required by the program.

```{admonition} Concept
[Custom Rules](../concepts/custom-rules.md) · [Transforms](../concepts/transforms.md) · [Primitives](../concepts/primitives.md)
```

```python
import autoform as af


calls = []


@af.custom
def bracket(text: str) -> str:
    return af.format("[{}]", text)


@bracket.set_pushforward
def pushforward_bracket(in_tree, /, *, call):
    primals, tangents = in_tree
    (text_tangent,) = tangents

    # keep the forward value and define the tangent behavior
    output = call(*primals)
    tangent = af.format("bracket change: {}", text_tangent)
    return output, tangent


@bracket.set_pullback
def pullback_bracket(in_tree, /, *, call):
    del call
    (primals, output), feedback = in_tree
    (text,) = primals

    # turn output feedback into feedback for the input text
    text_feedback = af.format("{} via {} from {}", feedback, output, text)
    return (text_feedback,)


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

output, tangent = af.pushforward(ir).call(("alpha",), ("make it direct",))
print(output)
print(tangent)

output, (text_feedback,) = af.pullback(ir).call(("alpha",), "too decorated")
print(output)
print(text_feedback)

batched = af.batch(ir)

print(batched.call(["a", "b"]))
print(calls)
```

Each rule receives one `in_tree` argument:

| Hook | `in_tree` shape | Return shape |
| --- | --- | --- |
| `set_pushforward` | `(primals, tangents)` | `(output, tangent)` |
| `set_pullback` | `((primals, output), feedback)` | input-shaped feedback |
| `set_batch` | `(batch_size, axes, values)` | `(output, output_axes)` |

For batch, `output_axes` has the same [pytree](../concepts/pytrees.md) shape as the output and marks which
output leaves are batched.

Add only the rules the program needs. If a custom boundary should run under
scheduled async execution, add the matching async rule.
