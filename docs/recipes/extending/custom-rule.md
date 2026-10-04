# Custom Rules

Use {py:func}`custom <autoform.custom>` when a traceable helper function
should appear as one boundary in the [IR](../../concepts/the-ir.md). Add transform rules for the
[transforms](../../concepts/transforms.md) required by the program.

```{admonition} Concept
[Custom Rules](../../concepts/custom-rules.md) · [Transforms](../../concepts/transforms.md) · [Primitives](../../concepts/primitives.md)
```

There are three kinds of rules to separately register: forward, backward, and batch.

```python
import autoform as af


calls = []


@af.custom
def bracket(text: str) -> str:
    return "[" + text + "]"


@bracket.set_pushforward
def pushforward_bracket(in_tree, /, *, call):
    primals, tangents = in_tree
    (text_tangent,) = tangents

    # keep the forward value and define the tangent behavior
    output = call(*primals)
    tangent = "bracket change: " + text_tangent
    return output, tangent


@bracket.set_pullback
def pullback_bracket(in_tree, /, *, call):
    del call
    (primals, output), feedback = in_tree
    (text,) = primals

    # turn output feedback into feedback for the input text
    text_feedback = feedback + " via " + output + " from " + text
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
    return [("<" + text + ">") for text in texts], True


def clean(text: str) -> str:
    return bracket(text)


ir = af.trace(clean)("text")

output, tangent = af.pushforward(ir).call(("alpha",), ("input change",))
print(output)
print(tangent)
assert output == "[alpha]"
assert tangent == "bracket change: input change"

output, (text_feedback,) = af.pullback(ir).call(("alpha",), "output feedback")
print(output)
print(text_feedback)
assert output == "[alpha]"
assert text_feedback == "output feedback via [alpha] from alpha"

batched = af.batch(ir)

outputs = batched.call(["a", "b"])
print(outputs)
print(calls)
assert outputs == ["<a>", "<b>"]
assert calls == ["batch"]
```

Each rule receives one `in_tree` argument:

| Hook | `in_tree` shape | Return shape |
| --- | --- | --- |
| `set_pushforward` | `(primals, tangents)` | `(output, tangent)` |
| `set_pullback` | `((primals, output), feedback)` | input-shaped feedback |
| `set_batch` | `(batch_size, axes, values)` | `(output, output_axes)` |

For batch, `output_axes` has the same [pytree](../../concepts/pytrees.md) shape as the output and marks which
output leaves are batched.

The custom batch rule deliberately replaces brackets with angle brackets. A rule that should preserve the primal program would keep the original brackets.

Add only the rules the program needs. If a custom boundary should run under
scheduled async execution, add the matching async rule. Runtime calls that need
concrete Python values belong in [Primitive Definitions](writing-primitives.md),
not in function bodies decorated with {py:func}`custom <autoform.custom>`.
