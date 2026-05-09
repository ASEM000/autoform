# Custom Rules

Most functions do not need custom rules. The default transform behavior traces through the function body and applies primitive rules inside it.

Reach for `@af.custom` when you need one of these:

- a sub-function should be treated as an atomic boundary by a transform;
- a domain-specific rule is more correct than the default decomposition;
- a domain-specific rule is more efficient than tracing through the body.

## Mental Model

`custom` is a decorator on a Python function. It wraps the function as a primitive-like boundary. Direct calls still behave like the original function, but transforms can stop at that boundary and use a registered rule.

```python
import autoform as af


@af.custom
def bracket(text: str) -> str:
    return af.format("[{}]", text)
```

With no registered rules, transforms fall back to the body behavior. Register a rule only for the transform you want to override.

## A Custom Batch Rule

```python
@bracket.set_batch
def bracket_batch(in_tree, /, *, call):
    del call
    batch_size, axes, values = in_tree
    (texts,) = values
    (text_axis,) = axes

    assert text_axis is True
    assert batch_size == len(texts)

    return [af.format("<{}>", text) for text in texts], True


ir = af.trace(lambda text: bracket(text))("seed")
assert af.batch(ir).call(["a", "b"]) == ["<a>", "<b>"]
```

The rule receives one `in_tree` argument. For batch, that tree is `(batch_size, axes, values)`, and the rule returns `(outputs, output_axes)`.

## Available Rule Hooks

The wrapper exposes three sync/async pairs:

- `set_pushforward(rule)` / `aset_pushforward(rule)`;
- `set_pullback(rule)` / `aset_pullback(rule)`;
- `set_batch(rule)` / `aset_batch(rule)`.

The pullback hook overrides the backward sweep. The forward sweep still records the primal output and residuals needed by the backward rule.

Sync and async registrations are independent. If you register only `set_batch`, then `af.batch(ir).call(...)` uses your rule, while `await af.batch(ir).acall(...)` may use the default async behavior. Register both sides when both execution modes need the same custom semantics.

## Caveat

A custom rule is trusted. If the rule returns the wrong structure, wrong axes, or wrong cotangents, the transformed IR is wrong. Use custom rules for real boundaries, not as a general extension point for ordinary application code.
