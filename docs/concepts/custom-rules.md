# Custom Rules

```{admonition} Advanced
:class: info

Use custom rules when a traceable function boundary needs transform-specific behavior. Most functions should rely on the default behavior, where transforms trace through the function body.
```

Custom rules let a [transform](transforms.md) use a domain-specific rule for a traceable function. Without a custom rule, the transform uses the [primitive](primitives.md) rules in the function body.

The wrapped function body must still be traceable. Use {py:func}`custom <autoform.custom>` for a boundary around traceable `autoform` code. Use [Primitive Definitions](../recipes/extending/writing-primitives.md) for runtime work such as HTTP calls, database lookups, or libraries that require concrete Python values.

Use {py:func}`custom <autoform.custom>` when one of these applies:

- a sub-function should be treated as an atomic boundary by a transform;
- a domain-specific rule is more correct than the default decomposition;
- a domain-specific rule is more efficient than tracing through the body.

## Function Boundaries

{py:func}`custom <autoform.custom>` is a decorator on a traceable Python function. It wraps the function as a primitive-like boundary. Direct calls still behave like the original function, but [transforms](transforms.md) can stop at that boundary and use a registered rule. Mark a function boundary with the decorator:

```python
import autoform as af


@af.custom
def bracket(text: str) -> str:
    return "[" + text + "]"
```

With no registered rules, transforms fall back to the body behavior. Register a rule only for the transform to override.

## Pushforward Rule

Define how input changes affect the bracketed result:

```python
@bracket.set_pushforward
def bracket_pushforward(in_tree, /, *, call):
    primals, tangents = in_tree
    (text_tangent,) = tangents
    output = call(*primals)
    tangent = "bracket change: " + text_tangent
    return output, tangent


ir = af.trace(lambda text: bracket(text))("seed")
output, tangent = af.pushforward(ir).call(("hello",), ("make it direct",))

assert output == "[hello]"
assert tangent == "bracket change: make it direct"
```

The pushforward rule receives `(primals, tangents)` and returns
`(primal_output, tangent_output)`.

## Pullback Rule

Define how output feedback becomes feedback for the original text:

```python
@bracket.set_pullback
def bracket_pullback(in_tree, /, *, call):
    del call
    (primals, output), feedback = in_tree
    (text,) = primals
    text_feedback = feedback + " via " + output + " from " + text
    return (text_feedback,)


ir = af.trace(lambda text: bracket(text))("seed")
output, (text_feedback,) = af.pullback(ir).call(("hello",), "too decorated")

assert output == "[hello]"
assert text_feedback == "too decorated via [hello] from hello"
```

The pullback rule receives `((primals, output), feedback)` and returns
feedback with the same shape as the original inputs.

## Batch Rule

Define the behavior for a batch of texts:

```python
@bracket.set_batch
def bracket_batch(in_tree, /, *, call):
    del call
    batch_size, axes, values = in_tree
    (texts,) = values
    (text_axis,) = axes

    assert text_axis is True
    assert batch_size == len(texts)

    return [("<" + text + ">") for text in texts], True


ir = af.trace(lambda text: bracket(text))("seed")
assert af.batch(ir).call(["a", "b"]) == ["<a>", "<b>"]
```

The rule receives one `in_tree` argument. For batch, that tree is `(batch_size, axes, values)`, and the rule returns `(outputs, output_axes)`.

## Rule Hooks

The wrapper exposes three sync/async pairs:

- `set_pushforward(rule)` / `aset_pushforward(rule)`;
- `set_pullback(rule)` / `aset_pullback(rule)`;
- `set_batch(rule)` / `aset_batch(rule)`.

The pullback hook overrides the backward sweep. The forward sweep still records the primal output and residuals needed by the backward rule.

Sync and async registrations are independent. If only `set_batch` is registered, then {py:func}`batch <autoform.batch>` uses that rule, while `await af.batch(ir).acall(...)` may use the default async behavior. Register both sides when both execution modes need the same custom semantics.

## Rule Correctness

Transforms will trust that the rules provided are correct. It is possible to provide rules that are structurally correct but do not provide the correct feedback or result when used in a transform. Custom rules should be used to indicate boundaries between traceable subprograms. Custom rules should not be used as a general purpose extension point.
