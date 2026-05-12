# Custom Rules

```{admonition} Advanced
:class: info

Use custom rules when a traceable function boundary needs transform-specific behavior. Most functions should rely on the default behavior, where transforms trace through the function body.
```

Most functions do not need custom rules. The default [transform](transforms.md) behavior traces through the function body and applies [primitive](primitives.md) rules inside it.

The wrapped function body must still be traceable. Use {py:func}`custom <autoform.custom>` for a boundary around traceable `autoform` code. Use [Write a Primitive](../recipes/writing-primitives.md) for runtime work such as HTTP calls, database lookups, or libraries that require concrete Python values.

Reach for {py:func}`custom <autoform.custom>` when one of these applies:

- a sub-function should be treated as an atomic boundary by a transform;
- a domain-specific rule is more correct than the default decomposition;
- a domain-specific rule is more efficient than tracing through the body.

## Mental Model

{py:func}`custom <autoform.custom>` is a decorator on a traceable Python function. It wraps the function as a primitive-like boundary. Direct calls still behave like the original function, but [transforms](transforms.md) can stop at that boundary and use a registered rule.

```python
import autoform as af


@af.custom
def bracket(text: str) -> str:
    return af.format("[{}]", text)
```

With no registered rules, transforms fall back to the body behavior. Register a rule only for the transform to override.

## {py:func}`pushforward <autoform.pushforward>` Rule

```python
@bracket.set_pushforward
def bracket_pushforward(in_tree, /, *, call):
    primals, tangents = in_tree
    (text_tangent,) = tangents
    output = call(*primals)
    tangent = af.format("bracket change: {}", text_tangent)
    return output, tangent


ir = af.trace(lambda text: bracket(text))("seed")
output, tangent = af.pushforward(ir).call(("hello",), ("make it direct",))

assert output == "[hello]"
assert tangent == "bracket change: make it direct"
```

The pushforward rule receives `(primals, tangents)` and returns
`(primal_output, tangent_output)`.

## {py:func}`pullback <autoform.pullback>` Rule

```python
@bracket.set_pullback
def bracket_pullback(in_tree, /, *, call):
    del call
    (primals, output), feedback = in_tree
    (text,) = primals
    text_feedback = af.format("{} via {} from {}", feedback, output, text)
    return (text_feedback,)


ir = af.trace(lambda text: bracket(text))("seed")
output, (text_feedback,) = af.pullback(ir).call(("hello",), "too decorated")

assert output == "[hello]"
assert text_feedback == "too decorated via [hello] from hello"
```

The pullback rule receives `((primals, output), feedback)` and returns
feedback with the same shape as the original inputs.

## {py:func}`batch <autoform.batch>` Rule

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

## Rule Hooks

The wrapper exposes three sync/async pairs:

- `set_pushforward(rule)` / `aset_pushforward(rule)`;
- `set_pullback(rule)` / `aset_pullback(rule)`;
- `set_batch(rule)` / `aset_batch(rule)`.

The pullback hook overrides the backward sweep. The forward sweep still records the primal output and residuals needed by the backward rule.

Sync and async registrations are independent. If only `set_batch` is registered, then {py:func}`batch <autoform.batch>` uses that rule, while `await af.batch(ir).acall(...)` may use the default async behavior. Register both sides when both execution modes need the same custom semantics.

## Rule Correctness

A custom rule is trusted. If the rule returns the wrong structure, wrong axes, or wrong cotangents, the transformed IR is wrong. Use custom rules for traceable subprogram boundaries, not as a general extension point for ordinary application code.
