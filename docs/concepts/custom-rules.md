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


ir = af.trace(bracket)("seed")
assert ir.call("alpha") == "[alpha]"
assert af.batch(ir).call(["a", "b"]) == ["[a]", "[b]"]
```

With no registered rules, transforms fall back to the body behavior. Register a rule only for the transform to override.

The [Custom Rules recipe](../recipes/extending/custom-rule.md) shows how to register and check rules for each transform.

## Rule Hooks

The wrapper exposes three sync/async pairs:

- `set_pushforward(rule)` / `aset_pushforward(rule)`;
- `set_pullback(rule)` / `aset_pullback(rule)`;
- `set_batch(rule)` / `aset_batch(rule)`.

The pullback hook overrides the backward sweep. The forward sweep still records the primal output and residuals needed by the backward rule.

Sync and async registrations are independent. If only `set_batch` is registered, then {py:func}`batch <autoform.batch>` uses that rule, while `await af.batch(ir).acall(...)` may use the default async behavior. Register both sides when both execution modes need the same custom semantics.

## Rule Correctness

Transforms will trust that the rules provided are correct. It is possible to provide rules that are structurally correct but do not provide the correct feedback or result when used in a transform. Custom rules should be used to indicate boundaries between traceable subprograms. Custom rules should not be used as a general purpose extension point.
