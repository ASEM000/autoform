# Primitive Definitions

```{admonition} Advanced
:class: info

`autoform` is also a low-level extensible framework. Most code should stay on the public primitives and transforms, but primitive authoring is available when an operation needs to become part of the IR system itself. This recipe uses `autoform.extend`.
```

A primitive is the right boundary for runtime work that needs concrete values: HTTP calls, retrieval systems, databases, calculators, or libraries that cannot run on traced placeholders. The function wrapper stays small; the behavior lives in registered rules.

```{admonition} Concept
[Primitives](../../concepts/primitives.md) · [Transforms](../../concepts/transforms.md)
```

## Execution and Tracing

Define a wrapper and register its concrete and abstract rules:

```python
import autoform as af
import autoform.extend as afe


lookup_p = afe.Prim("lookup")


def lookup(query: str) -> str:
    return lookup_p.bind(query)


def impl_lookup(query: str, /) -> str:
    return "result for " + query


def abstract_lookup(query, /):
    del query
    return af.string.StrAVal()


afe.register_impl(lookup_p, impl_lookup)
afe.register_abstract(lookup_p, abstract_lookup)


ir = af.trace(lookup)("seed")
assert ir.call("query text") == "result for query text"
```

The wrapper `lookup(...)` is what traced programs call. During tracing, `lookup_p.bind(...)` records one equation. During execution, `impl_lookup(...)` receives the concrete runtime value.

The abstract rule runs at trace time. It must return the output shape and abstract value without calling the runtime implementation. Built-in scalar outputs use explicit avals such as `af.string.StrAVal()`, `af.numeric.IntAVal()`, `af.numeric.FloatAVal()`, and `af.numeric.BoolAVal()`.

In order to allow a new runtime type to be used, one needs to define the abstract value type for this runtime type and register the mapping from concrete to abstract values:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    text: str


class SearchResultAVal(afe.AVal):
    __slots__ = []


afe.register_trace_type(SearchResult, lambda value: SearchResultAVal())
```

Registration lets the type enter {py:func}`trace <autoform.trace>` as a dynamic leaf. Add primal, tangent, and cotangent space mappings for the transforms it should support. See [Array Extension](array-extension.md) for a type with shape metadata and AD rules.

## Rules

Each registry has a separate purpose:

| Registry | Purpose |
| --- | --- |
| `impl_rules` | Sync execution for `.call(...)`. |
| `abstract_rules` | Trace-time output shape and abstract value. |
| `batch_rules` | Behavior under {py:func}`batch <autoform.batch>`. |
| `push_rules` | Behavior under {py:func}`pushforward <autoform.pushforward>`. |
| `pull_fwd_rules` | Forward sweep used by {py:func}`pullback <autoform.pullback>`. |
| `pull_bwd_rules` | Backward sweep used by {py:func}`pullback <autoform.pullback>`. |

Register only the behavior the primitive needs. Applying a transform that reaches a primitive without the matching rule raises an error from the rule registry.

## Batch Rule

Handle a batch of queries or one shared query:

```python
def batch_lookup(in_tree, /):
    batch_size, axes, values = in_tree
    del batch_size
    query_axis = axes
    queries = values

    if not query_axis:
        return lookup_p.bind(queries), False

    return [lookup_p.bind(query) for query in queries], True


afe.register_batch(lookup_p, batch_lookup)


assert af.batch(ir).call(["a", "b"]) == ["result for a", "result for b"]
```

The batch rule receives the batch size, the input axes, and the input values. It returns `(output, output_axes)`.

## Pullback Rule

Keep the query and result as residuals for the backward rule:

```python
def pull_fwd_lookup(query: str, /):
    output = lookup_p.bind(query)
    return output, (query, output)


def pull_bwd_lookup(in_tree, /):
    (query, output), feedback = in_tree
    return "Improve query '" + query + "'. Feedback: " + feedback + ". Result: " + output


afe.register_pullback_fwd(lookup_p, pull_fwd_lookup)
afe.register_pullback_bwd(lookup_p, pull_bwd_lookup)


output, (query_feedback,) = af.pullback(ir).call(
    ("query text",),
    "output feedback",
)
assert output == "result for query text"
assert (
    query_feedback == "Improve query 'query text'. Feedback: output feedback. "
    "Result: result for query text"
)
```

The forward sweep returns the normal output plus residuals. The backward sweep receives those residuals and the output feedback, then returns feedback with the same shape as the primitive input.

## Async Execution

Register async implementations when an IR containing the primitive should run with `.acall(...)`:

```python
async def aimpl_lookup(query: str, /) -> str:
    return impl_lookup(query)


afe.register_aimpl(lookup_p, aimpl_lookup)
```

Register async rules with the corresponding function, such as `afe.register_abatch`.
