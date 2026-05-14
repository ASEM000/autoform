# Write a Primitive

```{admonition} Advanced
:class: info

`autoform` is also a low-level extensible framework. Most code should stay on the public primitives and transforms, but primitive authoring is available when an operation needs to become part of the IR system itself. This recipe uses `autoform.extend`.
```

A primitive is the right boundary for runtime work that needs concrete values: HTTP calls, retrieval systems, databases, calculators, or libraries that cannot run on traced placeholders. The function wrapper stays small; the behavior lives in registered rules.

## Minimal Shape

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
    return afe.StrAVal()


afe.impl_rules.set(lookup_p, impl_lookup)
afe.abstract_rules.set(lookup_p, abstract_lookup)


ir = af.trace(lookup)("seed")
assert ir.call("recursion") == "result for recursion"
```

The wrapper `lookup(...)` is what traced programs call. During tracing, `lookup_p.bind(...)` records one equation. During execution, `impl_lookup(...)` receives the concrete runtime value.

The abstract rule runs at trace time. It must return the output shape and abstract value without calling the runtime implementation. Built-in scalar outputs use explicit avals such as `afe.StrAVal()`, `afe.IntAVal()`, `afe.FloatAVal()`, and `afe.BoolAVal()`.

For new runtime value types, define an `afe.AVal` subclass that carries the abstract metadata you need, then register the trace type:

```python
class SearchResultAVal(afe.AVal):
    __slots__ = ["fields"]

    def __init__(self, fields: tuple[str, ...]):
        self.fields = fields


afe.register_trace_type(
    SearchResult,
    lambda value: SearchResultAVal(tuple(value.fields)),
)
```

This allows values of the Python type to enter `af.trace` and teaches `avalof(...)` how to infer their abstract value.

## Rules by Phase

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

```python
def batch_lookup(in_tree, /):
    batch_size, axes, values = in_tree
    del batch_size
    query_axis = axes
    queries = values

    if not query_axis:
        return lookup_p.bind(queries), False

    return [lookup_p.bind(query) for query in queries], True


afe.batch_rules.set(lookup_p, batch_lookup)


assert af.batch(ir).call(["a", "b"]) == ["result for a", "result for b"]
```

The batch rule receives the batch size, the input axes, and the input values. It returns `(output, output_axes)`.

## Pullback Rule

```python
def pull_fwd_lookup(query: str, /):
    output = lookup_p.bind(query)
    return output, (query, output)


def pull_bwd_lookup(in_tree, /):
    (query, output), feedback = in_tree
    return af.format("Improve query '{}'. Feedback: {}. Result: {}", query, feedback, output)


afe.pull_fwd_rules.set(lookup_p, pull_fwd_lookup)
afe.pull_bwd_rules.set(lookup_p, pull_bwd_lookup)


output, (query_feedback,) = af.pullback(ir).call(("recursion",), "too broad")
assert output == "result for recursion"
assert (
    query_feedback == "Improve query 'recursion'. Feedback: too broad. Result: result for recursion"
)
```

The forward sweep returns the normal output plus residuals. The backward sweep receives those residuals and the output feedback, then returns feedback with the same shape as the primitive input.

## Async Execution

Register async implementations when an IR containing the primitive should run with `.acall(...)`:

```python
async def aimpl_lookup(query: str, /) -> str:
    return impl_lookup(query)


afe.impl_rules.aset(lookup_p, aimpl_lookup)
```

Async transform rules use the corresponding `aset(...)` registry method. For example, `afe.batch_rules.aset(...)` registers async {py:func}`batch <autoform.batch>` behavior.
