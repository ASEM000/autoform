# Write a Primitive

```{admonition} Advanced
:class: info

`autoform` is also a low-level extensible framework. Most code should stay on the public primitives and transforms, but primitive authoring is available when an operation needs to become part of the IR system itself. This recipe uses `autoform.core` internals.
```

A primitive is the right boundary for runtime work that needs concrete values: HTTP calls, retrieval systems, databases, calculators, or libraries that cannot run on traced placeholders. The function wrapper stays small; the behavior lives in registered rules.

## Minimal Shape

```python
import autoform as af


lookup_p = af.core.Prim("lookup")


def lookup(query: str) -> str:
    return lookup_p.bind(query)


def impl_lookup(query: str, /) -> str:
    return "result for " + query


def abstract_lookup(query, /):
    del query
    return af.core.StrAVal()


af.core.impl_rules.set(lookup_p, impl_lookup)
af.core.abstract_rules.set(lookup_p, abstract_lookup)


ir = af.trace(lookup)("seed")
assert ir.call("recursion") == "result for recursion"
```

The wrapper `lookup(...)` is what traced programs call. During tracing, `lookup_p.bind(...)` records one equation. During execution, `impl_lookup(...)` receives the concrete runtime value.

The abstract rule runs at trace time. It must return the output shape and abstract value without calling the runtime implementation. Built-in scalar outputs use explicit avals such as `af.core.StrAVal()`, `af.core.IntAVal()`, `af.core.FloatAVal()`, and `af.core.BoolAVal()`.

For new runtime value types, define an `af.core.AVal` subclass that carries the abstract metadata you need, then register the conversion rules:

```python
class SearchResultAVal(af.core.AVal):
    __slots__ = ["fields"]

    def __init__(self, fields: tuple[str, ...]):
        self.fields = fields


af.core.aval_rules[SearchResult] = lambda value: SearchResultAVal(tuple(value.fields))
af.core.aval_types[SearchResultAVal] = lambda aval: SearchResult
```

`aval_rules` maps concrete trace inputs to abstract values. `aval_types` powers `af.core.typeof(...)` checks inside primitive rules. The `aval_types` rule is callable so richer avals can inspect their own metadata.

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


af.core.batch_rules.set(lookup_p, batch_lookup)


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


af.core.pull_fwd_rules.set(lookup_p, pull_fwd_lookup)
af.core.pull_bwd_rules.set(lookup_p, pull_bwd_lookup)


output, (query_feedback,) = af.pullback(ir).call(("recursion",), "too broad")
assert output == "result for recursion"
assert query_feedback == "Improve query 'recursion'. Feedback: too broad. Result: result for recursion"
```

The forward sweep returns the normal output plus residuals. The backward sweep receives those residuals and the output feedback, then returns feedback with the same shape as the primitive input.

## Async Execution

Register async implementations when an IR containing the primitive should run with `.acall(...)`:

```python
async def aimpl_lookup(query: str, /) -> str:
    return impl_lookup(query)


af.core.impl_rules.aset(lookup_p, aimpl_lookup)
```

Async transform rules use the corresponding `aset(...)` registry method. For example, `af.core.batch_rules.aset(...)` registers async {py:func}`batch <autoform.batch>` behavior.
