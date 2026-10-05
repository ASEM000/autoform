# Primitives and Rules

Primitives are named operations that can be recorded in the [IR](programs-and-ir.md#the-ir) when a function is [traced](tracing.md). Examples include {py:func}`concat <autoform.string.concat>`, {py:func}`fill <autoform.lm.fill>`, {py:func}`switch <autoform.switch>`, {py:func}`checkpoint <autoform.checkpoint>`, and {py:func}`factor <autoform.factor>`.

Primitives are important because [transforms](transforms.md) need to select appropriate rules based on the identity of the primitive.[^primitive-identity] For example, the {py:func}`pullback <autoform.pullback>` transform needs to know how to push feedback through a {py:func}`fill <autoform.lm.fill>` primitive. If {py:func}`fill <autoform.lm.fill>` was written as a standard Python function, the function body may include operations for which {py:func}`pullback <autoform.pullback>` has no rules. These operations would need to either be executed when tracing the function body or throw an exception when a concrete runtime value is required.

## Rule Registries

Rules for execution, tracing, and [transforms](transforms.md) are registered
through `autoform.extend`. The following registries cover execution, tracing,
batching, and differentiation:

| Registry | Purpose |
| --- | --- |
| `impl_rules` | Run the primitive with concrete values. |
| `abstract_rules` | Compute the abstract values of the outputs during tracing. |
| `batch_rules` | Return outputs and their batch axes for {py:func}`batch <autoform.batch>`. |
| `push_rules` | Return the output and its tangent for {py:func}`pushforward <autoform.pushforward>`. |
| `pull_fwd_rules` | Return the output and saved residuals for {py:func}`pullback <autoform.pullback>`. |
| `pull_bwd_rules` | Produce input feedback from residuals and output feedback. |

The forward sweep records residuals. The backward sweep uses those residuals and the output cotangent to produce input cotangents.

## Built-in Primitives

| Group | Primitives | Purpose |
| --- | --- | --- |
| Strings | {py:func}`concat <autoform.string.concat>`, {py:func}`match <autoform.string.match>` | String concatenation and equality checks.[^string-format] |
| Language models | {py:func}`fill <autoform.lm.fill>` | Replace specs in a pytree with generated values while retaining the surrounding context. See [Language Models](../language-models.md). |
| Numbers | [Numeric primitives](../api/primitives.md#numeric) | Scalar arithmetic and comparisons. |
| Control flow | {py:func}`switch <autoform.switch>`, {py:func}`while_loop <autoform.while_loop>`, {py:func}`fixpoint <autoform.fixpoint>` | Select branches or run bounded loops. See [Control Flow](control-flow.md). |
| Gradient flow | {py:func}`stop_gradient <autoform.stop_gradient>` | Preserve the input value while blocking tangents and cotangents. |
| Dependencies | {py:func}`depends <autoform.depends>` | Add execution dependencies without changing the returned value. |
| Intermediate values | {py:func}`checkpoint <autoform.checkpoint>` | Tag values for {py:func}`collect <autoform.collect>` or {py:func}`inject <autoform.inject>`. |
| Path weights | {py:func}`factor <autoform.factor>` | Multiply the path weight collected by {py:func}`weight <autoform.weight>`. |

## Primitive Definitions

Primitives handle runtime work that needs concrete values. A wrapper binds the primitive, an execution rule performs the work, and an abstract rule describes the output when tracing. These examples use the low-level `autoform.extend` API.

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

The abstract rule describes the output without calling the runtime implementation. Here it returns the abstract value for a built-in string. A new runtime type needs its own abstract value and registered spaces; see [Array Extension](../recipes/extending/array-extension.md) for a complete example.

### Primitive Transform Rules

Then rules need to be provided for each transform. For batching, a single rule is provided that is given the batch size, input axes, and input values, and must return the output and output axes. In this case, the rule supports either batching the queries or using a single shared one.

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

For pullback, two rules are needed, one for each sweep. The forward sweep records the residuals, and the backward sweep takes the residuals and the output feedback and returns the input feedback. Note that the input and output must match those provided to the primitive boundary.

```python
def pull_fwd_lookup(query: str, /):
    output = lookup_p.bind(query)
    return output, (query, output)


def pull_bwd_lookup(in_tree, /):
    (query, output), feedback = in_tree
    request = "Improve query '" + query + "'. Feedback: " + feedback
    return request + ". Result: " + output


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

Sync and async implementations are registered separately. An async implementation lets the containing IR use `.acall(...)`. Transforms used during async execution need matching async rules, such as `register_abatch` and `register_apullback_bwd`.

```python
async def aimpl_lookup(query: str, /) -> str:
    return impl_lookup(query)


afe.register_aimpl(lookup_p, aimpl_lookup)
```

## Custom Rules

{py:func}`custom <autoform.custom>` creates a boundary around a function that can already be traced. Ordinary calls run the body. Transforms use registered rules when available and otherwise use the body.

```python
@af.custom
def bracket(text: str) -> str:
    return "[" + text + "]"


bracket_ir = af.trace(bracket)("text")
assert bracket_ir.call("text") == "[text]"
```

### Rule Hooks

The hooks are given the `in_tree` and a `call` function that will call the original function. The pushforward hook should return the primal output and the tangent, and the pullback hook should return the input feedback.[^async-hooks]

```python
@bracket.set_pushforward
def pushforward_bracket(in_tree, /, *, call):
    primals, tangents = in_tree
    (text_tangent,) = tangents
    return call(*primals), "bracket change: " + text_tangent


@bracket.set_pullback
def pullback_bracket(in_tree, /, *, call):
    del call
    (primals, output), feedback = in_tree
    (text,) = primals
    return (feedback + " via " + output + " from " + text,)


output, tangent = af.pushforward(bracket_ir).call(
    ("text",),
    ("input change",),
)
assert (output, tangent) == ("[text]", "bracket change: input change")

output, (text_feedback,) = af.pullback(bracket_ir).call(
    ("text",),
    "output feedback",
)
assert text_feedback == "output feedback via [text] from text"
```

The batch hook should also return the tree for the output axes. Here, it keeps the original brackets whether the input is batched or shared.

```python
@bracket.set_batch
def batch_bracket(in_tree, /, *, call):
    batch_size, axes, values = in_tree
    del batch_size
    (text_axis,) = axes
    (texts,) = values
    if not text_axis:
        return call(texts), False
    return [call(text) for text in texts], True


assert af.batch(bracket_ir).call(["a", "b"]) == ["[a]", "[b]"]
assert af.batch(bracket_ir, in_axes=False).call("a") == "[a]"
```

| Hook | `in_tree` shape | Return shape |
| --- | --- | --- |
| `set_pushforward` | `(primals, tangents)` | `(output, tangent)` |
| `set_pullback` | `((primals, output), feedback)` | Input-shaped feedback. |
| `set_batch` | `(batch_size, axes, values)` | `(output, output_axes)` |

The pullback hook replaces the backward sweep, while the forward sweep records the primal inputs and output.

### Rule Correctness

Transforms trust the registered rules. Incorrect rules can produce incorrect results or feedback even when the returned structure is valid. Check the intended behavior of the function, including shared inputs and async execution when applicable.

[^primitive-identity]: If two different primitives have the same name, the primitives remain different keys when selecting rules.

[^string-format]: The {py:func}`format <autoform.string.format>` helper resolves `{name}` fields
    from keyword arguments by composing {py:func}`concat <autoform.string.concat>`
    calls. It has no primitive or transform rules of its own. Attributes or indexed
    values must be selected before being passed as keyword arguments.

[^async-hooks]: The async hooks are `aset_pushforward`, `aset_pullback`, and `aset_batch`. Sync hooks do not replace async behavior; register both for matching behavior.
