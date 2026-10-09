# Primitives and Rules

A transform needs a defined meaning for each operation it encounters. Primitives provide those identifiable operations in a [traced](tracing.md) program. Each call becomes an equation in the [IR](programs-and-ir.md#the-ir), and registered rules describe how that primitive executes or transforms. Examples include {py:func}`concat <autoform.string.concat>`, {py:func}`fill <autoform.lm.fill>`, {py:func}`switch <autoform.switch>`, {py:func}`checkpoint <autoform.checkpoint>`, and {py:func}`factor <autoform.factor>`.

Primitive identity determines which rule a [transform](transforms.md) selects.[^primitive-identity] This is especially useful for work that cannot run on a tracing placeholder. A language model request, for example, needs concrete data and an external service. Treating {py:func}`fill <autoform.lm.fill>` as a primitive gives tracing an output description and gives {py:func}`pullback <autoform.pullback>` a defined feedback rule, without tracing the provider implementation.

## Rule Registries

Rules for execution, tracing, and [transforms](transforms.md) are registered through `autoform.extend`. The following registries cover execution, tracing, batching, and differentiation:

| Registry | Purpose |
| --- | --- |
| `impl_rules` | Execution with concrete input values. |
| `abstract_rules` | Output abstract values used during tracing. |
| `batch_rules` | Outputs and output batch axes for {py:func}`batch <autoform.batch>`. |
| `push_rules` | The output and its tangent for {py:func}`pushforward <autoform.pushforward>`. |
| `pull_fwd_rules` | The output and saved residuals for {py:func}`pullback <autoform.pullback>`. |
| `pull_bwd_rules` | Input feedback computed from residuals and output feedback. |

A pullback separates the original computation from feedback propagation. Its forward rule returns both the result and any residuals needed later. The backward rule receives those residuals together with the output cotangent and returns input cotangents.

## Built-in Primitives

| Group | Primitives | Purpose |
| --- | --- | --- |
| Strings | {py:func}`concat <autoform.string.concat>`, {py:func}`match <autoform.string.match>` | String concatenation and equality checks.[^string-format] |
| Language models | {py:func}`fill <autoform.lm.fill>` | Generated values replacing specifications in a pytree, with surrounding context retained. [Language Models](../language-models.md) describes the interface. |
| Numbers | [Numeric primitives](../api/primitives.md#numeric) | Scalar arithmetic and comparisons. |
| Control flow | {py:func}`switch <autoform.switch>`, {py:func}`while_loop <autoform.while_loop>`, {py:func}`fixpoint <autoform.fixpoint>` | Runtime branch selection and bounded iteration, described in [Control Flow](control-flow.md). |
| Gradient flow | {py:func}`stop_gradient <autoform.stop_gradient>` | An unchanged forward value with tangents and cotangents blocked. |
| Dependencies | {py:func}`depends <autoform.depends>` | Execution dependencies added without changing the returned value. |
| Intermediate values | {py:func}`checkpoint <autoform.checkpoint>` | Named values available to {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>`. |
| Path weights | {py:func}`factor <autoform.factor>` | Contributions to the path weight returned by {py:func}`weight <autoform.weight>`. |

## Primitive Definitions

A primitive definition separates its public call from its implementation. The wrapper binds the primitive, the execution rule performs work on concrete values, and the abstract rule describes the output during tracing. The example below uses `autoform.extend` to expose a lookup operation through these three interfaces.

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

The abstract rule describes the output without calling the runtime implementation. Here it returns the abstract value for a built-in string. A new runtime type needs its own abstract value and registered spaces. [Array Extension](../recipes/extending/array-extension.md) provides a complete example.

### Primitive Transform Rules

Execution and tracing alone do not define transformation behavior. A batching rule receives the batch size, input axes, and input values, then returns the result and output axes. The lookup example supports either a batch of queries or one query shared across the batch.

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

A pullback needs a forward rule and a backward rule. The forward rule evaluates the lookup and retains the query and result as residuals. The backward rule uses those residuals and the output critique to return feedback shaped like the primitive input. The rule defines the meaning of that feedback for the lookup.

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

An existing traceable function can also have special transformation behavior. The {py:func}`custom <autoform.custom>` decorator preserves a boundary around that function. Ordinary calls still execute its body; supported transforms use a registered hook when one is available and otherwise transform the body.

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
    p, t = in_tree
    (t_text,) = t
    return call(*p), "bracket change: " + t_text


@bracket.set_pullback
def pullback_bracket(in_tree, /, *, call):
    del call
    (p, output), feedback = in_tree
    (text,) = p
    return (feedback + " via " + output + " from " + text,)


output, t = af.pushforward(bracket_ir).call(
    ("text",),
    ("input change",),
)
assert (output, t) == ("[text]", "bracket change: input change")

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

The pullback hook replaces the backward sweep of the pullback transform. The forward sweep still records the primal inputs and output.

### Rule Correctness

Transforms rely on the meaning supplied by registered rules. A structurally valid result can still be mathematically or semantically wrong. Useful checks compare the rule with the intended operation behavior, including shared inputs, batched inputs, and asynchronous calls where supported.

[^primitive-identity]: If two different primitives have the same name, the primitives remain different keys when selecting rules.

[^string-format]: The {py:func}`format <autoform.string.format>` helper resolves `{name}` fields from keyword arguments by composing {py:func}`concat <autoform.string.concat>` calls. It has no primitive or transform rules of its own. Attributes or indexed values must be selected before being passed as keyword arguments.

[^async-hooks]: The async hooks are `aset_pushforward`, `aset_pullback`, and `aset_batch`. Matching synchronous and asynchronous behavior requires registrations for both paths.
