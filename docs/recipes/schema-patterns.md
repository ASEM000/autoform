# Use Schema Patterns

{py:func}`lm_schema_call <autoform.lm_schema_call>` returns a structured value whose
shape can be transformed like any other [pytree](../concepts/pytrees.md). Dataclass examples use [Optree's dataclass integration](https://optree.readthedocs.io/en/latest/dataclasses.html).

```{admonition} Concept
[Schemas](../concepts/schemas.md) · [Pytrees](../concepts/pytrees.md) · [Transforms](../concepts/transforms.md) · [Primitives](../concepts/primitives.md)
```

## Route with {py:class}`Enum <autoform.Enum>`

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Route:
    tool: str
    answer: str


route_schema = Route(tool=af.Enum("search", "done"), answer=af.Str())


def choose_route(question: str) -> Route:
    prompt = af.format("Choose search or done for this question:\n{}", question)
    msg = dict(role="user", content=prompt)
    return af.lm_schema_call([msg], model="gpt-5.5", schema=route_schema)
```

Use {py:class}`Enum <autoform.Enum>` for finite decisions that should feed {py:func}`switch <autoform.switch>`, status fields, or
other control values.

## Use Nested Arguments

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class SearchArgs:
    query: str
    limit: int


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class SearchDecision:
    tool: str
    args: SearchArgs


decision_schema = SearchDecision(
    tool=af.Enum("search", "done"), args=SearchArgs(query=af.Str(), limit=af.Int(min=1, max=5))
)


def choose_search(question: str) -> SearchDecision:
    prompt = af.format("Choose the next tool call for:\n{}", question)
    msg = dict(role="user", content=prompt)
    return af.lm_schema_call([msg], model="gpt-5.5", schema=decision_schema)
```

Nested dataclasses keep related fields together and still register as one
pytree-shaped schema.

## Build Schema Variants

Schemas are [pytrees](../concepts/pytrees.md), so regular pytree utilities can derive call-specific variants before a call.

```python
import optree
import autoform as af


schema = {"answer": af.Str(), "confidence": af.Float(min=0, max=1)}

extract_schema = optree.tree_map(
    lambda leaf: leaf @ af.Doc("Extract only facts explicitly present in the input."),
    schema,
    namespace=af.PYTREE_NAMESPACE,
)

critique_schema = optree.tree_map(
    lambda leaf: leaf @ af.Doc("Critique the draft and report uncertainty conservatively."),
    schema,
    namespace=af.PYTREE_NAMESPACE,
)
```

Both schemas return `{"answer": ..., "confidence": ...}`. The downstream code stays fixed while each LM call receives different field guidance.

For a one-off call, the schema can stay inline:

```python
messages = [dict(role="user", content="Classify this request.")]
result = af.lm_schema_call(
    messages,
    model="gpt-5.5",
    schema={
        "kind": af.Enum("question", "request") @ af.Doc("Request type."),
        "reply": af.Str() @ af.Doc("Short reply."),
    },
)
```

## Represent Optional-Like Fields

There is no special optional schema node. Model presence explicitly.

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class MaybeAnswer:
    state: str
    text: str


maybe_schema = MaybeAnswer(state=af.Enum("present", "absent"), text=af.Str())
```

This keeps the output shape stable for {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>`, and {py:func}`switch <autoform.switch>`.

## Send Field Feedback

Structured outputs also work with {py:func}`pullback <autoform.pullback>`.

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    score: float


summary_schema = Summary(title=af.Str(max=80), score=af.Float(min=0, max=1))


def summarize(topic: str) -> Summary:
    prompt = af.format("Summarize {}.", topic)
    msg = dict(role="user", content=prompt)
    return af.lm_schema_call([msg], model="gpt-5.5", schema=summary_schema)


ir = af.trace(summarize)("recursion")
feedback = Summary(title="too vague", score="overconfident")
output, (topic_hint,) = af.pullback(ir).call(("recursion",), feedback)

print(output)
print(topic_hint)
```

The feedback value has the same dataclass shape as the output. Each field can
carry its own critique, and the backward rule summarizes that critique for the
inputs that produced the structured response.

Malformed provider output raises a parsing error during execution. Keep schemas
small and concrete: finite choices with {py:class}`Enum <autoform.Enum>`, bounded numbers with {py:class}`Int <autoform.Int>` or
{py:class}`Float <autoform.Float>`, and field descriptions with {py:class}`Doc <autoform.Doc>`.
