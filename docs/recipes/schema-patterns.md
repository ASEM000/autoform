# Schema Patterns

[`lm_schema_call`](../concepts/schemas.md) returns a structured value whose
shape can be transformed like any other [pytree](../concepts/pytrees.md).

## Enum Routing

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Route:
    tool: str
    answer: str


tool = af.Enum("search", "done") @ af.Doc("Next action.")
answer = af.Str() @ af.Doc("Final answer when tool is done.")
route_schema = Route(tool=tool, answer=answer)


def choose_route(question: str) -> Route:
    prompt = af.format("Choose search or done for this question:\n{}", question)
    msg = dict(role="user", content=prompt)
    return af.lm_schema_call([msg], model="gpt-5.2", schema=route_schema)
```

Use `Enum` for finite decisions that should feed `switch`, status fields, or
other control values.

## Nested Arguments

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


query = af.Str() @ af.Doc("Search query.")
limit = af.Int(min=1, max=5) @ af.Doc("Number of results.")
args_schema = SearchArgs(query=query, limit=limit)
tool_schema = af.Enum("search", "done") @ af.Doc("Selected tool.")
decision_schema = SearchDecision(tool=tool_schema, args=args_schema)


def choose_search(question: str) -> SearchDecision:
    prompt = af.format("Choose the next tool call for:\n{}", question)
    msg = dict(role="user", content=prompt)
    return af.lm_schema_call([msg], model="gpt-5.2", schema=decision_schema)
```

Nested dataclasses keep related fields together and still register as one
pytree-shaped schema.

## Optional-Like Fields

There is no special optional schema node. Model presence explicitly.

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class MaybeAnswer:
    state: str
    text: str


state = af.Enum("present", "absent") @ af.Doc("Whether text is present.")
text = af.Str() @ af.Doc("Answer text, or empty string when absent.")
maybe_schema = MaybeAnswer(state=state, text=text)
```

This keeps the output shape stable for `batch`, `pullback`, and `switch`.

## Field Feedback

Structured outputs also work with `pullback`.

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    score: float


title = af.Str(max=80) @ af.Doc("Short title.")
score = af.Float(min=0, max=1) @ af.Doc("Confidence.")
summary_schema = Summary(title=title, score=score)


def summarize(topic: str) -> Summary:
    prompt = af.format("Summarize {}.", topic)
    msg = dict(role="user", content=prompt)
    return af.lm_schema_call([msg], model="gpt-5.2", schema=summary_schema)


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
small and concrete: finite choices with `Enum`, bounded numbers with `Int` or
`Float`, and field descriptions with `Doc`.
