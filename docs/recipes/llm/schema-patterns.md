# Schema Patterns

Structured model results can contain routing choices, nested arguments, and numerical scores.
Build structured results from {py:func}`fill <autoform.lm.fill>` specifications in ordinary containers or registered dataclasses.
These examples assume familiarity with [Getting Started](../../getting-started.md).

```{admonition} Concept
[Schemas](../../concepts/schemas.md) · [Pytrees](../../concepts/pytrees.md) · [Transforms](../../concepts/transforms.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Routing

To represent a finite choice, use an enum:

```python
import optree
import autoform as af

model = "model-name"


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Route:
    tool: str
    answer: str


route_schema = Route(
    tool=af.lm.Enum(
        "search",
        "done",
        desc="Select search if information is missing.",
    ),
    answer=af.lm.Str(desc="Answer only for the done route."),
)


def choose_route(question: str) -> Route:
    content = dict(question=question, route=route_schema)
    return af.lm.fill(content, model=model)["route"]


route_ir = af.trace(choose_route)("question text")
```

The result is a `Route` with generated fields. Its `tool` can select a branch through {py:func}`switch <autoform.switch>`.
See [Tool-Use Agent](tool-use-agent.md) for a complete loop.

## Nested Arguments

To structure the arguments to a tool call, use a nested dataclass:

```python
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class SearchArgs:
    query: str
    limit: int


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class SearchDecision:
    tool: str
    args: SearchArgs


decision_schema = SearchDecision(
    tool=af.lm.Enum("search", "done"),
    args=SearchArgs(query=af.lm.Str(), limit=af.lm.Int(min=1, max=5)),
)


def choose_search(question: str) -> SearchDecision:
    content = dict(question=question, decision=decision_schema)
    return af.lm.fill(content, model=model)["decision"]
```

Each time this is called, it will return a `SearchDecision` with the nested dataclass `SearchArgs`. Note that the generated field `limit` will be an integer between 1 and 5.

## Field Instructions

Alternatively, the same result structure can be used with different descriptions:

```python
extract_schema = {
    "answer": af.lm.Str(desc="extraction instructions"),
    "confidence": af.lm.Float(min=0, max=1, desc="confidence instructions"),
}
critique_schema = {
    "answer": af.lm.Str(desc="critique instructions"),
    "confidence": af.lm.Float(min=0, max=1, desc="confidence instructions"),
}
```

Both results have an `answer` string and a `confidence` float. Downstream code can use the same fields while the calls receive different instructions.

## Presence Fields

Represent an optional value with a status and a value field:

```python
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class MaybeAnswer:
    state: str
    text: str


maybe_schema = MaybeAnswer(
    state=af.lm.Enum("present", "absent"),
    text=af.lm.Str(desc="Return an empty string when absent."),
)
```

This keeps the container structure fixed for transforms. Caller code interprets `state` rather than expecting a field to disappear.

## Field Feedback

To give structured feedback, construct a container that matches the output, but with the leaf types as the cotangent type of the output:

```python
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    score: float


summary_schema = Summary(
    title=af.lm.Str(max=80, desc="title instructions"),
    score=af.lm.Float(min=0, max=1, desc="clarity instructions"),
)


def summarize(topic: str) -> Summary:
    content = dict(topic=topic, summary=summary_schema)
    return af.lm.fill(content, model=model)["summary"]


ir = af.trace(summarize)("topic text")
feedback = Summary(title="title feedback", score=1.0)
output, (topic_feedback,) = af.pullback(ir).call(("topic text",), feedback)
print(output)
print(topic_feedback)
```

The title receives a textual critique. The score receives a numerical cotangent; `1.0` weights its contribution to the pullback.
The LM rule uses both to generate feedback for the string input `topic`.
See [Schemas](../../concepts/schemas.md) for validation and description behavior.
