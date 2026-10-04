# Schemas

A schema describes the values a language model should generate.
{py:func}`fill <autoform.lm.fill>` accepts a [pytree](pytrees.md) containing context values and specifications.
```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

The context is preserved, and each specification is replaced by a parsed value:

```python
import autoform as af

content = dict(
    topic="topic text",
    answer=af.lm.Str(desc="answer instructions"),
    score=af.lm.Float(min=0, max=1, desc="confidence instructions"),
)
result = af.lm.fill(content, model="model-name")
print(result["answer"], result["score"])
```

The result has the same dictionary structure as `content`.
Its `topic` remains `"topic text"`; `answer` is a string and `score` is a float.
The model route must support the JSON Schema response format used by the
[LiteLLM Responses API](https://docs.litellm.ai/docs/response_api).

```{raw} html
:file: ../assets/schema-fill.svg
```

## Specifications

The built-in specifications describe scalar values:

| Specification | Parameters | Generated value |
| --- | --- | --- |
| {py:class}`Str <autoform.lm.Str>` | `desc`, `min`, `max`, `pattern` | A string with optional length and pattern constraints. |
| {py:class}`Int <autoform.lm.Int>` | `desc`, `min`, `max` | An integer with optional bounds. |
| {py:class}`Float <autoform.lm.Float>` | `desc`, `min`, `max` | A floating-point number with optional bounds. |
| {py:class}`Bool <autoform.lm.Bool>` | `desc` | A boolean. |
| {py:class}`Enum <autoform.lm.Enum>` | `*values`, `desc` | One value from a non-empty set of JSON scalar values of the same type. |

Parameters other than enum values are keyword-only.
Specifications validate the parsed result. Malformed JSON, incorrect types, or values outside the constraints raise an error during execution.

## Descriptions

Use the `desc` keyword argument to provide instructions to the model for how to generate a value for that field:

```python
schema = {
    "kind": af.lm.Enum("summary", "definition", desc="kind instructions"),
    "text": af.lm.Str(desc="answer instructions"),
}
```

These descriptions are used in the JSON Schema description field when making a request to the provider. The expression `spec @ description` returns a copy of the specification with a different description:

```python
def explain(topic: str, instruction: str) -> str:
    content = dict(topic=topic, answer=af.lm.Str() @ instruction)
    return af.lm.fill(content, model="model-name")["answer"]


ir = af.trace(explain)("topic text", "answer instructions")
```

Here `instruction` is a runtime input. Its text can change between calls and receive feedback through a pullback.

## Containers

Specifications can appear in dictionaries, tuples, lists, and registered dataclasses.
A fixed list of four specifications generates four values:

```python
content = dict(topic="topic text", scores=[af.lm.Float(min=0, max=1)] * 4)
result = af.lm.fill(content, model="model-name")
assert len(result["scores"]) == 4
```

Currently, the structure of the container must be fixed, but it is possible to make something close to a variable length result by using a bounded number of slots in a container along with a count or a status.

Register dataclasses in the `autoform` pytree namespace before using instances as containers:

```python
import optree


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Decision:
    tool: str
    answer: str


decision = Decision(tool=af.lm.Enum("search", "done"), answer=af.lm.Str())
```

The specifications belong to the instance `decision`. The class defines the returned container.
See [Pytrees](pytrees.md) for registration details.

## Feedback

A pullback receives feedback with the same container structure as the program output.
The feedback type comes from each output leaf's registered cotangent space.
For an output such as `{"text": "answer text", "score": 0.8}`, use text feedback for `text` and a numerical cotangent for `score`:

```python
feedback = {"text": "answer feedback", "score": 1.0}
```

The LM backward rule uses that feedback to generate cotangents for the inputs.
A string input receives text feedback; a floating-point input receives a numerical cotangent.
See [Transforms](transforms.md) and [Schema Patterns](../recipes/llm/schema-patterns.md).
