# Schemas

An `autoform` schema is a [pytree](pytrees.md) whose leaves are schema specs. It is instance-first: the schema is a value shape, not a separate output class definition. {py:func}`lm_schema_call <autoform.lm_schema_call>` returns the same pytree structure, with each schema leaf replaced by a parsed value.

```python
answer_schema = {"text": af.Str(min=1), "score": af.Float(min=0, max=1)}
```

## Leaf Types

| Python | Args | Meaning |
| --- | --- | --- |
| {py:class}`Str <autoform.Str>` | `min=None, max=None, pattern=None` | A string, optionally constrained by length or regex. |
| {py:class}`Int <autoform.Int>` | `min=None, max=None` | An integer, optionally range constrained. |
| {py:class}`Float <autoform.Float>` | `min=None, max=None` | A number, optionally range constrained. |
| {py:class}`Bool <autoform.Bool>` | none | A boolean. |
| {py:class}`Enum <autoform.Enum>` | `*values` | One of a non-empty set of JSON scalar values of the same type. |

## Descriptions

Use {py:class}`Doc <autoform.Doc>` with the `@` operator to attach descriptions:

```python
schema = {
    "kind": af.Enum("summary", "definition") @ af.Doc("Kind."),
    "text": af.Str() @ af.Doc("Text."),
}
```

The descriptions become JSON Schema descriptions in the provider request.

## Pytree Shapes

A schema can have any [pytree](pytrees.md) shape that `autoform` can walk. It does not need a special schema class. Dictionaries, tuples, lists, and registered dataclasses all work.

```python
schema = {"route": (af.Enum("search", "done"), af.Str()), "answer": af.Str()}
```

{py:func}`lm_schema_call <autoform.lm_schema_call>` returns the same shape with schema leaves replaced by parsed values. The example above returns a dictionary whose `"route"` value is a tuple and whose `"answer"` value is a string.

Fixed-size repeated fields are just list-shaped schemas:

```python
score_schema = [af.Float(min=0, max=1)] * 4
schema = {"scores": score_schema}
```

This is a fixed-size schema. The parsed result has the same list shape:

```python
result = af.lm_schema_call(messages, model="gpt-5.5", schema=schema)
assert isinstance(result["scores"], list)
assert len(result["scores"]) == 4
```

For variable-length output, choose a bounded representation in the schema, such as a fixed number of slots plus a count or status field.

## Define a Custom Pytree

Schema trees can use any registered custom pytree. For dataclasses, use [Optree's dataclass integration](https://optree.readthedocs.io/en/latest/dataclasses.html):

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Decision:
    tool: str
    answer: str


decision_schema = Decision(tool=af.Enum("search", "done"), answer=af.Str())
```

The schema is the instance `decision_schema`, not the class `Decision`.

## Transform Schema Trees

Because schemas are [pytrees](pytrees.md), project code can build one base schema and derive call-specific variants with pytree utilities. `tree_map` changes the schema value before tracing or execution; it is not post-processing model output. Use the `autoform` namespace when mapping over registered dataclasses.

```python
import optree
import autoform as af


base_schema = {"answer": af.Str(), "confidence": af.Float(min=0, max=1)}

extract_schema = optree.tree_map(
    lambda leaf: leaf @ af.Doc("Extract only facts explicitly present in the input."),
    base_schema,
    namespace=af.PYTREE_NAMESPACE,
)

critique_schema = optree.tree_map(
    lambda leaf: leaf @ af.Doc("Critique the draft and report uncertainty conservatively."),
    base_schema,
    namespace=af.PYTREE_NAMESPACE,
)
```

Both calls return the same shape, so downstream code still reads `result["answer"]` and `result["confidence"]`. Only the schema guidance changes.

The same pattern works for dataclass-shaped schemas:

```python
plain_decision_schema = Decision(tool=af.Enum("search", "done"), answer=af.Str())

documented_decision_schema = optree.tree_map(
    lambda leaf: leaf @ af.Doc("Decision field."),
    plain_decision_schema,
    namespace=af.PYTREE_NAMESPACE,
)
```

This is useful when several calls share the same shape but differ by descriptions, ranges, or other schema leaf metadata.

The same shape-preserving rule is what makes schemas compose with `autoform` transforms:

- {py:func}`batch <autoform.batch>` returns a batched version of the schema-shaped output.
- {py:func}`pullback <autoform.pullback>` accepts feedback with the same schema shape.
- `tree_map` can derive schema variants before the IR is traced or executed.

## Schema Calls

Pass the schema value to {py:func}`lm_schema_call <autoform.lm_schema_call>`. The result has the same pytree shape, with schema leaves replaced by parsed provider output.[^schema-provider-support]

```python
import autoform as af


schema = {"text": af.Str(), "score": af.Float(min=0, max=1)}

messages = [dict(role="user", content="Explain recursion.")]
result = af.lm_schema_call(messages, model="gpt-5.5", schema=schema)

print(result["text"], result["score"])
```

For provider routing, retries, aliases, or fallback chains, use {py:func}`lm_client <autoform.lm_client>`. See [Configure LiteLLM Routing](../recipes/litellm-config.md).

## Schema Pullback

When a schema call is used inside {py:func}`pullback <autoform.pullback>`, feedback is still text. The output cotangent should match the schema shape, with text feedback at leaves. For example, feedback might be `{"text": "too terse", "score": "overconfident"}`.

If the provider returns malformed structured output, {py:func}`lm_schema_call <autoform.lm_schema_call>` raises while parsing the response.

[^schema-provider-support]: {py:func}`lm_schema_call <autoform.lm_schema_call>` sends a JSON Schema response format through the active LiteLLM client. Provider and model support for strict structured output can differ, so use a model route that supports the requested response format when schemas are required.
