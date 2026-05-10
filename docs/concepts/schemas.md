# Schemas

An `autoform` schema is a Python instance that describes the structured value expected from an LM. It is instance-first: the schema is a value shape, not a separate output class definition.

```python
answer_schema = {"text": af.Str(min=1), "score": af.Float(min=0, max=1)}
```

The result of `lm_schema_call` has the same [pytree](pytrees.md) shape, with schema leaves replaced by parsed values.

## Leaf Types

- `Str(min=None, max=None, pattern=None)`: a string, optionally constrained by length or regex.
- `Int(min=None, max=None)`: an integer, optionally range constrained.
- `Float(min=None, max=None)`: a number, optionally range constrained.
- `Bool()`: a boolean.
- `Enum(*values)`: one of a non-empty set of JSON scalar values of the same type.

## Descriptions

Use `Doc` with the `@` operator to attach descriptions:

```python
kind = af.Enum("summary", "definition") @ af.Doc("Kind of answer.")
text = af.Str() @ af.Doc("Short answer text.")
schema = {"kind": kind, "text": text} @ af.Doc("Answer object.")
```

The descriptions become JSON Schema descriptions in the provider request.

## Pytree Composition

Schema trees can use [Optree-registered dataclasses](https://optree.readthedocs.io/en/latest/dataclasses.html):

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Decision:
    tool: str
    answer: str


tool = af.Enum("search", "done") @ af.Doc("Tool to call next.")
answer = af.Str() @ af.Doc("Answer when done.")
decision_schema = Decision(tool=tool, answer=answer)
```

The schema is the instance `decision_schema`, not the class `Decision`.

## Schema Calls

`lm_schema_call` uses the active LM client. By default, that is [LiteLLM's completion API](https://docs.litellm.ai/docs/completion), so pass a model name configured in the active environment. For routing, retries, aliases, or fallback chains, install a [`litellm.Router`](https://docs.litellm.ai/docs/routing) with [`af.lm_client(...)`](../recipes/litellm-config.md).

```python
import autoform as af


text = af.Str() @ af.Doc("One-sentence answer.")
score = af.Float(min=0, max=1) @ af.Doc("Confidence.")
schema = {"text": text, "score": score}

messages = [dict(role="user", content="Explain recursion.")]
result = af.lm_schema_call(messages, model="gpt-5.2", schema=schema)

print(result["text"], result["score"])
```

With a router:

```python
from litellm import Router
import autoform as af


params = {"model": "gpt-5.2"}
model_list = [dict(model_name="docs-model", litellm_params=params)]
router = Router(model_list=model_list)

with af.lm_client(router):
    messages = [dict(role="user", content="Explain recursion.")]
    result = af.lm_schema_call(messages, model="docs-model", schema=schema)
```

Under the hood, `lm_schema_call` sends a JSON Schema response format, parses the JSON response, and rebuilds the original pytree shape.

## Schema Pullback

When a schema call is used inside [`pullback`](transforms.md), feedback is still text. The output cotangent should match the schema shape, with text feedback at leaves. For example, feedback might be `{"text": "too terse", "score": "overconfident"}`.

If the provider returns malformed structured output, `lm_schema_call` raises while parsing the response.
