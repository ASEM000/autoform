# Structured Results

Plain `lm_call` returns text. That is fine for prose, but many LM programs need a value the rest of Python can use without another parsing step: a label, a score, a route decision, or a short extracted field.

Use [`lm_schema_call`](../concepts/schemas.md) when the LM output should have a known shape.

```python
import optree
import autoform as af


# register the dataclass in autoform's pytree namespace
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Summary:
    title: str
    kind: str
    confidence: float
```

That two-line pattern, the decorator plus `namespace=af.PYTREE_NAMESPACE`, registers the dataclass as an `autoform` [pytree](../concepts/pytrees.md). The schema is an instance of that class:

```python
# build schema leaves first so the schema instance stays compact
title = af.Str(max=80) @ af.Doc("Short title.")
kind = af.Enum("definition", "analogy", "warning") @ af.Doc("Best category.")
confidence = af.Float(min=0, max=1) @ af.Doc("Confidence score.")
summary_schema = Summary(title=title, kind=kind, confidence=confidence)
```

Now write the function normally:

```python
def summarize(topic: str) -> Summary:
    prompt = af.format("Summarize {} for a technical audience.", topic)
    msg = dict(role="user", content=prompt)
    # return a summary value, not a raw string
    return af.lm_schema_call([msg], model="gpt-5.2", schema=summary_schema)
```

Trace it:

```python
# trace once with a placeholder topic
ir = af.trace(summarize)("placeholder topic")
```

The schema is a static parameter of the recorded `lm_schema_call`. The returned fields are still part of the [IR](../concepts/the-ir.md) output tree, so transforms can walk them like ordinary Python structure.

Execute it with a configured provider:

```python
# execute with a real topic and use the returned fields
result = ir.call("recursion")
print(result.title)
print(result.kind)
print(result.confidence)
```

The returned value is a `Summary`, not a string blob.

Schemas also work with transforms:

- `batch(ir)` returns a batched `Summary` tree.
- `pullback(ir)` accepts feedback with the same schema shape.
- `sched(ir)` can schedule schema calls like any other primitive.

For `pullback`, feedback lands on fields:

```python
# feedback has the same shape as the structured output
feedback = Summary(title="too vague", kind="classification is wrong", confidence="overconfident")
output, (topic_hint,) = af.pullback(ir).call(("recursion",), feedback)
```

The field feedback is summarized into a prompt-feedback request for the input message. For the full schema model, see [Schemas](../concepts/schemas.md).
