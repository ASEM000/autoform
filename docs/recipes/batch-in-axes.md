# Vectorize Inputs with `in_axes`

[`batch`](../concepts/transforms.md) needs to know which inputs are batched and
which inputs should be reused for every example. That is what `in_axes` describes.

## Broadcast One Input

```python
import autoform as af


def label(topic: str, prefix: str) -> str:
    return af.format("{}: {}", prefix, topic)


# topic is batched, prefix is reused for every topic
ir = af.trace(label)("recursion", "topic")
batched = af.batch(ir, in_axes=(True, False))
result = batched.call(["recursion", "gravity", "memoization"], "topic")

print(result)
```

`True` means "this leaf has a batch axis." `False` means "broadcast this leaf."

## Batch a Nested Input

`in_axes` can match a nested [pytree](../concepts/pytrees.md).

```python
import autoform as af


def render(request: dict[str, str]) -> str:
    return af.format("{}: {}", request["system"], request["topic"])


# the single function argument is a dict, so the axes sit inside a one-item tuple
example = {"system": "explain briefly", "topic": "recursion"}
axes = ({"system": False, "topic": True},)
requests = {"system": "explain briefly", "topic": ["recursion", "gravity"]}

ir = af.trace(render)(example)
batched = af.batch(ir, in_axes=axes)
print(batched.call(requests))
```

The output batch length is inferred from the batched leaves. All batched leaves
must agree on length. Broadcast leaves are passed through unchanged to each
per-example execution.

## Pair Two Batches

```python
import autoform as af


def score(answer: str, rubric: str) -> str:
    return af.format("answer: {}\nrubric: {}", answer, rubric)


# both leaves are batched, so examples are paired by position
ir = af.trace(score)("a", "r")
batched = af.batch(ir, in_axes=(True, True))
answers = ["short answer", "long answer"]
rubrics = ["prefer detail", "prefer brevity"]

print(batched.call(answers, rubrics))
```

Use `in_axes=True` when every input leaf is batched. Use an explicit pytree of
booleans when some leaves should be reused.
