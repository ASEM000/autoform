# Batched Inputs

A batch can vary some inputs while reusing others. Use `in_axes` with {py:func}`batch <autoform.batch>` to select which input leaves vary for each example.

```{admonition} Concept
[Transforms](../../concepts/transforms.md) · [Pytrees](../../concepts/pytrees.md)
```

## Shared Input

Reuse one prefix for every topic:

```python
import autoform as af


def label(topic: str, prefix: str) -> str:
    return prefix + ": " + topic


# topic is batched, prefix is reused for every topic
ir = af.trace(label)("recursion", "topic")
batched = af.batch(ir, in_axes=(True, False))
result = batched.call(["recursion", "gravity", "memoization"], "topic")

print(result)
```

The result is `["topic: recursion", "topic: gravity", "topic: memoization"]`.
`True` marks a batched leaf; `False` reuses the same value for every example.

## Nested Input

For a dictionary input, keep the instruction shared and batch only the topic:

```python
import autoform as af


def render(request: dict[str, str]) -> str:
    return request["system"] + ": " + request["topic"]


# a single dict argument needs axes inside a one-item tuple
example = {"system": "answer instructions", "topic": "recursion"}
axes = ({"system": False, "topic": True},)
requests = {"system": "answer instructions", "topic": ["recursion", "gravity"]}

ir = af.trace(render)(example)
batched = af.batch(ir, in_axes=axes)
print(batched.call(requests))
```

The output batch length is inferred from the batched leaves. All batched leaves
must agree on length. Broadcast leaves are passed through unchanged to each
per-example execution.

## Paired Inputs

Pair each answer with the rubric at the same position:

```python
import autoform as af


def score(answer: str, rubric: str) -> str:
    return "answer: " + answer + "\nrubric: " + rubric


# both leaves are batched, so examples are paired by position
ir = af.trace(score)("a", "r")
batched = af.batch(ir, in_axes=(True, True))
answers = ["answer text 1", "answer text 2"]
rubrics = ["rubric instructions 1", "rubric instructions 2"]

print(batched.call(answers, rubrics))
```

Note that when all leaves of the input will be batched, `in_axes=True` can be used instead of constructing a pytree of booleans. However, when some leaves will be reused, a pytree of booleans must be used to indicate which leaves should be batched.
