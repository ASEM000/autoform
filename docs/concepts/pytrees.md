# Pytrees

A pytree is a nested structure of containers and leaves. A string is a leaf; a tuple or registered dataclass is a container. `autoform` uses this structure to batch selected fields and return feedback in the same container shape.

`autoform` uses [Optree's pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html)
for traversal and registration.

```{raw} html
:file: ../assets/pytree-structure.svg
```

## Registration

Without registering a pytree, an object is considered a leaf, which is opaque to `autoform`. This means when using transforms like {py:func}`batch <autoform.batch>`, it won't know which fields of the object should be batched together. Similarly, when using transforms like {py:func}`pullback <autoform.pullback>`, it won't know how to assign cotangents to the fields of the object.

By registering a pytree, it can be used in the same way as other pytrees, like tuples and dicts.

## Namespace

`autoform` reserves {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`.
Register project dataclasses in that namespace so `autoform` and project code
agree on the same tree rules:

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    topic: str
    draft: str


state = State(topic="dna", draft="short")
upper = optree.tree_map(str.upper, state, namespace=af.PYTREE_NAMESPACE)

assert upper == State(topic="DNA", draft="SHORT")
```

Use the same namespace for both operations: use the same {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`
when registering dataclasses with [Optree's dataclass decorator](https://optree.readthedocs.io/en/latest/dataclasses.html)
and when calling [Optree pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html).

## Static Metadata Fields

Not every dataclass field should be a transform leaf. Use
`optree.dataclasses.field(pytree_node=False)` for static metadata that belongs
to the object but should not be mapped, batched, or receive feedback:

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class F32Array:
    values: tuple[float, ...]
    shape: tuple[int, ...] = optree.dataclasses.field(pytree_node=False)
    dtype: str = optree.dataclasses.field(default="float32", pytree_node=False)


array = F32Array(values=(1.0, 2.0), shape=(2,))
scaled = optree.tree_map(lambda x: x * 10, array, namespace=af.PYTREE_NAMESPACE)

assert scaled == F32Array(values=(10.0, 20.0), shape=(2,))
```

Here, `values` will be considered a child of the pytree, and will be mapped over when using transforms or other utilities from [optree](https://optree.readthedocs.io/en/latest/pytree.html). However, `shape` and `dtype` will be considered metadata, and will be stored in the structure of the pytree, and returned when flattening and unflattening. This is useful for storing metadata about the pytree, like the shape and dtype of an array, or the name and backend of a model, etc.

## Manual Flatten / Unflatten

For other classes, define flatten and unflatten functions to register the class as a pytree.

```python
import optree
import autoform as af


class State:
    def __init__(self, topic: str, draft: str):
        self.topic = topic
        self.draft = draft


def flatten_state(state: State):
    children = (state.topic, state.draft)
    metadata = None
    return children, metadata


def unflatten_state(metadata, children):
    topic, draft = children
    return State(topic=topic, draft=draft)


optree.register_pytree_node(
    State,
    flatten_state,
    unflatten_state,
    namespace=af.PYTREE_NAMESPACE,
)

state = State(topic="dna", draft="short")
upper = optree.tree_map(str.upper, state, namespace=af.PYTREE_NAMESPACE)

assert upper.topic == "DNA"
assert upper.draft == "SHORT"
```

The flatten rule returns two things:

- `children`: values that Optree and `autoform` should walk recursively;
- `metadata`: hashable static data stored in the tree spec and passed back to
  the unflatten rule.

`optree.dataclasses.field(pytree_node=False)` is the dataclass form of putting a
field in `metadata` instead of `children`.

The unflatten function receives the metadata and transformed children of the pytree, then returns a new instance of the class.

## Method-Bearing Pytrees

A registered class can also have methods. The fields are pytree leaves; the
methods are ordinary Python behavior.

Methods can use the fields to build a program. Registration lets transforms act on those fields when the object is an input or output:

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Explainer:
    instruction: str
    style: str

    def __call__(self, topic: str) -> str:
        return self.instruction + "\nStyle: " + self.style + "\nTopic: " + topic
```

The placement of the object matters:

- As an explicit input or output, transforms can see the registered fields.
- As a closed-over value, the object behaves like trace-time configuration.

Use the explicit-input form when {py:func}`batch <autoform.batch>`,
{py:func}`pullback <autoform.pullback>`, or {py:func}`pushforward <autoform.pushforward>`
should act on the module fields.

The [pytree module recipe](../recipes/core/pytree-modules.md) applies the
same dataclass pattern to method-bearing module objects.

## Transform Behavior

Once `State` is a pytree, an IR can accept and return it. Transforms see the
leaves:

- {py:func}`batch <autoform.batch>` can vectorize over `State(topic=[...], draft=[...])`.
- {py:func}`pullback <autoform.pullback>` can return field-shaped feedback such as `State(topic="topic feedback", draft="draft feedback")`.
- {py:func}`while_loop <autoform.while_loop>` can carry structured state as long as the body input and output structures match.
- {py:func}`fixpoint <autoform.fixpoint>` can iterate structured state as long as the step input and output structures match.

## Leaves

In general, leaves of pytrees should be values, or traced values.

- strings, ints, floats, bools;
- schema outputs;
- other registered pytrees;
- values produced by `autoform` primitives.

Leaves should not be things that are only meaningful at runtime, or only in the context of a trace.

- open files;
- sockets;
- closures;
- tracers leaked from another trace.

Pytrees describe structure. Pytree registration does not make a value serializable, replayable, or
safe to mutate.

Schemas are adjacent but different: a schema describes structured LM output. A
pytree describes how `autoform` walks user data. See [Schemas](schemas.md) for
schema output.
