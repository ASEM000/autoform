# Pytrees

Transforms need to walk values. A string is one leaf. A tuple is a container. A
registered dataclass can be a container too.

That container/leaf view is called a pytree. `autoform` uses pytrees so
transforms can apply per-leaf logic to user-defined structures.

`autoform` uses [Optree's pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html)
for traversal and registration.

## Registration

Without pytree registration, a custom object is opaque. {py:func}`batch <autoform.batch>`
cannot know which fields are batched. {py:func}`pullback <autoform.pullback>` cannot
route cotangents into the right fields.

With registration, the object becomes part of the same tree machinery as tuples
and dictionaries.

## {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`

`autoform` reserves {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`.
Register project dataclasses in that namespace so `autoform` and project code
agree on the same tree rules.

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

That is the canonical pattern: use the same {py:data}`PYTREE_NAMESPACE <autoform.PYTREE_NAMESPACE>`
when registering dataclasses with [Optree's dataclass decorator](https://optree.readthedocs.io/en/latest/dataclasses.html)
and when calling [Optree pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html).

## Static Metadata Fields

Not every dataclass field should be a transform leaf. Use
`optree.dataclasses.field(pytree_node=False)` for static metadata that belongs
to the object but should not be mapped, batched, or receive feedback.

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

`values` is a pytree child, so transforms and Optree utilities can walk it.
`shape` and `dtype` are metadata, so they are stored in the tree structure and
preserved when the object is rebuilt. This is useful for array shapes, dtypes,
model names, backends, and other configuration that should remain fixed while
the leaves change.

## Manual Flatten / Unflatten

For non-dataclass classes, register the class with explicit flatten and
unflatten functions.

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


optree.register_pytree_node(State, flatten_state, unflatten_state, namespace=af.PYTREE_NAMESPACE)

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

The unflatten rule receives the metadata and transformed children, then rebuilds
the object.

## Method-Bearing Pytrees

A registered class can also have methods. The fields are pytree leaves; the
methods are ordinary Python behavior.

This makes an object a transform-visible container: fields hold the values that
transforms can walk, while methods provide the callable surface that builds the
text-space program.

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Explainer:
    instruction: str
    style: str

    def __call__(self, topic: str) -> str:
        return af.format("{}\nstyle: {}\ntopic: {}", self.instruction, self.style, topic)
```

The placement of the object matters:

- As an explicit input or output, transforms can see the registered fields.
- As a closed-over value, the object behaves like trace-time configuration.

Use the explicit-input form when {py:func}`batch <autoform.batch>`,
{py:func}`pullback <autoform.pullback>`, or {py:func}`pushforward <autoform.pushforward>`
should act on the module fields.

The [object-oriented module recipe](../recipes/pytree-modules.md) applies the
same dataclass pattern to method-bearing module objects.

## Transform Behavior

Once `State` is a pytree, an IR can accept and return it. Transforms see the
leaves:

- {py:func}`batch <autoform.batch>` can vectorize over `State(topic=[...], draft=[...])`.
- {py:func}`pullback <autoform.pullback>` can return field-shaped feedback such as `State(topic="be more specific", draft="too terse")`.
- {py:func}`while_loop <autoform.while_loop>` can carry structured state as long as the body input and output structures match.

## Leaf Guidelines

Use leaves that are ordinary values or traced values:

- strings, ints, floats, bools;
- schema outputs;
- other registered pytrees;
- values produced by `autoform` primitives.

Avoid leaves that are runtime resources or trace-local implementation details:

- open files;
- sockets;
- closures;
- tracers leaked from another trace.

Pytrees describe structure. They do not make a value serializable, replayable, or
safe to mutate.

Schemas are adjacent but different: a schema describes structured LM output. A
pytree describes how `autoform` walks user data. See [Schemas](schemas.md) for
schema output.
