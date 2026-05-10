# Pytrees

Transforms need to walk values. A string is one leaf. A tuple is a container. A registered dataclass can be a container too.

That container/leaf view is called a pytree. `autoform` uses pytrees so transforms can apply per-leaf logic to user-defined structures.

`autoform` uses [Optree's pytree utilities](https://optree.readthedocs.io/en/latest/pytree.html) for traversal and registration.

## Why Registration Matters

Without pytree registration, a custom object is opaque. `batch` cannot know which fields are batched. `pullback` cannot route cotangents into the right fields.

With registration, the object becomes part of the same tree machinery as tuples and dictionaries.

## The Namespace

`autoform` reserves `af.PYTREE_NAMESPACE`. Register project dataclasses in that namespace so `autoform` and project code agree on the same tree rules.

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

That is the canonical pattern: pass `af.PYTREE_NAMESPACE` to [Optree's dataclass decorator](https://optree.readthedocs.io/en/latest/dataclasses.html).

## What This Enables

Once `State` is a pytree, an IR can accept and return it. Transforms see the leaves:

- `batch` can vectorize over `State(topic=[...], draft=[...])`.
- `pullback` can return field-shaped feedback such as `State(topic="be more specific", draft="too terse")`.
- `while_loop` can carry structured state as long as the body input and output structures match.

## Good Leaves

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

Pytrees describe structure. They do not make a value serializable, replayable, or safe to mutate.

Schemas are adjacent but different: a schema describes structured LM output. A pytree describes how `autoform` walks user data. See [Schemas](schemas.md) for schema output.
