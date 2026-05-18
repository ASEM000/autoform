# Build an Array Extension

```{admonition} Advanced
:class: info

This recipe uses `autoform.extend`, the low-level extension API. Use it when a
runtime value type should become part of the traced IR system.
```

`autoform` starts with text-space programs, but the extension API lets other
feedback spaces participate in the same IR machinery. A richer extension can
also define boundaries where spaces meet, for example turning textual feedback
into numerical cotangents, or summarizing numerical signals back into text-space
feedback.

Arrays are not built in. This recipe uses NumPy as the concrete runtime, but the
same extension pattern applies to any value space that can define trace-time
avals, zeros, cotangent accumulation, and primitive rules.

```{admonition} Concept
[Primitives](../concepts/primitives.md) · [Transforms](../concepts/transforms.md) ·
[Pytrees](../concepts/pytrees.md)
```

This example keeps arrays as atomic leaves. The {py:func}`batch <autoform.batch>`
example batches over a Python list of arrays, not over the leading axis of one
stacked array.

## Define the Array Domain

```python
import functools as ft

import numpy as np

import autoform as af
import autoform.extend as afe


class ArrayAVal(afe.AVal):
    __slots__ = ["shape", "dtype"]

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def __repr__(self):
        return f"ArrayAVal(shape={self.shape!r}, dtype={self.dtype!r})"

    def __eq__(self, other):
        return (
            isinstance(other, ArrayAVal) and self.shape == other.shape and self.dtype == other.dtype
        )

    def __hash__(self):
        return hash((type(self), self.shape, self.dtype.str))


def aval_rule(value):
    return ArrayAVal(value.shape, value.dtype)


afe.register_trace_type(np.ndarray, aval_rule)
afe.register_zero(ArrayAVal, lambda aval: np.zeros(aval.shape, dtype=aval.dtype))
afe.register_cotangent_accumulator(ArrayAVal, lambda cs, aval: sum(cs[1:], cs[0]))
```

`ArrayAVal` is the trace-time description. It carries only the information the
primitive rules need: shape and dtype. The zero and cotangent accumulator rules
make reverse-mode feedback concrete for array leaves.

## Register Binary Array Primitives

Each primitive needs rules for ordinary execution, abstraction, forward-mode AD,
reverse-mode AD, and batching.

```python
def array_aval(value):
    return value if isinstance(value, ArrayAVal) else afe.avalof(value)


def result_aval(x, y, op):
    ax = array_aval(x)
    ay = array_aval(y)
    result = op(np.ones(ax.shape, dtype=ax.dtype), np.ones(ay.shape, dtype=ay.dtype))
    return ArrayAVal(result.shape, result.dtype)


def register_binary(name, op, push_rule, pull_rule):
    prim = afe.Prim(name)

    def bind(x, y):
        return prim.bind((x, y))

    def impl(in_tree):
        return op(*in_tree)

    def abstract(in_tree):
        return result_aval(*in_tree, op)

    def push(in_tree):
        (x, y), (tx, ty) = in_tree
        tx, ty = afe.materialize((tx, ty))
        return bind(x, y), push_rule(x, y, tx, ty)

    def pull_fwd(in_tree):
        x, y = in_tree
        return bind(x, y), (x, y)

    def pull_bwd(in_tree):
        (x, y), g = in_tree
        return pull_rule(x, y, afe.materialize(g))

    def batch_rule(in_tree):
        batch_size, in_batched, in_values = in_tree
        if afe.batch_spec(in_values, in_batched) is None:
            return bind(*in_values), False

        x, y = in_values
        bx, by = in_batched
        x_at = ft.partial(afe.batch_index, x, bx)
        y_at = ft.partial(afe.batch_index, y, by)
        return [bind(x_at(i), y_at(i)) for i in range(batch_size)], True

    afe.impl_rules.set(prim, impl)
    afe.abstract_rules.set(prim, abstract)
    afe.push_rules.set(prim, push)
    afe.pull_fwd_rules.set(prim, pull_fwd)
    afe.pull_bwd_rules.set(prim, pull_bwd)
    afe.batch_rules.set(prim, batch_rule)
    return bind
```

The `bind` wrapper is the function traced programs call. During tracing it
stages a primitive equation; during execution the registered implementation rule
receives real NumPy arrays.

## Add Operators

```python
a_add = register_binary(
    "a_add",
    lambda x, y: x + y,
    lambda x, y, tx, ty: a_add(tx, ty),
    lambda x, y, g: (g, g),
)
a_sub = register_binary(
    "a_sub",
    lambda x, y: x - y,
    lambda x, y, tx, ty: a_sub(tx, ty),
    lambda x, y, g: (g, -g),
)
a_mul = register_binary(
    "a_mul",
    lambda x, y: x * y,
    lambda x, y, tx, ty: a_add(a_mul(tx, y), a_mul(x, ty)),
    lambda x, y, g: (a_mul(g, y), a_mul(g, x)),
)
a_div = register_binary(
    "a_div",
    lambda x, y: x / y,
    lambda x, y, tx, ty: a_div(a_sub(a_mul(tx, y), a_mul(x, ty)), a_mul(y, y)),
    lambda x, y, g: (a_div(g, y), -a_div(a_mul(g, x), a_mul(y, y))),
)
a_matmul = register_binary(
    "a_matmul",
    lambda x, y: x @ y,
    lambda x, y, tx, ty: a_add(a_matmul(tx, y), a_matmul(x, ty)),
    lambda x, y, g: (a_matmul(g, y.T), a_matmul(x.T, g)),
)

afe.register_add(ArrayAVal, a_add)
afe.register_sub(ArrayAVal, a_sub)
afe.register_mul(ArrayAVal, a_mul)
afe.register_div(ArrayAVal, a_div)
afe.register_matmul(ArrayAVal, a_matmul)
```

The last five registrations connect traced Python syntax to the primitives. For
example, `x + y` stages `a_add` when `x` has `ArrayAVal`.

## Use the Extension

```python
def f(x, y):
    return ((x + y) * y) / x


x = np.array([1.0, 2.0])
y = np.array([3.0, 4.0])
ir = af.trace(f)(x, y)

np.testing.assert_allclose(ir.call(x, y), f(x, y))

p_out, t_out = af.pushforward(ir).call((x, y), (np.ones_like(x), np.zeros_like(y)))
np.testing.assert_allclose(p_out, f(x, y))
np.testing.assert_allclose(t_out, -y * y / (x * x))

out, (dx, dy) = af.pullback(ir).call((x, y), np.ones_like(x))
np.testing.assert_allclose(out, f(x, y))
np.testing.assert_allclose(dx, -y * y / (x * x))
np.testing.assert_allclose(dy, (x + 2 * y) / x)
```

Batching uses an outer Python batch container. Each element is still a NumPy
array leaf.

```python
batched = af.batch(ir, in_axes=(True, True))
outs = batched.call([x, x + 1], [y, y + 1])

np.testing.assert_allclose(outs[0], f(x, y))
np.testing.assert_allclose(outs[1], f(x + 1, y + 1))
```

Matrix multiplication works through the same primitive pattern:

```python
def mm(a, b):
    return a @ b


a = np.eye(2)
b = np.array([[2.0, 0.0], [0.0, 3.0]])
mm_ir = af.trace(mm)(a, b)

pb_out, (da, db) = af.pullback(mm_ir).call((a, b), np.ones((2, 2)))
np.testing.assert_allclose(pb_out, a @ b)
np.testing.assert_allclose(da, np.ones((2, 2)) @ b.T)
np.testing.assert_allclose(db, a.T @ np.ones((2, 2)))
```

This is intentionally not a full NumPy backend. Broadcasting pullbacks,
reductions, dtype policy, scalar promotion, and stacked-array batch semantics
need additional rules. The point of the recipe is the extension shape: define an
aval, register value behavior, define primitives, then attach transform rules.
