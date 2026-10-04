# Array Extension

```{admonition} Advanced
:class: info

This recipe uses `autoform.extend`, the low-level extension API. Use it when a
runtime value type should become part of the traced IR system.
```

To add support for a new array type in the tracing and transformations, one needs to register an abstract value type and a set of rules for operating on those values. Here’s a recipe that demonstrates how to add support for NumPy arrays, using numerical arrays for tangents and cotangents.

Arrays are not a built-in feature, so this is a recipe. It uses NumPy arrays as a concrete runtime, but could be adapted to other types by providing different implementations of abstract value types, zero creation, feedback accumulation, primitive rules, etc.

```{admonition} Concept
[Primitives](../../concepts/primitives.md) · [Transforms](../../concepts/transforms.md) ·
[Pytrees](../../concepts/pytrees.md)
```

This example keeps arrays as atomic leaves. The {py:func}`batch <autoform.batch>`
example batches over a Python list of arrays, not over the leading axis of one
stacked array.

## Abstract Value

Describe an array by its shape and dtype:

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
        return type(self) is type(other) and self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self):
        return hash((type(self), self.shape, self.dtype.str))

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def accum(self, cotangents):
        return sum(cotangents[1:], cotangents[0])


def aval_rule(value):
    return ArrayAVal(value.shape, value.dtype)


afe.register_trace_type(np.ndarray, aval_rule)
afe.primal_s.set(ArrayAVal, lambda aval: aval)
afe.tangent_s.set(ArrayAVal, lambda aval: aval)
afe.cotangent_s.set(ArrayAVal, lambda aval: aval)
```

`ArrayAVal` is the trace-time description. It carries only the information the
primitive rules need: shape and dtype. The `zero()` method constructs a zero
array with that shape and dtype. The `accum()` method combines feedback
contributions for array leaves.

## Operation Rules

Each primitive needs rules for ordinary execution, abstraction, forward-mode AD,
reverse-mode AD, and batching. The following helper registers those rules for one binary operation:

```python
def array_aval(value):
    return value if isinstance(value, ArrayAVal) else afe.avalof(value)


def result_aval(x, y, op):
    ax = array_aval(x)
    ay = array_aval(y)
    result = op(
        np.ones(ax.shape, dtype=ax.dtype),
        np.ones(ay.shape, dtype=ay.dtype),
    )
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
        tx, ty = afe.materialize_zeros((tx, ty))
        return bind(x, y), push_rule(x, y, tx, ty)

    def pull_fwd(in_tree):
        x, y = in_tree
        return bind(x, y), (x, y)

    def pull_bwd(in_tree):
        (x, y), g = in_tree
        return pull_rule(x, y, afe.materialize_zeros(g))

    def batch_rule(in_tree):
        batch_size, in_batched, in_values = in_tree
        if afe.batch_spec(in_values, in_batched) is None:
            return bind(*in_values), False

        x, y = in_values
        bx, by = in_batched
        x_at = ft.partial(afe.batch_index, x, bx)
        y_at = ft.partial(afe.batch_index, y, by)
        return [bind(x_at(i), y_at(i)) for i in range(batch_size)], True

    afe.register_impl(prim, impl)
    afe.register_abstract(prim, abstract)
    afe.register_pushforward(prim, push)
    afe.register_pullback_fwd(prim, pull_fwd)
    afe.register_pullback_bwd(prim, pull_bwd)
    afe.register_batch(prim, batch_rule)
    return bind
```

The `bind` wrapper is the function traced programs call. During tracing it
stages a primitive equation; during execution the registered implementation rule
receives real NumPy arrays.

## Operators

Register arithmetic and matrix multiplication, then connect these operations to Python operators:

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
    lambda x, y, g: (g, a_neg(g)),
)


def a_neg(x):
    return a_sub(x, a_add(x, x))


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
    lambda x, y, g: (a_div(g, y), a_neg(a_div(a_mul(g, x), a_mul(y, y)))),
)
a_matmul = register_binary(
    "a_matmul",
    lambda x, y: x @ y,
    lambda x, y, tx, ty: a_add(a_matmul(tx, y), a_matmul(x, ty)),
    lambda x, y, g: (a_matmul(g, y.T), a_matmul(x.T, g)),
)

afe.register_dunder(afe.Dunder.ADD, ArrayAVal, a_add)
afe.register_dunder(afe.Dunder.SUB, ArrayAVal, a_sub)
afe.register_dunder(afe.Dunder.MUL, ArrayAVal, a_mul)
afe.register_dunder(afe.Dunder.DIV, ArrayAVal, a_div)
afe.register_dunder(afe.Dunder.MATMUL, ArrayAVal, a_matmul)
```

The final registrations connect traced Python syntax to the primitives through
one dunder rule table. For example, `x + y` stages `a_add` when `x` has
`ArrayAVal`.

## Execution and AD

Check the output and derivatives of a two-input program:

```python
def f(x, y):
    return ((x + y) * y) / x


x = np.array([1.0, 2.0])
y = np.array([3.0, 4.0])
ir = af.trace(f)(x, y)

np.testing.assert_allclose(ir.call(x, y), f(x, y))

p_out, t_out = af.pushforward(ir).call(
    (x, y),
    (np.ones_like(x), np.zeros_like(y)),
)
np.testing.assert_allclose(p_out, f(x, y))
np.testing.assert_allclose(t_out, -y * y / (x * x))

out, (dx, dy) = af.pullback(ir).call((x, y), np.ones_like(x))
np.testing.assert_allclose(out, f(x, y))
np.testing.assert_allclose(dx, -y * y / (x * x))
np.testing.assert_allclose(dy, (x + 2 * y) / x)
```

Batching uses an outer Python batch container. Each element is still a NumPy
array leaf:

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

This example is limited to floating-point arrays of the same shape, and two-dimensional matrix multiplications. One would need to register additional rules to handle broadcasting in pullback, reduction operations, dtype polymorphism and scalar promotion, to define batch semantics over stacked arrays rather than lists of arrays, etc. But regardless of the details of those rules, the rules would be registered in the same order as here: first an abstract value type, then rules for manipulating those values, then some primitives, and finally rules for transforming those primitives.
