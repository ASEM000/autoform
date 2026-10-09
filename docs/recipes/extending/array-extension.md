# Array Extension

A domain extension needs to explain both its values and its operations to `autoform`. This recipe adds NumPy arrays as atomic leaves, registers arithmetic primitives, and supplies rules for batching and numerical differentiation. A small program then uses ordinary Python operators while the transforms work through the registered rules. NumPy calculations provide reference results for the examples.

```{admonition} Concept
[Primitives and Rules](../../concepts/primitives-and-rules.md) · [Transforms](../../concepts/transforms.md) · [Types and Spaces](../../concepts/types-and-spaces.md)
```

This example only supports floating-point arrays, requires arrays to have the same shape, and only supports matrix multiply in 2D. The arrays are treated as atomic leaves, and the batches are Python lists of arrays. Supporting other dtypes, broadcasting shapes, reduction operations, and scalar promotion would require additional rules.

## Abstract Value

Tracing needs an array description before an operation has produced a concrete result. `ArrayAVal` records the shape and dtype and supplies the zero and accumulation behavior used by feedback. The primal, tangent, and cotangent spaces all use this same array representation in the example:

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
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __hash__(self):
        return hash((type(self), self.shape, self.dtype.str))

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def accum(self, c):
        return sum(c[1:], c[0])


def aval_rule(value):
    return ArrayAVal(value.shape, value.dtype)


afe.register_trace_type(np.ndarray, aval_rule)
afe.primal_s.set(ArrayAVal, lambda aval: aval)
afe.tangent_s.set(ArrayAVal, lambda aval: aval)
afe.cotangent_s.set(ArrayAVal, lambda aval: aval)
```

## Operation Rules

Registering a value type does not define how array operations transform. Each operation also needs execution and abstract rules, followed by the rules required for batching and differentiation. The helper below collects these registrations for a binary operation. The forward pullback rule retains both inputs as residuals, and the backward rule uses those values with output feedback:

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

## Operators

The helper can now describe several operations without repeating the registration code. Each definition provides the NumPy computation together with its pushforward and pullback formulas. Dunder registrations connect these primitives to the corresponding Python operators:

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

The final registrations connect traced Python syntax to the primitives through one dunder rule table. For example, `x + y` stages `a_add` when `x` has `ArrayAVal`.

## Execution and AD

The first program combines addition, multiplication, and division. Its execution is compared with NumPy, then its pushforward and pullback are compared with the expected derivatives. These checks exercise the composition of the registered rules across several operations:

```python
def f(x, y):
    return ((x + y) * y) / x


x = np.array([1.0, 2.0])
y = np.array([3.0, 4.0])
ir = af.trace(f)(x, y)

np.testing.assert_allclose(ir.call(x, y), f(x, y))

out_p, out_t = af.pushforward(ir).call(
    (x, y),
    (np.ones_like(x), np.zeros_like(y)),
)
np.testing.assert_allclose(out_p, f(x, y))
np.testing.assert_allclose(out_t, -y * y / (x * x))

out, (dx, dy) = af.pullback(ir).call((x, y), np.ones_like(x))
np.testing.assert_allclose(out, f(x, y))
np.testing.assert_allclose(dx, -y * y / (x * x))
np.testing.assert_allclose(dy, (x + 2 * y) / x)
```

Batching adds an outer Python container while preserving each NumPy array as one leaf. The two arrays at the same batch position form the inputs for one execution. This is distinct from treating an axis inside an array as the batch dimension:

```python
batched = af.batch(ir, in_axes=(True, True))
outs = batched.call([x, x + 1], [y, y + 1])

np.testing.assert_allclose(outs[0], f(x, y))
np.testing.assert_allclose(outs[1], f(x + 1, y + 1))
```

Matrix multiplication uses the same registration pattern, with transpose operations in its pullback formula. The following check compares both input gradients with the corresponding NumPy matrix products:

```python
def mm(a, b):
    return a @ b


a = np.eye(2)
b = np.array([[2.0, 0.0], [0.0, 3.0]])
mm_ir = af.trace(mm)(a, b)

out_pb, (da, db) = af.pullback(mm_ir).call((a, b), np.ones((2, 2)))
np.testing.assert_allclose(out_pb, a @ b)
np.testing.assert_allclose(da, np.ones((2, 2)) @ b.T)
np.testing.assert_allclose(db, a.T @ np.ones((2, 2)))
```

The assertions check the output, the numerical values of the tangents and cotangents, the values of the batched executions, and the derivatives of the matrix product.
