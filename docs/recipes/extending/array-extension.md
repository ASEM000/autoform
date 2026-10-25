# Array Extension

To extend the domain of `autoform`, one needs to explain to `autoform` what the new values are and how to operate on them. In this recipe, a new array domain is registered as an atomic leaf, along with a few simple arithmetic primitives. These primitives are then extended to support batching and numerical differentiation, allowing the following examples to work.

```{admonition} Concept
[Primitives and Rules](../../concepts/primitives-and-rules.md) · [Transforms](../../concepts/transforms.md) · [Types and Spaces](../../concepts/types-and-spaces.md)
```

This example is very simple: it only supports floating-point arrays, arrays need to have the same shape, and matmul is only supported for 2D arrays. Arrays will be treated as atomic values, so they will appear as leaves and batches will be made by constructing Python lists of arrays. Making this extension more fully featured would require support for other dtypes, broadcasting rules for shape, rules for reductions, and rules for promoting scalars.

## Abstract Value

To trace a function, `autoform` needs to know what an array looks like before the function is actually run. The following AVal describes arrays to `autoform`. It also describes how to construct a zero, and how to add two values together, which are both needed for handling feedback. In this case, the primal, tangent, and cotangent spaces use the same array representation.

```python

import numpy as np

import autoform as af
import autoform.extend as afe


class AValArray(afe.AVal):
    __slots__ = ["shape", "dtype"]

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def __repr__(self):
        return f"AValArray(shape={self.shape!r}, dtype={self.dtype!r})"

    def __eq__(self, other):
        return type(self) is type(other) and self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self):
        return hash((type(self), self.shape, self.dtype.str))

    def zero(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def add(self, left, right):
        return left + right

    def join(self, other):
        match other:
            case (
                np.ndarray(shape=shape, dtype=dtype)
                | AValArray(shape=shape, dtype=dtype)
                | afe.Zero(aval=AValArray(shape=shape, dtype=dtype))
            ):
                if self.shape == shape and self.dtype == dtype:
                    return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")


def aval_rule(value):
    return AValArray(value.shape, value.dtype)


afe.register_trace_type(np.ndarray, aval_rule)
afe.register_trace_type(AValArray, lambda aval: aval)
afe.s_primal.set(AValArray, lambda aval: aval)
afe.s_tangent.set(AValArray, lambda aval: aval)
afe.s_cotangent.set(AValArray, lambda aval: aval)
```

Finally, `join` needs to be able to handle concrete arrays, array AVals, and symbolic zeros carrying array AVals. In this example, the shape and dtype of the array must match exactly, since there is no broader array description. A mismatch raises `TypeError`. The checker calls `expected.join(value)` to make sure the result matches the expected AVal. Under the hood, `autoform` accumulates concrete contributions by doing a `reduce(add, ..., zero())`, producing an array of the correct shape and dtype.

## Operation Rules

Now that values have been registered, the operations need to be registered. Every operation has an execution rule and an abstract rule, and then additional rules for each transform. For brevity, a helper function is used to register binary operations. This example always uses both inputs as residuals in the forward pullback, so the backward rule will be given both of those values as well as the feedback from the output.

```python
def array_aval(value):
    return value if isinstance(value, AValArray) else afe.avalof(value)


def result_aval(x, y, op):
    ax = array_aval(x)
    ay = array_aval(y)
    result = op(
        np.ones(ax.shape, dtype=ax.dtype),
        np.ones(ay.shape, dtype=ay.dtype),
    )
    return AValArray(result.shape, result.dtype)


def register_binary(name, op, push_rule, pull_rule):
    p_op = afe.Prim(name)

    def bind(x, y):
        return p_op.bind((x, y))

    def impl(in_tree):
        return op(*in_tree)

    def abstract(in_tree):
        return result_aval(*in_tree, op)

    def push(in_tree):
        (x, y), (tx, ty) = in_tree
        tx, ty = afe.instantiate((tx, ty))
        return bind(x, y), push_rule(x, y, tx, ty)

    def pull_fwd(in_tree):
        x, y = in_tree
        return bind(x, y), (x, y)

    def pull_bwd(in_tree):
        (x, y), g = in_tree
        g = afe.instantiate(g)
        return pull_rule(x, y, g)

    def batch_rule(in_tree):
        spec, in_axes, in_values = in_tree
        items = spec.unstack(in_axes, in_values)
        results = (bind(x, y) for x, y in items)
        return spec.stack(True, results), True

    afe.r_impl.set(p_op, impl)
    afe.r_abstract.set(p_op, abstract)
    afe.r_push.set(p_op, push)
    afe.r_pull_fwd.set(p_op, pull_fwd)
    afe.r_pull_bwd.set(p_op, pull_bwd)
    afe.r_batch.set(p_op, batch_rule)
    return bind
```

## Operators

With this, a few operations can be registered, along with their pushforward/pullback rules. The dunder rules will allow these primitives to be used with Python operators on traceable values.

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

afe.register_dunder(afe.Dunder.ADD, AValArray, a_add)
afe.register_dunder(afe.Dunder.SUB, AValArray, a_sub)
afe.register_dunder(afe.Dunder.MUL, AValArray, a_mul)
afe.register_dunder(afe.Dunder.DIV, AValArray, a_div)
afe.register_dunder(afe.Dunder.MATMUL, AValArray, a_matmul)
```

In fact, these are the last rules that need to be registered. Thanks to the dunder rule table, a registered Python operator used on a traceable value with an `AValArray` stages a call to the corresponding primitive operation.

## Execution and AD

Now, let’s see if execution and autodiff work! For brevity, expected values are computed with NumPy.

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

For this example, batches are made by putting arrays into Python containers. To batch two executions, one supplies two arrays for each input, and the arrays at the same index will be passed to the function together.

```python
batched = af.batch(ir, in_axes=(True, True))
outs = batched.call([x, x + 1], [y, y + 1])

np.testing.assert_allclose(outs[0], f(x, y))
np.testing.assert_allclose(outs[1], f(x + 1, y + 1))
```

Let’s also try out matrix multiplication, making sure the gradients of both inputs are correct.

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
