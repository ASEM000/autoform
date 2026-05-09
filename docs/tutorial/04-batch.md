# Batch Inputs

You traced `explain` once. Now run the same IR over several topics.

The direct Python version is a loop:

```python
# this is just a python loop around the ir
outputs = [ir.call(topic) for topic in ["DNA", "gravity", "recursion"]]
```

That works, but the loop is not itself an IR. [`batch`](../concepts/transforms.md) gives you an IR that accepts batched inputs:

```python
import autoform as af


topics = ["DNA", "gravity", "recursion"]

# transform the ir so the input can be a list
batched_ir = af.batch(ir)
outputs = batched_ir.call(topics)

for topic, output in zip(topics, outputs):
    print(topic, "->", output)
```

`batched_ir` is a new IR, not a list. Calling it still executes the provider calls, and the result is a list of strings with the same length as `topics`.

The result is equivalent to the list comprehension, but the representation is different:

- the list comprehension gives you a Python list of results;
- `af.batch(ir)` gives you a transformed IR that can be transformed again.

That second point is the reason to use `batch` in `autoform`. The batched form can be composed with `pullback`, `sched`, or other IR transforms.

By default, every [input leaf](../concepts/pytrees.md) is batched. For functions with multiple inputs, `in_axes` controls which leaves are batched and which are broadcast:

```python
# batch the first input and broadcast the second
batched = af.batch(two_arg_ir, in_axes=(True, False))
```

That means: batch over the first positional input, reuse the second input for every item.
