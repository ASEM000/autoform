# Compose Transforms

You have [`batch`](../concepts/transforms.md). You have [`pullback`](../concepts/transforms.md). How do you do both at once: run prompt-feedback gradients across many inputs?

Without composition, you own the loop:

```python
# manual version: run one pullback per input
pb_ir = af.pullback(ir)

outputs = []
hints = []

for topic, critique in zip(topics, critiques):
    output, (topic_hint,) = pb_ir.call((topic,), critique)
    outputs.append(output)
    hints.append(topic_hint)
```

That is the glue you would actually write: pair every input with its critique, run the backward pass, unpack the one input cotangent, and keep the result aligned with the original batch.

## `composed = af.batch(af.pullback(ir))`

```python
# compose the transforms instead
composed = af.batch(af.pullback(ir))
```

*`pullback` returned an IR. `batch` accepts an IR. So `batch(pullback(ir))` is just function composition, and there was nothing special we had to add for them to compose.*

Run it:

```python
topics = ["DNA", "gravity", "recursion"]
critiques = ["too terse", "too abstract", "too long"]

# the original inputs are one positional tree: (topics,)
composed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = composed.call((topics,), critiques)

for topic, hint in zip(topics, topic_hints):
    print(topic, "->", hint)
```

The call shape follows from `pullback`:

- the original input tree is one positional input, so batched topics are passed as `(topics,)`;
- output feedback is batched as `critiques`;
- input feedback has the same structure as the original input tree, so it returns `(topic_hints,)`.

Both [transforms](../concepts/transforms.md) are `IR -> IR`. `pullback` does not know `batch` will be applied next. `batch` does not know the IR came from `pullback`. The type does the work.

Order matters:

- `batch(pullback(ir))` means many independent pullback calls at once.
- `pullback(batch(ir))` means feedback for the batched function as a whole.

Every other IR transform composes the same way. `sched(batch(pullback(ir)))` is a real expression. So is `batch(sched(ir))`.
