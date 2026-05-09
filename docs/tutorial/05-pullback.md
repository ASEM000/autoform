# Pullback

[`pullback`](../concepts/transforms.md) sends feedback backward through an IR.

The situation is:

- you have an output;
- you have a critique of that output;
- you want feedback for the inputs that contributed to it.

In autodiff terms, that critique is a [cotangent](../reference/glossary.md). In `autoform`, cotangents are usually text.

Start from the same `ir`:

```python
# build an ir that returns the output and input feedback
pb_ir = af.pullback(ir)

inputs = ("quantum entanglement",)
output, grad = pb_ir.call(inputs, "too technical")

print(output)
print(grad)
```

The call shape is:

```text
original inputs + output feedback -> output + input feedback
```

For `explain(topic)`, the original input tree has one positional input, so the first argument is a one-item tuple: `("quantum entanglement",)`. The second argument is feedback on the output: `"too technical"`.

The result has the same structure:

- `output` is the model output from the forward run;
- `grad` is a one-item tuple containing text feedback for `topic`.

For example, `grad` might suggest narrowing the topic, asking for less jargon, or adding an audience constraint. The exact text depends on the active model, because the backward pass through `lm_call` is itself an LM call.

Pullback becomes more useful as the program grows. If the IR has several LM calls, the output feedback flows backward through every recorded step. Each [primitive rule](../concepts/primitives.md) decides how to translate feedback for its output into feedback for its inputs.

Cotangent shapes must match output shapes. If the function returns a tuple, pass tuple-shaped feedback. If it returns a [schema](../concepts/schemas.md)-shaped object, pass feedback with the same [pytree](../concepts/pytrees.md) shape.
