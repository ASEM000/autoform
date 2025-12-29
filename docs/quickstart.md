# 🚀 Quickstart

## 📦 Installation

```bash
pip install autoform
```

## 💡 Core Concepts

Autoform traces Python functions into an **IR** (Intermediate Representation), then applies transformations like batching and differentiation.

```python
import autoform as af

class Summary(af.Struct):
    text: str

def summarize(topic):
    prompt = af.format("Summarize {} in one sentence.", topic)
    return af.struct_lm_call(
        [dict(role="user", content=prompt)],
        model="openai/gpt-4o-mini",
        struct=Summary,
    )

ir = af.build_ir(summarize)("topic")
result = af.call(ir)("cats")
print(result)
# Summary(text="Cats are domesticated...")
```

## ⚡ Transformations

*Process multiple inputs in parallel:*

```python
batched_ir = af.batch(ir, in_axes=list)
results = af.call(batched_ir)(["cats", "dogs", "birds"])
# Summary(text=["Cats are...", "Dogs are...", "Birds are..."])
```

*Compute textual gradients - feedback on how to improve inputs:*

```python
pb_ir = af.pullback(ir)
output, gradient = af.call(pb_ir)(("cats", Summary(text="too short")))
# gradient = "Try 'domestic cats behavior' for more detail"
```

*Gather values checkpointed during execution:*

```python
def with_checkpoint(topic):
    result = summarize(topic)
    af.checkpoint(result.text, collection="summaries", name="text")
    return result

ir = af.build_ir(with_checkpoint)("topic")
output, collected = af.collect(ir, collection="summaries")("cats")
# collected = {"text": ["Cats are..."]}
```

*Transformations compose naturally:*

```python
# Batched pullback - gradients for multiple inputs at once
batched_pb_ir = af.batch(
    af.pullback(ir),
    in_axes=(list, Summary.model_construct(text=list))
)

outputs, gradients = af.call(batched_pb_ir)((
    ["cats", "dogs"],
    Summary.model_construct(text=["too short", "more detail"])
))
```

## 🔗 Next Steps

- **[Internals](examples/internals)** - How `autoform` works
- **[Iterative Refinement](examples/iterative_refinement)** - Loops with early exit
- **[API Reference](api)** - Full documentation
