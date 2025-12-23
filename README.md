# autoform

**JAX-style function transformations for LLM programs**

> ⚠️ **Early Development**: API may change.

## Install

```bash
git clone https://github.com/ASEM000/autoform.git
cd autoform
uv sync
```

## Example

Write an LLM pipeline once. Transform it — **batch** (parallel `batch_completion`), **backprop** (semantic gradients), or **both**.

```python
import autoform as af

def research_and_write(topic: str) -> str:
    # step 1: research
    prompt1 = af.format("List 3 key facts about: {}", topic)
    notes = af.lm_call([dict(role="user", content=prompt1)], model="gpt-4.1")
    # step 2: write using the research
    prompt2 = af.format("Write a paragraph using: {}", notes)
    article = af.lm_call([dict(role="user", content=prompt2)], model="gpt-4.1")
    return article

# trace the program
ir = af.build_ir(research_and_write, "example")

# transform: batch (uses litellm.batch_completion, not a loop)
batch_ir = af.batch_ir(ir, in_axes=list)
articles = af.run_ir(batch_ir, ["AI safety", "quantum computing", "climate"])

# compose transforms: batched semantic backprop
pb_ir = af.pullback_ir(ir)
batch_pb_ir = af.batch_ir(pb_ir, in_axes=(list, list))
topics = ["AI safety", "quantum computing", "climate"]
feedbacks = ["too brief", "good", "needs examples"]
outputs, input_grads = af.run_ir(batch_pb_ir, (topics, feedbacks))
```

## More Examples

- [examples/research_and_write.py](examples/research_and_write.py) — multi-step pipeline with batching
- [examples/semantic_backprop.py](examples/semantic_backprop.py) — TextGrad-style custom backward passes