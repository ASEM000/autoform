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

## API Reference (WIP)

### IR Transformation

| Name | Description |
|------|-------------|
| `build_ir` | Trace a Python function into an IR |
| `run_ir` | Execute an IR with given inputs |
| `pushforward_ir` | Transform IR for forward-mode differentiation |
| `pullback_ir` | Transform IR for reverse-mode differentiation (semantic gradients) |
| `batch_ir` | Transform IR for batch execution |
| `iter_ir` | Execute IR with streaming output |
| `arun_ir` | Execute IR asynchronously |
| `dce_ir` | Dead code elimination pass |
| `fold_ir` | Constant folding pass |

### Primitives

| Name | Description |
|------|-------------|
| `format` | String formatting with traced values |
| `concat` | Concatenate strings |
| `lm_call` | Call LLM via LiteLLM |
| `struct_lm_call` | Call LLM with structured output (Pydantic) |
| `ir_call` | Call an IR as a traced operation |
| `switch` | Branch on a key to select IR |
| `stop_gradient` | Block gradient flow |
| `mark` | Identity with a tag |