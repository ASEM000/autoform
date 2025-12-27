# autoform

**composable function transformations for LLM programs**

> ⚠️ **early development**: API may change.

```bash
pip install autoform
```


## Example

Write a multi-agent pipeline once. get **per-agent semantic gradients** on a **batched dataset** in one call.

```python
import autoform as af

class Verdict(af.Struct):  # pydantic model for structured output
    decision: str
    reasoning: str

def judge_debate(topic: str) -> Verdict:
    """three agents debate, one judges. we can optimize any of them."""
    # agent 1: argue for
    pro = af.format("Argue FOR: {}", topic)
    pro = af.sow(pro, tag="prompts", name="pro")  # tag for collection
    msg = dict(role="user", content=pro)
    pro = af.lm_call([msg], model="gpt-4.1")

    # agent 2: argue against
    con = af.format("Argue AGAINST: {}", topic)
    con = af.sow(con, tag="prompts", name="con")
    msg = dict(role="user", content=con)
    con = af.lm_call([msg], model="gpt-4.1")

    # agent 3: judge
    prompt = af.format("PRO: {}\nCON: {}\nWho wins?", pro, con)
    prompt = af.sow(prompt, tag="prompts", name="judge")
    msg = dict(role="user", content=prompt)
    return af.struct_lm_call([msg], model="gpt-4o", struct=Verdict)

# trace
ir = af.build_ir(judge_debate)("...")

# batch: parallel topics
batch_ir = af.batch_ir(ir, in_axes=list)
verdicts = af.run_ir(batch_ir, ["pineapple on pizza", "cats vs dogs", "morning vs night"])

# gradients: feedback on output -> feedback on input
pb_ir = af.pullback_ir(ir)
feedback = Verdict(decision="too one-sided", reasoning="pro was weak")
verdict, grad = af.run_ir(pb_ir, ("pineapple on pizza", feedback))

# batched gradients
batch_pb = af.batch_ir(pb_ir, in_axes=(list, list))

# harvest: collect prompts
reaped_ir = af.reap_ir(ir, tag="prompts")
verdict, prompts = af.run_ir(reaped_ir, "pineapple on pizza")

# split: isolate judge for testing
ir_before, ir_after = af.split_ir(ir, tag="prompts", name="judge")

# composition: nest IRs (inlined during tracing)
def meta(topic):
    v1, v2 = af.run_ir(ir, topic), af.run_ir(ir, topic)
    return af.format("{} vs {}", v1.decision, v2.decision)
meta_ir = af.build_ir(meta)("...")

# transforms compose
batch_batch = af.batch_ir(af.batch_ir(ir, in_axes=list), in_axes=list)
grad_grad = af.pullback_ir(af.pullback_ir(ir))
```

## More

- [examples/research_and_write.py](examples/research_and_write.py)
- [examples/semantic_backprop.py](examples/semantic_backprop.py)