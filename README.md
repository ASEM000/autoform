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
    pro = af.checkpoint(pro, collection="debug", name="pro")  # tag for collection
    msg = dict(role="user", content=pro)
    pro = af.lm_call([msg], model="gpt-4.1")

    # agent 2: argue against
    con = af.format("Argue AGAINST: {}", topic)
    con = af.checkpoint(con, collection="debug", name="con")
    msg = dict(role="user", content=con)
    con = af.lm_call([msg], model="gpt-4.1")

    # agent 3: judge
    prompt = af.format("PRO: {}\nCON: {}\nWho wins?", pro, con)
    prompt = af.checkpoint(prompt, collection="debug", name="judge")
    msg = dict(role="user", content=prompt)
    return af.struct_lm_call([msg], model="gpt-4o", struct=Verdict)

# trace
ir = af.build_ir(judge_debate)("...")

# execute
verdict = af.call_ir(ir)("pineapple on pizza")

# batch: parallel topics
batch_ir = af.batch_ir(ir, in_axes=list)
verdicts = af.call_ir(batch_ir)(["pineapple on pizza", "cats vs dogs", "morning vs night"])

# gradients: feedback on output -> feedback on input
pb_ir = af.pullback_ir(ir)
feedback = Verdict(decision="too one-sided", reasoning="pro was weak")
verdict, grad = af.call_ir(pb_ir)(("pineapple on pizza", feedback))

# batched gradients
batch_pb = af.batch_ir(pb_ir, in_axes=(list, list))

# harvest: collect prompts at runtime
verdict, prompts = af.collect_ir(ir, collection="debug")("pineapple on pizza")
# prompts: {'pro': '...', 'con': '...', 'judge': '...'}

# plant: inject cached values
verdict = af.inject_ir(ir, collection="debug", values={"pro": "cached response"})("pineapple on pizza")

# transforms compose
batch_batch = af.batch_ir(af.batch_ir(ir, in_axes=list), in_axes=list)
grad_grad = af.pullback_ir(af.pullback_ir(ir))
```

## More

- [examples/research_and_write.py](examples/research_and_write.py)
- [examples/semantic_backprop.py](examples/semantic_backprop.py)