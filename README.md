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

# Pydantic model for structured LLM output
class Verdict(af.Struct):
    decision: str
    reasoning: str

def judge_debate(topic: str) -> Verdict:
    """three agents debate, one judges. we can optimize any of them."""

    # agent 1: argue for
    pro_prompt = af.format("Argue FOR: {}", topic)
    # tag for later collection/injection
    pro_prompt = af.sow(pro_prompt, tag="prompts", name="pro")
    # call LLM
    pro = af.lm_call([dict(role="user", content=pro_prompt)], model="gpt-4.1")

    # agent 2: argue against
    con_prompt = af.format("Argue AGAINST: {}", topic)
    con_prompt = af.sow(con_prompt, tag="prompts", name="con")
    con = af.lm_call([dict(role="user", content=con_prompt)], model="gpt-4.1")

    # agent 3: judge the debate
    judge_prompt = af.format("Judge this debate:\nPRO: {}\nCON: {}\nWho wins?", pro, con)
    judge_prompt = af.sow(judge_prompt, tag="prompts", name="judge")
    return af.struct_lm_call([dict(role="user", content=judge_prompt)], model="gpt-4o", struct=Verdict)

# trace (pass dummy input to build IR)
ir = af.build_ir(judge_debate, "...")

# batch: process multiple topics in parallel
batch_ir = af.batch_ir(ir, in_axes=list)
topics = ["pineapple on pizza", "cats vs dogs", "morning vs night"]
verdicts = af.run_ir(batch_ir, topics)

# semantic gradients: feedback on output → feedback on input
pb_ir = af.pullback_ir(ir)
feedback = Verdict(decision="too one-sided", reasoning="pro argument was weak")
verdict, topic_grad = af.run_ir(pb_ir, ("pineapple on pizza", feedback))

# batched gradients
batch_pb = af.batch_ir(pb_ir, in_axes=(list, list))
topics = ["pineapple on pizza", "cats vs dogs"]
feedbacks = [Verdict(decision="too short", reasoning=""), Verdict(decision="balanced", reasoning="")]
verdicts, grads = af.run_ir(batch_pb, (topics, feedbacks))

# harvest: see the prompts
reaped_ir = af.reap_ir(ir, tag="prompts")
verdict, prompts = af.run_ir(reaped_ir, "pineapple on pizza")
# prompts["pro"], prompts["con"], prompts["judge"]

# split: isolate the judge for testing
ir_before_judge, ir_judge = af.split_ir(ir, tag="prompts", name="judge")

# composition: nest IRs inside each other (automatically inlined during tracing)
def meta_debate(topic):
    v1 = af.run_ir(ir, topic)  # first debate
    v2 = af.run_ir(ir, topic)  # second debate  
    return af.format("Debate 1: {}\nDebate 2: {}", v1.decision, v2.decision)

meta_ir = af.build_ir(meta_debate, "...")

# transformations compose: batch the batched, gradient the gradiented
batch_batch = af.batch_ir(af.batch_ir(ir, in_axes=list), in_axes=list)  # nested batching
grad_grad = af.pullback_ir(af.pullback_ir(ir))  # second-order
```

## More

- [examples/research_and_write.py](examples/research_and_write.py)
- [examples/semantic_backprop.py](examples/semantic_backprop.py)