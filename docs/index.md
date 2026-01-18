# `autoform`

**Composable function transformations for LLM programs**

```bash
pip install autoform
```

## Example

Batched semantic gradients: feedback on N outputs → feedback on N inputs.

```python
import autoform as af

def answer(query):
    return af.lm_call(f"Answer: {query}", model="gpt-4o")

ir = af.trace(answer)("...") # trace

# 3 queries + 3 critiques -> 3 answers + 3 improvement hints
queries = ["What is AI?", "Explain DNS", "Define recursion"]
critiques = ["too technical", "too long", "perfect"]

batched_pb = af.batch(af.pullback(ir), in_axes=(True, True)) # compose

answers, hints = af.call(batched_pb)((queries, critiques))
```

Trace once. Batch it. Differentiate it. **Compose them.**

## Why

LLM programs are hard to optimize:
- **debugging**: which agent caused the bad output?
- **optimization**: how do you improve prompts systematically?
- **batching**: how do you run N inputs without rewriting code?

autoform solves this with **function transformations**. Trace once, transform freely.


## Full Example

Multi-agent pipeline with checkpoints, batching, gradients, and debugging:

```python
import autoform as af

class Verdict(af.Struct):
    decision: str
    reasoning: str

def judge_debate(topic: str) -> Verdict:
    """Three agents debate, one judges."""

    # agent 1: argue for
    pro = af.format("Argue FOR: {}", topic)
    pro = af.mark(pro, collection="debug", name="pro")
    msgs = [dict(role="user", content=pro)]
    pro = af.lm_call(msgs, model="gpt-4o")

    # agent 2: argue against  
    con = af.format("Argue AGAINST: {}", topic)
    con = af.mark(con, collection="debug", name="con")
    msgs = [dict(role="user", content=con)]
    con = af.lm_call(msgs, model="gpt-4o")

    # agent 3: judge
    prompt = af.format("PRO: {}\nCON: {}\nWho wins?", pro, con)
    prompt = af.mark(prompt, collection="debug", name="judge")
    msgs = [dict(role="user", content=prompt)]
    return af.struct_lm_call(msgs, model="gpt-4o", struct=Verdict)

ir = af.trace(judge_debate)("...")  # trace (no execution)

# run once
verdict = af.call(ir)("pineapple on pizza")

# batch: parallel topics
batched = af.batch(ir, in_axes=True)
verdicts = af.call(batched)(["pineapple on pizza", "cats vs dogs", "tabs vs spaces"])

# gradients: feedback -> input improvement
pb_ir = af.pullback(ir)
verdict, grad = af.call(pb_ir)(("pineapple on pizza", Verdict(decision="biased", reasoning="pro was weak")))

# collect: capture checkpointed values
with af.collect(collection="debug") as captured:
    verdict = af.call(ir)("pineapple on pizza")

# inject: override checkpointed values
with af.inject(collection="debug", values=captured):
    verdict = af.call(ir)("pineapple on pizza")

# explain how batches are placed
in_axes = (True, Verdict.model_construct(decision=True, reasoning=True))

# compose freely
batched_grads = af.batch(af.pullback(ir), in_axes=in_axes)
```

> ⚠️ **early development**: API may change.

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

examples/chain_of_thought
examples/multi_agent_debate
examples/iterative_refinement
examples/internals
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api
```
