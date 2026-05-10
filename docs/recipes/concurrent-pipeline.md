# Run An LM Pipeline Concurrently

[`sched`](../concepts/transforms.md) turns independent equations into async
gather steps. The Python function can stay sequential.

```python
import asyncio
import time
import autoform as af


def research(topic: str) -> str:
    summary_prompt = af.format("Summarize {} in two sentences.", topic)
    analogy_prompt = af.format("Give one concrete analogy for {}.", topic)
    summary_msg = dict(role="user", content=summary_prompt)
    analogy_msg = dict(role="user", content=analogy_prompt)

    # these two calls only depend on topic
    summary = af.lm_call([summary_msg], model="gpt-5.2")
    analogy = af.lm_call([analogy_msg], model="gpt-5.2")

    join_template = "Combine these notes.\nsummary: {}\nanalogy: {}"
    join_prompt = af.format(join_template, summary, analogy)
    join_msg = dict(role="user", content=join_prompt)
    combined = af.lm_call([join_msg], model="gpt-5.2")

    final_prompt = af.format("Rewrite this as a crisp answer:\n{}", combined)
    final_msg = dict(role="user", content=final_prompt)
    return af.lm_call([final_msg], model="gpt-5.2")


# trace once, then choose the execution form
ir = af.trace(research)("recursion")
scheduled = af.sched(ir)

start = time.perf_counter()
sequential = ir.call("recursion")
sequential_s = time.perf_counter() - start

start = time.perf_counter()
parallel = asyncio.run(scheduled.acall("recursion"))
parallel_s = time.perf_counter() - start

print(sequential)
print(parallel)
print(f"sequential: {sequential_s:.2f}s")
print(f"scheduled:  {parallel_s:.2f}s")
```

```{mermaid}
flowchart TD
    topic[/topic/] --> summary[["LM: summary"]]
    topic --> analogy[["LM: analogy"]]
    summary --> join[["LM: combine notes"]]
    analogy --> join
    join --> final[["LM: final answer"]]
```

Only the first two LM calls can overlap. The combine call waits for both, and
the final call waits for the combined text.

Measured speed depends on provider latency, provider-side rate limits, and the
active LiteLLM client. The property that matters is the dependency graph: if two
equations do not depend on each other, `sched(...).acall(...)` can run them in
the same async level.
