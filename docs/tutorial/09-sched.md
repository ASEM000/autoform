# Schedule Independent Calls

Independent LM calls do not need to wait for each other. The Python function can stay ordinary and sequential:

```python
import asyncio
import time
import autoform as af


# write the function sequentially; scheduling happens after tracing
def compare(topic: str) -> str:
    explain_prompt = af.format("Explain {} in one sentence.", topic)
    example_prompt = af.format("Give one concrete example of {}.", topic)
    explain_msg = dict(role="user", content=explain_prompt)
    example_msg = dict(role="user", content=example_prompt)

    explanation = af.lm_call([explain_msg], model="gpt-5.2")
    example = af.lm_call([example_msg], model="gpt-5.2")

    combine_prompt = af.format("Combine these into a concise answer:\n{}\n{}", explanation, example)
    combine_msg = dict(role="user", content=combine_prompt)
    return af.lm_call([combine_msg], model="gpt-5.2")
```

There is no `async def` in `compare`. The first two calls read only `topic`, so they are independent. The final call depends on both results.

```{mermaid}
flowchart TD
    topic["topic"] --> explain["LM: explain"]
    topic --> example["LM: example"]
    explain --> combine["LM: combine"]
    example --> combine
```

Trace the function:

```python
# trace once
ir = af.trace(compare)("placeholder topic")
```

Run the original IR synchronously:

```python
# measure the original ir
start = time.perf_counter()
sequential = ir.call("recursion")
sequential_s = time.perf_counter() - start
```

Schedule the IR with [`sched`](../concepts/transforms.md) and run it asynchronously:

```python
# schedule independent equations and run asynchronously
scheduled = af.sched(ir)

start = time.perf_counter()
parallel = asyncio.run(scheduled.acall("recursion"))
parallel_s = time.perf_counter() - start

print(f"sequential: {sequential_s:.2f}s")
print(f"scheduled:  {parallel_s:.2f}s")
```

The scheduled form groups independent [equations](../concepts/the-ir.md) into `gather` steps. With `scheduled.acall(...)`, those groups use `asyncio.gather`, so the two first LM calls can overlap. The final LM call still waits for both inputs.

The measured speedup depends on provider latency, provider-side rate limits, and the active LiteLLM client. The invariant is the dependency structure: independent equations can share a scheduling level; dependent equations cannot.

Compare the two pieces of code:

```python
# compare stays a normal function
def compare(topic: str) -> str:
    ...


scheduled = af.sched(ir)
parallel = asyncio.run(scheduled.acall("recursion"))
```

The execution mode is a property of how you run the IR, not of the original function. You did not rewrite the function into async Python. You changed the IR that executes it.

`sched` is another `IR -> IR` transform, so it composes with the earlier transforms:

```python
# transforms still compose after scheduling
fast_batch = af.sched(af.batch(ir))
fast_feedback = af.sched(af.batch(af.pullback(ir)))
```

Custom primitives need matching async behavior when they should run under `acall`. If you add custom rules, define the async rule alongside the synchronous rule so scheduled async execution does the same work. See [Custom Rules](../concepts/custom-rules.md).
