# Why `autoform`

An LM program written in ordinary Python tends to harden around the first way you run it. The rewrite pressure usually shows up as the same few demands:

- **Evaluate on 100 inputs**: write a batched loop.
- **Route prompt-tuning feedback through every LM call**: thread critiques backward by hand.
- **Run independent calls concurrently**: rewrite with `async def` and `asyncio.gather`, then make every caller async too.
- **Inspect a bad intermediate value**: wrap each step manually or split the function apart.

Each new requirement becomes another version of the same program: batched rewrite, feedback rewrite, async rewrite, debugging rewrite.

`autoform` factors those requirements differently. You write the function once, normally, then put different mechanisms around the same traced structure:

- **Trace once**: `af.trace(func)(*example_args)` captures the function as an IR.
- **Transform the IR**: `batch`, `pullback`, and `sched` each take an IR and produce another IR.
- **Wrap execution**: `collect` / `inject` capture or replace intermediates; `lm_client` changes provider routing.

The IR transforms compose because their input and output type is the same. The contexts wrap execution without changing the original function.

```python
ir = af.trace(explain)("...")        # capture once

af.batch(ir)                         # 100 inputs at once
af.pullback(ir)                      # text feedback flows backward
af.sched(ir)                         # independent calls run concurrently
af.batch(af.pullback(ir))            # batched prompt optimization
```

*the original `explain` was not modified, was not rewritten, did not know any of this would happen.*

## One Task, Two Shapes

Suppose you have a three-step pipeline and you want batched prompt feedback: run the pipeline over many topics, collect critiques on the outputs, then route text feedback backward to the corresponding inputs.

``````{tab-set}


`````{tab-item} autoform
````python
ir = af.trace(pipeline)("...")
transformed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = transformed.call((topics,), critiques)
````
`````


`````{tab-item} Manual
````python
results = []
hints = []

for topic, critique in zip(topics, critiques):
    prompt1 = build_prompt(topic)
    step1 = call_lm(prompt1)

    prompt2 = build_followup(step1)
    step2 = call_lm(prompt2)

    prompt3 = build_answer(step1, step2)
    answer = call_lm(prompt3)

    c_answer = critique
    c_step1, c_step2 = critique_join(step1, step2, c_answer)
    c_prompt2 = critique_followup(prompt2, c_step2)
    c_prompt1 = critique_start(prompt1, c_step1)
    c_topic = critique_topic(topic, c_prompt1)

    results.append(answer)
    hints.append(c_topic)
````
`````

``````

The right column is not a special combined feature. `pullback(ir)` returns an IR. `batch(...)` accepts an IR. The composition is ordinary Python function composition applied to a traced program.

## How This Differs From Other LM Frameworks

- **LangChain / LangGraph**: chain construction and chain execution are unified: you build a chain object and call it. `autoform` separates the phases. Tracing produces an inert IR, transforms reshape it, and execution is a later step. The tradeoff is one extra concept, the IR, in exchange for transforms composing by ordinary function composition.
- **DSPy**: programs are described with signatures and modules. DSPy then uses examples and a metric to search for better instructions, select few-shot demonstrations, or fine-tune model weights, returning a tuned version of the program. `autoform` exposes a traced program as an IR, so feedback, batching, and scheduling are transforms you can compose directly. The tradeoff is that DSPy gives you optimization algorithms; `autoform` gives you the substrate for writing and composing transformations.
- **Outlines / Instructor / Pydantic AI**: the design center is structured output for one LM call. `autoform`'s `lm_schema_call` covers structured output, but the schema call is one node inside a traceable program. That means structured output composes with batching, pullback, and scheduling. If structured output is the whole task, a narrower tool may be the right choice.

## When To Reach For It

- You are building an agent or multi-step LM pipeline that you expect to iterate on.
- You want prompt optimization where text feedback flows backward through the full program.
- You need batched evaluation over many inputs, and you want that batch form to compose with other transforms.
- You need to debug intermediate values without splitting the function into a test-only version.
- You want to explore concurrent execution without rewriting the program as async Python.

## When Not To

- You have a one-shot script and no expectation of reuse.
- Your program is a single LM call and structured output is the only requirement.
- You are optimizing one latency-critical request where an abstraction layer is not acceptable.
- Your project cannot take on a trace/IR/execute model yet.

Next, read [Getting Started](../tutorial/index.md), or go deeper on the model in [Trace, IR, Execute](trace-ir-execute.md).

```{warning}
API may change before a stable release.
```
