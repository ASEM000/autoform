# Why `autoform`

An LM program written in ordinary Python often hardens around the first way it runs. Later requirements tend to ask for the same logic in new shapes:

- Evaluate on 100 inputs: write a batched loop.
- Route prompt-tuning feedback through every LM call: thread critiques backward by hand.
- Run independent calls concurrently: rewrite with `async def` and `asyncio.gather`.
- Inspect a bad intermediate value: wrap each step manually or split the function apart.

Each new requirement becomes another version of the same program: batched rewrite, feedback rewrite, async rewrite, debugging rewrite.

`autoform` factors those requirements differently:

- {py:func}`trace <autoform.trace>` captures the function as an IR.
- {py:func}`batch <autoform.batch>`, {py:func}`pullback <autoform.pullback>`, and {py:func}`sched <autoform.sched>` transform that IR into another IR.
- {py:func}`collect <autoform.collect>`, {py:func}`inject <autoform.inject>`, and {py:func}`lm_client <autoform.lm_client>` wrap execution without changing the function.

The IR transforms compose because their input and output type is the same. The contexts wrap execution without changing the original function.

```python
ir = af.trace(explain)("...")  # capture once

af.batch(ir)  # 100 inputs at once
af.pullback(ir)  # text feedback flows backward
af.sched(ir)  # independent calls run concurrently
af.batch(af.pullback(ir))  # batched prompt optimization
```

*the original `explain` was not modified, was not rewritten, did not know any of this would happen.*

## One Task, Two Shapes

Suppose a three-step pipeline needs batched prompt feedback: run the pipeline over many topics, collect critiques on the outputs, then route text feedback backward to the corresponding inputs.

``````{tab-set}


`````{tab-item} autoform
````python
ir = af.trace(pipeline)("...")
transformed = af.batch(af.pullback(ir))
outputs, (topic_hints,) = transformed.call((topics,), critiques)
````
`````


`````{tab-item} Plain Python Rewrite
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

The rewritten version is not replaced by a special combined feature. {py:func}`pullback <autoform.pullback>` returns an IR. {py:func}`batch <autoform.batch>` accepts an IR. Composition is ordinary Python function composition applied to a traced program.

## Adjacent LM Frameworks

| Framework family | Architectural choice | `autoform` choice |
| --- | --- | --- |
| LangChain<br>LangGraph | Build a chain object and call it. | Separate trace, transform, and execute phases. The extra concept is the IR; the payoff is ordinary composition between transforms. |
| DSPy | Describe programs with signatures and modules, then use examples and metrics to tune them. | Expose the traced program as IR data, so feedback, batching, and scheduling are directly *composable* transforms. |
| TextGrad | Center the interface on textual-gradient optimization with an autograd-style workflow. | Treat text feedback as {py:func}`pullback <autoform.pullback>`, one IR transform that composes with batching, scheduling, and other transforms. |
| Microsoft Trace | Center agent training on a traced computation graph with trainable values and generative optimizers. | Keep trace, transform, and execute separate; training-style feedback is one use of the IR, not the whole interface. |
| Outlines<br>Instructor<br>Pydantic AI | Focus on structured output for one LM call. | Put structured output inside a traceable program, so it can compose with batching, pullback, and scheduling. |

## Project Fit

| Good fit | Poor fit |
| --- | --- |
| Agents or multi-step LM pipelines are expected to evolve. | The program is a one-shot script. |
| Text feedback should flow backward through the full program. | Structured output for one LM call is the whole task. |
| Batched evaluation should compose with other transforms. | One latency-critical request cannot afford another layer. |
| Debugging or concurrency experiments should not require rewrites. | The project cannot take on a trace/IR/execute model yet. |

Next, read [Getting Started](tutorial/index.md), or go deeper on the model in [Trace, IR, Execute](concepts/trace-ir-execute.md).

```{warning}
[API Reference](api/index.md) may change before a stable release.
```
