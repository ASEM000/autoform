# Inspect Intermediates

Tracing gives you an IR, but debugging often starts with a smaller question: what did the middle of the program produce?

Put a [`checkpoint`](../concepts/intercepts.md) at the value you want to inspect:

```python
import autoform as af


def explain_then_rewrite(topic: str) -> str:
    draft_prompt = af.format("Draft a one-sentence explanation of {}.", topic)
    draft_msg = dict(role="user", content=draft_prompt)
    step1 = af.lm_call([draft_msg], model="gpt-5.2")
    # mark the intermediate value for runtime inspection
    step1 = af.checkpoint(step1, key="step1", collection="debug")

    rewrite_prompt = af.format("Rewrite for a beginner: {}", step1)
    rewrite_msg = dict(role="user", content=rewrite_prompt)
    return af.lm_call([rewrite_msg], model="gpt-5.2")
```

Trace once:

```python
# trace once, then choose the execution context later
ir = af.trace(explain_then_rewrite)("placeholder topic")
```

`checkpoint` is transparent during ordinary execution. Without a context, this just runs both LM calls:

```python
# no collection context means checkpoint is transparent
result = ir.call("recursion")
```

Wrap execution with [`collect`](../concepts/intercepts.md) to capture checkpointed values:

```python
# collect captured intermediates during execution
with af.collect(collection="debug") as captured:
    result = ir.call("recursion")

print(captured["step1"])
```

The captured value is a list because the same key can be reached more than once. In this function there is one `step1` value per run.

Use [`inject`](../concepts/intercepts.md) when you want to replace an intermediate and keep the rest of the IR unchanged:

```python
# replace step1 only for this execution
with af.inject(collection="debug", values={"step1": ["Recursion is a function calling itself."]}):
    result = ir.call("recursion")
```

This execution still enters the IR at the same input and still runs the downstream rewrite call. The checkpointed `step1` value is replaced before the downstream prompt is built.

That makes `collect` and `inject` useful for tight debugging loops:

- capture the intermediate value from a failing run;
- edit or replace that value;
- rerun the downstream part of the same IR;
- keep the original Python function intact.

`collect` and `inject` are [runtime context managers](../concepts/intercepts.md). They do not modify the IR object. The same `ir` can run normally, run under `collect`, or run under `inject` depending on the execution context around `ir.call(...)` or `ir.acall(...)`.

For more detail, see [Intercepts](../concepts/intercepts.md).
