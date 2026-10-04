# Trace, IR, Execute

There are three main ways to interact with `autoform`: passing a Python function to `autoform` to create an IR, applying transforms to an IR to create new IRs, and executing an IR on actual inputs.

```{raw} html
:file: ../assets/program-lifecycle.svg
```

## Trace

Passing a function to `autoform.trace` starts a trace:

```python
import autoform as af


def label(topic: str) -> str:
    prompt = "Explain " + topic
    return "Prompt: " + prompt


ir = af.trace(label)("topic text")
```

The argument `"topic text"` tells {py:func}`trace <autoform.trace>` that the input is a string. Tracing replaces it with a placeholder and runs the function body once to record its operations. Later calls can supply another string. See [Tracing Semantics](tracing-semantics.md) for static and dynamic input rules.

During that run:

- calls to `autoform` primitives such as {py:func}`concat <autoform.string.concat>`, {py:func}`fill <autoform.lm.fill>`, {py:func}`switch <autoform.switch>`, and {py:func}`while_loop <autoform.while_loop>` become IR equations;
- ordinary Python that depends only on concrete or static values runs immediately and is baked into the trace;
- Python control flow that depends on a traced value is not available as a normal `if` or variable-length loop.

When the path depends on runtime data, use explicit [control-flow primitives](primitives.md): {py:func}`switch <autoform.switch>` for branching and {py:func}`while_loop <autoform.while_loop>` for loops.

## The IR

An IR is the recorded program. It has input variables, equations, and output variables. Most code does not import or construct the IR classes directly; {py:func}`trace <autoform.trace>` returns the IR.

The example above records this logical structure:

```text
input: topic
equations:
  prompt = concat("Explain ", topic)
  output = concat("Prompt: ", prompt)
output: output
```

Read it as data flow:

- `topic` is the runtime input;
- the first {py:func}`concat <autoform.string.concat>` equation builds `prompt`;
- the second adds the `"Prompt: "` prefix;
- `output` is the returned value.

## Execute

IRs can be executed synchronously with the `.call(...)` method.

```python
output = ir.call("another topic text")
print(output)
# Prompt: Explain another topic text
```

At runtime, the input `"another topic text"` is provided, and then the equations are walked and each primitive is dispatched to the appropriate implementation rule.

The async method runs the same IR through async primitive rules:

```python
import asyncio

output = asyncio.run(ir.acall("another topic text"))
print(output)
# Prompt: Explain another topic text
```

The original function was not written as `async def`. Execution mode is chosen at the call site.

## Transform

A transform is a function from IR to IR. Given one traced program, several transformed versions can be created without re-running `label`:

```python
batched = af.batch(ir)

outputs = batched.call(["topic text 1", "topic text 2", "topic text 3"])
print(outputs)
# [
#     "Prompt: Explain topic text 1",
#     "Prompt: Explain topic text 2",
#     "Prompt: Explain topic text 3",
# ]
```

To compute input feedback for several examples, compose the transforms:

```python
feedback_batch = af.batch(af.pullback(ir))
```

{py:func}`pullback <autoform.pullback>` returns an IR. {py:func}`batch <autoform.batch>` accepts an IR. Neither transform needs to know how the original Python function was written.

## Execution Modes

Finally, note again that the decision of whether to run in sync or async mode is made at the IR boundary:

- `ir.call(...)` runs synchronously.
- `await ir.acall(...)` runs asynchronously.
- {py:func}`sched <autoform.sched>` returns a scheduled IR where async execution is usually the useful path, because independent equations can run concurrently.
- `acall` is available even without {py:func}`sched <autoform.sched>`, and `call` is available even after {py:func}`sched <autoform.sched>`.

The Python function stays the same in both modes. Independent equations can overlap when a scheduled IR runs asynchronously:

```{raw} html
:file: ../assets/execution-modes.svg
```

## Tracing Limits

- Python `if` on a traced value: use {py:func}`switch <autoform.switch>` for runtime decisions. If the branch should be fixed while tracing, mark the controlling input {ref}`static <static-and-dynamic-inputs>` or use {ref}`fold <trace-time-decisions>`.
- Loops with runtime-dependent length: use {py:func}`while_loop <autoform.while_loop>`; ordinary Python loops are only appropriate when the iteration structure is known at trace time.
- Mutating closure state: pass state through the function inputs and outputs instead, preferably as registered [pytrees](pytrees.md) for structured state.
