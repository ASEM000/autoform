# Trace, IR, Execute

`autoform` has three phases:

Trace records the function using example arguments. Transform rewrites the recorded IR. Execute runs the final IR with real inputs.

```{mermaid}
flowchart TD
    func["Python function + example args"] --> trace["Trace"]
    trace --> ir["IR"]
    ir --> transform["Transform"]
    transform --> transformed_ir["IR"]
    transformed_ir --> execute["Execute"]
    execute --> output["output"]
```

That split is the core model. The function is ordinary Python; the trace is data; transforms rewrite that data; execution happens later.

## Trace

Tracing starts with a normal function:

```python
import autoform as af


def label(topic: str) -> str:
    prompt = af.format("Explain {}.", topic)
    return af.concat("Prompt: ", prompt)


ir = af.trace(label)("DNA")
```

The argument `"DNA"` is not the real input for later runs. It is a shape/type witness. {py:func}`trace <autoform.trace>` uses it to build placeholder input variables with the right Python leaf types, then runs the function body once under the tracing interpreter. See [Tracing Semantics](tracing-semantics.md) for static and dynamic input rules.

During that run:

- calls to `autoform` primitives such as {py:func}`format <autoform.format>`, {py:func}`concat <autoform.concat>`, {py:func}`lm_call <autoform.lm_call>`, {py:func}`switch <autoform.switch>`, and {py:func}`while_loop <autoform.while_loop>` become IR equations;
- ordinary Python that depends only on concrete or static values runs immediately and is baked into the trace;
- Python control flow that depends on a traced value is not available as a normal `if` or variable-length loop.

When the path depends on runtime data, use explicit [control-flow primitives](primitives.md): {py:func}`switch <autoform.switch>` for branching and {py:func}`while_loop <autoform.while_loop>` for loops.

## The IR

An IR is the recorded program. It has input variables, equations, and output variables. Most code does not import or construct the IR classes directly; {py:func}`trace <autoform.trace>` returns the IR.

The example above records this logical structure:

```text
input: topic
equations:
  prompt = format(topic, template="Explain {}.")
  output = concat("Prompt: ", prompt)
output: output
```

Read it as data flow:

- `topic` is the runtime input;
- the first equation records the {py:func}`format <autoform.format>` primitive;
- the second equation records the {py:func}`concat <autoform.concat>` primitive;
- `output` is the returned value.

For an LM program, the same mechanism records {py:func}`lm_call <autoform.lm_call>` as an equation instead of calling the provider during tracing.

## Execute

The IR has two execution methods:

```python
output = ir.call("gravity")
print(output)
# prompt: Explain gravity.
```

The runtime input `"gravity"` replaces the traced placeholder input. Execution walks the equations in order and dispatches each primitive to its registered implementation rule.

The async method runs the same IR through async primitive rules:

```python
import asyncio

output = asyncio.run(ir.acall("recursion"))
print(output)
# prompt: Explain recursion.
```

The original function was not written as `async def`. Execution mode is chosen at the call site.

## Transform

A transform is a function from IR to IR. Given one traced program, several transformed versions can be created without re-running `label`:

```python
batched = af.batch(ir)

outputs = batched.call(["DNA", "gravity", "recursion"])
print(outputs)
# ['Prompt: Explain DNA.', 'Prompt: Explain gravity.', 'Prompt: Explain recursion.']
```

The same idea is why composition works:

```python
optimized_batch = af.batch(af.pullback(ir))
```

{py:func}`pullback <autoform.pullback>` returns an IR. {py:func}`batch <autoform.batch>` accepts an IR. Neither transform needs to know how the original Python function was written.

## Execution Axis

Execution mode is chosen at the IR boundary:

- `ir.call(...)` runs synchronously.
- `await ir.acall(...)` runs asynchronously.
- {py:func}`sched <autoform.sched>` returns a scheduled IR where async execution is usually the useful path, because independent equations can run concurrently.
- `acall` is available even without {py:func}`sched <autoform.sched>`, and `call` is available even after {py:func}`sched <autoform.sched>`.

The choice is made where the IR runs, not where the function is defined.
Use `.call(...)` for sync execution and `.acall(...)` for async execution.
This avoids the [function-coloring](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/) split: the original Python function stays ordinary, while `.call(...)` and `.acall(...)` choose execution at the IR boundary.

```{mermaid}
flowchart TD
    func["Python function"] --> trace["trace(func)(...)"]
    trace --> ir["IR"]
    ir --> batch["batch"]
    ir --> pullback["pullback"]
    ir --> sched["sched"]
    batch --> transformed_ir["transformed IR"]
    pullback --> transformed_ir
    sched --> transformed_ir
    transformed_ir --> call[".call(...)"]
    transformed_ir --> acall[".acall(...)"]
    call --> output["output"]
    acall --> output
```

## Gotchas

- Python `if` on a traced value: use {py:func}`switch <autoform.switch>` for runtime decisions. If the branch should be fixed while tracing, mark the controlling input {ref}`static <static-and-dynamic-inputs>` or use {ref}`fold <trace-time-decisions>`.
- Loops with runtime-dependent length: use {py:func}`while_loop <autoform.while_loop>`; ordinary Python loops are only appropriate when the iteration structure is known at trace time.
- Mutating closure state: pass state through the function inputs and outputs instead, preferably as registered [pytrees](pytrees.md) for structured state.

Next, read [The IR](the-ir.md) for the IR structure in more detail.
