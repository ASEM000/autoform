# Trace, IR, Execute

`autoform` has three phases:

- **Trace**: run a Python function once with placeholder values and record `autoform` primitive calls.
- **Transform**: take the recorded IR and produce another IR, without re-running the original Python function.
- **Execute**: run the IR with concrete runtime inputs, synchronously or asynchronously.

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

The argument `"DNA"` is not the real input you intend to run forever. It is a shape/type witness. `trace` uses it to build placeholder input variables with the right Python leaf types, then runs the function body once under the tracing interpreter.

During that run:

- calls to `autoform` primitives such as `format`, `concat`, `lm_call`, `switch`, and `while_loop` become IR equations;
- ordinary Python that depends only on concrete or static values runs immediately and is baked into the trace;
- Python control flow that depends on a traced value is not available as a normal `if` or variable-length loop.

When the path depends on runtime data, use explicit control-flow primitives: `switch` for branching and `while_loop` for loops.

## The IR

An IR is the recorded program. It has input variables, equations, and output variables. You usually do not import or construct the IR classes directly; you get an IR from `af.trace(...)`.

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
- the first equation records the `format` primitive;
- the second equation records the `concat` primitive;
- `output` is the returned value.

For an LM program, the same mechanism records `lm_call` as an equation instead of calling the provider during tracing.

## Execute

The IR has two execution methods:

```python
output = ir.call("gravity")
print(output)
# Prompt: Explain gravity.
```

The runtime input `"gravity"` replaces the traced placeholder input. Execution walks the equations in order and dispatches each primitive to its registered implementation rule.

The async method runs the same IR through async primitive rules:

```python
import asyncio

output = asyncio.run(ir.acall("recursion"))
print(output)
# Prompt: Explain recursion.
```

The original function was not written as `async def`. Execution mode is chosen at the call site.

## Transform

A transform is a function from IR to IR. Given one traced program, you can create several transformed versions without re-running `label`:

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

`pullback(ir)` returns an IR. `batch(...)` accepts an IR. Neither transform needs to know how the original Python function was written.

## Execution Mode Is Separate

Execution mode is its own axis:

- `ir.call(...)` runs synchronously.
- `await ir.acall(...)` runs asynchronously.
- `af.sched(ir)` returns a scheduled IR where async execution is usually the useful path, because independent equations can run concurrently.
- `acall` is available even without `sched`, and `call` is available even after `sched`.

The choice is made where you run the IR, not where you define the function. Engineers may recognize this as avoiding the function-coloring problem: the function itself does not become permanently sync or async.

```{mermaid}
flowchart TD
    F["Python func"] --> T["trace(func)(...)"]
    T --> IR["IR"]
    IR --> B["batch"]
    IR --> P["pullback"]
    IR --> S["sched"]
    B --> IR2["IR'"]
    P --> IR2
    S --> IR2
    IR2 --> C[".call(...)"]
    IR2 --> A["await .acall(...)"]
    C --> O["output"]
    A --> O
```

## Common Gotchas

- **Python `if` on a traced value**: use `switch`, or mark the controlling input static if the branch is intentionally fixed at trace time.
- **Loops with runtime-dependent length**: use `while_loop`; ordinary Python loops are only appropriate when the iteration structure is known at trace time.
- **Mutating closure state**: pass state through the function inputs and outputs instead, preferably as registered pytrees for structured state.

Next, read `concepts/the-ir.md` for the IR structure in more detail.
