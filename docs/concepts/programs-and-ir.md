# Programs and IR

To run a program in `autoform`, trace a Python function to an intermediate representation (IR). Then, the IR can be transformed and executed for different inputs to take advantage of batched execution, feedback, and other transformations.

```{raw} html
:file: ../assets/program-lifecycle.svg
```

## Trace

Trace a function by passing it to {py:func}`trace <autoform.trace>`:

```python
import autoform as af


def label(topic: str) -> str:
    prompt = "Explain " + topic
    return "Prompt: " + prompt


ir = af.trace(label)("topic text")
```

Here, the input `"topic text"` tells {py:func}`trace <autoform.trace>` to expect a string input. During tracing, the string is replaced with a placeholder and the function body is run to record the operations performed. Later, the traced program can be run with a different string. See [Tracing](tracing.md) for which inputs must be static or can be dynamic.

## The IR

The IR is a tree of inputs, a tree of outputs, and a list of equations. Each equation records a primitive, its inputs and outputs, and additional static parameters and tags. Variables represent runtime values; literals are stored directly in the IR. The input and output trees preserve the structure of the pytrees passed to or returned from the Python function (see [Pytrees](pytrees.md#pytrees)).

The example above produces an IR with the following logical structure:

```text
input: topic
equations:
  prompt = concat("Explain ", topic)
  output = concat("Prompt: ", prompt)
output: output
```

Here, the first equation produces the `prompt` and the second produces the `output` returned. Each equation represents a single operation.

```text
out_vars = primitive(in_vars; static_params)
```

```{raw} html
:file: ../assets/ir-dataflow.svg
```

The IR records primitive calls, and does not store source code or provider responses. Calling a language model is recorded as an equation, but the call is only performed when the IR is run. Programs generally obtain IRs using {py:func}`trace <autoform.trace>` and need not worry about the internal classes.

## Execute

Run an IR by calling its `.call(...)` method:

```python
output = ir.call("another topic text")
print(output)
# Prompt: Explain another topic text
```

At runtime, the input `"another topic text"` is provided, and the equations are walked to dispatch each primitive to its implementation rule.

IRs can also be run asynchronously using the `.acall(...)` method, and other execution modes, checkpointing, caching, and manual stepping are available. See [Execution](execution.md) for details.

## Transform

Transformations take an IR and return a new IR. Apply transforms to create multiple variants without needing to re-run `label`:

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

Here, to compute the feedback for the inputs, the transforms are composed.

```python
feedback_batch = af.batch(af.pullback(ir))
```

Notice that {py:func}`pullback <autoform.pullback>` returns an IR and {py:func}`batch <autoform.batch>` takes an IR, and neither is aware of the original Python function.

## Tags

During tracing, use tags to attach metadata to equations for custom scheduling or running. Tags do not change the computation. Tag a region using {py:func}`tag <autoform.tag>`:

```python
def tagged_label(topic: str) -> str:
    with af.tag("draft"):
        prompt = "Explain " + topic
    return "Prompt: " + prompt


ir = af.trace(tagged_label)("topic text")
assert "draft" in ir.eqns[0].tags
assert "draft" not in ir.eqns[1].tags
```

Tags accumulate for nested blocks, but are not applied outside the block. Tags can be any hashable type, including strings, numbers, tuples of hashable types, and frozen dataclasses.

Tags can be used as the `cond` argument to {py:func}`sched <autoform.sched>`:

```python
scheduled = af.sched(ir, cond=lambda eqn: "draft" in eqn.tags)
assert scheduled.call("topic text") == "Prompt: Explain topic text"
```

Tags can also be accessed on equations during [manual execution](execution.md#manual-execution).
