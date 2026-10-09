# Programs and IR

A Python function describes the computation to perform. To change how that computation runs or derive another computation from it, `autoform` first records it as an intermediate representation (IR). This separates writing a program from executing and transforming it. The example below follows a short text-formatting function through tracing, execution, and composition.

```{raw} html
:file: ../assets/program-lifecycle.svg
```

## Trace

The function builds a prompt from a topic using two string concatenations. Passing it to {py:func}`trace <autoform.trace>` records these operations with a placeholder for the topic:

```python
import autoform as af


def label(topic: str) -> str:
    prompt = "Explain " + topic
    return "Prompt: " + prompt


ir = af.trace(label)("topic text")
```

Here, the input `"topic text"` tells {py:func}`trace <autoform.trace>` to expect a string input. During tracing, the string is replaced with a placeholder and the function body is run to record the operations performed. Later, the traced program can be run with a different string. [Tracing](tracing.md) explains which inputs must be static or can be dynamic.

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

The first equation produces `prompt`, and the second uses that value to produce the result. This dependency is explicit in the IR, so a transform can work with the computation without inspecting Python source. Each equation has the following general form:

```text
out_vars = primitive(in_vars; static_params)
```

```{raw} html
:file: ../assets/ir-dataflow.svg
```

The IR stores primitive calls rather than a copy of the Python source. Under ordinary tracing, a language model call is recorded without making a provider request; the request happens when the IR executes. The {py:func}`trace <autoform.trace>` interface constructs these records from the function.

## Execute

Execution supplies a concrete value for the topic placeholder. The `.call(...)` method runs the recorded operations and returns the final result:

```python
output = ir.call("another topic text")
print(output)
# Prompt: Explain another topic text
```

At runtime, the input `"another topic text"` is provided, and the equations are walked to dispatch each primitive to its implementation rule.

IRs can also be run asynchronously using the `.acall(...)` method, and other execution modes, checkpointing, caching, and manual stepping are available. [Execution](execution.md) describes these runtime interfaces.

## Transform

A transform takes this recorded program and returns a new IR. For example, batching creates a version that accepts several topics together. The transform works from the existing IR and does not need to run the original Python function again:

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

The same IR can also produce a feedback program. Composing the transforms below first creates a pullback, then batches that feedback computation:

```python
feedback_batch = af.batch(af.pullback(ir))
```

{py:func}`pullback <autoform.pullback>` produces the IR that {py:func}`batch <autoform.batch>` consumes. The transforms share this representation rather than requiring separate versions of the original function. Supported combinations depend on the registered rules.

## Tags

Some runners need to recognize a particular part of a program, such as a draft that requires review. Tags attach this metadata to equations during tracing without changing the computed values. A {py:func}`tag <autoform.tag>` context marks the operations recorded inside its block:

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

A scheduling condition can inspect those tags to select equations. Here, the `cond` argument to {py:func}`sched <autoform.sched>` selects the region marked as a draft:

```python
scheduled = af.sched(ir, cond=lambda eqn: "draft" in eqn.tags)
assert scheduled.call("topic text") == "Prompt: Explain topic text"
```

Tags can also be accessed on equations during [manual execution](execution.md#manual-execution).
