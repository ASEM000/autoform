# The IR

```{admonition} Advanced
:class: info

This page describes the IR as a concept so transform behavior is easier to reason about. Most code should get an IR from {py:func}`trace <autoform.trace>`, then use public transforms and execution methods rather than constructing internal IR classes directly.
```

An `autoform` IR is an equation list. Each equation has this conceptual shape:

```text
out_vars = primitive(in_vars; static_params)
```

The IR is not Python source and it is not bytecode. It is a small data structure that records the parts of a function that `autoform` can transform and execute later.

## IR Components

- input and output trees describe the runtime values entering and leaving the program;
- equations record one primitive call, its input tree, its output tree, static parameters, and tags;
- primitive names identify operations such as {py:func}`format <autoform.format>`, {py:func}`concat <autoform.concat>`, and {py:func}`lm_call <autoform.lm_call>`;
- the whole IR is the input tree, the equation list, and the output tree.

Most code should get an IR from {py:func}`trace <autoform.trace>`, transform it, and run it. Direct construction of the internal IR classes is not needed.

## Worked Example

Start with a short function:

```python
import autoform as af


def label(topic: str) -> str:
    prompt = af.format("Explain {}.", topic)
    return af.concat("Prompt: ", prompt)


ir = af.trace(label)("DNA")
```

The trace contains this logical equation list:

```text
input: topic
equations:
  prompt = format(topic, template="Explain {}.")
  output = concat("Prompt: ", prompt)
output: output
```

Read it left to right:

- `topic` is the runtime input.
- {py:func}`format <autoform.format>` produces `prompt`.
- {py:func}`concat <autoform.concat>` consumes the literal `"Prompt: "` and `prompt`, then produces `output`.
- `output` is the function output.

Literal values can appear directly in an equation. Runtime values are represented
by placeholders until execution supplies concrete inputs.

## IR Operations

Once an IR exists, there are two broad operations:

- Transform it: {py:func}`batch <autoform.batch>`, {py:func}`pushforward <autoform.pushforward>`, {py:func}`pullback <autoform.pullback>`, {py:func}`sched <autoform.sched>`, and {py:func}`dce <autoform.dce>` consume an IR and return another IR.
- Execute it: `.call(...)` and `.acall(...)` run the equation list with concrete inputs.

That is why the trace/transform/execute split matters. A transform does not need the original Python function. It only needs the equation list.

## Non-Goals

- It is not a graph database. The main representation is an ordered equation list.
- It is not Python source. Recovering arbitrary Python syntax from it is not supported.
- It is not a provider call log. An {py:func}`lm_call <autoform.lm_call>` is one equation whose implementation runs later.
- It is not the usual [public API](../api/index.md) for application code. It is the substrate that makes the public transforms compose.

## IR Inspection

For execution-time diagnostics, prefer {py:func}`checkpoint <autoform.checkpoint>` with {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>`.
If an expected operation is missing, the original function probably used
ordinary Python outside an `autoform` primitive. If an operation is present but
not used, {py:func}`dce <autoform.dce>` may be able to remove it.

The public workflow is still trace, transform, execute. Inspect these internals for debugging, analysis, or transform implementation work.
