# The IR

```{admonition} Advanced
:class: info

This page describes the IR as a concept so transform behavior is easier to reason about. Most code should get an IR from {py:func}`trace <autoform.trace>`, then use public transforms and execution methods rather than constructing internal IR classes directly.
```

An `autoform` IR is a list of equations. Each equation is conceptually of the form:

```text
out_vars = primitive(in_vars; static_params)
```

Equations capture individual operations for later execution or transformation. Literal values are stored directly in the IR, variables are values which are provided or calculated at run time.

## IR Components

An `autoform` IR contains the following:

- input and output trees describe the runtime values entering and leaving the program;
- equations record one primitive call, its input tree, its output tree, static parameters, and tags;
- primitive keys identify operations such as {py:func}`concat <autoform.string.concat>` and {py:func}`fill <autoform.lm.fill>`;
- the whole IR is the input tree, the equation list, and the output tree.

Most code should get an IR from {py:func}`trace <autoform.trace>`, transform it, and run it. Direct construction of the internal IR classes is not needed.

## Example

Start with a short function:

```python
import autoform as af


def label(topic: str) -> str:
    prompt = "Explain " + topic
    return "Prompt: " + prompt


ir = af.trace(label)("topic text")
```

The IR which traces this function contains the following list of equations (logically speaking):

```text
input: topic
equations:
  prompt = concat("Explain ", topic)
  output = concat("Prompt: ", prompt)
output: output
```

The equations express the data flow of the function:

```{raw} html
:file: ../assets/ir-dataflow.svg
```

- `topic` is the runtime input.
- The first {py:func}`concat <autoform.string.concat>` equation builds `prompt`.
- {py:func}`concat <autoform.string.concat>` consumes the literal `"Prompt: "` and `prompt`, then produces `output`.
- `output` is the function output.

Further, literal values are directly included in the equation, and run-time values are represented as placeholders:

## IR Operations

An IR supports transformation and execution:

- Transform it: {py:func}`batch <autoform.batch>`, {py:func}`pushforward <autoform.pushforward>`, {py:func}`pullback <autoform.pullback>`, {py:func}`sched <autoform.sched>`, {py:func}`dce <autoform.dce>`, and {py:func}`weight <autoform.weight>` consume an IR and return another IR.
- Execute it: `.call(...)` and `.acall(...)` run the equation list with concrete inputs.

Transforms use the recorded operations and registered rules. A transform does not need to run the original Python function again.

## Limits

The operations and registered behavior are:

- It is not a graph database. The main representation is an ordered equation list.
- It is not Python source. Recovering arbitrary Python syntax from it is not supported.
- It is not a provider call log. An {py:func}`fill <autoform.lm.fill>` is one equation whose implementation runs later.
- It is not the usual [public API](../api/index.md) for application code. Public transforms operate on it.

## IR Inspection

For diagnostic purposes at execution time, {py:func}`checkpoint <autoform.checkpoint>` can be used in conjunction with {py:func}`collect <autoform.collect>` and {py:func}`inject <autoform.inject>`. If an expected operation does not appear in the IR, the most likely cause is that the traced function executed plain python code rather than an `autoform` primitive. If an operation appears to not be used, running {py:func}`dce <autoform.dce>` may eliminate it.

While it can be useful to inspect the equation list for debugging, analysis, or implementing a transform, most applications should only need to use the publicly-exposed transforms and execution methods.
