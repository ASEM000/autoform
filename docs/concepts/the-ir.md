# The IR

An `autoform` IR is an equation list. Each equation has this conceptual shape:

```text
out_vars = primitive(in_vars; static_params)
```

The IR is not Python source and it is not bytecode. It is a small data structure that records the parts of your function that `autoform` can transform and execute later.

## The Pieces

- input and output trees describe the runtime values entering and leaving the program;
- equations record one primitive call, its input tree, its output tree, static parameters, and tags;
- primitive names identify operations such as `format`, `concat`, and `lm_call`;
- the whole IR is the input tree, the equation list, and the output tree.

Most code should get an IR from `af.trace(...)`, transform it, and run it. You do
not need to construct the internal IR classes directly.

## A Worked Example

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
- `format(...)` produces `prompt`.
- `concat(...)` consumes the literal `"Prompt: "` and `prompt`, then produces `output`.
- `output` is the function output.

Literal values can appear directly in an equation. Runtime values are represented
by placeholders until execution supplies concrete inputs.

## What You Can Do With It

Once you have an IR, there are three broad operations:

- **Transform it**: `batch`, `pushforward`, `pullback`, `sched`, and `dce` consume an IR and return another IR.
- **Execute it**: `.call(...)` and `.acall(...)` run the equation list with concrete inputs.

That is why the trace/transform/execute split matters. A transform does not need the original Python function. It only needs the equation list.

## What It Is Not

- It is not a graph database. The main representation is an ordered equation list.
- It is not Python source. You cannot recover arbitrary Python syntax from it.
- It is not a provider call log. An `lm_call` is one equation whose implementation runs later.
- It is not the public API you usually write against. It is the substrate that makes the public transforms compose.

## Inspecting An IR

For execution-time diagnostics, prefer checkpoints with `collect` and `inject`.
If an expected operation is missing, the original function probably used
ordinary Python outside an `autoform` primitive. If an operation is present but
not used, `dce(ir)` may be able to remove it.

The public workflow is still trace, transform, execute. Inspect these internals when you are debugging, analyzing, or writing a transform.
