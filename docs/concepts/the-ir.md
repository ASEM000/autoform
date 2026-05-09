# The IR

An `autoform` IR is an equation list. Each equation has this conceptual shape:

```text
out_vars = primitive(in_vars; static_params)
```

The IR is not Python source and it is not bytecode. It is a small data structure that records the parts of your function that `autoform` can transform and execute later.

## The Pieces

- **`IRVar`**: a typed placeholder for a value that will exist at execution time.
- **`Prim`**: a named operation such as `format`, `concat`, or `lm_call`.
- **`IREqn`**: one equation: a primitive, input tree, output tree, static parameters, and tags.
- **`IR`**: the whole program: input tree, equation list, and output tree.

These classes live in `autoform.core`. They are useful for inspection and analysis, but they are not part of the recommended everyday user surface. Most code should get an IR from `af.trace(...)`, transform it, and run it.

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

Literal values can appear directly in an equation. Runtime values are represented as `IRVar` leaves.

## What You Can Do With It

Once you have an IR, there are three broad operations:

- **Query it**: helpers in `autoform.analysis` can find variable producers, equation dependencies, and liveness.
- **Transform it**: `batch`, `pushforward`, `pullback`, `sched`, and `dce` consume an IR and return another IR.
- **Execute it**: `.call(...)` and `.acall(...)` run the equation list with concrete inputs.

That is why the trace/transform/execute split matters. A transform does not need the original Python function. It only needs the equation list.

## What It Is Not

- It is not a graph database. The main representation is an ordered equation list.
- It is not Python source. You cannot recover arbitrary Python syntax from it.
- It is not a provider call log. An `lm_call` is one equation whose implementation runs later.
- It is not the public API you usually write against. It is the substrate that makes the public transforms compose.

## Inspecting An IR

For execution-time diagnostics, prefer checkpoints with `collect` and `inject`. For transform work, helpers in `autoform.analysis` can inspect producers, dependencies, and liveness. If an expected operation is missing, the original function probably used ordinary Python outside an `autoform` primitive. If an operation is present but not used, `dce(ir)` may be able to remove it.

The public workflow is still trace, transform, execute. Inspect these internals when you are debugging, analyzing, or writing a transform.
