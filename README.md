<div align="center">

# `autoform`


[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ASEM000/autoform/actions/workflows/ci.yml/badge.svg)](https://github.com/ASEM000/autoform/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ASEM000/autoform/graph/badge.svg?token=Z0JBHSC3ZK)](https://codecov.io/gh/ASEM000/autoform)


[Documentation](https://autoform.readthedocs.io) ·
[Getting started](https://autoform.readthedocs.io/en/latest/getting-started.html) ·
[Recipes](https://autoform.readthedocs.io/en/latest/recipes/) ·
[API reference](https://autoform.readthedocs.io/en/latest/api/)

</div>

`autoform` is a research framework for program transformation and optimization.

The goal of the project is to explore:

- Programs over user-defined types and operations.
- User-defined feedback and rules for propagating it through programs.
- Composable transformations for building optimization methods.

_Specifically, how can ideas from programming languages, deep learning, optimization, and compiler design be used in text-space programs?_

## Installation

`autoform` requires Python 3.12 or later.

```bash
pip install git+https://github.com/ASEM000/autoform.git
```

## Getting Started

`autoform` is composed of three building blocks: **types**, **operations** on those types, and **transformations** of programs using those operations.

The examples below compose `batch` and `pullback` over numerical and text programs.
The same transformation interfaces apply to both, with each operation supplying
its own rules.

<details>
<summary>Numerical example</summary>

The numerical example uses $f(x, y) = xy$, where $x, y \in \mathbb{R}$ are represented by
Python `float`. `pullback` computes gradients, and `batch` composed with
`pullback` computes them for multiple input pairs.

```python
import autoform as af


# --- program ---
def multiply(x: float, y: float) -> float:
    return x * y


# --- trace ---
# trace the program to intermediate representation (IR).
scalar_ir = af.trace(multiply)(3.0, 4.0)  # supply example argument

# --- batch ---
# apply the same program to several input pairs.
lhs, rhs = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
batched_scalar_ir = af.batch(scalar_ir)
print(batched_scalar_ir.call(lhs, rhs))  # [4.0, 10.0, 18.0]

# --- pullback ---
# propagate output feedback (a cotangent) backward to the inputs.
# for z = x*y, dz/dx = y and dz/dy = x. a seed of 1.0 gives these derivatives.
pullback_ir = af.pullback(scalar_ir)
value, (cotangent_x, cotangent_y) = pullback_ir.call((3.0, 4.0), 1.0)
print(value, cotangent_x, cotangent_y)  # 12.0 4.0 3.0

# --- composition ---
# pullback returns another IR, which can be composed with batch.
batched_pullback_ir = af.batch(af.pullback(scalar_ir))
values, (cotangents_x, cotangents_y) = batched_pullback_ir.call((lhs, rhs), [1.0, 1.0, 1.0])
print(values)  # [4.0, 10.0, 18.0]
print(cotangents_x)  # [4.0, 5.0, 6.0]
print(cotangents_y)  # [1.0, 2.0, 3.0]

# --- optimization ---
# update x toward multiply(x, 2) = 6 using squared-error feedback.
x, target = 1.0, 6.0
prediction = scalar_ir.call(x, 2.0)
_, (feedback, _) = pullback_ir.call((x, 2.0), 2.0 * (prediction - target))
candidate = x - 0.1 * feedback
print(scalar_ir.call(candidate, 2.0))  # 5.2
```

</details>

<details>
<summary>Text example</summary>

The text example uses $f(x) = \mathrm{LM}(p(x))$, where $x \in \Sigma^*$
is an instruction represented by Python `str`, and $p$ concatenates it with a
topic. `pullback` propagates textual feedback to the instruction; composing it
with `batch` computes feedback for multiple instructions.

<details>
<summary>Model setup</summary>

`autoform` using LiteLLM as backend client. Setting up the model for the following example requireds 
1. Filling `model` name.
2. Export the corresponding API to the environment. 

| Provider | `model` | Environment variables |
| --- | --- | --- |
| [OpenAI GPT-5.6](https://docs.litellm.ai/docs/providers/openai) | `openai/gpt-5.6` | `export OPENAI_API_KEY="api_key_here"` |
| [Anthropic Claude Sonnet 5](https://docs.litellm.ai/docs/providers/anthropic) | `anthropic/claude-sonnet-5` | `export ANTHROPIC_API_KEY="api_key_here"` |
| [Google Gemini Flash](https://docs.litellm.ai/docs/providers/gemini) | `gemini/gemini-flash-latest` | `export GEMINI_API_KEY="api_key_here"` |
| [Ollama — local Qwen2.5 3B](https://docs.litellm.ai/docs/providers/ollama) | `ollama_chat/qwen2.5:3b` | `export OLLAMA_API_BASE="http://localhost:11434"` |


</details>

```python
# --- model setup ---
import os
import autoform as af

os.environ["OPENAI_API_KEY"] = "api_key_here"
model = "openai/gpt-5.6"


# --- program ---
def explain(instruction: str) -> str:
    prompt = instruction + "\n<topic> program transformation"
    return af.lm.complete([dict(role="user", content=prompt)], model=model)


# --- trace ---
text_ir = af.trace(explain)("...")  # supply example argument
text_feedback_ir = af.pullback(text_ir)
instruction = "Explain the following topic briefly."
critique = "Provide formal explanation."

# --- batch ---
instructions = ["Explain in 200 words.", "Explain in 50 words."]
print(af.batch(text_ir).call(instructions))

# --- pullback ---
# the LM backward rule builds feedback from the input, output, and critique.
answer, (feedback,) = text_feedback_ir.call((instruction,), critique)

# --- composition ---
batched_text_feedback_ir = af.batch(af.pullback(text_ir))
answers, (feedbacks,) = batched_text_feedback_ir.call(
    (instructions,),
    ["Provide formal explanation.", "Provide simple example."],
)
print(feedbacks)

# --- optimization ---
# ask a model to revise the instruction (the program argument) using backward feedback.
revision_request = (
    "Revise the instruction based on the current intstruction and the feedback.\n"
    "<instruction>:\n" + instruction + "\n<feedback>:\n" + feedback
)
candidate = af.lm.complete([dict(role="user", content=revision_request)], model=model)

print(candidate)
```

</details>

## Walkthrough

To learn more about `autoform`'s building blocks: types, operations, transformations, and execution model, see [concepts](https://autoform.readthedocs.io/en/latest/concepts/index.html).


`autoform` supports custom types and operations, user-defined feedback rules,
and composable transformations for building optimization methods. Language model
workflows and tool-using agents can be expressed as programs over text and
structured values, combining model calls, tool calls, and control flow. see
[recipes](https://autoform.readthedocs.io/en/latest/recipes/index.html)

## Citation

If you use `autoform` in your research, please cite it:

```bibtex
@software{autoform,
  author       = {Asem, Mahmoud},
  title        = {AutoForm: A Research Framework for Program Transformation and Optimization},
  year         = {2026},
  url          = {https://github.com/ASEM000/autoform},
  doi          = {10.5281/zenodo.18071950},
  publisher    = {Zenodo},
  license      = {Apache-2.0}
}
```

> [!WARNING]
> Early developement: expect breaking changes.
