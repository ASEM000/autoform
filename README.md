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

`autoform` is a framework for composable program transformations over user-defined types and operations.
Users define how feedback passes through operations and combine transformations to build optimization methods.

The project explores how to apply ideas from programming languages, deep learning,
optimization, and compiler design to text programs.

## Installation

`autoform` requires Python 3.12 or later.

```bash
pip install git+https://github.com/ASEM000/autoform.git
```

## Getting Started

`autoform` has three main pieces: **types**, **operations** on these types, and **transformations** of programs.

The following example optimizes a program with string and numerical inputs.
It updates a text rubric $r$ and a numerical scale $s$.
The language model generates a score $\mathrm{LM}(r, x)$ from the rubric $r$ and a fixed example $x$.
As in regression, the loss is the squared error between the scaled score and a fixed target $y$:

$$
\mathcal{L}(r, s) = (s \, \mathrm{LM}(r, x) - y)^2.
$$

With step size $\eta$, update the scale using the gradient for the generated score:

$$
s \leftarrow s - \eta \, \nabla_s \mathcal{L}(r, s).
$$

The model updates the rubric using textual feedback from `pullback`.
Replace `"model-name"` with a LiteLLM model name.
Replace the example and instruction labels with text for the task.

```python
import autoform as af

model = "model-name"
example = "example text"
target = 8.0  # illustrative reference score


def grading_loss(rubric: str, scale: float) -> float:
    content = dict(
        rubric=rubric,
        example=example,
        score=af.lm.Float(min=0, max=10, desc="grading instructions"),
    )
    result = af.lm.fill(content, model=model)
    error = scale * result["score"] - target
    return error * error


rubric = "rubric instructions"
scale = 0.8

# transform the program to return output along with
# textual feedback on rubric and numerical feedback (gradient) on scale
pullback = af.pullback(af.trace(grading_loss)(rubric, scale))

for step in range(3):
    loss, (rubric_fdbk, scale_fdbk) = pullback.call((rubric, scale), 1.0)
    print(step, loss)
    scale -= 0.005 * scale_fdbk
    content = dict(
        rubric=rubric,
        rubric_feedback=rubric_fdbk,
        new_rubric=af.lm.Str(desc="rubric update instructions"),
    )
    rubric = af.lm.fill(content, model=model)["new_rubric"]

print(scale, rubric)
```

## More

The [concepts guide](https://autoform.readthedocs.io/en/latest/concepts/index.html)
explains types, operations, feedback rules, and program transformations.
It describes how a Python program is traced into an intermediate representation (IR).
It then explains how the IR is transformed and executed.

The [recipes](https://autoform.readthedocs.io/en/latest/recipes/index.html)
show how to combine these pieces for specific tasks.
Examples cover batching, control flow, prompt optimization, and programs with model and tool calls.
The examples also show how to inspect execution and add custom types, operations, and feedback rules.

## Citation

Please cite AutoForm in research that uses it:

```bibtex
@software{autoform,
  author       = {Asem, Mahmoud},
  title        = {AutoForm: Extensible Framework for Program Transformations over User-Defined Types},
  year         = {2026},
  url          = {https://github.com/ASEM000/autoform},
  doi          = {10.5281/zenodo.18071950},
  publisher    = {Zenodo},
  license      = {Apache-2.0}
}
```

> [!WARNING]
> Early development. Expect API changes that break existing code.
