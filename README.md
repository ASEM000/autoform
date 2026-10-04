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

`autoform` is an extensible framework for program transformations over user-defined types and operations.
Rules for individual operations compose to transform entire programs.
The result can be executed or transformed again.

Programs can mix text, numbers, and user-defined structures.
Extensions define how to represent changes or feedback for each type,
and how transformations handle its operations.
These building blocks support program optimization and other applications.

## Installation

`autoform` requires Python 3.12 or later.

```bash
pip install git+https://github.com/ASEM000/autoform.git
```

## Getting Started

The example optimizes a text rubric and a numerical scale.
A language model scores an example using the rubric.
The loss is the squared error between the scaled score and a target.
A single `pullback` returns textual feedback for the rubric and a numerical gradient for the scale.

Replace `"model-name"` with a LiteLLM model name,
and replace the example and instruction labels with text for the task.

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
explains tracing, types, spaces, and transformation rules.

The [recipes](https://autoform.readthedocs.io/en/latest/recipes/index.html)
show batching, control flow, prompt optimization, model and tool calls, and extensions.

## Citation

Cite `autoform` in research that uses it:

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

> **Warning**
>
> Early development. Expect API changes that break existing code.
