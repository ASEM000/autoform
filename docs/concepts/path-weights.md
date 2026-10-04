# Path Weights

A *path* is one execution of some IR for some set of inputs at runtime. Inputs can lead to different paths through the IR due to branches, loops, batched inputs, LM calls, etc.

A *path weight* is extra information attached to the execution of a path, often used to decide if paths should be kept/rejected, or how good a generated path is, without changing the output of the program itself.

Use {py:func}`factor <autoform.factor>` to contribute a score to the path:

```python
import autoform as af


def label_path(label: str, weight: float) -> str:
    af.factor(weight, name="score")
    return label
```

During ordinary execution, `factor` is a no-output effect. During
{py:func}`weight <autoform.weight>` execution, each reached factor
multiplies the returned path weight:

```python
ir = af.trace(label_path)("p1", 1.0)
output, path_weight = af.weight(ir).call("p1", 0.9)
```

Mathematically, for a single path `p` (concrete execution):

```{math}
\mathrm{path\_weight}(p) =
\prod_{i \in \mathrm{reached}(p)} w_i
```

That value is the accumulated score for that concrete path.

```{raw} html
:file: ../assets/path-weight-channels.svg
```

## Accumulation

Two reached factors contribute the product of the factor values:

```python
def program(x: str, a: float, b: float) -> str:
    af.factor(a, name="a")
    af.factor(b, name="b")
    return x


ir = af.trace(program)("x1", 1.0, 1.0)
output, path_weight = af.weight(ir).call("x1", 0.5, 0.25)
```

`factor` contributes to the path-weight channel. It does not change the
function's returned value.

For the call above:

| Value | Result |
| --- | --- |
| `output` | `"x1"` |
| `path_weight` | `0.125` |

Both multipliers are included because both `factor` calls are reached on the
same concrete execution path.

## Batch Path Scoring

To score multiple paths, where each path should start with an independent path weight, wrap the {py:func}`weight
<autoform.weight>` call in a {py:func}`batch <autoform.batch>` call:

```python
single_factor_ir = af.trace(label_path)("p1", 1.0)
scored = af.batch(af.weight(single_factor_ir), in_axes=(True, True))

labels = ["p1", "p2"]
weights = [0.9, 0.2]
outputs, path_weights = scored.call(labels, weights)
```

Which is a common pattern for scoring paths.

1. Prepare the batched inputs.
2. Run `batch(weight(ir))`.
3. Use the returned `path_weights` in the caller.

Order matters:

| Expression | Meaning |
| --- | --- |
| `batch(weight(ir))` | Score many paths independently. The result has one weight per path. |
| `weight(batch(ir))` | Score one batched path. Reached factors across the whole batched execution multiply into one weight. |

## Boundaries

`factor` does not produce a value in the user program. It only contributes to the
path weight returned by {py:func}`weight <autoform.weight>`.

`weight(ir)` does not change the original output. It wraps that output with a
second value:

```python
output, path_weight = af.weight(ir).call(...)
```

Factors must be finite, non-negative numbers. A zero factor makes the path weight zero; later operations still run.

The returned `path_weight` is an ordinary Python number. Caller code decides what
to do with it after the IR call returns.

## Probability Reading

Probability is an interpretation layer over the same execution result. Treat
each candidate path as a candidate `x`, and treat each reached `factor` as
evidence compatibility. Then the path weight can be used as a likelihood-style
score.

In probability terms, the variables are:

| Term | Meaning |
| --- | --- |
| Candidate `x` | A concrete value being scored. |
| Evidence `e` | An observed condition used to score candidates. |
| Prior `P(x)` | The probability of candidate `x` before using `e`. |
| Likelihood `L(e \| x)` | How likely candidate `x` would be to generate evidence `e`. |
| Path weight `w(x)` | The value returned by `weight(ir)` for candidate `x`. |
| Posterior `P(x \| e)` | The normalized result after combining the prior and path weight. |

When the product of the factors represents the likelihood of the evidence given the candidate, the path weight can stand in for `L(e | x)`.

For exact enumeration, caller code can compute:

```{math}
\mathrm{mass}(x) = P(x)\,w(x)
```

```{math}
P(x \mid e) = \frac{\mathrm{mass}(x)}{\sum_{x'} \mathrm{mass}(x')}
```

`autoform` returns `w(x)`. The prior, aggregation, and normalization stay in
the caller.

The posterior reading depends on the meaning of the factors. If the factors are
calibrated likelihood terms, the normalized masses have the form of a posterior.
If the factors are heuristic scores, the same calculation is a normalized
decision score. Normalization requires a positive total mass.

If candidates are sampled from the prior instead of enumerated once, the prior
is already represented by sample frequency. In that case, aggregate the returned
path weights by candidate and normalize those masses.

The candidate source determines how to calculate mass:

| Candidate source | Caller-side mass |
| --- | --- |
| Enumerate each unique candidate once | `prior_mass * path_weight` |
| Sample candidates from the prior | Sum `path_weight` over samples with the same output |
