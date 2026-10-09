# Types and Spaces

Programs can combine strings, numbers, and structured objects, but a transformation needs more than the ability to store those values. It also needs a representation for the information carried through the transformed program. A numerical derivative and a written critique are different kinds of information. Abstract values describe the data in the original program, while spaces specify its representation for a particular role.

A small review-formatting program makes the distinction concrete. The comment is text and the score is a number. The program adds a prefix to the comment and converts a score out of five into a score out of ten:

```python
import autoform as af


def format_review(comment: str, score: float) -> tuple[str, float]:
    return "Review: " + comment, score * 2.0


ir = af.trace(format_review)("review text", 2.0)
assert ir.call("review text", 2.0) == ("Review: review text", 4.0)
```

## Abstract Values

During tracing, an abstract value describes a runtime value without requiring its concrete contents. The comment has a string abstract value, and the score has a floating-point abstract value. Operation rules use these descriptions to determine output types before execution. A new value type needs an abstract value and the appropriate operation rules, as illustrated by [Array Extension](../recipes/extending/array-extension.md).

(spaces)=
(defining-feedback-spaces)=
## Spaces

A space maps an abstract value to a representation for a particular role. The primal space describes the original values. Tangent spaces describe the changes carried forward by {py:func}`pushforward <autoform.pushforward>`, and cotangent spaces describe the feedback carried backward by {py:func}`pullback <autoform.pullback>`. These are examples of spaces, rather than a requirement that all information resemble a numerical gradient.

The review program has two outputs, so its pullback receives a pair of feedback values. Text feedback belongs to the formatted comment, and numerical feedback belongs to the converted score. The string rule passes the critique to the comment; the multiplication rule doubles the score feedback:

```python
output, (comment_feedback, score_feedback) = af.pullback(ir).call(
    ("review text", 2.0),
    ("comment feedback", 1.0),
)
assert output == ("Review: review text", 4.0)
assert comment_feedback == "comment feedback"
assert score_feedback == 2.0
```

In this example, the cotangent has the same type as the primal value. A custom type can use a different representation: the cotangent for a document could be a critique of its structure and content.

## Zero and Accumulation

A value can contribute to several later operations, each returning its own feedback. Accumulation combines those contributions when the backward computation reaches the shared value. Zero represents the absence of a contribution. For the comment, zero is the empty string and accumulation concatenates text. For the score, zero is `0.0` and accumulation adds numbers.[^symbolic-zeros]

These definitions let the transform handle unused values and shared inputs without knowing the details of each feedback representation. The operation rules determine how feedback reaches the inputs, and the cotangent type supplies zero and accumulation. A custom critique type needs these definitions just as a numerical type does.

[^symbolic-zeros]: `autoform` can keep zero symbolic until an operation needs a concrete value.
