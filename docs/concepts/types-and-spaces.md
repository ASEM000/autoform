# Types and Spaces

Programs often combine strings, numbers, and structured objects. To transform such a program, `autoform` needs to know how to represent each value and the changes or feedback associated with it. Abstract values describe the data in the program. Spaces describe how that data is represented for a particular transformation.

Consider a program that formats a review. It prefixes a written comment and converts a score out of five to a score out of ten:

```python
import autoform as af


def format_review(comment: str, score: float) -> tuple[str, float]:
    return "Review: " + comment, score * 2.0


ir = af.trace(format_review)("review text", 2.0)
assert ir.call("review text", 2.0) == ("Review: review text", 4.0)
```

## Abstract Values

When `autoform` traces a program, it describes each runtime value with an abstract value. The comment has a string abstract value, and the score has a floating-point abstract value. These descriptions let operation rules determine the types of their results during tracing. Additional value types need their own abstract values and operation rules, as shown in [Array Extension](../recipes/extending/array-extension.md).

(spaces)=
(defining-feedback-spaces)=
## Spaces

A space maps an abstract value to a representation for a particular role. The primal space describes values in the original program. Tangent spaces describe changes carried forward by {py:func}`pushforward <autoform.pushforward>`, and cotangent spaces describe feedback carried backward by {py:func}`pullback <autoform.pullback>`. A custom leaf type may use different representations in each space.

The two outputs receive different kinds of feedback. The pullback passes text feedback to the comment and doubles the numerical feedback for the score:

```python
output, (comment_feedback, score_feedback) = af.pullback(ir).call(
    ("review text", 2.0),
    ("comment feedback", 1.0),
)
assert output == ("Review: review text", 4.0)
assert comment_feedback == "comment feedback"
assert score_feedback == 2.0
```

In this example, the feedback (cotangent) type is the same as the primal (value) type. A custom type can use a different representation. For example, the cotangent for a document may be a critique of its structure and content.

## Zero and Accumulation

A transform needs a zero when a value receives no feedback, and accumulation when several uses of a value contribute feedback. For the comment, zero is the empty string and accumulation concatenates text. For the score, zero is `0.0` and accumulation adds numbers.[^symbolic-zeros] These definitions belong to the cotangent type of each value.

An operation rule determines how feedback reaches each input. A custom cotangent type must define its own zero and accumulation.

[^symbolic-zeros]: `autoform` can keep zero symbolic until an operation needs a concrete value.
