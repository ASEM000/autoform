# Types and Spaces

Data in programs includes a variety of types including strings, numbers, and structured objects. To transform a program, AutoForm also needs a way to represent information in the transformed program, e.g. the numerical derivative or a critique of the results. Abstract values are used to describe information in the original program, and spaces describe how to represent that information in a particular role.

Below is a simple program that formats a review into a standard form:

```python
import autoform as af


def format_review(comment: str, score: float) -> tuple[str, float]:
    return "Review: " + comment, score * 2.0


ir = af.trace(format_review)("review text", 2.0)
assert ir.call("review text", 2.0) == ("Review: review text", 4.0)
```

## Abstract Values

Abstract values are descriptions of what types of values might appear at a particular location in a program at runtime. When tracing a program, the abstract values describe what type each value in the original program has, e.g. a string for the comment and a floating point number for the score. These are then used with rules for each operation to determine the type of the outputs. When extending AutoForm to a new type of value, new abstract values and operation rules will need to be defined - see the [Array Extension](../recipes/extending/array-extension.md) recipe for an example.

## Joins and Checking

Abstract values can be joined by calling ``aval.join(other)``. The other value can be another AVal, a concrete value, or a symbolic zero. The result should be an AVal that describes both inputs. Note that when one input is a concrete value, it only "describes" itself, e.g. joining a float range with a float outside the range will expand the bounds of the range but a float inside the range won't change it.

Abstract value joining is defined directly on the ``AVal`` class and should either return an AVal or raise a ``TypeError``. There is no separate join registration and the call is not reflected. Methods for built-in scalar AVals also accept the corresponding concrete type, AVal type (including subclasses), and symbolic zero carrying that AVal type. Joining on Int AVals specifically rejects booleans. Subclasses of AVals inherit the join methods, but their value types and space mappings still need explicit registration for tracing and spaces. AVals that refine other AVals with additional constraints should override ``join`` to compare/combine the metadata.

Checking can be implemented using joining - the check is valid if joining the expected AVal with the value gives back the expected AVal (i.e. the value doesn't expand the description). Symbolic zeros are passed directly to ``join``, so each implementation handles the zeros it supports. Checking is done the same way for both execution and tracing, and the original value is returned after checking. When refining an AVal with metadata, the metadata should be included both when joining (to ensure e.g. that an expanded range or different shape results in a different AVal) and in the equality check (to ensure that two AVals that describe different values are not equal). Note that when joining two AVals, if both directions are supported they should both result in an AVal that describes the same set of values. However, when checking, AutoForm will only call the method on the expected AVal. Defining a ``join`` method on an AVal does not define rules for propagating it through primitives.

Joining can also be called on ``ABatch`` instances. When joining two batches, the bases of the batches are joined. When joining a batch with a raw container, all of the immediate children of the container are joined with the base of the batch. This allows for e.g. batches of batches or batches of user-defined pytrees without requiring special cases in the checking code. Note that joining a batch with an empty container will result in the same batch - the container describes nothing so the base is not expanded. Joining will raise a ``TypeError`` for an unrelated AVal, a scalar leaf in place of a container, or a child that cannot be joined with the base. AVals that refine composite types with additional constraints should override ``join`` to compare/combine the metadata about the container.

(spaces)=
(defining-feedback-spaces)=
## Spaces

A space describes a representation for an abstract value in a certain role. The primal space describes the original value. When applying {py:func}`pushforward <autoform.pushforward>` to a program, the information being passed is in the tangent spaces. Similarly, when applying {py:func}`pullback <autoform.pullback>` to a program, the information being passed is in the cotangent spaces.

Since there are two outputs to the program, the pullback will receive a pair of values describing feedback about each of the outputs. In this case, the feedback on the text is associated with the formatted comment and the feedback on the numerical score is associated with the converted score. Using the rules for strings and multiplication, the critique is passed through to the comment and the score feedback is doubled:

```python
output, (comment_feedback, score_feedback) = af.pullback(ir).call(
    ("review text", 2.0),
    ("comment feedback", 1.0),
)
assert output == ("Review: review text", 4.0)
assert comment_feedback == "comment feedback"
assert score_feedback == 2.0
```

In this case, the cotangent has the same type as the primal value. However, for a custom type it may be useful to have a different type for the cotangent, e.g. a critique of a document.

## Zero and Accumulation

In a program, a value may be used in several places, resulting in multiple pieces of feedback. When the values are accumulated, a zero value represents the lack of any feedback. For the comment, the zero value is the empty string and accumulation is concatenation. For the score, the zero value is 0.0 and accumulation is addition.[^symbolic-zeros]

To support this, AutoForm needs to know how to accumulate cotangent values. The rules for each operation determine how feedback reaches its inputs. To do the accumulation, the framework will need to know how to get the zero value and how to add two values together for a particular type. This information is provided by the AVal for the cotangent. For a custom type, these will need to be defined for the cotangent type.

[^symbolic-zeros]: AutoForm will attempt to keep the zero value symbolic until an operation needs a concrete value.

Getting the zero value and adding values together is defined directly on the ``AVal`` class. When accumulating multiple contributions, AutoForm will use ``ft.reduce(aval.add, values, aval.zero())`` with the concrete types. This allows the zero value and addition to use any information from the metadata, e.g. shape and dtype. Subclasses of AVals inherit the methods using standard Python inheritance. The methods on the base AVal will raise ``NotImplementedError`` - they are not needed for checking, but if they are implemented then the zeros and results of addition should pass checking. Subclasses that refine an AVal can override the methods to change the behavior. Spaces can also map the AVal to change how it is described when accumulating, e.g. in the tangent or cotangent spaces.

When doing an accumulation, symbolic zeros and single live cotangents can be passed through without needing to call these methods. When checking at the boundary, symbolic zeros and concrete contributions will be passed to ``join`` unchanged.

When a rule for a transform needs a concrete zero, the local zero value will need to be computed concretely. This can be done by calling ``zero.aval.zero()``. See the [Array Extension](../recipes/extending/array-extension.md) recipe for a full example.
