# Why `autoform` ?

There are a number of methods and tools that aim to optimize programs over text spaces (e.g. programs with LM calls). Each of these methods has some way of evaluating a program and propagating a signal between operations in the program to make updates to it.

Writing this evaluation machinery out by hand is similar to writing forward and backward passes for a neural network by hand. If a model is written out by hand, a change in the model may require changes in both the forward and the backward. On the other hand, automatic differentiation (autodiff) frameworks allow users to define new operations and then the user can combine operations in new ways without having to define a new backward pass. The backward pass is composed from the rules for the individual operations.

`autoform` follows the same principle: define transformation rules for individual operations, then compose these rules to transform the whole program. Types can be numbers, strings, or user-defined structures, each with its own representation of changes or feedback. This provides a low-level foundation for higher-level frameworks and methods, including program optimization. The transformed program can be transformed again.

```{raw} html
:file: assets/program-optimization.svg
```

(defining-optimization-methods)=
## Program Optimization

Many optimization methods share basic tasks: evaluating a program and propagating feedback to its inputs. Rebuilding this machinery for each method becomes harder when programs combine text, numbers, and custom data structures. `autoform` provides spaces and transformation rules to make it reusable. Each optimization method supplies an objective, evaluation data, and an update policy, while different methods can share the same traced program and operation rules.

See [Getting Started](getting-started.md) and [Trace, IR, Execute](concepts/trace-ir-execute.md) for examples of tracing and transforming programs.

(defining-feedback-spaces)=
## Spaces

In `autoform`, a space is the way of representing a type from a program in a particular role. The primal space is the normal way of representing a type in the program. Other spaces might be the tangent space (for the {py:func}`pushforward <autoform.pushforward>` transform) or the cotangent space (for the {py:func}`pullback <autoform.pullback>` transform). Tangent values carry changes forward through operations; cotangent values carry sensitivities or feedback backward.

Explicitly defining spaces allows the same transform to work with different types of values. For example, when running the {py:func}`pullback <autoform.pullback>`, the type of the feedback for each program value needs to be specified. The cotangent space maps each program type to its feedback type. For numerical values this might be the gradient and for summaries this might be the set of missing facts and the set of unsupported claims.

An extension can use `Summary` as the primal type and `SummaryFeedback` as its cotangent type:

```python
summary = Summary(text="summary text")
feedback = SummaryFeedback(
    missing={"missing fact"},
    unsupported={"unsupported claim"},
)
```

A value can receive no feedback, or receive feedback from several uses. Zero represents the absent contribution. Accumulation combines multiple contributions for the same value. Numerical differentiation uses zero and addition for this purpose.

A summary can be used in two different reviews, one which checks if it is factual and one which ensures that it covers enough of the article. Thus the feedback needs to be combined. This can be done by taking the union of the sets of missing facts and the union of the sets of unsupported claims. The zero for this feedback type contains two empty sets. Combining this empty feedback with another review leaves that review unchanged.

An update operation revises the summary based on the feedback. An extension defines the types and their space mappings, zero and accumulation, and rules for transforming operations on these types. See [Primitives](concepts/primitives.md) and [Array Extension](recipes/extending/array-extension.md) for examples of type and rule registration.

(handling-mixed-types)=

A single program can have multiple spaces. A language model grades an example according to a text rubric. Its score is multiplied by a numerical scale, and the loss is the squared error against a target. A {py:func}`pullback <autoform.pullback>` gives a numerical gradient for the scale and textual feedback for the rubric through the LM rule. Each input can then be updated using its type of feedback.

## Transform Composition

There are two levels of composition in `autoform` : operations can be composed into programs, and transforms can be composed. When a transform is applied to a program, it returns an intermediate representation (IR) that can be transformed again.

Batching the pullback of this program gives feedback for multiple rubrics and scales:

```python
batched_feedback = af.batch(af.pullback(ir))
```

The {py:func}`batch <autoform.batch>` transform operates on the feedback IR returned by {py:func}`pullback <autoform.pullback>`. The original program stays the same. Extensions can compose with other types and transforms when the required rules are registered, although transform order can affect the result. See [Transforms](concepts/transforms.md) for more information.

Note that `autoform` is in early stages of development, and breaking API changes may be made that require changes to existing code.
