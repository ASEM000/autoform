# Why `autoform` ?

A growing number of methods and tools are being developed to optimize LLM programs and prompts. These methods all have some mechanism for evaluating a program and propagating some signal between operations to make updates.

Writing this evaluation machinery out by hand is similar to writing forward and backward passes for a neural network by hand. If a model is written out by hand, a change in the model may require changes in both the forward and the backward. On the other hand, automatic differentiation (autodiff) frameworks allow users to define new operations and then the user can combine operations in new ways without having to define a new backward pass. The backward pass is composed from the rules for the individual operations. Check out this tutorial from PyTorch to see the difference between writing backward passes by hand and using autodiff [4] .

`autoform` provides building blocks to define transformations on programs that use strings, numbers, and user-defined types. To define a transformation, one needs to define rules for how to transform individual operations, and these rules are combined to transform larger programs. The result of a transformation is itself a program, which may itself be transformed. A key application of these building blocks is to define optimization methods, but many other uses are possible.

## Defining feedback spaces

When working with programs composed of text, it may be natural to use some structure in the feedback other than just a string. For example, when reviewing a document, one may want to associate each comment with a particular section of the document, and perhaps also want to keep track of which reviewer made each comment and what criteria was used to evaluate the document.

To support this, one can use `autoform` ’s extension mechanism to create a document type and an associated review type. The definition of the value and feedback space would look something like this:

| | Example review of a document |
| --- | --- |
| Value | `Document(sections={"introduction": "...", "methods": "...", "conclusion": "..."})` |
| Feedback | `Review(comments=[Comment(section="methods", issue="Why not compare to baseline X?", source="peer review"), Comment(section="conclusion", issue="Claim Y is not supported by the results.", source="self review")])` |
| Zero | `Review(comments=[])` |
| Accumulate | Combine two reviews, keeping track of which comments came from which review and which section they are associated with. |

One would then need to define how feedback for these types flows through operations. For example, one may have an operation that assembles sections into a document. The pullback of this operation would take a review of the document and return a review for each section. For the example review above, the pullback might return `Review(comments=[Comment(section="methods", issue="Why not compare to baseline X?", source="peer review")])` for the methods section and `Review(comments=[Comment(section="conclusion", issue="Claim Y is not supported by the results.", source="self review")])` for the conclusion section. One would also need a way to update the sections given the feedback.

This is just an example of how one might want to extend `autoform` . To actually define this extension one needs to register the types and rules for how to accumulate them and how they interact with operations. See Primitives or check out the array extension for an example of how to register these types and rules.

## Handling mixed types

In the example from the previous section, a rubric (text) was used to grade responses, and a scale (numerical) was used to adjust the rubric. The language model returned a numerical score, which was then used in numerical operations to compute the squared error.

When the pullback {py:func}`autoform.pullback` was taken, the function returned both text feedback for the rubric and a numerical gradient for the scale. The feedback for the scale was a number all the way through the loss and scale operations, but then was used to generate text feedback for the rubric when calling the language model. Finally, in the update loop, a language model was used to update the rubric, and a gradient step was used to update the scale.

This example shows how `autoform` can handle multiple types of values and feedback in the same program, and even allows for custom types to be used.

## Composing transforms

There are two levels of composition in `autoform` : operations can be composed into programs, and transforms can be composed. When a transform is applied to a program, it returns an intermediate representation (IR) that can be transformed again.

For example, in the example from the previous section, one could compute the feedback for a batch of rubrics and scales like this:

```python
ir = af.trace(grading_loss)(rubric, scale)
batched_feedback = af.batch(af.pullback(ir))
```

Here, the batch transform {py:func}`autoform.batch` is applied to the result of the pullback transform {py:func}`autoform.pullback` , which was applied to the IR of the `grading_loss` function. The same definition of the function was used for both the non-batched and batched version, and the transforms were composed to get the desired behavior. Similarly, if one defines an extension, the extension will work with other types and transforms as long as the necessary rules are defined, although the order of transforms may be important. See Transforms for more information.

## Defining optimization methods

`autoform` is a low-level library that operates on types, rules, and transforms. This low-level interface allows for a wide variety of higher-level frameworks and methods to be built on top of it. One example of a higher-level framework is defining optimization methods for LLM programs and prompts. For an example of this kind of framework, see DSPy’s optimizers [5] .

To define an optimization method, one needs a way to propagate feedback through a program, which is what `autoform` provides. However, there are many other things one can do with a traced program. See Getting Started and Trace, IR, Execute for more information.

Note that `autoform` is in early stages of development, and breaking API changes may be made that require changes to existing code.
