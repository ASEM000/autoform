# Building Blocks

A program transformation derives another computation from a program that has already been defined. For example, a grading program can combine a language model call with arithmetic. A pullback of that program can return text feedback for a rubric and numerical feedback for a points adjustment. Batching can then apply the feedback program across examples. `autoform` supports this pattern for text, numbers, and user-defined types.

The framework provides three building blocks:

1. **Types and spaces** describe the values in a program and the representations used for changes or feedback. These representations can differ: a document can receive a structured critique rather than another document.
2. **Operations** describe computations on those values. Execution rules perform the work, while transformation rules define the corresponding batching, feedback, or other behavior.
3. **Transformations** combine the rules for individual operations into a new program. A transformed program has the same IR interface and can participate in further supported transformations.

```{raw} html
:file: assets/autoform-building-blocks.svg
```

## Programs

Tracing turns a Python function into an intermediate representation (IR) that records operations and dependencies. Transforms work with this representation, so the original function does not need a separate implementation for each behavior. The resulting IR can run synchronously or asynchronously. [Programs and IR](concepts/programs-and-ir.md) introduces that sequence.

## Composition

Composition happens at two levels. Operations combine to form a program, and transforms combine to derive new programs. For example, `batch(pullback(ir))` first creates a feedback program and then applies it across examples. The available rules and the order of transformation determine which combinations are supported.

These building blocks support higher-level methods, including program optimization. [Motivation and Applications](motivation-and-applications.md) describes these applications. [A First Program](a-first-program.md) traces, runs, and transforms a language model program.

`autoform` is in early development. Breaking API changes may require updates to existing code.
