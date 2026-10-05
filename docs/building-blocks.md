# Building Blocks

`autoform` transforms programs that operate on text, numbers, and user-defined types. For example, a grading program can call a language model, take a text rubric as input, and perform arithmetic on its score. A `pullback` computes input feedback in the space defined for each type, and `batch` runs that feedback program across examples.

The framework provides three building blocks:

1. **Types and spaces** represent values and their changes or feedback.
2. **Operations** define computations on those values, with rules for execution and supported transforms.
3. **Transformations** use spaces and operation rules to produce a new program, which can itself be transformed.

```{raw} html
:file: assets/autoform-building-blocks.svg
```

## Programs

Tracing converts a Python function into an intermediate representation (IR). The IR records operations and their data dependencies, and can be transformed and run synchronously or asynchronously. See [Programs and IR](concepts/programs-and-ir.md) for details.

## Composition

Operations compose into programs, and transformations compose with each other. For example, `batch(pullback(ir))` computes input feedback for a batch of examples. Both transforms produce IR, so the result can be transformed again. Registered types, spaces, and operation rules determine how each transform handles values.

These building blocks support higher-level methods, including program optimization. [Motivation and Applications](motivation-and-applications.md) describes their uses. [A First Program](a-first-program.md) traces, runs, and transforms a language model program.

Note that `autoform` is in early stages of development, and breaking API changes may be made that require changes to existing code.
