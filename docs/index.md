```{include} ../README.md
:end-before: <picture id="grading-program">
```

```{raw} html
:file: assets/grading-program.svg
```

```{include} ../README.md
:start-after: <!-- end grading-program -->
:end-before: <picture id="grading-loss">
```

```{raw} html
:file: assets/grading-loss.svg
```

```{include} ../README.md
:start-after: <!-- end grading-loss -->
:end-before: <picture id="mixed-feedback">
```

```{raw} html
:file: assets/mixed-feedback.svg
```

```{include} ../README.md
:start-after: <!-- end mixed-feedback -->
:end-before: <picture id="batched-feedback">
```

```{raw} html
:file: assets/batched-feedback.svg
```

```{include} ../README.md
:start-after: <!-- end batched-feedback -->
```

`````{div} sd-p-3 sd-border sd-rounded-2

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} A First Program
:link: a-first-program
:link-type: doc

Trace, run, and transform a language model program.
```

```{grid-item-card} Concepts
:link: concepts/index
:link-type: doc

Programs, spaces, rules, and transforms.
```

```{grid-item-card} Recipes
:link: recipes/index
:link-type: doc

Tool use, tool ranking, human review, and array extensions.
```

```{grid-item-card} API Reference
:link: api/index
:link-type: doc

API signatures, parameters, and return values.
```

````

`````

```{toctree}
:maxdepth: 2
:caption: Introduction
:hidden:

a-first-program
building-blocks
motivation-and-applications
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/foundations
concepts/computation-and-transforms
concepts/execution-overview
```

```{toctree}
:maxdepth: 2
:caption: Resources
:hidden:

recipes/index
api/index
reference/changelog
reference/glossary
```
