```{include} ../README.md
```

`````{div} sd-p-3 sd-border sd-rounded-2

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Installation, tracing, execution, and transformations.
```

```{grid-item-card} Concepts
:link: concepts/index
:link-type: doc

Types, operations, feedback rules, and program transformations.
```

```{grid-item-card} Recipes
:link: recipes/index
:link-type: doc

Batching, control flow, model and tool calls, and custom types.
```

```{grid-item-card} Reference
:link: api/index
:link-type: doc

API signatures, parameters, and return values.
```

````

`````

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

why-autoform
getting-started
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

concepts/programs
concepts/tracing
concepts/transforms-and-rules
concepts/execution
concepts/values
concepts/control-flow-and-scoring
```

```{toctree}
:maxdepth: 2
:caption: Recipes
:hidden:

recipes/core/index
recipes/llm/index
recipes/execution/index
recipes/extending/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
reference/changelog
reference/glossary
```
