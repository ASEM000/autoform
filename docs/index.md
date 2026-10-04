# `autoform`

```{include} ../README.md
:start-after: "</div>"
:end-before: "## Installation"
```

## Example

```{include} ../README.md
:start-after: "## Getting Started"
:end-before: "## More"
```

Model calls require [provider credentials](getting-started.md).

## Documentation

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

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting-started
```

```{toctree}
:maxdepth: 2
:caption: Concepts
:hidden:

why-autoform
concepts/index
```

```{toctree}
:maxdepth: 2
:caption: Recipes
:hidden:

recipes/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
reference/changelog
reference/glossary
```

```{warning}
Early development. Expect API changes that break existing code.
```
