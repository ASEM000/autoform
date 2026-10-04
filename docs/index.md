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

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

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
