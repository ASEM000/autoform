# Tool Ranking

A request can match several available tools, and the strongest match may still be too weak to justify a choice. This recipe evaluates each candidate against the request and conversation history, combines those ratings with prior weights, and leaves the final decision to caller code. A threshold determines whether the highest-ranked tool is selected or clarification is requested.

```{admonition} Concept
<a href="../../concepts/transforms.html?transform=weight#weight">Path Weights</a> · [Language Models](../../language-models.md) · [Transforms](../../concepts/transforms.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

## Candidates

The candidate tools and prior weights are fixed by the application before scoring. Keeping that choice outside the traced function allows the same scoring program to evaluate a different candidate set later. NumPy handles the prior weights and the final normalization:

```python
import numpy as np
import autoform as af


MODEL = "model-name"
tools = ["web_search", "code_interpreter", "file_reader", "ask_user"]
tool_descriptions = [
    "Search current public information.",
    "Run code to inspect data or compute results.",
    "Read workspace files.",
    "Request missing information.",
]
prior = np.array([0.3, 0.3, 0.2, 0.2])
```

## Scores

The model assesses two aspects of each candidate: its fit to the current request and its fit to the history. Both ratings are constrained to `[0, 1]`. Each becomes a factor in the traced program, so the path weight is the product of those ratings, `request_fit * history_fit`:

```python
fit_schema = {
    "request_fit": af.lm.Float(
        min=0,
        max=1,
        desc="request relevance instructions",
    ),
    "history_fit": af.lm.Float(
        min=0,
        max=1,
        desc="history relevance instructions",
    ),
    "reasoning": af.lm.Str(max=200, desc="score explanation instructions"),
}


def judge_tool(tool: str, description: str, request: str, history: str):
    content = dict(
        tool=tool,
        description=description,
        request=request,
        history=history,
        judgment=fit_schema,
    )
    judgment = af.lm.fill(content, model=MODEL)["judgment"]

    af.factor(judgment["request_fit"], name="request")
    af.factor(judgment["history_fit"], name="history")

    return {"tool": tool, "judgment": judgment}
```

## Batching

The scoring function is traced once with representative inputs. Applying {py:func}`weight <autoform.weight>` adds a path weight to its result, and {py:func}`batch <autoform.batch>` runs that scoring program for every candidate. This order keeps each candidate's weight separate:

```python
request = "request text"
history = "conversation history"

ir = af.trace(judge_tool)("web_search", tool_descriptions[0], request, history)
score_tools = af.batch(af.weight(ir), in_axes=(True, True, False, False))

outputs, path_weights = score_tools.call(
    tools,
    tool_descriptions,
    request,
    history,
)
```

## Normalization

Each tool appears once in the candidate set, so its path weight is multiplied by its explicit prior mass. Dividing the resulting masses by the total mass gives comparable decision scores. The calculation stays outside the IR because candidate selection and aggregation belong to the application.

```python
masses = prior * np.array(path_weights)
total = np.sum(masses)
if total <= 0:
    raise ValueError("No candidate has positive mass; review the scores.")
normalized_scores = masses / total

best_idx = int(np.argmax(normalized_scores))
best_tool = tools[best_idx]
top_score = normalized_scores[best_idx]
```

The model ratings are heuristic measures of fit. The normalized values are decision scores, not calibrated probabilities. A posterior interpretation would require factors with a likelihood meaning; normalization alone does not provide that meaning.

## Selection

The selection threshold makes the application policy explicit. A score above the threshold selects the highest-ranked tool; a lower score leads to a clarification request.[^human-review]

```python
threshold = 0.7

if top_score > threshold:
    print(f"Using {best_tool} ({top_score:.2f})")
else:
    print(f"Uncertain: top tool is {best_tool} at {top_score:.2f}")
    for tool, score in zip(tools, normalized_scores, strict=True):
        print(f"  {tool}: {score:.3f}")
    print("Route to a clarification step.")
```

The following illustrative ratings show how the policy behaves when one tool leads the ranking but does not clear the threshold:

| Tool | `request_fit` | `history_fit` | Path weight | Unnormalized mass | Normalized score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `web_search` | 0.8 | 0.7 | 0.56 | 0.168 | 0.512 |
| `code_interpreter` | 0.3 | 0.6 | 0.18 | 0.054 | 0.165 |
| `file_reader` | 0.6 | 0.8 | 0.48 | 0.096 | 0.293 |
| `ask_user` | 0.1 | 0.5 | 0.05 | 0.010 | 0.030 |

The top tool is `web_search`, but the normalized score is below `0.7`, so the caller asks for clarification instead of acting.

[^human-review]: The low-score branch can hand control to a human review runner. [Human Review](../execution/human-review.md) demonstrates execution that pauses, collects feedback, and resumes.
