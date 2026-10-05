# Tool Ranking

A request may have several candidate tools. Score each candidate against the request and history, combine the scores with prior weights, and normalize the resulting weights. Select the best tool only when its decision score exceeds the threshold; otherwise request clarification.

```{admonition} Concept
[Path Weights](../../concepts/path-weights.md) · [Language Models](../../language-models.md) · [Transforms](../../concepts/transforms.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Candidates

The candidate set stays outside the traced function. The prior and normalization
are ordinary NumPy arrays:

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

The schema asks the LM for two values in `[0, 1]`. Each value becomes a factor,
so the path weight is `request_fit * history_fit`:

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

Trace once with representative values, then compose {py:func}`batch
<autoform.batch>` around {py:func}`weight <autoform.weight>`:

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

The set of tools is enumerated only once, and the weights can be combined with the explicit prior mass over tools.

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

These LM ratings are heuristic fit scores. Read the normalized values as decision scores, not calibrated probabilities. A posterior interpretation requires factors that represent likelihoods.

## Selection

Use a threshold to decide whether to use a tool or ask for clarification:[^human-review]

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

For example, if the LM returned these scores:

| Tool | `request_fit` | `history_fit` | Path weight | Unnormalized mass | Normalized score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `web_search` | 0.8 | 0.7 | 0.56 | 0.168 | 0.512 |
| `code_interpreter` | 0.3 | 0.6 | 0.18 | 0.054 | 0.165 |
| `file_reader` | 0.6 | 0.8 | 0.48 | 0.096 | 0.293 |
| `ask_user` | 0.1 | 0.5 | 0.05 | 0.010 | 0.030 |

The top tool is `web_search`, but the normalized score is below `0.7`, so the
caller asks for clarification instead of acting.

[^human-review]: The low-score branch can hand control to a human review runner. See
    [Human Review](../execution/human-review.md) for the
    pattern where execution pauses, collects feedback, and resumes.
