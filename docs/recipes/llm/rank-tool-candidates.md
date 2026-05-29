# Rank Tool Candidates with Path Weights

Use this recipe to rank a fixed set of candidate tools against a request. The
traced function asks an LM for two fit scores, records those scores
with {py:func}`factor <autoform.factor>`, and {py:func}`weighted
<autoform.weighted>` returns one path weight per tool.

This recipe uses live provider calls. It requires an active LM client that
supports structured output. Change the `MODEL` constant to use a different
LiteLLM model.

```{admonition} Concept
[Path Weights](../../concepts/path-weights.md) · [Schemas](../../concepts/schemas.md) · [Transforms](../../concepts/transforms.md)
```

## Define Candidates and Priors

The candidate set stays outside the traced function. The prior and normalization
are ordinary NumPy arrays.

```python
import numpy as np
import autoform as af


MODEL = "gpt-5.5"
tools = ["web_search", "code_interpreter", "file_reader", "ask_user"]
tool_descriptions = [
    "Search the web for current public information.",
    "Run code to inspect data, logs, or computed results.",
    "Read files already available in the workspace.",
    "Ask the user for missing context before acting.",
]
prior = np.array([0.3, 0.3, 0.2, 0.2])
```

## Score One Tool

The schema asks the LM for two values in `[0, 1]`. Each value becomes a factor,
so the path weight is `request_fit * history_fit`.

```python
fit_schema = {
    "request_fit": af.Float(min=0, max=1)
    @ af.Doc("How well this tool handles the request. 0 is not at all; 1 is perfect."),
    "history_fit": af.Float(min=0, max=1)
    @ af.Doc("How consistent this tool is with the conversation history."),
    "reasoning": af.Str(max=200) @ af.Doc("One sentence explaining the scores."),
}


def judge_tool(tool: str, description: str, request: str, history: str):
    prompt = af.format(
        "Tool: {0}\n"
        "Description: {1}\n\n"
        "User request: {2}\n\n"
        "Conversation history: {3}\n\n"
        "Rate how well this tool fits the request and history.",
        tool,
        description,
        request,
        history,
    )
    msg = dict(role="user", content=prompt)
    judgment = af.lm_schema_call([msg], model=MODEL, schema=fit_schema)

    af.factor(judgment["request_fit"], name="request")
    af.factor(judgment["history_fit"], name="history")

    return {"tool": tool, "judgment": judgment}
```

The returned dictionary is the ordinary program output. The factors contribute
only to the path weight returned by `weighted(ir)`.

## Score Every Tool

Trace once with representative values, then compose {py:func}`batch
<autoform.batch>` around {py:func}`weighted <autoform.weighted>`.

```python
request = "<request body>"
history = "<history body>"

ir = af.trace(judge_tool)("web_search", tool_descriptions[0], request, history)
score_tools = af.batch(af.weighted(ir), in_axes=(True, True, False, False))

outputs, path_weights = score_tools.call(
    tools,
    tool_descriptions,
    request,
    history,
)
```

`batch(weighted(ir))` scores each tool independently. The first two inputs vary
by candidate; `request` and `history` are broadcast to every candidate.

## Normalize the Scores

The tools were enumerated exactly once, so combine each path weight with its
explicit prior mass.

```python
masses = prior * np.array(path_weights)
normalized_scores = masses / np.sum(masses)

best_idx = int(np.argmax(normalized_scores))
best_tool = tools[best_idx]
confidence = normalized_scores[best_idx]
```

If the LM scores are calibrated likelihood terms, `normalized_scores` has the
form of a posterior. If they are heuristic fit scores, read it as a normalized
decision score.

## Act or Ask

Use the normalized score to choose between acting and asking for clarification.

```python
threshold = 0.7

if confidence > threshold:
    print(f"Using {best_tool} ({confidence:.2f})")
else:
    print(f"Uncertain: top tool is {best_tool} at {confidence:.2f}")
    for tool, score in zip(tools, normalized_scores, strict=True):
        print(f"  {tool}: {score:.3f}")
    print("Route to a clarification step.")
```

The low-confidence branch can hand control to a human review runner. See
[Add Human Feedback with Walk](../execution/human-review-walk.md) for the
pattern where execution pauses, collects feedback, and resumes.

For example, if the LM returned these scores:

| Tool | `request_fit` | `history_fit` | Path weight | Unnormalized mass | Normalized score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `web_search` | 0.8 | 0.7 | 0.56 | 0.168 | 0.512 |
| `code_interpreter` | 0.3 | 0.6 | 0.18 | 0.054 | 0.165 |
| `file_reader` | 0.6 | 0.8 | 0.48 | 0.096 | 0.293 |
| `ask_user` | 0.1 | 0.5 | 0.05 | 0.010 | 0.030 |

The top tool is `web_search`, but the normalized score is below `0.7`, so the
caller asks for clarification instead of acting.
