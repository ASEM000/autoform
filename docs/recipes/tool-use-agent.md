# Build a Tool-Use Agent

Build an agent as one traced function, then use [IR transforms](../concepts/transforms.md) around it. The agent below can ask for a local search observation, finish with `done`, and keep its loop state in a registered [pytree](../concepts/pytrees.md).

```{mermaid}
flowchart TD
    Q[/question/] --> I[(State)]
    I --> C{"continue?"}
    C -- yes --> B["body_ir"]
    B --> D{"tool?"}
    D -- search --> S["search branch"]
    S --> H["new history"]
    H --> C
    D -- done --> R[/result/]
    C -- no --> R
```

## Build the Agent

```python
import json
import optree
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import autoform as af


# decision is the structured output returned by the lm
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Decision:
    tool: str
    args: str
    answer: str
    status: str


# state is the value carried through the loop
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    history: str
    result: str
    status: str


# build the schema as a value-shaped instance
decision_schema = Decision(
    tool=af.Enum("search", "done") @ af.Doc("Tool to call next."),
    args=af.Str() @ af.Doc("Argument for the selected tool."),
    answer=af.Str() @ af.Doc("Current answer for the user."),
    status=af.Enum("continue", "done") @ af.Doc("Whether another step is needed."),
)


# custom marks this as one external tool boundary in the ir
# the body still runs normally during execution
@af.custom
def wikipedia_search(query: str) -> str:
    # ordinary python can live inside the boundary
    params = {"action": "opensearch", "format": "json", "limit": 3, "search": query}
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "autoform-docs/0.1"})
    with urlopen(request, timeout=10) as response:
        _, titles, summaries, links = json.loads(response.read().decode())
    pairs = zip(titles, summaries, links)
    rows = [f"{title}: {summary} ({link})" for title, summary, link in pairs]
    return "\n".join(rows) or "No results."


# batch receives the batch size, input axes, and input values
@wikipedia_search.set_batch
def batch_wikipedia_search(in_tree, /, *, call):
    batch_size, axes, values = in_tree
    del batch_size
    (queries,) = values
    (query_axis,) = axes

    # if query is broadcast, call the tool once and mark the output unbatched
    if not query_axis:
        return call(queries), False

    # if query is batched, call the tool once per query and mark output batched
    return [call(query) for query in queries], True


# pullback receives the primal output and feedback on that output
@wikipedia_search.set_pullback
def pullback_wikipedia_search(in_tree, /, *, call):
    del call
    (_, output), feedback = in_tree

    # return one cotangent because wikipedia_search has one input: query
    hint = af.format("Improve the search query. Feedback: {}. Result: {}", feedback, output)
    return (hint,)


def search_tool(query: str, history: str) -> str:
    result = wikipedia_search(query)
    return af.format("{}\nsearch({}): {}", history, query, result)


def done_tool(answer: str, history: str) -> str:
    return af.format("{}\ndone: {}", history, answer)


# trace each branch once; switch chooses between these at runtime
search_ir = af.trace(search_tool)("query", "history")
done_ir = af.trace(done_tool)("answer", "history")
tool_branches = {"search": search_ir, "done": done_ir}


def should_continue(state: State) -> bool:
    return state.status == "continue"


def step(state: State) -> State:
    system = "Use search when needed. Use done when the answer is ready."
    user = af.format("Question and history:\n{}", state.history)
    messages = [dict(role="system", content=system), dict(role="user", content=user)]
    decision = af.lm_schema_call(messages, model="gpt-5.2", schema=decision_schema)
    history = af.switch(decision.tool, tool_branches, decision.args, state.history)
    return State(history=history, result=decision.answer, status=decision.status)


example = State(history="Question: What is autoform?", result="", status="continue")

# while_loop takes traced condition and body programs
cond_ir = af.trace(should_continue)(example)
body_ir = af.trace(step)(example)


def agent(question: str) -> str:
    history = af.format("Question: {}", question)
    init = State(history=history, result="", status="continue")
    # max_iters keeps the agent bounded
    final = af.while_loop(cond_ir, body_ir, init, max_iters=4)
    return final.result


# trace the whole agent once, then execute with a real question
agent_ir = af.trace(agent)("What is autoform?")
answer = agent_ir.call("What is autoform?")
print(answer)
```

The provider decides which branch to run by returning a [`Decision` schema value](../concepts/schemas.md). [`switch`](../concepts/primitives.md) dispatches to the traced tool branch at execution time. [`while_loop`](../concepts/primitives.md) keeps applying `body_ir` while `should_continue` returns true, capped by `max_iters`.

`wikipedia_search` is a [custom boundary](../concepts/custom-rules.md) around a real HTTP call. The registered batch and pullback rules tell `autoform` how that boundary behaves under the transforms used below.

## Transform the Agent

The agent is still one IR:

```python
# batch runs the same agent ir over many questions
questions = ["What is autoform?", "How does batching fit?"]
answers = af.batch(agent_ir).call(questions)
```

Feedback can flow through the full loop:

```python
# pullback turns output feedback into question feedback
pb_agent = af.pullback(agent_ir)
answer, (question_hint,) = pb_agent.call(("What is autoform?",), "too vague")
```

For real tools, keep the branch signature stable: each branch here is `(args, history) -> history`. External APIs, retrieval systems, and calculators should sit behind traceable adapters or a custom boundary with rules for the intended transforms. See [Custom Rules](../concepts/custom-rules.md).
