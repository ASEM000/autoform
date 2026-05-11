# Build a Tool-Use Agent

Build an agent as one traced function, then use [IR transforms](../concepts/transforms.md) around it. The agent below can ask for a local search observation, finish with `done`, and keep its loop state in a registered [pytree](../concepts/pytrees.md).

```{admonition} Concept
[Transforms](../concepts/transforms.md) · [Pytrees](../concepts/pytrees.md) · [Schemas](../concepts/schemas.md) · [Primitives](../concepts/primitives.md)
```

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
decision_schema = Decision(tool=af.Enum("search", "done"), args=af.Str(), answer=af.Str(), status=af.Enum("continue", "done"))


# primitive wrapper called by traced programs
wikipedia_search_p = af.core.Prim("wikipedia_search")


def wikipedia_search(query: str) -> str:
    return wikipedia_search_p.bind(query)


# runtime implementation receives concrete python values
def impl_wikipedia_search(query: str, /) -> str:
    params = {"action": "opensearch", "format": "json", "limit": 3, "search": query}
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "autoform-docs/0.1"})
    with urlopen(request, timeout=10) as response:
        _, titles, summaries, links = json.loads(response.read().decode())
    pairs = zip(titles, summaries, links)
    rows = [f"{title}: {summary} ({link})" for title, summary, link in pairs]
    return "\n".join(rows) or "No results."


# tracing needs output shape without running the http call
def abstract_wikipedia_search(query, /):
    del query
    return af.core.TypedAVal(str)


# batch receives the batch size, input axes, and input values
def batch_wikipedia_search(in_tree, /):
    batch_size, axes, values = in_tree
    del batch_size
    query_axis = axes
    queries = values

    # if query is broadcast, call the primitive once and mark output unbatched
    if not query_axis:
        return wikipedia_search_p.bind(queries), False

    # if query is batched, call the primitive once per query and mark output batched
    return [wikipedia_search_p.bind(query) for query in queries], True


# pullback forward sweep records the query and output as residuals
def pull_fwd_wikipedia_search(query: str, /):
    output = wikipedia_search_p.bind(query)
    return output, (query, output)


# pullback backward sweep turns output feedback into query feedback
def pull_bwd_wikipedia_search(in_tree, /):
    (query, output), feedback = in_tree
    return af.format("Improve the Wikipedia search query. Query: {}. Feedback: {}. Result: {}", query, feedback, output)


af.core.impl_rules.set(wikipedia_search_p, impl_wikipedia_search)
af.core.abstract_rules.set(wikipedia_search_p, abstract_wikipedia_search)
af.core.batch_rules.set(wikipedia_search_p, batch_wikipedia_search)
af.core.pull_fwd_rules.set(wikipedia_search_p, pull_fwd_wikipedia_search)
af.core.pull_bwd_rules.set(wikipedia_search_p, pull_bwd_wikipedia_search)


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

The provider decides which branch to run by returning a [`Decision` schema value](../concepts/schemas.md). {py:func}`switch <autoform.switch>` dispatches to the traced tool branch at execution time. {py:func}`while_loop <autoform.while_loop>` keeps applying `body_ir` while `should_continue` returns true, capped by `max_iters`.

`wikipedia_search` is a [primitive](../concepts/primitives.md) written with the same pattern as [Write a Primitive](writing-primitives.md). The HTTP call stays in the runtime implementation, while the abstract, {py:func}`batch <autoform.batch>`, and {py:func}`pullback <autoform.pullback>` rules tell `autoform` how the external tool behaves when tracing or transforming the IR.

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

For real tools, keep the branch signature stable: each branch here is `(args, history) -> history`.
