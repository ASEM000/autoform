# Build a Tool-Use Agent

Build an agent as one traced function, then use [IR transforms](../concepts/transforms.md) around it. The agent below can ask for a search observation, finish with `done`, keep its loop state in a registered [pytree](../concepts/pytrees.md), and run tool calls through async primitive rules.

```{admonition} Concept
[Transforms](../concepts/transforms.md) · [Pytrees](../concepts/pytrees.md) · [Schemas](../concepts/schemas.md) · [Primitives](../concepts/primitives.md)
```

```{mermaid}
flowchart TD
    question["question"] --> state["state"]
    state --> should_continue{"continue?"}
    should_continue -- "yes" --> step["agent step"]
    step --> tool{"tool?"}
    tool -- "search" --> search["search branch"]
    search --> state
    tool -- "done" --> result["result"]
    should_continue -- "no" --> result
```

## Build the Agent

```python
import asyncio
import optree
from urllib.parse import urlencode

import httpx
import autoform as af
import autoform.extend as afe


# decision is the structured output returned by the lm
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class Decision:
    tool: str
    args: str
    answer: str


# state is the value carried through the loop
@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    history: str
    result: str
    active: bool


# build the schema as a value-shaped instance
decision_schema = Decision(
    tool=af.Enum("search", "done")
    @ af.Doc("Use search if history has no search line. Use done if history already has search."),
    args=af.Str() @ af.Doc("Search query when tool is search. Empty when tool is done."),
    answer=af.Str() @ af.Doc("Final answer when tool is done. Empty when tool is search."),
)


# primitive wrapper called by traced programs
wikipedia_search_p = afe.Prim("wikipedia_search")


def wikipedia_search(query: str) -> str:
    return wikipedia_search_p.bind(query)


def wikipedia_url(query: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 3,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "exsentences": 2,
    }
    return "https://en.wikipedia.org/w/api.php?" + urlencode(params)


def format_wikipedia_response(payload) -> str:
    pages = payload.get("query", {}).get("pages", {})
    rows = [
        f"{page.get('title', 'Untitled')}: {page.get('extract', 'No extract.')}"
        for page in sorted(pages.values(), key=lambda page: page.get("index", 0))
    ]
    return "\n".join(rows) or "No results."


# async execution uses an async http client
async def aimpl_wikipedia_search(query: str, /) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(wikipedia_url(query))
        response.raise_for_status()
    return format_wikipedia_response(response.json())


# tracing needs output shape without running the http call
def abstract_wikipedia_search(query, /):
    del query
    return afe.StrAVal()


# async batch receives the batch size, input axes, and input values
async def abatch_wikipedia_search(in_tree, /):
    batch_size, axes, values = in_tree
    del batch_size
    query_axis = axes
    queries = values

    if not query_axis:
        return await wikipedia_search_p.abind(queries), False

    results = await asyncio.gather(*(wikipedia_search_p.abind(query) for query in queries))
    return list(results), True


# async pullback forward sweep records the same residuals
async def apull_fwd_wikipedia_search(query: str, /):
    output = await wikipedia_search_p.abind(query)
    return output, (query, output)


# async pullback backward sweep turns output feedback into query feedback
async def apull_bwd_wikipedia_search(in_tree, /):
    (query, output), feedback = in_tree
    return af.format(
        "Improve the Wikipedia search query. Query: {}. Feedback: {}. Result: {}",
        query,
        feedback,
        output,
    )


afe.impl_rules.aset(wikipedia_search_p, aimpl_wikipedia_search)
afe.abstract_rules.set(wikipedia_search_p, abstract_wikipedia_search)
afe.batch_rules.aset(wikipedia_search_p, abatch_wikipedia_search)
afe.pull_fwd_rules.aset(wikipedia_search_p, apull_fwd_wikipedia_search)
afe.pull_bwd_rules.aset(wikipedia_search_p, apull_bwd_wikipedia_search)


def search_tool(query: str, _answer: str, history: str) -> str:
    result = wikipedia_search(query)
    return af.format("{}\nsearch({}): {}", history, query, result)


def done_tool(_query: str, answer: str, history: str) -> str:
    return af.format("{}\ndone: {}", history, answer)


# trace each branch once; switch chooses between these at runtime
search_ir = af.trace(search_tool)("query", "answer", "history")
done_ir = af.trace(done_tool)("query", "answer", "history")
tool_branches = {"search": search_ir, "done": done_ir}


def should_continue(state: State) -> bool:
    return state.active


def step(state: State) -> State:
    system = "If the history contains a line that starts with search, choose done. Otherwise choose search."
    user = af.format("Question and history:\n{}", state.history)
    messages = [dict(role="system", content=system), dict(role="user", content=user)]
    decision = af.lm_schema_call(messages, model="gpt-5.5", schema=decision_schema)
    history = af.switch(decision.tool, tool_branches, decision.args, decision.answer, state.history)
    return State(history=history, result=decision.answer, active=decision.tool == "search")


example = State(history="Question: What is recursion?", result="", active=True)

# while_loop takes traced condition and body programs
cond_ir = af.trace(should_continue)(example)
body_ir = af.trace(step)(example)


def agent(question: str) -> str:
    history = af.format("Question: {}", question)
    init = State(history=history, result="", active=True)
    # max_iters keeps the agent bounded
    final = af.while_loop(cond_ir, body_ir, init, max_iters=4)
    return final.result


# trace the whole agent once, then execute with a real question
agent_ir = af.trace(agent)("What is recursion?")
answer = asyncio.run(agent_ir.acall("What is recursion?"))
print(answer)
```

The provider decides which branch to run by returning a [`Decision` schema value](../concepts/schemas.md). {py:func}`switch <autoform.switch>` dispatches to the traced tool branch at execution time, and the selected branch appends to the history. {py:func}`while_loop <autoform.while_loop>` keeps applying `body_ir` while `should_continue` returns true, capped by `max_iters`.

`wikipedia_search` is a [primitive](../concepts/primitives.md) written with the same pattern as [Write a Primitive](writing-primitives.md). The HTTP call stays in the async runtime implementation, while the abstract, {py:func}`batch <autoform.batch>`, and {py:func}`pullback <autoform.pullback>` rules tell `autoform` how the external tool behaves when tracing or transforming the IR.

## Transform the Agent

The agent is still one IR:

```python
# batch runs the same agent ir over many questions
questions = ["What is recursion?", "What is memoization?"]
answers = asyncio.run(af.batch(agent_ir).acall(questions))
```

Feedback can flow through the full loop:

```python
# pullback turns output feedback into question feedback
pb_agent = af.pullback(agent_ir)
answer, (question_hint,) = asyncio.run(pb_agent.acall(("What is recursion?",), "too vague"))
```

For real tools, keep the branch signature stable: each branch here is `(query, answer, history) -> history`.

## Sync Rules

The async and sync registries are independent. To support `.call(...)` for the
same primitive, add sync counterparts with the same input and output shapes:

- `afe.impl_rules.set(wikipedia_search_p, impl_wikipedia_search)`;
- `afe.batch_rules.set(wikipedia_search_p, batch_wikipedia_search)`;
- `afe.pull_fwd_rules.set(wikipedia_search_p, pull_fwd_wikipedia_search)`;
- `afe.pull_bwd_rules.set(wikipedia_search_p, pull_bwd_wikipedia_search)`.

The sync HTTP implementation can use `httpx.get(...)` or `httpx.Client`. The
async implementation above uses `httpx.AsyncClient` so `.acall(...)` can overlap
independent tool calls.
