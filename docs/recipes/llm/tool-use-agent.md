# Tool-Use Agent

A question may require information that is not available in the initial context. A tool-using program needs to choose a search, retain the result, and decide when to return an answer. This recipe combines a structured model decision, a search primitive, and a bounded loop into one traced program. The resulting IR can answer multiple questions or propagate answer feedback back to the input question.

```{admonition} Concept
[Transforms](../../concepts/transforms.md) · [Pytrees](../../concepts/pytrees.md#pytrees) · [Language Models](../../language-models.md) · [Primitives and Rules](../../concepts/primitives-and-rules.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

```{raw} html
:file: ../../assets/agent-loop.svg
```

## Decision and State

The model decision and the loop state have different roles. A `Decision` describes the next action: a search query or a final answer. `State` retains the history, latest result, and whether another step is needed. Both are registered pytrees so the model output and loop state can use structured values. The schema constrains the tool choice to `search` or `done`.

```python
import asyncio
import optree
from urllib.parse import urlencode

import httpx
import autoform as af
import autoform.extend as afe

model = "model-name"


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
    tool=af.lm.Enum(
        "search",
        "done",
        desc="Select done after a search result; otherwise select search.",
    ),
    args=af.lm.Str(desc="Search query; empty for done."),
    answer=af.lm.Str(desc="Final answer; empty for search."),
)
```

## Search Tool

An HTTP search requires a concrete query and cannot run on a tracing placeholder. The search is therefore exposed as a primitive. Its abstract rule reports a string output without making a request; its execution rule performs the request later. Batch and pullback rules explain how this external operation participates in the agent's transforms. The pullback rule supplies a simple textual suggestion for improving the query.

```python
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
        for page in sorted(
            pages.values(),
            key=lambda page: page.get("index", 0),
        )
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
    return af.string.StrAVal()


# async batch receives the batch size, input axes, and input values
async def abatch_wikipedia_search(in_tree, /):
    batch_size, axes, values = in_tree
    del batch_size
    query_axis = axes
    queries = values

    if not query_axis:
        return await wikipedia_search_p.abind(queries), False

    results = await asyncio.gather(
        *(wikipedia_search_p.abind(query) for query in queries)
    )
    return list(results), True


# async pullback forward sweep records the same residuals
async def apull_fwd_wikipedia_search(query: str, /):
    output = await wikipedia_search_p.abind(query)
    return output, (query, output)


# async pullback backward sweep turns output feedback into query feedback
async def apull_bwd_wikipedia_search(in_tree, /):
    (query, output), feedback = in_tree
    return (
        "Improve the Wikipedia search query. Query: "
        + query
        + ". Feedback: "
        + feedback
        + ". Result: "
        + output
    )


afe.register_aimpl(wikipedia_search_p, aimpl_wikipedia_search)
afe.register_abstract(wikipedia_search_p, abstract_wikipedia_search)
afe.register_abatch(wikipedia_search_p, abatch_wikipedia_search)
afe.register_apullback_fwd(wikipedia_search_p, apull_fwd_wikipedia_search)
afe.register_apullback_bwd(wikipedia_search_p, apull_bwd_wikipedia_search)
```

## Agent Loop

Each step asks the model to choose an action from the question and accumulated history. A switch runs the selected tool branch, and the resulting history becomes part of the next state. Selecting `search` leaves the loop active; selecting `done` stores the answer and stops further steps. The condition and body are traced separately before being assembled into the bounded loop.

```python
def search_tool(query: str, _answer: str, history: str) -> str:
    result = wikipedia_search(query)
    return history + "\nsearch(" + query + "): " + result


def done_tool(_query: str, answer: str, history: str) -> str:
    return history + "\ndone: " + answer


# trace each branch once; switch chooses between these at runtime
search_ir = af.trace(search_tool)("query", "answer", "history")
done_ir = af.trace(done_tool)("query", "answer", "history")
tool_branches = {"search": search_ir, "done": done_ir}


def should_continue(state: State) -> bool:
    return state.active


def step(state: State) -> State:
    system = "Select done after a search result; otherwise select search."
    messages = [
        dict(role="system", content=system),
        dict(role="user", content="Question and history:\n" + state.history),
    ]
    decision = af.lm.fill(
        dict(context=messages, output=decision_schema),
        model=model,
    )["output"]
    history = af.switch(
        decision.tool,
        tool_branches,
        decision.args,
        decision.answer,
        state.history,
    )
    return State(
        history=history,
        result=decision.answer,
        active=decision.tool == "search",
    )


example = State(history="Question: question text", result="", active=True)

# while_loop takes traced condition and body programs
cond_ir = af.trace(should_continue)(example)
body_ir = af.trace(step)(example)


def agent(question: str) -> str:
    history = "Question: " + question
    init = State(history=history, result="", active=True)
    # max_iters keeps the agent bounded
    final = af.while_loop(cond_ir, body_ir, init, max_iters=4)
    return final.result


# trace the whole agent once, then execute with a real question
agent_ir = af.trace(agent)("question text")
answer = asyncio.run(agent_ir.acall("question text"))
print(answer)
```

The result is the answer field from the final model decision. The loop ends when the model selects `done` or reaches `max_iters=4`. The iteration bound limits execution; it does not guarantee that an answer is complete. Model requests and HTTP searches occur when the program executes, using the asynchronous rules registered above.[^synchronous-rules]

## Transforms

The complete agent is an IR, so {py:func}`batch <autoform.batch>` can apply it to several questions. Each question has its own loop state and can reach the stopping condition on a different iteration:

```python
# batch runs the same agent ir over many questions
questions = ["question text 1", "question text 2"]
answers = asyncio.run(af.batch(agent_ir).acall(questions))
```

A {py:func}`pullback <autoform.pullback>` of the agent follows the computation performed by the loop to propagate answer feedback to the question. The registered search rule contributes text about the query and its result; the feedback has the meaning supplied by that rule, rather than a numerical derivative of the search service:

```python
# pullback turns output feedback into question feedback
pb_agent = af.pullback(agent_ir)
answer, (question_hint,) = asyncio.run(
    pb_agent.acall(("question text",), "answer feedback")
)
```

Both tool branches have the signature `(query, answer, history) -> history`. Keeping this interface consistent allows the switch to select a branch at runtime while preserving the structure expected by the loop and its transforms.

[^synchronous-rules]: This tool registers async rules. Synchronous execution also needs synchronous execution, batch, and pullback rules. [Primitive Definitions](../../concepts/primitives-and-rules.md#primitive-definitions) describes these registrations.
