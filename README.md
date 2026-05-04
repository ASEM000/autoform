<div align="center">

# `autoform`

**Trace once. Transform freely.**

Composable function transformations for LM programs.

*Think [JAX](https://github.com/jax-ml/jax), but for LM programs.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ASEM000/autoform/actions/workflows/ci.yml/badge.svg)](https://github.com/ASEM000/autoform/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ASEM000/autoform/graph/badge.svg?token=Z0JBHSC3ZK)](https://codecov.io/gh/ASEM000/autoform)

[Quickstart](#quickstart) - [Transforms](#transforms) - [Concurrency](#concurrency) - [Debugging](#debugging) - [Agent](#agent) - [Docs](https://autoform.readthedocs.io)

</div>

```bash
pip install git+https://github.com/ASEM000/autoform.git
```

## Quickstart
```python
import autoform as af


def explain(topic: str) -> str:
    prompt = af.format("Explain {} in one paragraph.", topic)
    msg = dict(role="user", content=prompt)
    return af.lm_call([msg], model="gpt-5.2")


ir = af.trace(explain)("...")  # capture structure, no execution
```

Now transform it:
```python
# execute
output = ir.call("quantum entanglement")

# batch: n inputs
outputs = af.batch(ir).call(["DNA", "gravity", "recursion"])

# pushforward: propagate input perturbations forward
output, tangent = af.pushforward(ir).call(("quantum entanglement", "add more examples"))

# pullback: propagate output feedback backward
output, grad = af.pullback(ir).call(("quantum entanglement", "too technical"))

# compose: batched differentiation
topics = ["DNA", "gravity", "recursion"]
critiques = ["too technical", "too brief", "too abstract"]
outputs, hints = af.batch(af.pullback(ir)).call((topics, critiques))
```

The last line is the point: `batch(pullback(ir))`, transformations compose.

## Transforms

<div align="center">

| Transform | What it does |
|-----------|--------------|
| `batch` | Vectorize over inputs |
| `pushforward` | Forward-mode-like transform |
| `pullback` | Reverse-mode-like transform |
| `custom` | Optional custom transform rules for a function boundary |
| `sched` | Auto-concurrent execution |

</div>

## Concurrency

`sched` analyzes the IR's dependency graph, groups independent equations into parallel stages, and `acall` runs each stage concurrently.

```mermaid
flowchart LR
    subgraph before["sequential"]
        direction TB
        A1[format] --> B1[LLM explain]
        A2[format] --> B2[LLM facts]
        B1 --> C1[format]
        B2 --> C1
        C1 --> D1[LLM synthesize]
    end

    before -- "sched" --> after

    subgraph after["scheduled"]
        direction TB
        A3[format] & A4[format]
        subgraph "gather (concurrent)"
            B3[LLM explain] & B4[LLM facts]
        end
        A3 --> B3
        A4 --> B4
        B3 & B4 --> C2[format]
        C2 --> D2[LLM synthesize]
    end
```

```python
scheduled = af.sched(ir)
result = await scheduled.acall("DNA")
```

## Debugging

Checkpoint intermediate values. Substitute on re-execution.
```python
def pipeline(x: str) -> str:
    msg1 = dict(role="user", content=x)
    step1 = af.lm_call([msg1], model="gpt-5.2")
    step1 = af.checkpoint(step1, key="step1", collection="debug")

    msg2 = dict(role="user", content=step1)
    step2 = af.lm_call([msg2], model="gpt-5.2")
    return step2


ir = af.trace(pipeline)("...")

# capture
with af.collect(collection="debug") as captured:
    result = ir.call("input")

# substitute step1 value
with af.inject(collection="debug", values=dict(step1=["modified"])):
    result = ir.call("input")
```

## Agent

Trace a tool-use agent once, then differentiate, batch, or schedule it with no code changes. Because the agent is a pure traced function, `pullback` propagates natural-language feedback backward through every LLM call, and `batch` vectorizes over inputs. Compose them: `batch(pullback(ir))` gives batched prompt optimization of the full agent graph.

```mermaid
flowchart TD
    Q([question]) --> cond{cond}
    cond -- continue --> LLM
    cond -- done --> result([result])

    subgraph body
        LLM -- Decision --> SW{switch tool}
        subgraph "traced branches"
            SW --> search[search] & calc[calc] & dn[done]
        end
        search & calc & dn --> nh(new_history)
    end

    nh -- State --> cond
```

```python
from typing import Literal

import optree  # tree manipulation (https://optree.readthedocs.io/en/latest/)
import autoform as af


treelib = optree.pytree.reexport(namespace=af.PYTREE_NAMESPACE)


@treelib.dataclasses.dataclass
class Decision:
    tool: str
    args: str
    answer: str
    status: Literal["continue", "done"]


@treelib.dataclasses.dataclass
class State:
    history: str
    result: str
    status: Literal["continue", "done"]


# - Str, Enum are used to define the schema of the LLM's decision output
# - Doc is used to guide the LLM's output with natural language descriptions
#   of each field.
decision_schema = Decision(
    tool=af.Enum("search", "calc", "done") @ af.Doc("Tool to call next."),
    args=af.Str() @ af.Doc("Tool arguments."),
    answer=af.Str() @ af.Doc("Current answer."),
    status=af.Enum("continue", "done") @ af.Doc("Whether to continue."),
)


def search(query: str, history: str) -> str: ...


def calc(expression: str, history: str) -> str: ...


def done(answer: str, history: str) -> str: ...


# each tool branch is traced independently; switch dispatches at runtime.
tool_branches = dict(
    search=af.trace(search)("...", "..."),  # (args, history) -> new_history
    calc=af.trace(calc)("...", "..."),
    done=af.trace(done)("...", "..."),
)


def cond(state: State):
    # continue if status is "continue", else stop
    # used by the while loop to determine when to stop iterating
    return af.match(state.status, "continue")


def body(state: State):
    messages = [
        dict(role="system", content="You are a tool-use agent."),
        dict(role="user", content=state.history),
    ]
    d = af.lm_schema_call(messages, model="gpt-5.2", schema=decision_schema)
    new_history = af.switch(d.tool, tool_branches, d.args, state.history)
    return State(history=new_history, result=d.answer, status=d.status)


example_state = State(history="...", result="", status="continue")
cond_ir = af.trace(cond)(example_state)
body_ir = af.trace(body)(example_state)


def agent(question: str):
    init = State(history=question, result="", status="continue")
    return af.while_loop(cond_ir, body_ir, init, max_iters=5).result


agent_ir = af.trace(agent)("...")

# pullback: propagate text feedback backward through every LLM call
# batch: vectorize over multiple questions
# compose them: batched prompt optimization of the full agent
af.batch(af.pullback(agent_ir))
```

---

> ⚠️ **Early development**: API may change.
