# Control Flow

`autoform` [control-flow primitives](primitives.md) keep branches and loops visible to the [IR](the-ir.md).
Use these operations when a branch condition or loop count depends on runtime inputs.
For loops that should stop when repeated application reaches a stable state, see
[Fixed Points](fixpoint.md).

## Branches

Select a traced branch from the runtime `kind`:

```python
import autoform as af


def brief(text: str) -> str:
    return "brief: " + text


def detailed(text: str) -> str:
    return "detailed: " + text


branches = {
    "brief": af.trace(brief)("seed"),
    "detailed": af.trace(detailed)("seed"),
}


def route(kind: str, text: str) -> str:
    return af.switch(kind, branches, text)


ir = af.trace(route)("brief", "recursion")
print(ir.call("detailed", "recursion"))
```

The above would print `detailed: recursion`. Note that all branches must have the same input and output structure and types.

## Loops

```{raw} html
:file: ../assets/loop-state.svg
```

Carry a structured state through the loop:

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    text: str
    status: str


def keep_going(state: State) -> bool:
    return state.status == "continue"


def add_step(state: State) -> State:
    text = state.text + "!"
    return State(text=text, status="done")


example = State(text="go", status="continue")
cond_ir = af.trace(keep_going)(example)
body_ir = af.trace(add_step)(example)

result = af.while_loop(cond_ir, body_ir, example, max_iters=3)
print(result)
```

The result is `State(text="go!", status="done")`. The body changes the status after one iteration.

The loop state is a registered [pytree](pytrees.md), using [Optree's dataclass integration](https://optree.readthedocs.io/en/latest/dataclasses.html).

## Feedback Boundaries

Use {py:func}`stop_gradient <autoform.stop_gradient>` to keep one input fixed during AD:

```python
import autoform as af


def combine(locked: str, editable: str) -> str:
    locked = af.stop_gradient(locked)
    return locked + "\n" + editable


ir = af.trace(combine)("terms:", "draft answer")
inputs = ("terms:", "draft answer")
output, (locked_feedback, editable_feedback) = af.pullback(ir).call(
    inputs,
    "make clearer",
)

print(output)
print(locked_feedback)
print(editable_feedback)
```

In the above example, the forward value of `locked` is unchanged, but the feedback value is `""`, while the feedback for the other input is `"make clearer"`.

## Dependencies

Use {py:func}`depends <autoform.depends>` to delay a result until another traced value is available:

```python
import autoform as af


def ordered(topic: str) -> str:
    audit = "audit " + topic
    answer = "answer " + topic
    # return answer through a barrier that also waits for audit
    return af.depends(answer, audit)


ir = af.trace(ordered)("recursion")
scheduled = af.sched(ir)
print(scheduled.call("recursion"))
```

Use {py:func}`depends <autoform.depends>` when a result should not become available until another traced
value has also been evaluated, even though the returned value does not consume it directly. It does not
force the computation that produces the returned value to start after the dependencies; a scheduler may still
run independent producers concurrently and place the `depends` barrier after those producers.
