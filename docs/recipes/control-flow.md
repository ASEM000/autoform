# Use Control Flow Inside A Traced Function

`autoform` [control-flow primitives](../concepts/primitives.md) keep branches and loops visible to the [IR](../concepts/the-ir.md).
Use them when the branch condition or loop state is part of the traced program.

## Route With `switch`

```python
import autoform as af


def brief(text: str) -> str:
    return af.format("brief: {}", text)


def detailed(text: str) -> str:
    return af.format("detailed: {}", text)


branches = {
    "brief": af.trace(brief)("seed"),
    "detailed": af.trace(detailed)("seed"),
}


def route(kind: str, text: str) -> str:
    return af.switch(kind, branches, text)


ir = af.trace(route)("brief", "recursion")
print(ir.call("detailed", "recursion"))
```

## Repeat With `while_loop`

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
    text = af.concat(state.text, "!")
    return State(text=text, status="done")


example = State(text="go", status="continue")
cond_ir = af.trace(keep_going)(example)
body_ir = af.trace(add_step)(example)

result = af.while_loop(cond_ir, body_ir, example, max_iters=3)
print(result)
```

The loop state is a registered [pytree](../concepts/pytrees.md), using [Optree's dataclass integration](https://optree.readthedocs.io/en/latest/dataclasses.html).

## Block Feedback With `stop_gradient`

```python
import autoform as af


def combine(locked: str, editable: str) -> str:
    locked = af.stop_gradient(locked)
    return af.format("{}\n{}", locked, editable)


ir = af.trace(combine)("terms:", "draft answer")
inputs = ("terms:", "draft answer")
output, (locked_feedback, editable_feedback) = af.pullback(ir).call(inputs, "make clearer")

print(output)
print(locked_feedback)
print(editable_feedback)
```

The forward value of `locked` is unchanged. Feedback for that input is blocked.

## Force Ordering With `depends`

```python
import autoform as af


def ordered(topic: str) -> str:
    audit = af.format("audit {}", topic)
    answer = af.format("answer {}", topic)
    # return answer, but make it depend on audit
    return af.depends(answer, audit)


ir = af.trace(ordered)("recursion")
scheduled = af.sched(ir)
print(scheduled.call("recursion"))
```

Use `depends` when a value must be computed before another value even though
the second value does not consume it directly.
