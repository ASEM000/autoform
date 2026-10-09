# Control Flow

A program may need to choose a branch or repeat a step using values that are only available during execution. [Control-flow primitives](primitives-and-rules.md) keep those choices visible in the [IR](programs-and-ir.md#the-ir). `switch` selects a branch, `while_loop` continues while a condition holds, and `fixpoint` repeats a step until the state is stable. Both loops also have an iteration limit.[^loop-choice]

## Branches

A runtime branch has several possible computations with a common interface. The {py:func}`switch <autoform.switch>` primitive selects one traced branch using the supplied key. In this example, `kind` determines which format is applied to the text:

```python
import autoform as af


def brief(text: str) -> str:
    return "brief: " + text


def detailed(text: str) -> str:
    return "detailed: " + text


branches = {
    "brief": af.trace(brief)("topic text"),
    "detailed": af.trace(detailed)("topic text"),
}


def route(kind: str, text: str) -> str:
    return af.switch(kind, branches, text)


ir = af.trace(route)("brief", "topic text")
print(ir.call("detailed", "topic text"))
```

The result is `detailed: topic text`. All branches must have the same input and output structure and types.

## Conditional Loops

Some loops have an explicit stopping condition, such as a status flag that changes after a revision. The {py:func}`while_loop <autoform.while_loop>` primitive evaluates its condition before each step. Execution ends when the condition is false or `max_iters` steps have run. An initially false condition skips the body entirely.

```{raw} html
:file: ../assets/loop-state.svg
```

The loop carries a dataclass containing text and a status. The body returns an updated state, and the condition reads its status to decide whether another step is needed. Registering the dataclass as a [pytree](pytrees.md#pytrees) exposes those fields to tracing and transforms:

```python
import optree


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class State:
    text: str
    status: str


def keep_going(state: State) -> bool:
    return state.status == "draft"


def revise(state: State) -> State:
    return State(text="Revision: " + state.text, status="stable")


example = State(text="draft text", status="draft")
cond_ir = af.trace(keep_going)(example)
body_ir = af.trace(revise)(example)

result = af.while_loop(cond_ir, body_ir, example, max_iters=3)
print(result)
```

The result is `State(text="Revision: draft text", status="stable")`.

## Fixed Points

A different stopping criterion is stability: another application of the step no longer changes the state. The {py:func}`fixpoint <autoform.fixpoint>` primitive checks for this after each step. Its step program accepts `(state, theta)` and returns a state with the same pytree structure. The state changes during iteration, while `theta` supplies parameters that remain fixed throughout the loop.

The example reuses the `State` dataclass. The target text is passed as `theta` because it stays fixed, while the text in the state is updated. Two steps are needed: the first installs the target text, and the second produces the same state again.

```python
def rewrite_step(state: State, target: str) -> State:
    del state
    return State(text=target, status="stable")


step_ir = af.trace(rewrite_step)(example, "revised draft text")


def settle(init: State, target: str) -> State:
    return af.fixpoint(step_ir, init, target, max_iters=4)


ir = af.trace(settle)(example, "revised draft text")
result = ir.call(example, "revised draft text")
print(result)
```

The fixpoint will use structural equality by default to compare the previous state and the new state. The step program will always be executed at least once, and `max_iters` must be at least 1. If the fixpoint reaches `max_iters` before converging, then the final state from the last iteration will be returned, even if it is not stable.

### State and Stability

Stability can depend on the meaning of the state rather than equality of every field. An equivalence program receives `(previous_state, new_state)` after each step and returns a boolean. Here, it checks whether the new state has `status="stable"`, so the loop stops after the first step even though the text has changed.

```python
def is_stable(prev: State, new: State) -> bool:
    del prev
    return new.status == "stable"


equiv_ir = af.trace(is_stable)(example, example)


def settle_by_status(init: State, target: str) -> State:
    return af.fixpoint(step_ir, init, target, max_iters=4, equiv_ir=equiv_ir)


status_ir = af.trace(settle_by_status)(example, "revised draft text")
assert status_ir.call(example, "revised draft text").status == "stable"
```

### Pullback

The two loop primitives also differ in backward behavior. The {py:func}`pullback <autoform.pullback>` of `fixpoint` uses an implicit fixed-point rule instead of reversing the full sequence of iterations. Its forward sweep retains the returned state and `theta`. Its backward sweep evaluates the step pullback at that state and refines an adjoint equation for `adj_iters` iterations.

The state-to-state feedback is used to approximate the effect of the fixed-point relation. The rule has the following input-feedback behavior:

- feedback to `init_val` is zero;
- feedback to `theta` carries the output critique through the fixed-point step;
- `adj_iters=0` uses the direct step transpose at the fixed point;
- larger `adj_iters` include more state-to-state feedback before reading feedback for `theta`.

If the forward loop reaches its iteration limit, the returned state may not be a fixed point. The implicit rule is then an approximation at that state, rather than the derivative of the finite sequence of steps.

{py:func}`while_loop <autoform.while_loop>` uses feedback through the executed iterations, including feedback to the initial state. It also supports a body that is skipped when the initial condition is false.

The following call illustrates the fixed-point boundary: the pullback returns feedback for `theta` and a symbolic zero for the initial state. The structure of the state remains visible in the result.

```python
def final_text(init: State, target: str) -> str:
    final = af.fixpoint(step_ir, init, target, max_iters=4, adj_iters=1)
    return final.text


text_ir = af.trace(final_text)(example, "revised draft text")
output, (init_feedback, target_feedback) = af.pullback(text_ir).call(
    (example, "revised draft text"),
    "draft feedback",
)

print(output)
print(init_feedback)
print(target_feedback)
```

### Transform Behavior

`fixpoint` contains a nested step program and, optionally, an equivalence program. Transforming the outer program therefore also requires the appropriate rules for the operations inside `step_ir` and `equiv_ir`.

- {py:func}`batch <autoform.batch>` runs an independent fixed-point iteration for each batched item. `theta` can be batched or broadcast with `in_axes`.
- {py:func}`pullback <autoform.pullback>` uses the implicit fixed-point rule described above instead of storing every forward iteration.
- {py:func}`dce <autoform.dce>` keeps the loop-carried state live through the step, because the next iteration may need any state leaf.
- `.acall(...)` uses the async execution path for the step and equivalence IRs.

As with the other control-flow primitives, the step can include other primitives, including calls to language models. Each requested transform must have compatible rules for the operations in the step.


[^loop-choice]: **Loop Choice.** The stopping condition and backward behavior distinguish the two loop primitives:

    ````{container}

    | Primitive | Usage | Backward behavior |
    | --- | --- | --- |
    | {py:func}`while_loop <autoform.while_loop>` | an explicit condition controls repetition | feedback follows the loop rule for the executed state path |
    | {py:func}`fixpoint <autoform.fixpoint>` | repetition stops when the next state is stable | feedback approximates the implicit rule at the returned state and flows to `theta` |

    Solid arrows in the diagrams carry state values, and dashed arrows carry
    feedback. The states are $s_0$, $s_1$, and $s_2$, with corresponding feedback
    $g_0$, $g_1$, and $g_2$.

    The `while_loop` diagram shows two executed steps. The backward pass applies
    the body pullback at the saved states in reverse order, passing output
    feedback back to the initial state. The condition selects which steps run,
    but no feedback passes through the condition.

    ```{raw} html
    :file: ../assets/while-loop-pullback.svg
    ```

    The `fixpoint` diagram shows one round of feedback refinement (`adj_iters=1`).
    Both pullbacks use the same returned state and `theta`. The first computes
    feedback for the state, which is accumulated with the original output
    feedback. The second computes feedback for `theta` using that accumulated
    feedback. Feedback to the initial state is zero. Refinement can stop early
    if feedback stops changing. The returned state stays fixed throughout these
    backward computations.

    ```{raw} html
    :file: ../assets/fixpoint-pullback.svg
    ```

    ````
