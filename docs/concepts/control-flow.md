# Control Flow

Some programs require branching or looping based on runtime values. To support
this, `autoform` includes [control-flow primitives](primitives-and-rules.md)
represented in the [IR](programs-and-ir.md#the-ir). `switch` selects a branch,
`while_loop` repeats a step while a condition holds, and `fixpoint` repeats a
step until the state stabilizes. Both loops have a maximum number of iterations.[^loop-choice]

## Branches

The {py:func}`switch <autoform.switch>` primitive selects one of several traced
branches based on a runtime value. Here, `kind` selects the format of a text
label:

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

The result is `detailed: topic text`. All branches must have the same input and
output structure and types.

## Conditional Loops

The {py:func}`while_loop <autoform.while_loop>` primitive repeatedly executes a
step until a condition returns false or `max_iters` steps have run. The body
does not run if the condition is false at the beginning.

```{raw} html
:file: ../assets/loop-state.svg
```

Here, the loop state is a dataclass with text and a status. The body modifies
the state until the condition, defined by the status, becomes false. Optree
registers the dataclass as a [pytree](pytrees.md#pytrees) so that `autoform` can
access its fields:

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

The {py:func}`fixpoint <autoform.fixpoint>` primitive repeatedly applies a step program until the state becomes stable or the maximum number of iterations is reached. The step program must accept `(state, theta)` and must return a new state with the same pytree structure as the old state. The `theta` argument will be the same in every iteration.

The same `State` dataclass is used here. The target text is `theta` because it
stays fixed throughout the loop. This example takes two steps: the first sets
the text to the target, and the second confirms that the state is unchanged.

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

To customize how the fixpoint decides whether the state has converged, pass a program that determines whether the previous and new states are equivalent. This program will be called with the arguments `(previous_state, new_state)` after each call to the step program and must return a boolean. Here, the fixpoint will stop after the first step because the `status` field is the only field that the equivalence program checks.

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

{py:func}`pullback <autoform.pullback>` does not treat `fixpoint` as a fully
unrolled loop. The forward pass keeps the returned state and `theta`, including when the iteration limit is reached. The
backward pass applies the step pullback at the fixed point and refines the
adjoint equation for `adj_iters`.

This pullback rule is an approximation of the feedback through the returned state.

- feedback to `init_val` is zero;
- feedback to `theta` carries the output critique through the fixed-point step;
- `adj_iters=0` uses the direct step transpose at the fixed point;
- larger `adj_iters` include more state-to-state feedback before reading
  feedback for `theta`.

If the forward loop reaches its iteration limit, the returned state may not be a fixed point. The implicit rule is then an approximation at that state, rather than the derivative of the finite sequence of steps.

Use {py:func}`while_loop <autoform.while_loop>` if the initial state should
receive ordinary unrolled-iteration feedback, or if the loop must be allowed to
run zero times.

The pullback will return the feedback for `theta`, and the symbolic zero for the initial state. Here is an example using structured states.

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

`fixpoint` is a higher-order primitive, so transforms dispatch to rules for the
nested `step_ir` and optional `equiv_ir`.

- {py:func}`batch <autoform.batch>` runs an independent fixed-point iteration
  for each batched item. `theta` can be batched or broadcast with `in_axes`.
- {py:func}`pullback <autoform.pullback>` uses the implicit fixed-point rule
  described above instead of storing every forward iteration.
- {py:func}`dce <autoform.dce>` keeps the loop-carried state live through the
  step, because the next iteration may need any state leaf.
- `.acall(...)` uses the async execution path for the step and equivalence IRs.

As with the other control-flow primitives, the step can include other
primitives, including calls to language models. Each requested transform must
have compatible rules for the operations in the step.


[^loop-choice]: **Loop Choice.** Choose the primitive according to the stopping condition and backward rule:

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
