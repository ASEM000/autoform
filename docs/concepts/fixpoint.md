# Fixed Points

A primitive that applies a single step repeatedly in a loop until a fixed point, where the next application of the step no longer changes the state.

```text
new_state = step(state, theta)
```

Use {py:func}`fixpoint <autoform.fixpoint>` when the termination condition of a loop is stability of some state. This is useful for e.g. normalization, self-refinement, optimizers, etc.

## Step Shape

See the full program in [Fixed Points](../recipes/core/fixpoint.md).

```python
step_ir = af.trace(step)(example_state, example_theta)
```

Conceptually, `step_ir` has type `(State, Theta) -> State`.

- `State` is the value being iterated toward stability.
- `Theta` is the external input reused by every step.
- The step output must have the same [pytree](pytrees.md) structure as the state input.

The public API takes the form:

```python
result = af.fixpoint(step_ir, init_val, theta, max_iters=8)
```

`max_iters` (required, must be >= `1`) to ensure that the loop runs at least once (the step always runs before checking for stability). If the state never stabilizes, `fixpoint` returns the last state after running for `max_iters`.

## State and Stability

Separate the parts of the state that will be iterated over into `state` and the parts that will stay the same for every step into `theta`. For example, if each `step_ir` refines a draft answer based on a rubric, the draft would go in `state` and the rubric would go in `theta`. Similarly, for an optimizer, the optimizer state would go in `state` and the prompt/task input would go in `theta`.

By default, stability is structural equality between the previous state and the
new state. For semantic or field-level convergence, pass `equiv_ir`:

```python
equiv_ir = af.trace(lambda prev, new: new.status == "stable")(
    example_state,
    example_state,
)
result = af.fixpoint(
    step_ir,
    example_state,
    example_theta,
    max_iters=8,
    equiv_ir=equiv_ir,
)
```

`equiv_ir` must have shape `(State, State) -> Bool`. It receives the previous
state and the newly produced state after each step.

## Pullback

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

## Transform Behavior

`fixpoint` is a higher-order primitive, so transforms dispatch to rules for the
nested `step_ir` and optional `equiv_ir`.

- {py:func}`batch <autoform.batch>` runs an independent fixed-point iteration
  for each batched item. `theta` can be batched or broadcast with `in_axes`.
- {py:func}`pullback <autoform.pullback>` uses the implicit fixed-point rule
  described above instead of storing every forward iteration.
- {py:func}`dce <autoform.dce>` keeps the loop-carried state live through the
  step, because the next iteration may need any state leaf.
- `.acall(...)` uses the async execution path for the step and equivalence IRs.

The step can be any IR that `autoform` understands, meaning that it can contain other primitives, pytrees, custom rules, LM calls, etc.

## Loop Choice

Choose the primitive according to the stopping condition and backward rule:

| Primitive | Use when | Runs zero times? | Backward meaning |
| --- | --- | --- | --- |
| {py:func}`while_loop <autoform.while_loop>` | an explicit condition controls repetition | yes, if the condition is false initially | feedback follows the loop rule for the executed state path |
| {py:func}`fixpoint <autoform.fixpoint>` | repetition stops when the next state is stable | no | feedback approximates the implicit rule at the returned state and flows to `theta` |
