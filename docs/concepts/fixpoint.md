# Fixed Points

A fixed-point loop repeatedly applies the same traced step until applying that
step stops changing the state:

```text
new_state = step(state, theta)
```

Use {py:func}`fixpoint <autoform.fixpoint>` when the end condition is stability
of the state itself. Typical examples are normalization passes, self-refinement
loops, and optimizer steps that should run until the next candidate is
equivalent to the current one.

## Step Shape

The step is a traced IR with this shape:

```python
step_ir = af.trace(step)(example_state, example_theta)
```

Conceptually, `step_ir` has type `(State, Theta) -> State`.

- `State` is the value being iterated toward stability.
- `Theta` is the external input reused by every step.
- The step output must have the same [pytree](pytrees.md) structure as the state input.

The public call shape is:

```python
result = af.fixpoint(step_ir, init_val, theta, max_iters=8)
```

`max_iters` is required and must be at least `1`. The step always runs before the
stability check, so a fixed-point loop runs at least once. If the state never
stabilizes, `fixpoint` returns the last state produced within `max_iters`.

## State, Theta, And Stability

The split between `state` and `theta` is intentional. Put the iterated draft,
candidate, accumulator, or optimizer state in `state`. Put the stable context
for the step in `theta`: a rubric, target answer, prompt, task input, or other
value that every iteration should read.

By default, stability is structural equality between the previous state and the
new state. For semantic or field-level convergence, pass `equiv_ir`:

```python
equiv_ir = af.trace(lambda prev, new: af.match(new.status, "stable"))(example_state, example_state)
result = af.fixpoint(step_ir, example_state, example_theta, max_iters=8, equiv_ir=equiv_ir)
```

`equiv_ir` must have shape `(State, State) -> Bool`. It receives the previous
state and the newly produced state after each step.

## Pullback Through A Fixed Point

{py:func}`pullback <autoform.pullback>` does not treat `fixpoint` as a fully
unrolled loop. The forward pass keeps the converged state and `theta`. The
backward pass applies the step pullback at the fixed point and refines the
adjoint equation for `adj_iters`.

The result is deliberately shaped around the equilibrium:

- feedback to `init_val` is zero;
- feedback to `theta` carries the output critique through the fixed-point step;
- `adj_iters=0` uses the direct step transpose at the fixed point;
- larger `adj_iters` include more state-to-state feedback before reading
  feedback for `theta`.

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

The important invariant is that the step is still ordinary `autoform` IR. The
same primitives, pytrees, custom rules, and LM calls that work in a traced
function can appear inside the fixed-point step.

## Relationship To `while_loop`

| Primitive | Use when | Runs zero times? | Backward meaning |
| --- | --- | --- | --- |
| {py:func}`while_loop <autoform.while_loop>` | an explicit condition controls repetition | yes, if the condition is false initially | feedback follows the loop rule for the executed state path |
| {py:func}`fixpoint <autoform.fixpoint>` | repetition stops when the next state is stable | no | feedback is implicit at the equilibrium and flows to `theta` |
