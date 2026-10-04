# Fixed Points

To use a fixed point in a traced program, use the {py:func}`fixpoint <autoform.fixpoint>` function:

```{admonition} Concept
[Fixed Points](../../concepts/fixpoint.md) · [Pytrees](../../concepts/pytrees.md) · [Transforms](../../concepts/transforms.md)
```

## State and Step

The step receives the current state and an input reused across iterations, called `theta`. It returns the next state with the same pytree structure:

```python
import optree
import autoform as af


@optree.dataclasses.dataclass(namespace=af.PYTREE_NAMESPACE)
class RewriteState:
    draft: str
    status: str


def rewrite_step(state: RewriteState, target: str) -> RewriteState:
    del state
    return RewriteState(draft=target, status="stable")


example_state = RewriteState(draft="rough draft", status="draft")
step_ir = af.trace(rewrite_step)(example_state, "polished draft")
```

This step is deterministic so the example can run without an LM provider. In a
real refinement program, `rewrite_step` can call {py:func}`fill <autoform.lm.fill>`,
custom tools, or any other
traceable `autoform` code.

## Iteration

Build the fixed point inside the outer traced program:

```python
def settle(init: RewriteState, target: str) -> RewriteState:
    return af.fixpoint(step_ir, init, target, max_iters=4)


ir = af.trace(settle)(example_state, "polished draft")
result = ir.call(example_state, "polished draft")
print(result)
```

The fixed point here runs for two iterations: the first iteration advances the state from `"rough draft"` to `"polished draft"`, but the next iteration leaves the state the same. The fixed point uses structural equality by default to detect that the state has reached a fixed point and no further iterations are necessary.

## Stability Check

When stability is semantic, compare only the fields that matter:

```python
def is_stable(prev: RewriteState, new: RewriteState) -> bool:
    del prev
    return new.status == "stable"


equiv_ir = af.trace(is_stable)(example_state, example_state)


def settle_by_status(init: RewriteState, target: str) -> RewriteState:
    return af.fixpoint(step_ir, init, target, max_iters=4, equiv_ir=equiv_ir)
```

`equiv_ir` receives `(previous_state, new_state)` after each step. Here the loop
can stop as soon as the status field says the draft is stable, even if other
state fields changed.

## Feedback

The fixed-point pullback sends output feedback to the stable input `theta`, not
to the initial state:

```python
def final_draft(init: RewriteState, target: str) -> str:
    final = af.fixpoint(step_ir, init, target, max_iters=4, adj_iters=1)
    return final.draft


draft_ir = af.trace(final_draft)(example_state, "polished draft")
output, (init_feedback, target_feedback) = af.pullback(draft_ir).call(
    (example_state, "polished draft"),
    "make the final draft more concrete",
)

print(output)
print(init_feedback)
print(target_feedback)
```

The backward pass of the fixed point approximates this feedback using the state returned by the fixed point, even if the fixed point didn't converge and instead stopped after the maximum number of iterations. The `adj_iters` argument of {py:func}`fixpoint <autoform.fixpoint>` controls how much feedback should be accumulated from state to state inside the fixed point before reading it out for `theta`.

## LM Refinement

For an LM refinement loop, use the same roles:

- state: the current draft, route, score, or optimizer state;
- theta: the rubric, instruction, task input, or prompt being optimized;
- step IR: one refinement pass;
- equivalence IR: a deterministic check, schema judge, or cached semantic
  comparator.

Use `max_iters` as the hard budget. If non-convergence matters to downstream
code, include a status or counter in the state so the caller can distinguish a stable state from an unfinished result.
