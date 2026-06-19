# Iterate To A Fixed Point

Use {py:func}`fixpoint <autoform.fixpoint>` when a traced step should run until
the next state is stable.

```{admonition} Concept
[Fixed Points](../../concepts/fixpoint.md) · [Pytrees](../../concepts/pytrees.md) · [Transforms](../../concepts/transforms.md)
```

## Define The State And Step

The step receives the current state plus a stable input, conventionally called
`theta`, and returns the next state with the same pytree structure.

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
real refinement program, `rewrite_step` can call {py:func}`lm_call <autoform.lm_call>`,
{py:func}`lm_schema_call <autoform.lm_schema_call>`, custom tools, or any other
traceable `autoform` code.

## Use The Step In A Program

Build the fixed point inside the outer traced program:

```python
def settle(init: RewriteState, target: str) -> RewriteState:
    return af.fixpoint(step_ir, init, target, max_iters=4)


ir = af.trace(settle)(example_state, "polished draft")
result = ir.call(example_state, "polished draft")
print(result)
```

The first iteration moves from `"rough draft"` to `"polished draft"`. The next
iteration produces the same state, so the default structural equality check
stops the loop.

## Stop With A Custom Equivalence IR

When stability is semantic, compare only the fields that matter:

```python
def is_stable(prev: RewriteState, new: RewriteState) -> bool:
    del prev
    return af.match(new.status, "stable")


equiv_ir = af.trace(is_stable)(example_state, example_state)


def settle_by_status(init: RewriteState, target: str) -> RewriteState:
    return af.fixpoint(step_ir, init, target, max_iters=4, equiv_ir=equiv_ir)
```

`equiv_ir` receives `(previous_state, new_state)` after each step. Here the loop
can stop as soon as the status field says the draft is stable, even if other
state fields changed.

## Pull Feedback To `theta`

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

This is the main reason to use `fixpoint` instead of manually unrolling a loop:
the backward pass is about the equilibrium. `adj_iters` controls how much
state-to-state feedback is accumulated at the fixed point before feedback is
read out for `theta`.

## Adapt The Pattern To LM Refinement

For an LM refinement loop, keep the same shape:

- state: the current draft, route, score, or optimizer state;
- theta: the rubric, instruction, task input, or prompt being optimized;
- step IR: one refinement pass;
- equivalence IR: a deterministic check, schema judge, or cached semantic
  comparator.

Use `max_iters` as the hard budget. If non-convergence matters to downstream
code, include a status or counter in the state so the returned value records
that the budget was exhausted.
