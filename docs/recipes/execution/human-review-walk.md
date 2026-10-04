# Human Review

Use `ir.walk(...)` when human feedback belongs to the executor rather than the
traced program. Tags mark which primitive equations require review; the walk
runner pauses after those equations, asks for feedback, then continues execution
with the accepted or edited output.

```{admonition} Concept
[Walk](../../concepts/walk.md) · [Tags](../../concepts/tags.md) ·
[Trace, IR, Execute](../../concepts/trace-ir-execute.md)
```

## Mark the Review Point

Use {py:func}`tag <autoform.tag>` around the primitive call that should emit a
reviewable equation:

```python
import autoform as af


review = "human-review"


def draft_then_finalize(topic: str) -> str:
    draft = "draft for " + topic + ": method A maps x to y."
    with af.tag(review):
        draft = af.checkpoint(draft, key="draft", collection="review")

    return "final answer:\n" + draft


ir = af.trace(draft_then_finalize)("topic x")
```

The tag is attached to equations emitted during tracing. It does not change the
primitive result or execution order by itself.

## Feedback Function

A feedback function can accept the draft or return an edited value for downstream equations:

```python
def ask_for_feedback(draft: str) -> str:
    print("Output for review:")
    print(draft)
    note = input("Feedback (leave empty to accept): ")
    if not note:
        return draft
    return draft + "\nHuman feedback: " + note
```

The pause happens at `input(...)`. The runner decides when to call this
function.

```{raw} html
:file: ../../assets/human-review.svg
```

## Runner

The runner executes one equation at a time. When a yielded equation has the
review tag, it passes the equation output to `feedback(...)` before sending it
back into the generator:

```python
def run_with_human_feedback(ir, *args, tag: str, feedback):
    gen = ir.walk(*args)
    eqn, in_values = next(gen)

    while eqn is not None:
        out_values = eqn.bind(in_values, **eqn.params)

        if tag in eqn.tags:
            out_values = feedback(out_values)

        eqn, in_values = gen.send(out_values)

    return in_values
```

The pause happens before `gen.send(...)`. Execution continues only after the
runner sends the accepted or edited output back to the walk generator.

## Execution

Finally, run the program by calling the runner with the IR and the feedback function:

```python
result = run_with_human_feedback(
    ir,
    "topic x",
    tag=review,
    feedback=ask_for_feedback,
)

print(result)
```

Execution resumes after `run_with_human_feedback(...)` sends the accepted or
edited value back to the walk generator.

## Review Boundary

{py:func}`checkpoint <autoform.checkpoint>` and {py:func}`inject <autoform.inject>` should be used when review points need to be specified within the traced function. `ir.walk(...)` should be used when the review policy will be specified within a custom runner, and should not be specified in the traced function.
