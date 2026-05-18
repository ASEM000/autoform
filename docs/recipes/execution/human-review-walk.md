# Add Human Feedback with Walk

AutoForm is a small, low-level library; it does not prescribe an approval UI,
review queue, or feedback service. The core idea is that human feedback can be
an executor policy over traced primitive equations.

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
    with af.tag(review):
        draft = af.format("draft for {}: method A maps x to y.", topic)

    return af.format("final answer:\n{}", draft)


ir = af.trace(draft_then_finalize)("topic x")
```

The tag is attached to equations emitted during tracing. It does not change the
primitive result or execution order by itself.

## Ask for Feedback

The feedback function can block on user input. Returning the original value
accepts it; returning a different value patches what downstream equations see.

```python
def ask_for_feedback(draft: str) -> str:
    print("review output:")
    print(draft)
    note = input("feedback, or empty to accept: ")
    if not note:
        return draft
    return draft + "\nhuman feedback: " + note
```

The pause happens at `input(...)`. The runner decides when to call this
function.

## Pause on Tagged Equations

The runner executes one equation at a time. When a yielded equation has the
review tag, it passes the equation output to `feedback(...)` before sending it
back into the generator.

```python
def run_with_human_feedback(ir, *args, tag: str, feedback):
    gen = ir.walk(*args)
    ir_eqn, in_values = next(gen)

    while ir_eqn is not None:
        out_values = ir_eqn.bind(in_values, **ir_eqn.params)

        if tag in ir_eqn.tags:
            out_values = feedback(out_values)

        ir_eqn, in_values = gen.send(out_values)

    return in_values
```

The pause happens before `gen.send(...)`. Execution continues only after the
runner sends the accepted or edited output back to the walk generator.

## Run the Reviewed Program

Call the runner with the traced IR and the feedback function:

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

## Choose the Review Boundary

Use {py:func}`checkpoint <autoform.checkpoint>` and
{py:func}`inject <autoform.inject>` when the traced function should expose named
review points. Use `ir.walk(...)` when the review policy belongs to a custom
runner and should stay outside the traced function.
