# Human Review

A draft may need human review before the final answer is produced. Mark the review point and run the program. Execution pauses there to request a response: an empty response accepts the draft, while a note is appended before execution continues.

```{admonition} Concept
[Manual Execution](../../concepts/execution.md#manual-execution) · [Tags](../../concepts/programs-and-ir.md#tags) ·
[Programs and IR](../../concepts/programs-and-ir.md)
```

## Mark the Review Point

Mark the draft to be reviewed before it is finalized.

```python
import autoform as af


review = "human-review"


def draft_then_finalize(topic: str) -> str:
    draft = "draft for " + topic + ": draft text"
    with af.tag(review):
        draft = af.checkpoint(draft, key="draft", collection="review")

    return "final answer:\n" + draft


ir = af.trace(draft_then_finalize)("topic x")
```

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

After acceptance, the final answer includes the original draft. After a note is provided, the final answer also includes that note. The runner applies the review policy, so the same traced program can run without interactive review.
