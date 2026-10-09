# Human Review

Human review can be useful when a program needs an intermediate result checked before proceeding to a final answer. A draft, for example, may need approval or a correction before the next step uses it.

In this example, a program creates a draft and marks a point for review. A separate runner handles the interaction through [`IR.walk`](../../concepts/execution.md#manual-execution), pausing execution at that point and continuing with the reviewed value. The same traced program can therefore run with a terminal review, a different review interface, or no review.

```{admonition} Concept
[Manual Execution](../../concepts/execution.md#manual-execution) · [Tags](../../concepts/programs-and-ir.md#tags) ·
[Programs and IR](../../concepts/programs-and-ir.md)
```

(mark-the-review-point)=
## Review Point

The review point sits between draft creation and final answer assembly. A {py:func}`checkpoint <autoform.checkpoint>` normally returns the draft unchanged. The {py:func}`tag <autoform.tag>` attached during tracing marks its equation, allowing the runner to recognize where review is required.

```python
import autoform as af


review = "human-review"


def draft_then_finalize(topic: str) -> str:
    draft = "draft: " + topic
    with af.tag(review):
        draft = af.checkpoint(draft, key="draft", collection="review")

    return "final answer:\n" + draft


ir = af.trace(draft_then_finalize)("topic text")
```

## Feedback Function

The feedback function determines what happens at the review point. Its return value becomes the draft used by the remaining computation. Here, an empty response accepts the original draft, while a note is appended to it. The function could instead return an edited draft from another interface.[^review-types]

```python
def ask_for_feedback(draft: str) -> str:
    print("Draft for review:")
    print(draft)
    note = input("Review note (empty to accept): ")
    if not note:
        return draft
    return draft + "\nReview note: " + note
```

```{raw} html
:file: ../../assets/human-review.svg
```

## Runner

The runner uses [`IR.walk`](../../concepts/execution.md#manual-execution) to execute the traced program one equation at a time. At the tagged checkpoint, it calls the feedback function with the draft and waits for the result. Sending that result through `gen.send(...)` makes it the checkpoint output used by the next equation.

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

The generator retains the values already computed while waiting for review. Execution continues from the same point when the reviewed draft is sent back.

## Execution

The final call supplies the topic, the review tag, and the feedback function to the runner. A different feedback function changes the review interaction while the traced program stays the same.

```python
result = run_with_human_feedback(
    ir,
    "topic text",
    tag=review,
    feedback=ask_for_feedback,
)

print(result)
```

The printed answer contains either the accepted draft or the draft with the review note attached. Calling `ir.call(...)` runs the same program without invoking the review function.

## Tracing After Review

A review point can also separate work already completed from work that remains. For example, a longer program might start by generating an outline and a draft. After review, the draft is revised and the result is evaluated. Comparing different edits requires those last two steps to run again, while keeping the same outline. Tracing the remaining computation produces a new program that can run on each edited draft without repeating the earlier generation.

This version starts a fresh [`IR.walk`](../../concepts/execution.md#manual-execution) through the existing IR and stops at the tagged checkpoint. The same feedback function collects a reviewed draft. The generator remains paused, holding the values computed before review, until the reviewed draft is sent back during tracing.

```python
gen = ir.walk("topic text")
eqn, in_values = next(gen)

while review not in eqn.tags:
    out_values = eqn.bind(in_values, **eqn.params)
    eqn, in_values = gen.send(out_values)

draft = eqn.bind(in_values, **eqn.params)
reviewed_draft = ask_for_feedback(draft)
```

The `resume` function sends the reviewed draft to the generator and drives the remaining equations. [Tracing](../../concepts/programs-and-ir.md#trace) this function records those equations in a new IR.[^trace-generator] The reviewed draft determines the input type, but its text is not fixed in the new program. Earlier values needed by the remaining equations become fixed context. Human interaction happens before tracing, so running this IR does not prompt for review again.

```python
def resume(reviewed_draft: str) -> str:
    eqn, in_values = gen.send(reviewed_draft)
    while eqn is not None:
        out_values = eqn.bind(in_values, **eqn.params)
        eqn, in_values = gen.send(out_values)
    return in_values


remaining_ir = af.trace(resume)(reviewed_draft)
```

The remaining computation comes from the original IR. Its implementation does not need to be copied into a separate Python function: `resume` only drives the suspended walk. This is useful when a runner selects where to pause an existing program with several review points.

The new IR can run on the reviewed draft or another revision. Applying {py:func}`batch <autoform.batch>` runs the same remaining computation on several drafts, with the same context captured from the earlier execution. Each draft passes through the remaining steps; the earlier steps stay outside the new IR.

```python
result = remaining_ir.call(reviewed_draft)
drafts = [reviewed_draft, "alternative draft text"]
results = af.batch(remaining_ir).call(drafts)
print(results)
```

Other [transforms](../../concepts/transforms.md) can apply to this IR when the relevant rules are registered. For example, {py:func}`pullback <autoform.pullback>` can propagate feedback from the final answer to the reviewed draft. That feedback concerns the draft supplied to the new IR; it does not extend back to the original topic or the steps that created the draft.

In this small example, the remaining IR only adds the final-answer prefix. In the longer program described above, revision and evaluation calls would occupy that position. If those steps already form a separate Python function, tracing that function directly is simpler.

[^review-types]: [`IR.walk`](../../concepts/execution.md#manual-execution) checks program inputs and values sent through `gen.send(...)` against the types recorded during tracing. Here, the topic and reviewed draft must be strings; a numerical replacement raises `TypeError`. Python's `input()` returns a string, but annotations alone do not perform these runtime checks.

[^trace-generator]: Tracing consumes the suspended generator. The resulting IR is reusable; the generator and `resume` function are not.
