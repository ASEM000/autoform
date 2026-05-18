# Specialize a Program with Static Context

Use {py:func}`fold <autoform.fold>` when a structured LM call should run while
tracing and produce configuration for later dynamic inputs. This recipe builds
a domain-specific rewriter: the domain is fixed at trace time, and each draft
is supplied at execution time.

```{admonition} Concept
[Fold](../../concepts/fold.md) · [Tracing Semantics](../../concepts/tracing-semantics.md) · [Schemas](../../concepts/schemas.md)
```

## Trace-Time and Runtime

Tracing and execution happen at different times:

- Trace-time: {py:func}`trace <autoform.trace>` runs the Python function once
  to build an IR. Static inputs are concrete during this run. Work inside
  {py:func}`fold <autoform.fold>` also runs during this phase, so its result is
  embedded in the IR.
- Runtime: `.call(...)` or `.acall(...)` runs the traced IR with real dynamic
  inputs. Dynamic inputs replace the example values used for tracing. Folded
  work does not run again.

In this recipe, the domain and model are trace-time context. The draft is
runtime data.

## Build the Program

The first schema call creates a domain guide. It runs inside
{py:func}`fold <autoform.fold>`, so the guide becomes a concrete value in the
traced program. The second schema call runs later and consumes the dynamic
draft.

```python
import autoform as af


guide_schema = dict(
    audience=af.Str(),
    terms=[af.Str(), af.Str(), af.Str()],
    rule=af.Str(),
)

rewrite_schema = dict(text=af.Str())


def rewrite_for_domain(domain: str, model: str, draft: str) -> dict[str, str]:
    # trace-time: build a domain guide once and embed it in the ir.
    with af.fold():
        guide_messages = [
            dict(role="system", content="Derive a formal policy for the selected topic."),
            dict(role="user", content=domain),
        ]
        guide = af.lm_schema_call(
            guide_messages,
            model=model,
            schema=guide_schema,
        )

    # runtime: rewrite each draft using the guide embedded above.
    system = (
        f"Audience: {guide['audience']}\n"
        f"Preferred terms: {', '.join(guide['terms'])}\n"
        f"Rule: {guide['rule']}"
    )
    draft_prompt = af.format(
        "Apply the topic policy to statement S:\n{}",
        draft,
    )
    rewrite_messages = [
        dict(role="system", content=system),
        dict(role="user", content=draft_prompt),
    ]
    return af.lm_schema_call(
        rewrite_messages,
        model=model,
        schema=rewrite_schema,
    )
```

## Trace with Static Context

Mark the domain and model static, and the draft dynamic:

```python
# trace-time: the static domain and model specialize the ir.
ir = af.trace(rewrite_for_domain, static=(True, True, False))("topic x", "gpt-5.5", "seed draft")

# runtime: the same ir can run on new drafts.
result = ir.call("topic x", "gpt-5.5", "method A maps input x to output y under constraint C")

print(result["text"])
```

The folded schema call executes during tracing. Later calls only execute the
second schema call because the domain guide is already embedded in the IR.

## Changing Static Context Requires Retracing

The IR is specialized to the domain and model used during tracing. The same IR
can run with many drafts, but calling it with different static context is
invalid:

```python
ir.call("topic y", "gpt-5.5", "method A maps input x to output y under constraint C")
```

Expected error:

```text
AssertionError: Static input mismatch
```

This catches accidental reuse of an IR specialized for one domain or model as
if it were specialized for another.

## Compile a Rewriter for One Domain

When the domain and model are compile-time configuration, capture them before
tracing so callers only pass the draft:

```python
def compile_rewriter(domain: str, model: str):
    def rewrite(draft: str) -> dict[str, str]:
        return rewrite_for_domain(domain, model, draft)

    # trace-time: domain and model are captured by the closure.
    return af.trace(rewrite)("seed draft")


rewrite_topic_x = compile_rewriter("topic x", "gpt-5.5")

# runtime: callers only pass the draft.
result = rewrite_topic_x.call("procedure P estimates parameter theta from observations z")
```

The closure version has one runtime input: `draft`. There is no domain or model
argument to pass incorrectly. To change either value, compile another IR.

- Use `static=(True, True, False)` when the fixed values should stay visible in
  the traced signature. Runtime calls still pass `domain`, `model`, and
  `draft`; a different static value raises `Static input mismatch`.
- Use a closure when the fixed value is compile-time configuration. Runtime
  calls only pass `draft`; changing the domain or model requires retracing.
