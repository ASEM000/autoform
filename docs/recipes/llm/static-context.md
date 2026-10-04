# Static Context

A domain guide can be generated once and reused to rewrite many drafts.
{py:func}`fold <autoform.fold>` runs the guide-generation call while tracing and embeds the result in the IR.
Later executions generate only the rewritten draft.

```{admonition} Concept
[Fold](../../concepts/fold.md) · [Tracing Semantics](../../concepts/tracing-semantics.md) · [Schemas](../../concepts/schemas.md)
```

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

## Program

Generate a guide from the fixed domain, then use it with the runtime draft:

```python
import autoform as af


def rewrite_for_domain(domain: str, model: str, draft: str) -> str:
    with af.fold():
        content = dict(
            domain=domain,
            guide=af.lm.Str(desc="style guide instructions"),
        )
        guide = af.lm.fill(content, model=model)["guide"]
    content = dict(
        guide=guide,
        draft=draft,
        text=af.lm.Str(desc="rewrite instructions"),
    )
    return af.lm.fill(content, model=model)["text"]
```

Folded work requires concrete inputs. Mark the domain and model static while keeping the draft dynamic:

```python
model = "model-name"
ir = af.trace(rewrite_for_domain, static=(True, True, False))(
    "technical documentation",
    model,
    "draft text",
)
result = ir.call("technical documentation", model, "draft text")
print(result)
```

Tracing makes the first model request. The resulting IR contains the generated guide as a literal.
Each execution uses that guide to rewrite its draft.

## Static Inputs

Later calls must pass the same domain and model. This call raises `AssertionError: Static input mismatch`:

```python
ir.call("another domain", model, "draft text")
```

To change a static input, trace the function again.

## Closure

If callers should pass only the draft, capture the domain and model before tracing:

```python
def compile_rewriter(domain: str, model: str):
    def rewrite(draft: str) -> str:
        return rewrite_for_domain(domain, model, draft)

    return af.trace(rewrite)("draft text")


rewriter = compile_rewriter("technical documentation", model)
print(rewriter.call("draft text"))
```

The closure and static-input forms both specialize the IR.
The closure form removes the fixed configuration from the call signature.
Creating another rewriter makes another guide-generation request.
