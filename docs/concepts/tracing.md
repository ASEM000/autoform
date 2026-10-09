# Tracing

Tracing determines which work becomes part of a reusable program and which work happens while that program is being built. The Python function runs with placeholders for dynamic inputs. Calls to [primitives](primitives-and-rules.md) become [IR equations](programs-and-ir.md#the-ir), while ordinary Python code runs immediately. This distinction matters for branches, external calls, and mutable Python state.

(static-and-dynamic-inputs)=
## Static and Dynamic Inputs

The API for {py:func}`trace <autoform.trace>` is as follows:

```text
ir = af.trace(func, static=False)(*example_inputs)
```

`static` is a bool [pytree](pytrees.md#pytrees) with the same structure as the positional inputs to the function being traced. For example:

* `static=False` indicates all leaves of the input pytree are dynamic. This is the default.
* `static=True` indicates all leaves of the input pytree are static (the values must be the same on later calls).
* `static=(True, False)` means that, for a function with two arguments, the first input is static and the second is dynamic.

A fixed input can select a Python branch during tracing. Here, `kind` is static, so only the selected text-formatting branch is recorded. The text remains a dynamic input:

```python
import autoform as af


def label(kind: str, text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text


ir = af.trace(label, static=(True, False))("short", "seed")
assert ir.call("short", "topic text") == "Short: topic text"
```

Later calls to this IR must supply the same static value for `kind`. A different branch choice requires tracing another IR.

## Runtime Control Flow

A Python branch needs a concrete truth value while the function runs. A dynamic input supplies only a placeholder at that stage. The following function therefore cannot use its dynamic `kind` input to choose a Python branch:

```python
def dynamic_branch(kind: str, text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text
```

Tracing with `af.trace(dynamic_branch)("short", "topic text")` records the comparison, but Python cannot use the resulting placeholder to select a branch. Marking `kind` as static fixes the choice during tracing. A runtime choice instead belongs in a control-flow primitive such as {py:func}`switch <autoform.switch>` or {py:func}`while_loop <autoform.while_loop>`, as described in [Control Flow](control-flow.md).

## Fold

Some work is useful to perform while building the IR. A {py:func}`fold <autoform.fold>` block evaluates primitive implementations during tracing and stores the results as literals in the surrounding IR.[^fold-outside-tracing] Those concrete results can also select Python branches.[^trace-time-decisions]

### Static Context

A program may need the same context for many calls. For example, a domain guide can inform the rewriting of several drafts. Generating the guide while tracing makes that result part of the fixed context, so later calls can focus on rewriting the supplied draft.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

The domain and model are static inputs in this example. The guide is generated from those inputs inside the fold block, while the draft remains a runtime input:

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


model = "model-name"
ir = af.trace(rewrite_for_domain, static=(True, True, False))(
    "domain description",
    model,
    "draft text",
)
result = ir.call("domain description", model, "draft text")
print(result)
```

When preparing the IR, the guide is requested once. In subsequent executions, only the drafts are rewritten. If the domain or model changes, the IR can be prepared again. Subsequent calls must use the same domain and model.

### Dynamic Value Limits

Folded work needs concrete inputs. A placeholder for a dynamic value cannot supply the text needed by the concatenation below:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = text + " "
    return prefix
```

This example fails during tracing because `text` is dynamic. A static `text` input would provide a concrete value; leaving the concatenation outside {py:func}`fold <autoform.fold>` would instead record it for execution.

## Closures and Mutation

Values captured from a closure become fixed context when tracing uses them. Python mutations are not recorded as IR equations, so changing a list or object is not a way to express state updates in the recorded program. State that must change between calls belongs in the function inputs and outputs, optionally organized as a [pytree](pytrees.md#pytrees).

Python `print` calls also run during tracing, rather than on every later execution. [Checkpoints with collect](execution.md#checkpoints) capture values from the running IR. This distinction helps explain why a Python side effect may appear while building a program but not when calling it.

[^fold-outside-tracing]: {py:func}`fold <autoform.fold>` has no effect outside tracing.

[^trace-time-decisions]: **Trace-Time Decisions.** Fold can also be used to execute Python control flow at trace time:

    ```python
    increment = af.trace(lambda value: value + 1)(1.0)


    def route(text: str) -> str:
        with af.fold():
            label = increment.call(1.0)
        if label == 2:
            return "yes: " + text
        return "no: " + text


    ir = af.trace(route)("seed")
    assert ir.call("answer") == "yes: answer"
    ```

    In this example, the branch will be selected at trace time and only the selected branch will appear in the resulting IR.
