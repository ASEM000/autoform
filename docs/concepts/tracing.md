# Tracing

Tracing determines which work becomes part of a reusable program and which work happens while that program is being built. The Python function runs with placeholders for dynamic inputs. Calls to [primitives](primitives-and-rules.md) become [IR equations](programs-and-ir.md#the-ir), while ordinary Python code runs immediately. This distinction matters for branches, external calls, and mutable Python state.

(inputs-and-captured-values)=
## Inputs and Captured Values

The API for {py:func}`trace <autoform.trace>` is as follows:

```text
ir = af.trace(func)(*example_inputs)
```

Every leaf of the positional input [pytree](pytrees.md#pytrees) becomes a dynamic variable. Example inputs establish the expected types; later calls can supply different values of those types. Fixed configuration belongs in the function's enclosing scope and does not occupy an IR input position.

A captured value can select a Python branch during tracing. Here, `kind` is captured, so only the selected text-formatting branch is recorded. The text remains a dynamic input:

```python
import autoform as af


kind = "short"


def label(text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text


ir = af.trace(label)("seed")
assert ir.call("topic text") == "Short: topic text"
```

Later calls supply only the text. A different branch choice requires tracing another IR. To configure a reusable Python function, capture its fixed arguments in a wrapper, such as `af.trace(lambda text: format_label("short", text))("seed")`.

## Runtime Control Flow

A Python branch needs a concrete truth value while the function runs. A dynamic input supplies only a placeholder at that stage. The following function therefore cannot use its dynamic `kind` input to choose a Python branch:

```python
def dynamic_branch(kind: str, text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text
```

Tracing with `af.trace(dynamic_branch)("short", "topic text")` records the comparison, but Python cannot use the resulting placeholder to select a branch. Capturing `kind` in a wrapper fixes the choice during tracing. A runtime choice belongs in a control-flow primitive such as {py:func}`switch <autoform.switch>` or {py:func}`while_loop <autoform.while_loop>`, as described in [Control Flow](control-flow.md).

## Fold

Some work is useful to perform while building the IR. A {py:func}`fold <autoform.fold>` block evaluates primitive implementations during tracing and stores the results as literals in the surrounding IR.[^fold-outside-tracing] Those concrete results can also select Python branches.[^trace-time-decisions]

### Fixed Context

A program may need the same context for many calls. For example, a domain guide can inform the rewriting of several drafts. Generating the guide while tracing makes that result part of the fixed context, so later calls can focus on rewriting the supplied draft.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls. The `"model-name"` placeholder stands for a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers), with the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys) configured in the environment. Labels such as `"answer instructions"` stand for task-specific instructions.
```

The domain and model are captured values in this example. The guide is generated from them inside the fold block, while the draft remains a runtime input:

```python
import autoform as af


domain = "domain description"
model = "model-name"


def rewrite_for_domain(draft: str) -> str:
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


ir = af.trace(rewrite_for_domain)("draft text")
result = ir.call("draft text")
print(result)
```

When preparing the IR, the guide is requested once. In subsequent executions, only the drafts are rewritten. If the domain or model changes, prepare a new IR.

### Dynamic Value Limits

Folded work needs concrete inputs. A placeholder for a dynamic value cannot supply the text needed by the concatenation below:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = text + " "
    return prefix
```

This example fails during tracing because `text` is dynamic. Capture concrete text from the enclosing scope when the result should be fixed during tracing, or leave the concatenation outside {py:func}`fold <autoform.fold>` to record it for execution.

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
