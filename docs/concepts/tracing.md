# Tracing

When a function is traced, it will be executed exactly once using placeholder values for all dynamic inputs. Any invocations of `autoform` primitives (see the [list of primitives](primitives-and-rules.md)) are recorded as IR equations (see [the IR description](programs-and-ir.md#the-ir) for details). All other Python code will run immediately.

(static-and-dynamic-inputs)=
## Static and Dynamic Inputs

The API for {py:func}`trace <autoform.trace>` is as follows:

```text
ir = af.trace(func, static=False)(*example_inputs)
```

`static` is a bool [pytree](pytrees.md#pytrees) with the same structure as the positional inputs to the function being traced. For example:

* `static=False` indicates all leaves of the input pytree are dynamic. This is the default.
* `static=True` indicates all leaves of the input pytree are static (the values must be the same on later calls).
* `static=(True, False)` for a two argument function indicates the first input is static and the second input is dynamic.

To use an input to select a Python branch, mark it as `static`:

```python
import autoform as af


def label(kind: str, text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text


ir = af.trace(label, static=(True, False))("short", "seed")
assert ir.call("short", "topic text") == "Short: topic text"
```

Note that the value must be the same for all subsequent calls to the function.

## Runtime Control Flow

Python `if` statements and calls to `range()` expect concrete Python values. The example below would fail because `kind` is dynamic (i.e. it is a placeholder value during tracing).

```python
def dynamic_branch(kind: str, text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text
```

If this function is traced using `af.trace(dynamic_branch)("short", "topic text")` then the comparison in the if statement will be recorded as an IR equation. However, selecting the branch will fail because Python will try to run that code, and it requires a concrete value. To fix this, mark `kind` as `static` so the branch is fixed at trace time. Alternatively, use one of `autoform`'s control flow primitives like {py:func}`switch <autoform.switch>` or {py:func}`while_loop <autoform.while_loop>` to make the decision at runtime. See the [Control Flow](control-flow.md) page for details.

## Fold

Use a {py:func}`fold <autoform.fold>` block to run primitive implementations at trace time and use the result as a literal in the surrounding IR.[^fold-outside-tracing] Folded results can select Python branches during tracing.[^trace-time-decisions]

### Static Context

Multiple parts of a computation may require some shared information that is expensive to compute. For example, if a large domain guide is used to rewrite multiple drafts of an article, it should be generated once for all rewrites.

```{admonition} Model Setup
`autoform` uses LiteLLM for model calls.
Replace `"model-name"` with a model from [LiteLLM's provider reference](https://docs.litellm.ai/docs/providers).
Set the provider's [API key](https://docs.litellm.ai/docs/set_keys#setting-api-keys).
Replace labels such as `"answer instructions"` with text for the task.
```

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

The example below will fail because `text` is dynamic:

```python
def bad(text: str) -> str:
    with af.fold():
        prefix = text + " "
    return prefix
```

This will raise an error at trace time because `text` is not concrete. Either mark `text` as `static` or move the work outside of the {py:func}`fold <autoform.fold>` block.

## Closures and Mutation

All closed over variables will be captured at trace time. Mutating Python state is not recorded as an IR equation. If the function needs to communicate state between invocations, include that state in the function's inputs and outputs. If necessary, register the state as a [pytree](pytrees.md#pytrees).

Python `print` statements will execute at trace time. To capture values from later executions, use [checkpoints with collect](execution.md#checkpoints). If it appears that some work is not producing equations, it may be running as normal Python code.

[^fold-outside-tracing]: Note that {py:func}`fold <autoform.fold>` is a no-op context manager outside of trace time.

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
