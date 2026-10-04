# Tracing Semantics

At trace time, the function runs once with placeholder values for every dynamic input. Calls to [`autoform` primitives](primitives.md) are recorded as [IR equations](the-ir.md). Everything else is ordinary Python and runs immediately.

(static-and-dynamic-inputs)=
## Static and Dynamic Inputs

The {py:func}`trace <autoform.trace>` API is:

```text
ir = af.trace(func, static=False)(*example_inputs)
```

`static` is a bool [pytree](pytrees.md) matching the positional input structure:

- `static=False`: every input leaf is dynamic. This is the default.
- `static=True`: every input leaf is fixed at trace time.
- `static=(True, False)`: for a two-argument function, the first input is static and the second is dynamic.

To select a Python branch while tracing, mark the controlling input static:

```python
import autoform as af


def label(kind: str, text: str) -> str:
    if kind == "short":
        return "Short: " + text
    return "Long: " + text


ir = af.trace(label, static=(True, False))("short", "seed")
assert ir.call("short", "DNA") == "Short: DNA"
```

The value of the static input is recorded in the trace, and must be the same on subsequent calls.

## Traced Branches

A Python `if` needs a concrete condition. This function cannot be traced with a dynamic `kind`:

```python
def bad(kind: str, text: str) -> str:
    if kind == "short":  # wrong: kind is dynamic by default
        return "Short: " + text
    return "Long: " + text


ir = af.trace(bad)("short", "seed")
```

The comparison will be recorded, but python can’t pick a branch to follow using the placeholder value.

Use {py:func}`switch <autoform.switch>` when the branch is a runtime decision:

```python
short = af.trace(lambda text: "Short: " + text)("seed")
long = af.trace(lambda text: "Long: " + text)("seed")
branches = {"short": short, "long": long}


def routed(kind: str, text: str) -> str:
    return af.switch(kind, branches, text)


ir = af.trace(routed)("short", "seed")
assert ir.call("long", "DNA") == "Long: DNA"
```

## Runtime Loops

Python needs the number of iterations while tracing. A dynamic `n` cannot control `range`:

```python
def bad_repeat(n: int, text: str) -> str:
    out = text
    for _ in range(n):  # wrong when n is dynamic
        out = out + "!"
    return out
```

Python needs to know `n` in order to decide how many equations to make.

Use {py:func}`while_loop <autoform.while_loop>` when the loop condition is runtime data:

```python
def cond(state: tuple[str, str]) -> bool:
    text, target = state
    return text == target


def body(state: tuple[str, str]) -> tuple[str, str]:
    text, target = state
    return (text + "!"), target


cond_ir = af.trace(cond)(("seed", "target"))
body_ir = af.trace(body)(("seed", "target"))


def repeat(state: tuple[str, str]) -> tuple[str, str]:
    return af.while_loop(cond_ir, body_ir, state, max_iters=1)


loop_ir = af.trace(repeat)(("go", "go"))
assert loop_ir.call(("go", "go")) == ("go!", "go")
```

The outer trace records the loop as one primitive. Each run checks its condition and executes its body with runtime values.

## Runtime Value Inspection

A Python `print` runs while the function is traced:

```python
def noisy(text: str) -> str:
    prompt = "Explain " + text
    print(prompt)  # prints during tracing, not during every execution
    return prompt
```

Use [checkpoints](intercepts.md) (see [intercepts](intercepts.md)) for diagnostic output at execution time.

```python
def inspectable(text: str) -> str:
    prompt = "Explain " + text
    return af.checkpoint(prompt, key="prompt", collection="debug")


ir = af.trace(inspectable)("seed")
with af.collect(collection="debug") as captured:
    ir.call("recursion")

assert captured["prompt"] == ["Explain recursion"]
```

## Closures and Mutation

Values closed over by the function are captured at trace time. Mutation is handled by python and will not be recorded as IR equations:

Pass state as input and output to the function instead. If the state has structure, register it as a [pytree](pytrees.md) (see [pytrees](pytrees.md)).

Missing equations usually mean that some operation has been executed as python, instead of as an `autoform` primitive.
