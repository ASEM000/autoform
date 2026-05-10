# Tracing Semantics

At trace time, the function runs once with placeholder values for every dynamic input. Operations that hit [`autoform` primitives](primitives.md) are recorded as [IR equations](the-ir.md). Everything else is ordinary Python and runs immediately.

That rule explains most tracing surprises.

(static-and-dynamic-inputs)=
## Static and Dynamic Inputs

The trace API is:

```python
af.trace(func, /, *, static: Tree[bool] = False)
```

`static` is a bool [pytree](pytrees.md) matching the positional input structure:

- `static=False`: every input leaf is dynamic. This is the default.
- `static=True`: every input leaf is fixed at trace time.
- `static=(True, False)`: for a two-argument function, the first input is static and the second is dynamic.

Use static inputs when ordinary Python control flow should be selected while tracing:

```python
def label(kind: str, text: str) -> str:
    if kind == "short":
        return af.format("Short: {}", text)
    return af.format("Long: {}", text)


ir = af.trace(label, static=(True, False))("short", "seed")
assert ir.call("short", "DNA") == "Short: DNA"
```

The static value is part of the trace. Later calls must pass the same static value.

## Traced Branches

```python
def bad(kind: str, text: str) -> str:
    if kind == "short":  # wrong: kind is dynamic by default
        return af.format("Short: {}", text)
    return af.format("Long: {}", text)


ir = af.trace(bad)("short", "seed")
```

The comparison would need a concrete value while tracing. A dynamic input only carries abstract type information.

Use [`switch`](primitives.md) when the branch is a runtime decision:

```python
short = af.trace(lambda text: af.format("Short: {}", text))("seed")
long = af.trace(lambda text: af.format("Long: {}", text))("seed")
branches = {"short": short, "long": long}


def routed(kind: str, text: str) -> str:
    return af.switch(kind, branches, text)


ir = af.trace(routed)("short", "seed")
assert ir.call("long", "DNA") == "Long: DNA"
```

## Runtime Loops

```python
def bad_repeat(n: int, text: str) -> str:
    out = text
    for _ in range(n):  # wrong when n is dynamic
        out = af.concat(out, "!")
    return out
```

Python needs `n` while tracing to decide how many equations to create.

Use [`while_loop`](primitives.md) when the loop condition is runtime data:

```python
def cond(state: tuple[str, str]) -> bool:
    text, target = state
    return text == target


def body(state: tuple[str, str]) -> tuple[str, str]:
    text, target = state
    return af.concat(text, "!"), target


cond_ir = af.trace(cond)(("seed", "target"))
body_ir = af.trace(body)(("seed", "target"))
looped = af.while_loop(cond_ir, body_ir, ("go", "go"), max_iters=1)
```

The loop is now one explicit primitive in the surrounding IR.

## Runtime Value Inspection

```python
def noisy(text: str) -> str:
    prompt = af.format("Explain {}", text)
    print(prompt)  # prints during tracing, not during every execution
    return prompt
```

Use [checkpoints](intercepts.md) when execution-time diagnostics are needed:

```python
def inspectable(text: str) -> str:
    prompt = af.format("Explain {}", text)
    return af.checkpoint(prompt, key="prompt", collection="debug")


ir = af.trace(inspectable)("seed")
with af.collect(collection="debug") as captured:
    ir.call("recursion")

assert captured["prompt"] == ["Explain recursion"]
```

## Closures and Mutation

Closure values are captured by the Python function while tracing. Mutating a list inside a traced function is still Python mutation; it is not an IR equation.

Pass state through inputs and outputs instead. If the state is structured, register it as a [pytree](pytrees.md) so transforms can walk its leaves.

Missing equations usually mean the operation ran as Python instead of going through an `autoform` primitive.
