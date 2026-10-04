# Walk

```{admonition} Advanced
:class: info

Use this when manual equation stepping is needed for debugging, visualization, or a custom runner. For ordinary execution, prefer `ir.call(...)` or `await ir.acall(...)`.
```

`ir.walk(...)` is the manual execution interface on the object returned by {py:func}`trace <autoform.trace>`. It exposes the same equation stream that `.call(...)` and `.acall(...)` execute.

## Generator

The function `ir.walk(*inputs)` returns a generator:

| Step | Action | Result |
| --- | --- | --- |
| Start | `next(gen)` | The first equation and its concrete input values. |
| Continue | `gen.send(output_values)` | The next equation and its input values. |
| Finish | send the last equation output | `None` and the final output tree. |

Each yielded equation carries its primitive and parameters. The concrete input values have the same pytree shape as that equation's inputs.

## Example

To run each equation, and send the result back into the generator, simply do:

```python
import autoform as af


def program(text: str) -> str:
    text = text + "!"
    return "[" + text + "]"


ir = af.trace(program)("seed")

gen = ir.walk("world")
eqn, in_values = next(gen)

assert eqn.prim.name == "concat"
assert in_values == ("world", "!")

out_values = eqn.bind(in_values, **eqn.params)
eqn, in_values = gen.send(out_values)

assert eqn.prim.name == "concat"

out_values = eqn.bind(in_values, **eqn.params)
eqn, in_values = gen.send(out_values)

assert eqn.prim.name == "concat"
assert in_values == ("[world!", "]")

out_values = eqn.bind(in_values, **eqn.params)
eqn, output = gen.send(out_values)

assert eqn is None
assert output == "[world!]"
```

Here `eqn.bind(...)` runs the synchronous implementation of the primitive for that equation. If the runner is asynchronous, then `await eqn.abind(...)` should be used instead.

## Execution Boundary

The runner controls execution one equation at a time. It must send a value with the expected type and container structure before continuing. The built-in `.call(...)` and `.acall(...)` methods handle this sequence automatically.
