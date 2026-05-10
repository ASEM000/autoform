# Walk

```{admonition} Advanced
:class: info

Use this when you need manual equation stepping for debugging, visualization, or a custom runner. For ordinary execution, prefer `ir.call(...)` or `await ir.acall(...)`.
```

`ir.walk(...)` is the manual execution interface on the object returned by `af.trace(...)`. It exposes the same equation stream that `.call(...)` and `.acall(...)` execute for you.

Use it when you are building a debugger, visualizer, custom runner, or test harness that needs to observe or replace individual equation results. For ordinary execution, use `ir.call(...)` or `await ir.acall(...)`.

## The Contract

`ir.walk(*inputs)` returns a generator:

| Step | Action | Result |
| --- | --- | --- |
| Start | `next(gen)` | The first equation and its concrete input values. |
| Continue | `gen.send(output_values)` | The next equation and its input values. |
| Finish | send the last equation output | `None` and the final output tree. |

Each yielded equation carries its primitive and parameters. The concrete input values have the same pytree shape as that equation's inputs.

```python
import autoform as af


def program(text: str) -> str:
    text = af.concat(text, "!")
    return af.format("[{}]", text)


ir = af.trace(program)("seed")

gen = ir.walk("world")
ir_eqn, in_values = next(gen)

assert ir_eqn.prim.name == "concat"
assert in_values == ("world", "!")

out_values = ir_eqn.bind(in_values, **ir_eqn.params)
ir_eqn, in_values = gen.send(out_values)

assert ir_eqn.prim.name == "format"

out_values = ir_eqn.bind(in_values, **ir_eqn.params)
ir_eqn, output = gen.send(out_values)

assert ir_eqn is None
assert output == "[world!]"
```

`ir_eqn.bind(...)` runs the primitive's synchronous implementation for that one equation. Async runners can use `await ir_eqn.abind(...)` instead.

## What Walk Is Not

`walk` is not an IR transform. It does not take an IR and return another IR.

`walk` is not a context manager. It does not wrap trace-time or execution-time behavior around a block.

It is an advanced execution interface for the traced object. The built-in `.call(...)` and `.acall(...)` methods use the same stepping model.
