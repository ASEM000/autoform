# Eliminate Dead Computations With `dce`

[`dce`](../concepts/transforms.md) removes equations that cannot affect the
selected output.

```python
import autoform as af


def program(text: str) -> str:
    unused = af.format("unused: {}", text)
    used = af.format("used: {}", text)
    del unused
    return used


ir = af.trace(program)("seed")
cleaned = af.dce(ir)

print(cleaned.call("alpha"))
```

`dce` walks backward from the output [pytree](../concepts/pytrees.md). The unused `format` call is not on
that path, so the cleaned IR can drop it.

## Keep Part Of The Output

Use `out_used` when the original function returns more than needed.

```python
import autoform as af


def pair(text: str) -> tuple[str, str]:
    left = af.format("left: {}", text)
    right = af.format("right: {}", text)
    return left, right


ir = af.trace(pair)("seed")
left_only = af.dce(ir, out_used=(True, False))

print(left_only.call("alpha"))
```

The returned tree keeps the same shape. Output leaves removed by `out_used` are
returned as `None`.

Use `dce` after transforms or debugging edits when the IR contains work that no
longer contributes to the needed value.
