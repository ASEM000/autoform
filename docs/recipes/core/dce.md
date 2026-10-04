# Dead Code Elimination

{py:func}`dce <autoform.dce>` removes equations that cannot affect the
selected output. This can reduce work left by unused intermediate values or earlier transforms.

```{admonition} Concept
[Transforms](../../concepts/transforms.md) · [Pytrees](../../concepts/pytrees.md)
```

An example of removing a string operation that is no longer used:

```python
import autoform as af


def program(text: str) -> str:
    unused = "unused: " + text
    used = "used: " + text
    del unused
    return used


ir = af.trace(program)("seed")
cleaned = af.dce(ir)

print(cleaned.call("alpha"))
```

The {py:func}`dce <autoform.dce>` function walks backwards from the output, which can be a [pytree](../../concepts/pytrees.md). Because the {py:func}`concat <autoform.string.concat>` operation is not used in the selected part of the output, it is removed in the cleaned version of the IR, and the result of executing the IR is only `used: alpha`.

```{raw} html
:file: ../../assets/dead-code.svg
```

## Selected Outputs

Use `out_used` to select the required output leaves:

```python
import autoform as af


def pair(text: str) -> tuple[str, str]:
    left = "left: " + text
    right = "right: " + text
    return left, right


ir = af.trace(pair)("seed")
left_only = af.dce(ir, out_used=(True, False))

print(left_only.call("alpha"))
```

The returned tree keeps the same shape. Output leaves removed by `out_used` are
returned as `None`. This call prints `("left: alpha", None)`.

Primitives that are registered as not to be DCE’d, such as checkpoints and factors, will still be present after DCE.

Use {py:func}`dce <autoform.dce>` after transforms or debugging edits when the IR contains work that no
longer contributes to the needed value.
