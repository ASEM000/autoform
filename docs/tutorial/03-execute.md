# Execute The IR

The [IR](../concepts/the-ir.md) is the recipe. [Execution](../concepts/trace-ir-execute.md) runs that recipe with real inputs.

Using the `explain` function from the previous step:

```python
# run the traced ir with a real input
output = ir.call("quantum entanglement")
print(output)
```

This does hit the active LM provider. The runtime input replaces the placeholder string used during tracing, and the recorded `lm_call` equation executes.

Calling the IR again calls the provider again:

```python
# each call executes the recorded lm call again
first = ir.call("quantum entanglement")
second = ir.call("quantum entanglement")
```

`ir` is not a cached response. It is an executable program representation.

Every IR also has an [async execution](../concepts/trace-ir-execute.md) method:

```python
import asyncio

# use asyncio.run in a normal script
output = asyncio.run(ir.acall("quantum entanglement"))
```

You do not need async execution yet. It becomes useful when the scheduled form of an IR can run independent equations concurrently.
