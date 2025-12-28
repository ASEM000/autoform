# Quickstart

## Installation

```bash
pip install autoform
```

## Basic Usage

### Tracing

```python
import autoform as af

def greet(name):
    return af.concat("Hello, ", name)

# Trace to create IR
ir = af.build_ir(greet)("name")

# Execute
result = af.call(ir)("World")
print(result)  # "Hello, World"
```

### Batching

```python
batched_ir = af.batch(ir, in_axes=list)
results = af.call(batched_ir)(["Alice", "Bob", "Charlie"])
# ["Hello, Alice", "Hello, Bob", "Hello, Charlie"]
```

### Semantic Gradients

```python
pb_ir = af.pullback(ir)
primal, cotangent = af.call(pb_ir)(("World", "feedback"))
# primal = "Hello, World"
# cotangent = "..." (LLM-generated gradient)
```
