# Quickstart

## Installation

```bash
pip install autoform
```

## Core Concepts

Autoform lets you build **composable LLM programs** by tracing Python functions into an **IR** (Intermediate Representation), then applying transformations like batching and differentiation.

```python
import autoform as af
```

**Trace a Function**

```python
def greet(name):
    return af.concat("Hello, ", name, "!")

# Trace with example inputs -> IR
ir = af.build_ir(greet)("name")
print(ir)
```

**Execute the IR**

```python
result = af.call(ir)("World")
print(result)  # "Hello, World!"
```

**Batch Execution**

Process multiple inputs in parallel:

```python
batched_ir = af.batch(ir, in_axes=list)
results = af.call(batched_ir)(["Alice", "Bob", "Charlie"])
# ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]
```

**Semantic Gradients (Pullback)**

Compute "textual gradients" - feedback that flows backward through the program:

```python
pb_ir = af.pullback(ir)
output, gradient = af.call(pb_ir)(("World", "too formal"))
# output = "Hello, World!"
# gradient = "..." (LLM-generated suggestion to improve input)
```

**LLM Calls**

Use structured output with Pydantic:

```python
class Response(af.Struct):
    answer: str
    confidence: float

def ask(question):
    prompt = af.format("Answer: {}", question)
    return af.struct_lm_call(
        [{"role": "user", "content": prompt}],
        model="openai/gpt-4o-mini",
        struct=Response,
    )

ir = af.build_ir(ask)("question")
result = af.call(ir)("What is 2+2?")
print(result.answer, result.confidence)
```

## Next Steps

- **[Internals](examples/internals)**: Learn how primitives and rules work
- **[Iterative Refinement](examples/iterative_refinement)**: Build loops with early exit
- **[API Reference](api)**: Full function documentation
