# `autoform`

**JAX-style function transformations for LLM programs**

> ⚠️ **Early Development**: This project is under active development. The API is unstable and may change significantly between versions. Use at your own risk in production.

Build LLM pipelines with composable IR transforms: `pushforward`, `pullback`, and `batch`.


## Installation

```bash
git clone https://github.com/ASEM000/autoform.git
cd autoform
uv sync
```


## Quick Start

The core idea is simple: write your LLM program once, then **transform** it in powerful ways without rewriting any code.

```python
import autoform as af

# define a simple LLM program
def greet(name: str) -> str:
    prompt = af.format("Say hello to {} in a friendly way", name)
    messages = [{"role": "user", "content": prompt}]
    return af.lm_call(messages, model="openai/gpt-4o")

# build the ir — trace your program into an intermediate representation
# this captures the structure of your program as data
ir = af.build_ir(greet, "Alice")

# run it with different inputs without rebuilding
result = af.run_ir(ir, "Bob")  # "Hello Bob! Great to meet you!"

# transform: batch multiple inputs
# instead of calling the LLM 3 times, process all inputs in parallel
batch_ir = af.batch_ir(ir, in_axes=list)
results = af.run_ir(batch_ir, ["Alice", "Bob", "Charlie"])

# transform: semantic backpropagation
# given feedback on the output, get feedback on the input!
pb_ir = af.pullback_ir(ir)
output, input_grad = af.run_ir(pb_ir, ("Bob", "be more formal"))
# input_grad contains suggestions for how to change the input
```

---

## Core Concepts

### What Problem Does This Solve?

When you build LLM applications, you often want to:
- **Run the same logic with different inputs** (batching)
- **Understand how changes to inputs affect outputs** (sensitivity analysis)
- **Propagate feedback backwards** to improve earlier steps (optimization)

Normally, you'd have to manually implement each of these patterns. `autoform` lets you write your logic once and automatically derive these capabilities.

### Primitives

Everything in autoform is built from **primitives** — atomic operations with defined transform rules. Think of them as the "building blocks" that autoform knows how to differentiate and batch:

- `af.format(template, *args)` — string formatting (like Python's `.format()`)
- `af.concat(*args)` — joins strings together  
- `af.lm_call(messages, model=...)` — calls an LLM and returns the response
- `af.struct_lm_call(messages, model=..., struct=...)` — calls an LLM and returns structured data
- ... more to come soon

### Transforms

Transforms take an IR (your traced program) and return a new IR with enhanced capabilities:

| Transform | What It Does | Use Case |
|-----------|--------------|----------|
| `pushforward_ir(ir)` | Propagates "changes" forward through the program | Sensitivity analysis: "If I change X, how does Y change?" |
| `pullback_ir(ir)` | Propagates "feedback" backward through the program | Optimization: "Given feedback on output, what should I change about input?" |
| `batch_ir(ir)` | Makes the program process multiple inputs at once | Efficiency: Process 100 queries in parallel instead of sequentially |

**Transforms compose!** `batch_ir(pullback_ir(ir))` gives you batched backpropagation — get feedback for 100 examples simultaneously.

---

## Defining Custom Primitives

Want to add your own operations that work with all transforms? Define a primitive with its rules:

```python
import functools as ft
import autoform as af

# create the primitive
shout_p = af.Primitive("shout")

# user-facing function
def shout(text: str) -> str:
    """converts text to uppercase."""
    return af.bind(shout_p, text)

# implementation rule — what it actually does when run
@ft.partial(af.impl_rules.set, shout_p)
def impl_shout(text: str) -> str:
    return text.upper()

# eval rule — tells the tracer what shape/type the output has
@ft.partial(af.eval_rules.set, shout_p)
def eval_shout(text) -> af.Var:
    return af.Var()

# pushforward rule — how changes propagate forward
# "if input changes by tangent, output changes by tangent.upper()"
@ft.partial(af.push_rules.set, shout_p)
def push_shout(primal: str, tangent: str) -> tuple[str, str]:
    return primal.upper(), tangent.upper()

# pullback rules — how feedback propagates backward
# first: run forward and save anything needed for backward
@ft.partial(af.pull_fwd_rules.set, shout_p)
def pull_fwd_shout(text: str) -> tuple[str, str]:
    return text.upper(), text  # (output, residual_for_backward)

# then: given feedback on output, compute feedback on input
@ft.partial(af.pull_bwd_rules.set, shout_p)
def pull_bwd_shout(residual: str, cotangent: str) -> str:
    return cotangent  # feedback passes through unchanged

# batch rule — how to process multiple inputs at once
@ft.partial(af.batch_rules.set, shout_p)
def batch_shout(batch_size: int, in_batched: bool, text) -> tuple:
    if in_batched:
        return [t.upper() for t in text], True
    return text.upper(), False
```

Now your primitive works with every transform automatically:

```python
def program(x):
    return shout(af.format("Say: {}", x))

ir = af.build_ir(program, "hello")
af.run_ir(ir, "world")  # "SAY: WORLD"

# all transforms work automatically!
batch_ir = af.batch_ir(ir, in_axes=list)
af.run_ir(batch_ir, ["a", "b", "c"])  # ["SAY: A", "SAY: B", "SAY: C"]
```

---

## TextGrad-style Semantic Backpropagation

This is where `autoform` gets interesting for LLM applications.

**The idea**: Instead of computing numerical gradients (like in neural networks), we can use an LLM to generate *semantic* feedback. Given the output and some critique, the LLM can suggest how to improve each input.

Here's how to define an LLM primitive with a custom backward pass that uses another LLM call to generate feedback:

```python
import functools as ft
import litellm
import autoform as af

# define output structures (what the LLM should return)
class ResearchNotes(af.Struct):
    topic: str
    key_points: str
    sources: str

class Article(af.Struct):
    title: str
    body: str
    summary: str


# create primitive with custom backward pass
textgrad_style_lm_call_p = af.Primitive("textgrad_style_lm_call")

def textgrad_style_lm_call(messages, *, model: str, struct: type) -> af.Struct:
    roles = tuple(m["role"] for m in messages)
    contents = tuple(m["content"] for m in messages)
    return af.bind(textgrad_style_lm_call_p, contents, roles=roles, model=model, struct=struct)

# forward: normal llm call
@ft.partial(af.impl_rules.set, textgrad_style_lm_call_p)
def impl_textgrad_style(contents, *, roles, model, struct):
    return af.impl_rules.get(af.struct_lm_call_p)(
        contents, roles=roles, model=model, struct=struct
    )

# eval: needed for tracing/build_ir
@ft.partial(af.eval_rules.set, textgrad_style_lm_call_p)
def eval_textgrad_style(in_tree, *, struct, **params):
    return struct.model_construct(**{k: af.Var() for k in struct.model_fields})

# pullback forward: run and save residuals (inputs + output)
@ft.partial(af.pull_fwd_rules.set, textgrad_style_lm_call_p)
def pull_fwd_textgrad_style(contents, *, roles, model, struct):
    out = impl_textgrad_style(contents, roles=roles, model=model, struct=struct)
    residuals = (contents, roles, out)
    return out, residuals

# pullback backward: llm generates critique for inputs!
# use an LLM to generate semantic gradients
@ft.partial(af.pull_bwd_rules.set, textgrad_style_lm_call_p)
def pull_bwd_textgrad_style(residuals, cotangent_out, *, roles, model, struct):
    contents, original_roles, output = residuals
    
    # use llm to generate targeted feedback for each input
    input_gradients = []
    for i, content in enumerate(contents):
        critique_prompt = f"""
        Original input: {content}
        Output: {output}
        Downstream feedback: {cotangent_out}
        
        How should this input be improved?
        """
        resp = litellm.completion(
            messages=[{"role": "user", "content": critique_prompt}],
            model=model,
        )
        input_gradients.append(resp.choices[0].message.content)
    
    return tuple(input_gradients)

# batch: scale to multiple inputs efficiently!
@ft.partial(af.batch_rules.set, textgrad_style_lm_call_p)
def batch_textgrad_style(batch_size, in_batched, contents, *, roles, model, struct):
    # construct batched messages
    batched_msgs = []
    for b in range(batch_size):
        msgs = [
            {"role": r, "content": contents[i][b] if in_batched[i] else contents[i]} 
            for i, r in enumerate(roles)
        ]
        batched_msgs.append(msgs)
    
    # use litellm.batch_completion for parallel processing
    responses = litellm.batch_completion(messages=batched_msgs, model=model, response_format=struct)
    results = [resp.choices[0].message.parsed for resp in responses]
    return results, True
```

### Multi-Agent Pipeline with Backprop

Now here's the payoff — chain multiple LLM calls together, and backprop feedback through the entire pipeline:

```python
def multi_agent_pipeline(topic: str):
    # agent 1: researcher gathers information
    research_msgs = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user", "content": af.format("Research: {}", topic)},
    ]
    notes = textgrad_style_lm_call(research_msgs, model="openai/gpt-4o", struct=ResearchNotes)
    
    # agent 2: writer creates article from research
    writer_msgs = [
        {"role": "system", "content": "Write an article from these notes."},
        {"role": "user", "content": af.format("Notes: {}", notes.key_points)},
    ]
    article = textgrad_style_lm_call(writer_msgs, model="openai/gpt-4o", struct=Article)
    
    return article

# build the pipeline
ir = af.build_ir(multi_agent_pipeline, "AI safety")

# now: given feedback on the final article, get suggestions for the original topic!
pb_ir = af.pullback_ir(ir)

# feedback must match output structure
feedback = Article(
    title="Good title",
    body="Needs more technical depth on alignment.",
    summary="Concise."
)
output, input_feedback = af.run_ir(pb_ir, ("AI safety", feedback))
# input_feedback contains LLM-generated suggestions for improving the original topic!

# bonus: batch multiple topics
batch_ir = af.batch_ir(ir, in_axes=list)
articles = af.run_ir(batch_ir, ["AI safety", "quantum computing", "climate tech"])
```

---

## Batch Transform Deep Dive

The `batch_ir` transform is simple but powerful: it converts a program that handles ONE input into a program that handles MANY inputs in parallel.

```python
# write code for one input
def process(query: str) -> str:
    return af.format("[Processed: {}]", query)

ir = af.build_ir(process, "test")

# transform to handle many inputs
batch_ir = af.batch_ir(ir, in_axes=list)

# now run with a list!
results = af.run_ir(batch_ir, ["q1", "q2", "q3"])
# ["[Processed: q1]", "[Processed: q2]", "[Processed: q3]"]
```

### How Batch Rules Work

Each primitive defines how it handles batched inputs. The batch rule receives:
- `batch_size`: how many items are in the batch
- `in_batched`: which inputs are batched (some might be shared across all items)
- `in_tree`: the actual input values

```python
@ft.partial(af.batch_rules.set, my_primitive_p)
def batch_my_primitive(
    batch_size: int,        # number of items in batch
    in_batched: tuple,      # which inputs are batched (bool per input)
    in_tree,                # input values (lists if batched)
    **params                # static parameters
):
    # process all batch items
    results = []
    for b in range(batch_size):
        # get value: in_tree[i][b] if batched, else in_tree[i]
        val = in_tree[b] if in_batched else in_tree
        results.append(process_one(val, **params))
    
    # return (outputs, is_output_batched)
    return results, True
```

### Composing Transforms

Transforms compose naturally — you can batch a pullback, or pullback through a batch:

```python
ir = af.build_ir(my_program, "x")

# batched backprop: get gradients for many examples at once
batch_pb = af.batch_ir(af.pullback_ir(ir), in_axes=(list, list))
outputs, grads = af.run_ir(batch_pb, inputs, cotangents)

# batched pushforward: sensitivity analysis for many inputs
batch_pf = af.batch_ir(af.pushforward_ir(ir), in_axes=(list, list))
primals, tangents = af.run_ir(batch_pf, primals, tangents)
```

---

## How it Works

Under the hood, `autoform` works like a modern autodiff engine (think JAX or PyTorch), but designed for symbolic and LLM-based operations instead of tensors.

1. **Tracing & IR**: When you call `build_ir`, autoform runs your Python function with special "tracer" inputs. Every time you call a primitive (like `lm_call` or `format`), it records that operation. The result is an **Intermediate Representation (IR)** — a data structure representing your computation as a sequence of primitive calls.

2. **Execution via Interpreters**: `run_ir` takes this IR and executes it using an **Interpreter**. The default interpreter runs each primitive using its `impl_rules`. But you could write other interpreters — for visualization, cost estimation, or serialization.

3. **Transforms as IR Wrappers**: When you call `batch_ir(ir)` or `pullback_ir(ir)`, you don't modify the original IR. Instead, you create a *new* IR that wraps the original in a "higher-order primitive" (`batch_call`, `pullback_call`). When this wrapper primitive runs, it knows how to transform each inner primitive using that primitive's registered rules.

4. **Rule-Based Extensibility**: Every primitive defines its own rules for each transform. This means YOU can add new primitives — any LLM call, API request, or Python computation — and teach autoform how to batch or differentiate it by registering the appropriate rules.

---

## Contributing

`autoform` is in early development and we'd love your help!

### Share Ideas
- **Use Cases**: What multi-agent pipelines or LLM workflows would you like to optimize?
- **Design Feedback**: How can we make the API more intuitive?
- **Feature Requests**: What transforms or primitives would be most useful?

More to come soon