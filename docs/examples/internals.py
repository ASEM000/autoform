# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: af
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Autoform Internals
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASEM000/autoform/blob/main/docs/examples/internals.ipynb)
#
# This tutorial teaches autoform's architecture by building a custom primitive from scratch.

# %% [markdown]
# ## Setup (Colab only)
#
# Uncomment and run the following cell if running in Google Colab:

# %%
# # !pip install autoform
# import os
# os.environ["OPENAI_API_KEY"] = "your-key-here"

# %%
import functools as ft
import autoform as af
import autoform.core

# %% [markdown]
# ## 1. What is a Primitive?
#
# A **Primitive** is a named identifier for an atomic operation. It has no behavior on its own - behavior is defined separately in **rules**.
#
# Let's create a primitive called `shout` that will uppercase text:

# %%
# Create the primitive (just a name, no behavior yet)
shout_p = af.core.Primitive("shout")
print(shout_p)


# %% [markdown]
# ## 2. What is `bind`?
#
# Every primitive has a `bind` method. When called, it:
# 1. Takes inputs and parameters
# 2. Routes to the **active interpreter**
# 3. The interpreter looks up the appropriate **rule** and executes it
#
# We wrap `bind` in a user-friendly function:


# %%
def shout(text: str) -> str:
    """Uppercase the input text."""
    return shout_p.bind(text)  # bind routes to the active interpreter


# %% [markdown]
# If we try to call `shout` now, it will fail - we haven't defined any rules yet:

# %%
try:
    shout("hello")
except Exception as e:
    print(f"Error: {e}")
    print("We need to register an impl_rule first!")


# %% [markdown]
# ## 3. What are Rules?
#
# **Rules** define what happens when a primitive is called. Different rules are used in different contexts:
#
# | Rule Registry | When Used | Purpose |
# |---------------|-----------|----------|
# | `impl_rules` | Normal execution | Actually perform the operation |
# | `eval_rules` | Tracing (`build_ir`) | Return output type/shape |
# | `pull_fwd_rules` | Pullback (forward) | Execute and save residuals |
# | `pull_bwd_rules` | Pullback (backward) | Compute gradient from residuals |
#
# Let's register an **impl_rule** so our primitive works:


# %%
@ft.partial(af.core.impl_rules.def_rule, shout_p)
def impl_shout(text: str) -> str:
    """Implementation: uppercase the text."""
    return text.upper()


# %% [markdown]
# Now `shout` works:

# %%
result = shout("hello world")
print(result)


# %% [markdown]
# ## 4. How Does `bind` Know Which Rule to Use?
#
# The **active interpreter** determines which rule registry is consulted:
#
# ```
# shout_p.bind(text)
#        │
#        ▼
# ┌─────────────────────┐
# │  Active Interpreter │
# └─────────────────────┘
#        │
#        ├── EvalInterpreter ──────▶ impl_rules[shout_p]
#        ├── TracingInterpreter ───▶ eval_rules[shout_p]
#        └── PullbackInterpreter ──▶ pull_fwd/bwd_rules[shout_p]
# ```
#
# By default, `EvalInterpreter` is active, so `impl_rules` is used.

# %% [markdown]
# ## 5. Adding Tracing Support
#
# To use `build_ir`, we need an **eval_rule** that returns the output type without executing:


# %%
@ft.partial(af.core.eval_rules.def_rule, shout_p)
def eval_shout(text) -> af.core.Var:
    """Return a symbolic placeholder (Var) for the output."""
    return af.core.Var()


# %% [markdown]
# Now we can trace a function to build an IR:


# %%
def my_program(x):
    return shout(x)


ir = af.build_ir(my_program)("placeholder")
print("IR:")
print(ir)


# %% [markdown]
# ## 6. Adding Pullback Support
#
# Pullback (reverse-mode differentiation) requires two rules:
#
# 1. **pull_fwd_rule**: Run forward, save residuals needed for backward
# 2. **pull_bwd_rule**: Given residuals and output gradient, compute input gradient


# %%
@ft.partial(af.core.pull_fwd_rules.def_rule, shout_p)
def pull_fwd_shout(text: str):
    """Forward pass: return (output, residuals)."""
    output = text.upper()
    residuals = text  # save original input for backward
    return output, residuals


@ft.partial(af.core.pull_bwd_rules.def_rule, shout_p)
def pull_bwd_shout(residuals: str, out_grad: str) -> str:
    """Backward pass: compute input gradient from output gradient."""
    # For shout, gradient passes through unchanged (it's "linear")
    return out_grad


# %% [markdown]
# Now we can apply the pullback transformation:

# %%
pb_ir = af.pullback(ir)
print("Pullback IR:")
print(pb_ir)

# %%
# Execute: (input, output_gradient) -> (output, input_gradient)
output, input_grad = af.call(pb_ir)(("hello", "feedback"))
print(f"Output: {output}")
print(f"Input gradient: {input_grad}")

# %% [markdown]
# ## 7. Complete Example: LLM Primitive
#
# Now let's apply everything to build an LLM primitive with **semantic gradients** - where gradients are natural language feedback.

# %%
import autoform.lm

MODEL = "ollama/llama3:8b"  # or "openai/gpt-4o"


class Summary(af.Struct):
    text: str


# %%
# Step 1: Create the primitive
textgrad_p = af.core.Primitive("textgrad_lm")


# Step 2: Create user-facing function with bind
def textgrad_lm(prompt: str, *, model: str, struct: type):
    return textgrad_p.bind(prompt, model=model, struct=struct)


# %%
# Step 3: Register all rules


@ft.partial(af.core.impl_rules.def_rule, textgrad_p)
def impl_textgrad(prompt, *, model, struct):
    """Implementation: call LLM."""
    return af.core.impl_rules[af.lm.struct_lm_call_p](
        (prompt,), roles=("user",), model=model, struct=struct
    )


@ft.partial(af.core.eval_rules.def_rule, textgrad_p)
def eval_textgrad(prompt, *, struct, **_):
    """Tracing: return symbolic struct."""
    return struct.model_construct(**{k: af.core.Var() for k in struct.model_fields})


@ft.partial(af.core.pull_fwd_rules.def_rule, textgrad_p)
def pull_fwd_textgrad(prompt, *, model, struct):
    """Pullback forward: execute and save state."""
    out = impl_textgrad(prompt, model=model, struct=struct)
    return out, (prompt, out)


@ft.partial(af.core.pull_bwd_rules.def_rule, textgrad_p)
def pull_bwd_textgrad(residuals, cotangent, *, model, **_):
    """Pullback backward: LLM generates semantic gradient."""
    import litellm

    prompt, output = residuals
    critique = f"Input: {prompt}\nOutput: {output}\nFeedback: {cotangent}\nHow to improve input?"
    resp = litellm.completion(
        messages=[{"role": "user", "content": critique}], model=model, max_tokens=100
    )
    return resp.choices[0].message.content


# %%
# Step 4: Use it!
def summarize(topic: str) -> Summary:
    prompt = af.format("Summarize: {}", topic)
    return textgrad_lm(prompt, model=MODEL, struct=Summary)


ir = af.build_ir(summarize)("example")
pb_ir = af.pullback(ir)

feedback = Summary(text="too brief")
output, grad = af.call(pb_ir)(("AI safety", feedback))

print("Output:", output)
print("\nGradient:", grad)

# %% [markdown]
# ## Summary
#
# 1. **Primitive**: Named identifier, no behavior
# 2. **`bind`**: Routes calls to the active interpreter
# 3. **Rules**: Define behavior for each context
# 4. **Interpreter**: Determines which rule registry is used
#
# To create a new primitive:
# 1. `p = Primitive("name")`
# 2. `def my_func(...): return p.bind(...)`
# 3. Register `impl_rules`, `eval_rules`, `pull_fwd/bwd_rules` as needed
