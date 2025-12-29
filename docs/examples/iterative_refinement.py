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
# # 🔄 Iterative Refinement
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASEM000/autoform/blob/main/docs/examples/iterative_refinement.ipynb)
#
# This tutorial demonstrates `af.while_loop` with **true early exit** for iterative LLM workflows.

# %% [markdown]
# ## Setup (Colab only)
#
# Uncomment and run the following cell if running in Google Colab:

# %%
# # !pip install autoform
# import os
# os.environ["OPENAI_API_KEY"] = "your-key-here"

# %%
from typing import Literal

import autoform as af

MODEL = "openai/gpt-4o-mini"  # or "ollama/llama3.2:3b" for local

# %% [markdown]
# ## 1. The Problem
#
# LLM workflows often require **iterative refinement**: improve text until a quality threshold is met. The challenge: when processing batches, items complete at different rates.

# %% [markdown]
# ## 2. Define the State
#
# The loop state flows through each iteration. Input and output structures must match.


# %%
class ReviewState(af.Struct):
    code: str
    has_bugs: Literal["yes", "no"]  # constrained to exact values


# %% [markdown]
# ## 3. Define Condition and Body
#
# - **Condition**: Return `True` to continue, `False` to exit
# - **Body**: Transform state, checkpoint for observability


# %%
def should_continue(state: ReviewState) -> bool:
    """Loop while bugs remain."""
    return af.match(state.has_bugs, "yes")


def fix_one_bug(state: ReviewState) -> ReviewState:
    """Find and fix one bug, report if more remain."""
    prompt = af.format(
        "Review this code. Fix ONE bug if present.\n\n"
        "```\n{}\n```\n\n"
        "Return fixed code and whether MORE bugs remain.",
        state.code,
    )

    messages = [{"role": "user", "content": prompt}]
    result = af.struct_lm_call(messages, model=MODEL, struct=ReviewState)

    af.checkpoint(result.code, collection="iterations", name="code")
    return result


# %% [markdown]
# ## 4. Build the Loop
#
# Trace condition and body separately, then compose with `while_loop`:

# %%
dummy = ReviewState(code="...", has_bugs="yes")

cond_ir = af.build_ir(should_continue)(dummy)
body_ir = af.build_ir(fix_one_bug)(dummy)


def review_loop(init: ReviewState) -> ReviewState:
    return af.while_loop(cond_ir, body_ir, init, max_iters=5)


loop_ir = af.build_ir(review_loop)(dummy)
print(loop_ir)

# %% [markdown]
# ## 5. Single Execution

# %%
buggy = ReviewState(
    code="def divide(a, b):\n    return a / b  # No zero check",
    has_bugs="yes",
)

result, collected = af.collect(loop_ir, collection="iterations")(buggy)

print("Fixed code:")
print(result.code)
print(f"\nIterations: {len(collected.get('code', []))}")

# %% [markdown]
# ## 6. Batched Execution with Early Exit
#
# Items that finish early stop consuming LLM calls:

# %%
batched_ir = af.batch(
    loop_ir,
    in_axes=ReviewState.model_construct(code=list, has_bugs=list),
)

snippets = ReviewState.model_construct(
    code=[
        "def add(a, b):\n    return a + b",  # Clean
        "def divide(a, b):\n    return a / b",  # 1 bug
        "def get(d, k):\n    return d[k]",  # Multiple issues
    ],
    has_bugs=["yes", "yes", "yes"],
)

results, collected = af.collect(batched_ir, collection=("iterations", "batch"))(snippets)

for i, (code, bugs) in enumerate(zip(results.code, results.has_bugs)):
    print(f"Snippet {i + 1}: has_bugs={bugs}")
    print(f"  {code[:60]}..." if len(code) > 60 else f"  {code}")

print(f"\nTotal iterations across all items: {len(collected.get('code', []))}")

# %% [markdown]
# ## Summary
#
# 1. **State**: Define a structured state that flows through iterations
# 2. **Condition**: Returns `True` to continue, `False` to exit
# 3. **Body**: Transform state each iteration
# 4. **Early exit**: Items stop LLM calls when done, saving cost
#
# Use `while_loop` for any iterate-until-done pattern: refinement, search, agents.
