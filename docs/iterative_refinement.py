# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Iterative Refinement
#
# LLM workflows often require **iterative refinement**: improve text until a quality
# threshold is met, or fix bugs until the code is clean. The challenge is
# efficiency—when processing batches, items complete at different rates.
#
# This tutorial demonstrates `af.while_loop` with **true early exit**: items that
# finish stop consuming LLM calls immediately, saving cost and time.

# %%
import autoform as af

MODEL = "ollama/llama3.2:3b"  # or "openai/gpt-4o-mini"

# %% [markdown]
# ## The Cost Problem
#
# Consider reviewing 3 code snippets with a maximum of 5 refinement iterations:
#
# | Snippet | Bugs | Standard Batching | Early Exit |
# |---------|------|-------------------|------------|
# | 1 | 0 (clean) | 5 calls (4 wasted) | 1 call |
# | 2 | 1 | 5 calls (3 wasted) | 2 calls |
# | 3 | 5 | 5 calls | 5 calls |
# | **Total** | | **15 calls** | **8 calls** |


# %% [markdown]
# ## Implementation
#
# ### Define the Loop State
#
# The state flows through each iteration. Input and output structures must match.


# %%
class ReviewState(af.Struct):
    code: str
    has_bugs: str  # "yes" or "no"


# %% [markdown]
# ### Define Condition and Body
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

    # Each checkpoint = one iteration (collected later)
    af.checkpoint(result.code, collection="iterations", name="code")

    return result


# %% [markdown]
# ### Build the Loop

# %%
dummy = ReviewState(code="", has_bugs="yes")

cond_ir = af.build_ir(should_continue)(dummy)
body_ir = af.build_ir(fix_one_bug)(dummy)


def review_loop(init: ReviewState) -> ReviewState:
    return af.while_loop(cond_ir, body_ir, init, max_iters=5)


loop_ir = af.build_ir(review_loop)(dummy)
print(loop_ir)

# %% [markdown]
# ### Single Execution

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
# ### Batched Execution with Early Exit

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

results, collected = af.collect(batched_ir, collection="iterations")(snippets)

for i, (code, bugs) in enumerate(zip(results.code, results.has_bugs)):
    print(f"Snippet {i + 1}: has_bugs={bugs}")
    print(f"  {code[:60]}..." if len(code) > 60 else f"  {code}")

print(f"\nTotal iterations across all items: {len(collected.get('code', []))}")

# %% [markdown]
# ## Key Points
#
# | Feature | Benefit |
# |---------|---------|
# | True early exit | Items stop LLM calls when done |
# | Checkpoint collection | Track iteration counts without state bloat |
# | Composable | `batch`, `pullback`, `collect` all work together |
#
# Use `while_loop` for any iterate-until-done pattern: refinement, search, agents.
