"""TextGrad-style semantic backpropagation.

Shows how to define a custom primitive with an LLM-based backward pass.
When you call pullback on this primitive, instead of numeric gradients,
you get LLM-generated feedback on how to improve your inputs.
"""

import functools as ft
import autoform as af


# ==================================================================================================
# step 1: define output structures
# ==================================================================================================
# these are pydantic models that describe what the LLM should return.
# af.Struct is just pydantic.BaseModel with tracing support.


class ResearchNotes(af.Struct):
    topic: str
    key_points: str
    sources: str


class Article(af.Struct):
    title: str
    body: str
    summary: str


# ==================================================================================================
# step 2: create a custom primitive
# ==================================================================================================
# a primitive is the atomic unit of computation that autoform knows how to
# transform (batch, pullback, pushforward). you define rules for each transform.

textgrad_lm_call_p = af.Primitive("textgrad_lm_call")


def textgrad_lm_call(messages: list, *, model: str, struct: type) -> af.Struct:
    """LLM call that supports semantic backpropagation."""
    roles = tuple(m["role"] for m in messages)
    contents = tuple(m["content"] for m in messages)
    return textgrad_lm_call_p.bind(contents, roles=roles, model=model, struct=struct)


# ==================================================================================================
# step 3: define the rules
# ==================================================================================================


# impl_rule: what happens when you actually run this primitive
@ft.partial(af.impl_rules.def_rule, textgrad_lm_call_p)
def impl_textgrad(contents, *, roles, model, struct):
    return af.impl_rules[af.struct_lm_call_p](contents, roles=roles, model=model, struct=struct)


# eval_rule: tells the tracer what shape/type the output has (for build_ir)
@ft.partial(af.eval_rules.def_rule, textgrad_lm_call_p)
def eval_textgrad(in_tree, *, struct, **params):
    return struct.model_construct(**{k: af.Var() for k in struct.model_fields})


# pull_fwd_rule: forward pass of pullback - run and save what we need for backward
@ft.partial(af.pull_fwd_rules.def_rule, textgrad_lm_call_p)
def pull_fwd_textgrad(contents, *, roles, model, struct):
    out = impl_textgrad(contents, roles=roles, model=model, struct=struct)
    residuals = (contents, roles, out)  # save for backward
    return out, residuals


# pull_bwd_rule: backward pass - given feedback on output, generate feedback on inputs
# this is the "textgrad" part: we use an LLM to generate semantic gradients
@ft.partial(af.pull_bwd_rules.def_rule, textgrad_lm_call_p)
def pull_bwd_textgrad(residuals, cotangent_out, *, roles, model, struct):
    import litellm

    contents, original_roles, output = residuals

    input_gradients = []
    for content in contents:
        critique_prompt = f"""You will evaluate an input variable and provide feedback for optimization.

**Input Variable**: {content}

**Output produced**: {output}

**Loss/Feedback on Output**: {cotangent_out}

Provide specific, actionable feedback on how to improve the Input Variable to address the Loss/Feedback. Focus on what changes would lead to a better output."""
        resp = litellm.completion(
            messages=[dict(role="user", content=critique_prompt)], model=model
        )
        input_gradients.append(resp.choices[0].message.content)

    return tuple(input_gradients)


# batch_rule: how to process multiple inputs at once (uses batch_completion)
@ft.partial(af.batch_rules.def_rule, textgrad_lm_call_p)
def batch_textgrad(batch_size, in_batched, contents, *, roles, model, struct):
    import litellm

    try:
        batched_msgs = []
        for b in range(batch_size):
            msgs = [
                dict(role=r, content=contents[i][b] if in_batched[i] else contents[i])
                for i, r in enumerate(roles)
            ]
            batched_msgs.append(msgs)
        responses = litellm.batch_completion(
            messages=batched_msgs, model=model, response_format=struct
        )
        results = [
            struct.model_validate_json(resp.choices[0].message.content) for resp in responses
        ]
        return results, True
    except Exception as e:
        error_struct = struct.model_construct(**{k: f"[Error: {e}]" for k in struct.model_fields})
        return [error_struct for _ in range(batch_size)], True


# ==================================================================================================
# step 4: write your pipeline as a normal python function
# ==================================================================================================


def multi_agent_pipeline(topic: str) -> Article:
    """Two agents: researcher gathers notes, writer creates article."""
    # agent 1: researcher
    research_prompt = af.format("Research: {}", topic)
    research_msgs = [
        dict(role="system", content="You are a research assistant."),
        dict(role="user", content=research_prompt),
    ]
    notes = textgrad_lm_call(research_msgs, model="gpt-4.1", struct=ResearchNotes)

    # agent 2: writer (uses output from agent 1)
    writer_prompt = af.format("Notes: {}", notes.key_points)
    writer_msgs = [
        dict(role="system", content="Write an article from these notes."),
        dict(role="user", content=writer_prompt),
    ]
    article = textgrad_lm_call(writer_msgs, model="gpt-4.1", struct=Article)

    return article


# ==================================================================================================
# step 5: trace, transform, run
# ==================================================================================================

if __name__ == "__main__":
    # trace the pipeline into an IR
    ir = af.build_ir(multi_agent_pipeline)("example topic")

    # transform: pullback (semantic backprop)
    pb_ir = af.pullback_ir(ir)

    # transform: batch the pullback
    batch_pb_ir = af.batch_ir(pb_ir, in_axes=(list, list))

    # run: given a topic and feedback on the output, get feedback on the input
    feedback = Article(title="Good title", body="Needs more technical depth.", summary="Concise.")
    output, input_grad = af.run_ir(pb_ir, ("AI safety", feedback))
    print("output:", output)
    print("input gradient:", input_grad)
