import autoform as af

MODEL = "openai/gpt-4o"

# =============================================================================
# Example 1: Chain-of-Thought Gradients
# Shows how pullback provides feedback on each reasoning step
# =============================================================================


class Answer(af.Struct):
    reasoning: str
    answer: str


def chain_of_thought(question: str) -> Answer:
    # Step 1: Break down the problem
    step1_prompt = af.format("Break down this question into sub-problems:\n{}", question)
    msgs1 = [{"role": "user", "content": step1_prompt}]
    step1 = af.lm_call(msgs1, model=MODEL)
    step1 = af.mark(step1, collection="reasoning", name="breakdown")

    # Step 2: Solve each sub-problem
    step2_prompt = af.format("Given these sub-problems:\n{}\n\nSolve each one:", step1)
    msgs2 = [{"role": "user", "content": step2_prompt}]
    step2 = af.lm_call(msgs2, model=MODEL)
    step2 = af.mark(step2, collection="reasoning", name="solutions")

    # Step 3: Synthesize final answer
    step3_prompt = af.format(
        "Sub-problems:\n{}\n\nSolutions:\n{}\n\nProvide a final answer with reasoning:",
        step1,
        step2,
    )
    msgs3 = [{"role": "user", "content": step3_prompt}]
    return af.struct_lm_call(msgs3, model=MODEL, struct=Answer)


print("=" * 60)
print("EXAMPLE 1: Chain-of-Thought Gradients")
print("=" * 60)

# Trace
dummy = "..."
ir = af.build_ir(chain_of_thought)(dummy)
print("\n[1] IR created:")
print(ir)

# Run once with collect to see reasoning steps
print("\n[2] Running with collect to capture reasoning steps...")
result, captured = af.collect(ir, collection="reasoning")("What is 15% of 80?")
print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning[:100]}...")
print(f"Captured steps: {list(captured.keys())}")

# Pullback: get gradients for each input given feedback
print("\n[3] Running pullback to get improvement suggestions...")
pb_ir = af.pullback(ir)
critique = Answer(
    reasoning="The breakdown was good but solutions were too verbose", answer="correct"
)
output, grad = af.call(pb_ir)(("What is 15% of 80?", critique))
print(f"Output: {output.answer}")
print(f"Gradient (improvement hint): {grad[:200]}...")


# =============================================================================
# Example 2: Multi-Agent Debate with Collect/Inject
# Shows debugging a debate by capturing and replaying intermediate states
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 2: Multi-Agent Debate")
print("=" * 60)


class Position(af.Struct):
    argument: str
    confidence: str


def debate(topic: str) -> Position:
    # Agent 1: Proponent
    pro_prompt = af.format("Argue strongly FOR: {}", topic)
    msgs_pro = [{"role": "user", "content": pro_prompt}]
    pro = af.lm_call(msgs_pro, model=MODEL)
    pro = af.mark(pro, collection="debate", name="pro")

    # Agent 2: Opponent
    con_prompt = af.format("Argue strongly AGAINST: {}", topic)
    msgs_con = [{"role": "user", "content": con_prompt}]
    con = af.lm_call(msgs_con, model=MODEL)
    con = af.mark(con, collection="debate", name="con")

    # Agent 3: Synthesizer
    synth_prompt = af.format("PRO:\n{}\n\nCON:\n{}\n\nSynthesize a balanced position:", pro, con)
    msgs_synth = [{"role": "user", "content": synth_prompt}]
    return af.struct_lm_call(msgs_synth, model=MODEL, struct=Position)


# Trace
ir = af.build_ir(debate)(dummy)
print("\n[1] IR created")

# Run with collect to capture debate
print("\n[2] Running with collect...")
result, captured = af.collect(ir, collection="debate")("AI will replace programmers")
print(f"Final position: {result.argument[:100]}...")
print(f"Confidence: {result.confidence}")
print(f"Captured: {list(captured.keys())}")

# Inject: replay with different arguments
print("\n[3] Inject: replay with modified pro argument...")
modified = {
    "pro": ["AI is a tool that will augment programmers, not replace them."],
    "con": [captured["con"][0]],  # keep original con, wrapped in list
}
result2 = af.inject(ir, collection="debate", values=modified)("AI will replace programmers")
print(f"New position: {result2.argument[:100]}...")


# =============================================================================
# Example 3: Batched Pullback for Prompt Optimization
# Shows using batched gradients to optimize prompts
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 3: Batched Gradients for Prompt Analysis")
print("=" * 60)


class Summary(af.Struct):
    summary: str


def summarize(text: str) -> Summary:
    prompt = af.format("Summarize this in one sentence:\n{}", text)
    msgs = [{"role": "user", "content": prompt}]
    return af.struct_lm_call(msgs, model=MODEL, struct=Summary)


# Trace
ir = af.build_ir(summarize)(dummy)

# Batch + Pullback: get improvement hints for multiple inputs
print("\n[1] Creating batched pullback IR...")
pb_ir = af.pullback(ir)
batched_pb = af.batch(pb_ir, in_axes=(list, Summary.model_construct(summary=list)))

# Test cases with different critiques
inputs = [
    "Machine learning uses data to train models that make predictions.",
    "The internet connects billions of devices using TCP/IP protocols.",
    "Climate change affects weather patterns globally.",
]
critiques = Summary.model_construct(summary=["too technical", "perfect", "too vague"])

print("\n[2] Running batched pullback...")
outputs, grads = af.call(batched_pb)((inputs, critiques))

print("\n[3] Results:")
for i, (inp, out, grad) in enumerate(zip(inputs, outputs.summary, grads)):
    print(f"\nInput {i + 1}: {inp[:50]}...")
    print(f"Output: {out}")
    print(f"Gradient: {grad[:100]}...")


print("\n" + "=" * 60)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 60)
