"""Multi-step LLM pipeline with batching and composition.

This demonstrates:
1. A pipeline with multiple lm_calls (research -> write)
2. batch_ir: parallelize with batch_completion, not a loop
3. batch_ir(pullback_ir(ir)): batched semantic backpropagation
"""

import autoform as af


def research_and_write(topic: str) -> str:
    """Two-step pipeline: research facts, then write an article."""
    # step 1: research
    prompt1 = af.format("List 3 key facts about: {}", topic)
    notes = af.lm_call([dict(role="user", content=prompt1)], model="gpt-4.1")
    # step 2: write using the research
    prompt2 = af.format("Write a paragraph using: {}", notes)
    article = af.lm_call([dict(role="user", content=prompt2)], model="gpt-4.1")
    return article


if __name__ == "__main__":
    ir = af.build_ir(research_and_write)("example topic")
    batch_ir = af.batch_ir(ir, in_axes=list)
    pb_ir = af.pullback_ir(ir)
    batch_pb_ir = af.batch_ir(pb_ir, in_axes=(list, list))
    topics = ["AI safety", "quantum computing", "climate change"]
    articles = af.run_ir(batch_ir, topics)
    topics = ["AI safety", "quantum computing", "climate change"]
    feedbacks = ["too technical", "good overview", "needs more data"]
    outputs, input_grads = af.run_ir(batch_pb_ir, (topics, feedbacks))
