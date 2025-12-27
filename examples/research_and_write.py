"""Multi-step LLM pipeline with batching and composition.

This demonstrates:
1. A pipeline with multiple lm_calls (research -> write)
2. batch: parallelize with batch_completion, not a loop
3. batch(pullback(ir)): batched semantic backpropagation
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
    batched = af.batch(ir, in_axes=list)
    pb_ir = af.pullback(ir)
    batch_pb = af.batch(pb_ir, in_axes=(list, list))
    topics = ["AI safety", "quantum computing", "climate change"]
    articles = af.call(batched)(topics)
    feedbacks = ["too technical", "good overview", "needs more data"]
    outputs, input_grads = af.call(batch_pb)((topics, feedbacks))
