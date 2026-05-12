# Optimize Prompts with {py:func}`pullback <autoform.pullback>` Feedback

Use {py:func}`pullback <autoform.pullback>` when feedback on an output should become
feedback for the inputs that produced it. For prompt work, the input is often an
instruction string.

```{admonition} Concept
[Trace, IR, Execute](../concepts/trace-ir-execute.md) · [Transforms](../concepts/transforms.md)
```

```python
import autoform as af


def answer_with_instruction(instruction: str, topic: str) -> str:
    prompt = af.format("{}\n\nTopic: {}", instruction, topic)
    msg = dict(role="user", content=prompt)
    # use any model configured for the active lm client
    return af.lm_call([msg], model="gpt-5.5")


# trace with placeholder values
ir = af.trace(answer_with_instruction)("Explain clearly.", "recursion")
pb_ir = af.pullback(ir)

instruction = "Explain the topic in one short paragraph."
topic = "recursion"
critique = "too abstract; include one concrete example"

for step in range(3):
    # pullback returns the output and feedback for each original input
    output, (instruction_hint, topic_hint) = pb_ir.call((instruction, topic), critique)
    print("step", step)
    print(output)
    print("instruction hint:", instruction_hint)
    print("topic hint:", topic_hint)
    instruction = instruction + "\nrevision guidance: " + instruction_hint
```

The original function still has one job: make an LM call from an instruction and
a topic. {py:func}`pullback <autoform.pullback>` makes a second IR whose input is the original inputs plus
feedback on the output.

This is not a special optimizer. It is a feedback channel. The update policy
above is deliberately small: append the returned instruction hint to the next
instruction. A production loop can score examples, keep the best instruction,
or batch many `(topic, critique)` pairs with {py:func}`batch <autoform.batch>` and {py:func}`pullback <autoform.pullback>`.

See [Transforms](../concepts/transforms.md) for the call shapes and
[Primitives](../concepts/primitives.md) for how feedback moves through each
recorded operation.
