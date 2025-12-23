import functools as ft
import autoform.core as core


shout_p = core.Primitive("shout")


def shout(text: str) -> str:
    return core.bind(shout_p, text)


@ft.partial(core.impl_rules.set, shout_p)
def impl_shout(text: str) -> str:
    return text.upper()


@ft.partial(core.eval_rules.set, shout_p)
def eval_shout(text) -> core.Var:
    return core.Var()


@ft.partial(core.push_rules.set, shout_p)
def push_shout(primal: str, tangent: str) -> tuple[str, str]:
    return primal.upper(), tangent.upper()


@ft.partial(core.pull_fwd_rules.set, shout_p)
def pull_fwd_shout(text: str) -> tuple[str, str]:
    return text.upper(), text


@ft.partial(core.pull_bwd_rules.set, shout_p)
def pull_bwd_shout(residual: str, cotangent: str) -> str:
    return cotangent


@ft.partial(core.batch_rules.set, shout_p)
def batch_shout(batch_size: int, in_batched: bool, text) -> tuple[list[str], bool]:
    if in_batched:
        return [t.upper() for t in text], True
    else:
        return text.upper(), False


class TestCustomPrimitive:
    def test_basic_shout(self):
        result = shout("hello")
        assert result == "HELLO"

    def test_shout_ir_build_and_run(self):
        def program(x):
            return shout(x)

        ir = core.build_ir(program, "test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "shout"

        result = core.run_ir(ir, "world")
        assert result == "WORLD"

    def test_shout_pushforward(self):
        def program(x):
            return shout(x)

        ir = core.build_ir(program, "test")
        pf_ir = core.pushforward_ir(ir)
        primal_out, tangent_out = core.run_ir(pf_ir, ("hello", "bye"))
        assert primal_out == "HELLO"
        assert tangent_out == "BYE"

    def test_shout_pullback(self):
        def program(x):
            return shout(x)

        ir = core.build_ir(program, "test")
        pb_ir = core.pullback_ir(ir)
        output, grad_input = core.run_ir(pb_ir, ("hello", "feedback"))
        assert output == "HELLO"
        assert grad_input == "feedback"

    def test_shout_batch(self):
        def program(x):
            return shout(x)

        ir = core.build_ir(program, "test")
        batch_ir = core.batch_ir(ir, in_axes=list)
        result = core.run_ir(batch_ir, ["hello", "world", "test"])
        assert result == ["HELLO", "WORLD", "TEST"]

    def test_shout_chained(self):
        def program(x):
            step1 = core.format("Say: {}", x)
            step2 = shout(step1)
            return step2

        ir = core.build_ir(program, "hello")
        result = core.run_ir(ir, "world")
        assert result == "SAY: WORLD"


class ResearchNotes(core.Struct):
    topic: str
    key_points: str
    sources: str


class Article(core.Struct):
    title: str
    body: str
    summary: str


textgrad_style_lm_call_p = core.Primitive("textgrad_style_lm_call")


def textgrad_style_lm_call(
    messages: list[dict[str, str]],
    *,
    model: str,
    struct: type[core.Struct],
) -> core.Struct:
    roles = tuple(m["role"] for m in messages)
    contents = tuple(m["content"] for m in messages)
    return core.bind(textgrad_style_lm_call_p, contents, roles=roles, model=model, struct=struct)


@ft.partial(core.impl_rules.set, textgrad_style_lm_call_p)
def impl_textgrad_style_lm_call(
    contents: tuple, *, roles: tuple, model: str, struct: type[core.Struct]
):
    return core.impl_rules.get(core.struct_lm_call_p)(
        contents, roles=roles, model=model, struct=struct
    )


@ft.partial(core.eval_rules.set, textgrad_style_lm_call_p)
def eval_textgrad_style_lm_call(in_tree, *, struct: type[core.Struct], **params):
    return struct.model_construct(**{k: core.Var() for k in struct.model_fields})


@ft.partial(core.pull_fwd_rules.set, textgrad_style_lm_call_p)
def pull_fwd_textgrad_style_lm_call(
    contents: tuple, *, roles: tuple, model: str, struct: type[core.Struct]
):
    out = core.impl_rules.get(core.struct_lm_call_p)(
        contents, roles=roles, model=model, struct=struct
    )
    residuals = (contents, roles, out)
    return out, residuals


@ft.partial(core.pull_bwd_rules.set, textgrad_style_lm_call_p)
def pull_bwd_textgrad_style_lm_call(
    residuals: tuple,
    cotangent_out,
    *,
    roles: tuple,
    model: str,
    struct: type[core.Struct],
):
    contents, original_roles, output = residuals

    if hasattr(cotangent_out, "model_dump"):
        cotangent_str = str(cotangent_out.model_dump())
    else:
        cotangent_str = str(cotangent_out)

    if hasattr(output, "model_dump"):
        output_str = str(output.model_dump())
    else:
        output_str = str(output)

    input_gradients = []
    for i, content in enumerate(contents):
        critique_prompt = f"""You are providing feedback to improve an LLM prompt's input.

ORIGINAL INPUT (role: {original_roles[i]}):
{content}

LLM OUTPUT:
{output_str}

DOWNSTREAM FEEDBACK:
{cotangent_str}

Based on the downstream feedback, provide specific, actionable suggestions 
to improve this input. Be concise and targeted."""

        try:
            import litellm

            resp = litellm.completion(
                messages=[{"role": "user", "content": critique_prompt}],
                model=model,
            )
            input_gradients.append(resp.choices[0].message.content)
        except Exception:
            input_gradients.append(f"[Feedback propagated: {cotangent_str}]")

    return tuple(input_gradients)


@ft.partial(core.push_rules.set, textgrad_style_lm_call_p)
def push_textgrad_style_lm_call(
    primals: tuple, tangents: tuple, *, roles: tuple, model: str, struct: type[core.Struct]
):
    """Pushforward rule: propagate tangents through the LLM call.

    For LLM calls, we run the model on both primals and tangents independently.
    This is useful for sensitivity analysis — seeing how changes to inputs
    affect the outputs.
    """
    p_out = core.impl_rules.get(core.struct_lm_call_p)(
        primals, roles=roles, model=model, struct=struct
    )
    t_out = core.impl_rules.get(core.struct_lm_call_p)(
        tangents, roles=roles, model=model, struct=struct
    )
    return p_out, t_out


@ft.partial(core.batch_rules.set, textgrad_style_lm_call_p)
def batch_textgrad_style_lm_call(
    batch_size: int,
    in_batched: tuple[bool, ...],
    contents: tuple,
    *,
    roles: tuple,
    model: str,
    struct: type[core.Struct],
):
    batched_messages = []
    for b in range(batch_size):
        msgs = [
            {"role": r, "content": contents[i][b] if in_batched[i] else contents[i]}
            for i, r in enumerate(roles)
        ]
        batched_messages.append(msgs)

    try:
        import litellm

        responses = litellm.batch_completion(
            messages=batched_messages,
            model=model,
            response_format=struct,
        )
        results = [
            struct.model_validate_json(resp.choices[0].message.content) for resp in responses
        ]
    except Exception as e:
        results = [
            struct.model_construct(**{k: f"[Error: {e}]" for k in struct.model_fields})
            for _ in range(batch_size)
        ]

    return results, True


class TestTextGradStylePullback:
    def test_textgrad_style_ir_build(self):
        def program(topic: str):
            messages = [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": topic},
            ]
            return textgrad_style_lm_call(messages, model="openai/gpt-4o", struct=ResearchNotes)

        ir = core.build_ir(program, "test topic")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "textgrad_style_lm_call"

    def test_multi_agent_ir_build(self):
        def multi_agent_pipeline(topic: str):
            research_messages = [
                {
                    "role": "system",
                    "content": "You are a research assistant. Gather key information.",
                },
                {"role": "user", "content": core.format("Research this topic: {}", topic)},
            ]
            notes = textgrad_style_lm_call(
                research_messages, model="openai/gpt-4o", struct=ResearchNotes
            )

            writer_messages = [
                {
                    "role": "system",
                    "content": "You are a writer. Create an article from research notes.",
                },
                {"role": "user", "content": core.format("Notes: {}", notes.key_points)},
            ]
            article = textgrad_style_lm_call(writer_messages, model="openai/gpt-4o", struct=Article)

            return article

        ir = core.build_ir(multi_agent_pipeline, "AI safety")

        prim_names = [eqn.prim.name for eqn in ir.ireqns]
        assert prim_names.count("textgrad_style_lm_call") == 2
        assert prim_names.count("format") == 2

    def test_pullback_ir_construction(self):
        def pipeline(topic: str):
            messages = [
                {"role": "user", "content": core.format("Research: {}", topic)},
            ]
            return textgrad_style_lm_call(messages, model="openai/gpt-4o", struct=ResearchNotes)

        ir = core.build_ir(pipeline, "test")
        pb_ir = core.pullback_ir(ir)

        assert len(pb_ir.ireqns) == 1
        assert pb_ir.ireqns[0].prim.name == "pullback_call"


class TestMultiAgentComposition:
    def test_batch_multi_agent(self):
        def simple_agent(query: str):
            return core.format("[Processed: {}]", query)

        ir = core.build_ir(simple_agent, "test")
        batch_ir = core.batch_ir(ir, in_axes=list)

        results = core.run_ir(batch_ir, ["query1", "query2", "query3"])
        assert results == ["[Processed: query1]", "[Processed: query2]", "[Processed: query3]"]

    def test_pushforward_chained_agents(self):
        def chained_agents(x: str):
            step1 = core.format("[Agent1: {}]", x)
            step2 = core.format("[Agent2: {}]", step1)
            return step2

        ir = core.build_ir(chained_agents, "test")
        pf_ir = core.pushforward_ir(ir)

        primal_out, tangent_out = core.run_ir(pf_ir, ("input", "delta"))
        assert primal_out == "[Agent2: [Agent1: input]]"
        assert tangent_out == "[Agent2: [Agent1: delta]]"

    def test_pullback_chained_agents(self):
        def chained_agents(x: str):
            step1 = core.format("[Agent1: {}]", x)
            step2 = core.format("[Agent2: {}]", step1)
            return step2

        ir = core.build_ir(chained_agents, "test")
        pb_ir = core.pullback_ir(ir)

        output, input_grad = core.run_ir(pb_ir, ("input", "feedback"))
        assert output == "[Agent2: [Agent1: input]]"

        assert input_grad == "feedback"

    def test_batch_ir_construction_textgrad_style(self):
        """Test that batch_ir can be constructed for TextGrad primitive.

        This demonstrates the batch transform:
        1. Write a program that processes ONE input
        2. Call batch_ir() to transform it
        3. Now it can process MANY inputs at once!
        """

        def single_agent(topic: str):
            messages = [
                {"role": "user", "content": core.format("Analyze: {}", topic)},
            ]
            return textgrad_style_lm_call(messages, model="openai/gpt-4o", struct=ResearchNotes)

        ir = core.build_ir(single_agent, "test")

        batch_transformed = core.batch_ir(ir, in_axes=list)

        assert len(batch_transformed.ireqns) == 1
        assert batch_transformed.ireqns[0].prim.name == "batch_call"

    def test_nested_batch_and_pullback(self):
        """Test composing batch and pullback transforms.

        Shows how transforms compose: batch_ir(pullback_ir(ir))
        """

        def agent(x: str):
            return core.format("[Processed: {}]", x)

        ir = core.build_ir(agent, "test")

        pb_ir = core.pullback_ir(ir)
        batch_pb = core.batch_ir(pb_ir, in_axes=(list, list))

        inputs = ["a", "b", "c"]
        cotangents = ["g1", "g2", "g3"]
        outputs, input_grads = core.run_ir(batch_pb, inputs, cotangents)

        assert outputs == ["[Processed: a]", "[Processed: b]", "[Processed: c]"]
        assert input_grads == ["g1", "g2", "g3"]
