import pytest

import autoform as af
from autoform.core import Effect, EffectInterpreter, using_effect, using_interpreter
from autoform.intercept import checkpoint


class TestEffectBasics:
    def test_checkpoint_via_effects(self):
        result = checkpoint("hello", key="test", collection="debug")
        assert result == "hello"

    def test_effect_p_in_ir(self):
        def func(x):
            return checkpoint(x, key="my_key", collection="my_col")

        ir = af.trace(func)("test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "effect"
        assert ir.ireqns[0].effect.key == "my_key"
        assert ir.ireqns[0].effect.collection == "my_col"


class TestCollect:
    def test_collect_basic(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = af.call(ir)("hello")

        assert result == "hello"
        assert collected == {"val": ["hello"]}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_collect_basic_async(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = await af.acall(ir)("hello")

        assert result == "hello"
        assert collected == {"val": ["hello"]}

    def test_collect_filters_by_collection(self):
        def func(x):
            a = checkpoint(x, key="debug_val", collection="debug")
            b = checkpoint(a, key="other_val", collection="other")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = af.call(ir)("hello")

        assert result == "hello"
        assert collected == {"debug_val": ["hello"]}
        assert "other_val" not in collected

    def test_collect_all_when_no_collection_filter(self):
        def func(x):
            a = checkpoint(x, key="a", collection="one")
            b = checkpoint(a, key="b", collection="two")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection=...) as collected:
            af.call(ir)("hello")

        assert collected == {"a": ["hello"], "b": ["hello"]}


class TestInject:
    def test_inject_replaces_value(self):
        def func(x):
            return checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.trace(func)("test")

        with af.collect(collection="cache") as collected:
            normal = af.call(ir)("World")
        assert normal == "Hello, World"

        with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
            injected = af.call(ir)("World")
        assert injected == "CACHED"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_inject_replaces_value_async(self):
        def func(x):
            return checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
            injected = await af.acall(ir)("World")
        assert injected == "CACHED"

    def test_inject_partial(self):
        def func(x):
            a = checkpoint(x, key="first", collection="cache")
            b = checkpoint(af.concat(a, "!"), key="second", collection="cache")
            return b

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"first": ["INJECTED"]}):
            result = af.call(ir)("ignored")

        assert result == "INJECTED!"


class TestEffectsWithTransforms:
    def test_effects_through_batch(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        with af.collect(collection="debug") as collected:
            result = af.call(batched)(["a", "b", "c"])

        assert result == ["a", "b", "c"]
        assert collected == {"val": [["a", "b", "c"]]}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_effects_through_batch_async(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        with af.collect(collection="debug") as collected:
            result = await af.acall(batched)(["a", "b", "c"])

        assert result == ["a", "b", "c"]
        assert collected == {"val": [["a", "b", "c"]]}

    def test_effects_through_pushforward(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pf_ir = af.pushforward(ir)

        with af.collect(collection="debug") as collected:
            primal, tangent = af.call(pf_ir)(("primal", "tangent"))

        assert primal == "primal"
        assert tangent == "tangent"
        assert collected == {"val": ["primal", "tangent"]}

    def test_effects_through_pullback(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pb_ir = af.pullback(ir)

        with af.collect(collection="debug") as collected:
            primal, cotangent = af.call(pb_ir)(("primal", "cotangent"))

        assert primal == "primal"
        assert cotangent == "cotangent"
        assert collected == {"val": ["primal", "cotangent"]}


class TestHandlerComposition:
    def test_nested_handlers(self):
        def func(x):
            a = checkpoint(x, key="debug", collection="debug")
            b = checkpoint(a, key="cache", collection="cache")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as debug_collected:
            with af.collect(collection="cache") as cache_collected:
                result = af.call(ir)("hello")

        assert result == "hello"
        assert debug_collected == {"debug": ["hello"]}
        assert cache_collected == {"cache": ["hello"]}


class TestMultiShotContinuation:
    def test_multi_shot_collects_all(self):
        class MultiEffect(Effect):
            pass

        class MultiShotHandler:
            def __init__(self, alternatives: list):
                self.alternatives = alternatives
                self.results = []

            def __call__(self, prim, effect, in_tree, /):
                for v in self.alternatives:
                    result = yield (v, in_tree[1])
                    self.results.append(result)
                return self.results[-1]
                yield

        def program(x):
            with using_effect(MultiEffect()):
                return af.concat(x, "!")
            return x

        ir = af.trace(program)("test")

        handler = MultiShotHandler(alternatives=["A", "B", "C"])
        with using_interpreter(EffectInterpreter((MultiEffect, handler))):
            result = af.call(ir)("ignored")

        assert handler.results == ["A!", "B!", "C!"]
        assert result == "C!"

    def test_multi_shot_aggregation(self):
        class AggregateEffect(Effect):
            pass

        def aggregating_handler(prim, effect, in_tree, /):
            results = []
            for suffix in ["!", "?", "."]:
                left, _ = in_tree
                result = yield (left + suffix, "")
                results.append(result)
            return " | ".join(results)
            yield

        def program(x):
            with using_effect(AggregateEffect()):
                return af.concat(x, " appended")
            return x

        ir = af.trace(program)("test")

        with using_interpreter(EffectInterpreter((AggregateEffect, aggregating_handler))):
            result = af.call(ir)("Hello")

        assert result == "Hello! | Hello? | Hello."

    def test_single_shot_still_works(self):
        class SingleEffect(Effect):
            pass

        def single_shot_handler(prim, effect, in_tree, /):
            left, right = in_tree
            result = yield (left.upper(), right)
            return result + " modified"
            yield

        def program(x):
            with using_effect(SingleEffect()):
                return af.concat(x, " world")
            return x

        ir = af.trace(program)("test")

        with using_interpreter(EffectInterpreter((SingleEffect, single_shot_handler))):
            result = af.call(ir)("hello")

        assert result == "HELLO world modified"

    def test_skip_still_works(self):
        class SkipEffect(Effect):
            pass

        def skip_handler(prim, effect, value, /):
            return "SKIPPED"
            yield

        def program(x):
            with using_effect(SkipEffect()):
                return af.concat(x, " should not appear")
            return x

        ir = af.trace(program)("test")

        with using_interpreter(EffectInterpreter((SkipEffect, skip_handler))):
            result = af.call(ir)("ignored")

        assert result == "SKIPPED"

    def test_handler_receives_prim_and_params(self):
        class InspectEffect(Effect):
            pass

        captured = {}

        def inspect_handler(prim, effect, in_tree, /, **params):
            captured["prim_name"] = prim.name
            captured["params"] = params
            return (yield in_tree)

        def program(x):
            with using_effect(InspectEffect()):
                return af.format("hello {}", x)
            return x

        ir = af.trace(program)("test")

        with using_interpreter(EffectInterpreter((InspectEffect, inspect_handler))):
            result = af.call(ir)("world")

        assert result == "hello world"
        assert captured["prim_name"] == "format"
        assert "template" in captured["params"]
        assert captured["params"]["template"] == "hello {}"


class TestDefaultHandler:
    def test_passthrough(self):
        def handler(prim, effect, in_tree, /, **params):
            return (yield in_tree)

        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("hello")

        assert result == "hello!"

    def test_skip(self):
        def handler(prim, effect, in_tree, /, **params):
            return "SKIPPED"
            yield

        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("hello")

        assert result == "SKIPPED"

    def test_post_process(self):
        def handler(prim, effect, in_tree, /, **params):
            result = yield in_tree
            return result + " (intercepted)"

        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("hello")

        assert result == "hello! (intercepted)"

    def test_pre_process(self):
        def handler(prim, effect, in_tree, /, **params):
            left, right = in_tree
            return (yield (left.upper(), right))

        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("hello")

        assert result == "HELLO!"

    def test_effect_is_none_for_plain_primitives(self):
        captured_effects = []

        def handler(prim, effect, in_tree, /, **params):
            captured_effects.append(effect)
            return (yield in_tree)

        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            af.call(ir)("hello")

        assert captured_effects == [None]

    def test_receives_prim_and_params(self):
        captured = {}

        def handler(prim, effect, in_tree, /, **params):
            captured["prim_name"] = prim.name
            captured["params"] = params
            return (yield in_tree)

        def func(x):
            return af.format("hello {}", x)

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("world")

        assert result == "hello world"
        assert captured["prim_name"] == "format"
        assert captured["params"]["template"] == "hello {}"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_passthrough_async(self):
        def handler(prim, effect, in_tree, /, **params):
            return (yield in_tree)

        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = await af.acall(ir)("hello")

        assert result == "hello!"


class TestDefaultHandlerInterceptsAll:
    def test_intercepts_every_primitive(self):
        seen_prims = []

        def handler(prim, effect, in_tree, /, **params):
            seen_prims.append(prim.name)
            return (yield in_tree)

        def func(x):
            a = af.format("hello {}", x)
            return af.concat(a, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("world")

        assert result == "hello world!"
        assert seen_prims == ["format", "concat"]

    def test_intercepts_effect_primitives_too(self):
        seen_prims = []

        def handler(prim, effect, in_tree, /, **params):
            seen_prims.append(prim.name)
            return (yield in_tree)

        def func(x):
            a = af.concat(x, "!")
            return checkpoint(a, key="val", collection="debug")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=handler)):
            result = af.call(ir)("hello")

        assert result == "hello!"
        assert "concat" in seen_prims
        assert "effect" in seen_prims


class TestDefaultWithSpecificHandlers:
    def test_specific_handler_takes_priority(self):
        class MyEffect(Effect):
            pass

        default_calls = []
        specific_calls = []

        def default(prim, effect, in_tree, /, **params):
            default_calls.append(prim.name)
            return (yield in_tree)

        def specific(prim, effect, in_tree, /):
            specific_calls.append(prim.name)
            return (yield in_tree)

        def func(x):
            a = af.concat(x, "!")
            with using_effect(MyEffect()):
                b = af.concat(a, "?")
            return b

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter((MyEffect, specific), default=default)):
            result = af.call(ir)("hello")

        assert result == "hello!?"
        assert "concat" in default_calls
        assert "concat" in specific_calls
        assert len(default_calls) == 1
        assert len(specific_calls) == 1

    def test_default_handles_unmatched_effects(self):
        class UnhandledEffect(Effect):
            pass

        default_calls = []

        def default(prim, effect, in_tree, /, **params):
            default_calls.append((prim.name, type(effect).__name__))
            return (yield in_tree)

        def func(x):
            with using_effect(UnhandledEffect()):
                return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=default)):
            result = af.call(ir)("hello")

        assert result == "hello!"
        assert default_calls == [("concat", "UnhandledEffect")]

    def test_composes_with_collect(self):
        seen_prims = []

        def default(prim, effect, in_tree, /, **params):
            seen_prims.append(prim.name)
            return (yield in_tree)

        def func(x):
            a = af.concat(x, "!")
            return checkpoint(a, key="val", collection="debug")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=default)):
            with af.collect(collection="debug") as collected:
                result = af.call(ir)("hello")

        assert result == "hello!"
        assert collected == {"val": ["hello!"]}
        assert "concat" in seen_prims


class TestDefaultHandlerBackwardCompat:
    def test_no_default_falls_through(self):
        def func(x):
            return af.concat(x, "!")

        ir = af.trace(func)("test")

        with using_interpreter(EffectInterpreter(default=None)):
            result = af.call(ir)("hello")

        assert result == "hello!"

    def test_existing_code_without_default(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = af.call(ir)("hello")

        assert result == "hello"
        assert collected == {"val": ["hello"]}
