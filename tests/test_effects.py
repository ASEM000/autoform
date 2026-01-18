import pytest

import autoform as af
from autoform.core import Effect, EffectInterpreter, using_effect, using_interpreter
from autoform.harvest import (
    checkpoint,
)


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

            def __call__(self, effect, in_tree):
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

        def aggregating_handler(effect, in_tree):
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

        def single_shot_handler(effect, in_tree):
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

        def skip_handler(effect, value):
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
