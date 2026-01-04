import autoform as af
from autoform.core import Effect, using_effect
from autoform.effects import using_effect_handler
from autoform.harvest import (
    CheckpointEffect,
    CollectHandler,
    InjectHandler,
    checkpoint,
)


class TestEffectBasics:
    def test_checkpoint_via_effects(self):
        result = checkpoint("hello", key="test", collection="debug")
        assert result == "hello"

    def test_effect_p_in_ir(self):
        def func(x):
            return checkpoint(x, key="my_key", collection="my_col")

        ir = af.build_ir(func)("test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "effect"
        assert ir.ireqns[0].effect.key == "my_key"
        assert ir.ireqns[0].effect.collection == "my_col"


class TestCollectHandler:
    def test_collect_basic(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")

        handler = CollectHandler(collection="debug")
        with using_effect_handler({CheckpointEffect: handler}):
            result = af.call(ir)("hello")

        assert result == "hello"
        assert handler.collected == {"val": ["hello"]}

    def test_collect_filters_by_collection(self):
        def func(x):
            a = checkpoint(x, key="debug_val", collection="debug")
            b = checkpoint(a, key="other_val", collection="other")
            return b

        ir = af.build_ir(func)("test")

        handler = CollectHandler(collection="debug")
        with using_effect_handler({CheckpointEffect: handler}):
            result = af.call(ir)("hello")

        assert result == "hello"
        assert handler.collected == {"debug_val": ["hello"]}
        assert "other_val" not in handler.collected

    def test_collect_all_when_no_collection_filter(self):
        def func(x):
            a = checkpoint(x, key="a", collection="one")
            b = checkpoint(a, key="b", collection="two")
            return b

        ir = af.build_ir(func)("test")

        handler = CollectHandler(collection=...)
        with using_effect_handler({CheckpointEffect: handler}):
            af.call(ir)("hello")

        assert handler.collected == {"a": ["hello"], "b": ["hello"]}


class TestInjectHandler:
    def test_inject_replaces_value(self):
        def func(x):
            return checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.build_ir(func)("test")

        handler = CollectHandler(collection="cache")
        with using_effect_handler({CheckpointEffect: handler}):
            normal = af.call(ir)("World")
        assert normal == "Hello, World"

        handler = InjectHandler(collection="cache", values={"greeting": ["CACHED"]})
        with using_effect_handler({CheckpointEffect: handler}):
            injected = af.call(ir)("World")
        assert injected == "CACHED"

    def test_inject_partial(self):
        def func(x):
            a = checkpoint(x, key="first", collection="cache")
            b = checkpoint(af.concat(a, "!"), key="second", collection="cache")
            return b

        ir = af.build_ir(func)("test")

        handler = InjectHandler(collection="cache", values={"first": ["INJECTED"]})
        with using_effect_handler({CheckpointEffect: handler}):
            result = af.call(ir)("ignored")

        assert result == "INJECTED!"


class TestEffectsWithTransforms:
    def test_effects_through_batch(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")
        batched = af.batch(ir)

        handler = CollectHandler(collection="debug")
        with using_effect_handler({CheckpointEffect: handler}):
            result = af.call(batched)(["a", "b", "c"])

        assert result == ["a", "b", "c"]
        assert handler.collected == {"val": [["a", "b", "c"]]}

    def test_effects_through_pushforward(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")
        pf_ir = af.pushforward(ir)

        handler = CollectHandler(collection="debug")
        with using_effect_handler({CheckpointEffect: handler}):
            primal, tangent = af.call(pf_ir)(("primal", "tangent"))

        assert primal == "primal"
        assert tangent == "tangent"
        assert handler.collected == {"val": ["primal", "tangent"]}

    def test_effects_through_pullback(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")
        pb_ir = af.pullback(ir)

        handler = CollectHandler(collection="debug")
        with using_effect_handler({CheckpointEffect: handler}):
            primal, cotangent = af.call(pb_ir)(("primal", "cotangent"))

        assert primal == "primal"
        assert cotangent == "cotangent"
        assert handler.collected == {"val": ["primal", "cotangent"]}


class TestHandlerComposition:
    def test_nested_handlers(self):
        def func(x):
            a = checkpoint(x, key="debug", collection="debug")
            b = checkpoint(a, key="cache", collection="cache")
            return b

        ir = af.build_ir(func)("test")

        debug_handler = CollectHandler(collection="debug")
        with using_effect_handler({CheckpointEffect: debug_handler}):
            cache_handler = CollectHandler(collection="cache")
            with using_effect_handler({CheckpointEffect: cache_handler}):
                result = af.call(ir)("hello")

        assert result == "hello"
        assert debug_handler.collected == {"debug": ["hello"]}
        assert cache_handler.collected == {"cache": ["hello"]}


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
            with using_effect(MultiEffect(key="multi")):
                return af.concat(x, "!")
            return x

        ir = af.build_ir(program)("test")

        handler = MultiShotHandler(alternatives=["A", "B", "C"])
        with using_effect_handler({MultiEffect: handler}):
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
            with using_effect(AggregateEffect(key="agg")):
                return af.concat(x, " appended")
            return x

        ir = af.build_ir(program)("test")

        with using_effect_handler({AggregateEffect: aggregating_handler}):
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
            with using_effect(SingleEffect(key="single")):
                return af.concat(x, " world")
            return x

        ir = af.build_ir(program)("test")

        with using_effect_handler({SingleEffect: single_shot_handler}):
            result = af.call(ir)("hello")

        assert result == "HELLO world modified"

    def test_skip_still_works(self):

        class SkipEffect(Effect):
            pass

        def skip_handler(effect, value):
            return "SKIPPED"
            yield

        def program(x):
            with using_effect(SkipEffect(key="skip")):
                return af.concat(x, " should not appear")
            return x

        ir = af.build_ir(program)("test")

        with using_effect_handler({SkipEffect: skip_handler}):
            result = af.call(ir)("ignored")

        assert result == "SKIPPED"
