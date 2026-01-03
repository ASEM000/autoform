import autoform as af
from autoform.core import using_interpreter
from autoform.harvest import (
    CollectInterpreter,
    InjectInterpreter,
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

        handler = CollectInterpreter(collection="debug")
        with using_interpreter(handler):
            result = af.call(ir)("hello")

        assert result == "hello"
        assert handler.collected == {"val": ["hello"]}

    def test_collect_filters_by_collection(self):
        def func(x):
            a = checkpoint(x, key="debug_val", collection="debug")
            b = checkpoint(a, key="other_val", collection="other")
            return b

        ir = af.build_ir(func)("test")

        handler = CollectInterpreter(collection="debug")
        with using_interpreter(handler):
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

        handler = CollectInterpreter()
        with using_interpreter(handler):
            af.call(ir)("hello")

        assert handler.collected == {"a": ["hello"], "b": ["hello"]}


class TestInjectHandler:
    def test_inject_replaces_value(self):
        def func(x):
            return checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.build_ir(func)("test")

        handler = CollectInterpreter(collection="cache")
        with using_interpreter(handler):
            normal = af.call(ir)("World")
        assert normal == "Hello, World"

        handler = InjectInterpreter(collection="cache", values={"greeting": ["CACHED"]})
        with using_interpreter(handler):
            injected = af.call(ir)("World")
        assert injected == "CACHED"

    def test_inject_partial(self):
        def func(x):
            a = checkpoint(x, key="first", collection="cache")
            b = checkpoint(af.concat(a, "!"), key="second", collection="cache")
            return b

        ir = af.build_ir(func)("test")

        handler = InjectInterpreter(collection="cache", values={"first": ["INJECTED"]})
        with using_interpreter(handler):
            result = af.call(ir)("ignored")

        assert result == "INJECTED!"


class TestEffectsWithTransforms:
    def test_effects_through_batch(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")
        batched = af.batch(ir)

        handler = CollectInterpreter(collection="debug")
        with using_interpreter(handler):
            result = af.call(batched)(["a", "b", "c"])

        assert result == ["a", "b", "c"]
        assert handler.collected == {"val": [["a", "b", "c"]]}

    def test_effects_through_pushforward(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")
        pf_ir = af.pushforward(ir)

        handler = CollectInterpreter(collection="debug")
        with using_interpreter(handler):
            primal, tangent = af.call(pf_ir)(("primal", "tangent"))

        assert primal == "primal"
        assert tangent == "tangent"
        assert handler.collected == {"val": ["primal", "tangent"]}

    def test_effects_through_pullback(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.build_ir(func)("test")
        pb_ir = af.pullback(ir)

        handler = CollectInterpreter(collection="debug")
        with using_interpreter(handler):
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

        debug_handler = CollectInterpreter(collection="debug")
        with using_interpreter(debug_handler):
            cache_handler = CollectInterpreter(collection="cache")
            with using_interpreter(cache_handler):
                result = af.call(ir)("hello")

        assert result == "hello"
        assert debug_handler.collected == {"debug": ["hello"]}
        assert cache_handler.collected == {"cache": ["hello"]}
