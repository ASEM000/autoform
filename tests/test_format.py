import pytest

import autoform as af


class TestFormatBasic:
    def test_positional_single(self):
        result = af.format("Hello, {}!", "World")
        assert result == "Hello, World!"

    def test_positional_multiple(self):
        result = af.format("{} + {} = {}", "1", "2", "3")
        assert result == "1 + 2 = 3"

    def test_positional_indexed(self):
        result = af.format("{0} {1} {0}", "a", "b")
        assert result == "a b a"

    def test_kwargs_single(self):
        result = af.format("Hello, {name}!", name="World")
        assert result == "Hello, World!"

    def test_kwargs_multiple(self):
        result = af.format("{first} {last}", first="John", last="Doe")
        assert result == "John Doe"

    def test_mixed_positional_and_kwargs(self):
        result = af.format("{0}, {name}!", "Hi", name="World")
        assert result == "Hi, World!"

    def test_empty_args_only_kwargs(self):
        result = af.format("{a}{b}{c}", a="x", b="y", c="z")
        assert result == "xyz"


class TestFormatTrace:
    def test_trace_positional(self):
        def greet(name):
            return af.format("Hello, {}!", name)

        ir = af.trace(greet)("x")
        result = af.call(ir)("World")
        assert result == "Hello, World!"

    def test_trace_kwargs(self):
        def greet(name):
            return af.format("Hello, {name}!", name=name)

        ir = af.trace(greet)("x")
        result = af.call(ir)("World")
        assert result == "Hello, World!"

    def test_trace_mixed(self):
        def greet(greeting, name):
            return af.format("{}, {name}!", greeting, name=name)

        ir = af.trace(greet)("x", "y")
        result = af.call(ir)("Hi", "World")
        assert result == "Hi, World!"


class TestFormatBatch:
    def test_batch_positional(self):
        def greet(name):
            return af.format("Hello, {}!", name)

        ir = af.trace(greet)("x")
        batched = af.batch(ir)
        result = af.call(batched)(["Alice", "Bob", "Charlie"])
        assert result == ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]

    def test_batch_kwargs(self):
        def greet(name):
            return af.format("Hello, {name}!", name=name)

        ir = af.trace(greet)("x")
        batched = af.batch(ir)
        result = af.call(batched)(["Alice", "Bob", "Charlie"])
        assert result == ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]

    def test_batch_mixed_args_kwargs(self):
        def greet(greeting, name):
            return af.format("{0}, {name}!", greeting, name=name)

        ir = af.trace(greet)("x", "y")
        batched = af.batch(ir)
        result = af.call(batched)(["Hi", "Hello"], ["Alice", "Bob"])
        assert result == ["Hi, Alice!", "Hello, Bob!"]

    def test_batch_broadcast_positional(self):
        def greet(greeting, name):
            return af.format("{}: {}", greeting, name)

        ir = af.trace(greet)("x", "y")
        batched = af.batch(ir, in_axes=(False, True))
        result = af.call(batched)("Hello", ["Alice", "Bob"])
        assert result == ["Hello: Alice", "Hello: Bob"]

    def test_batch_broadcast_kwargs(self):
        def greet(greeting, name):
            return af.format("{greeting}: {name}", greeting=greeting, name=name)

        ir = af.trace(greet)("x", "y")
        batched = af.batch(ir, in_axes=(False, True))
        result = af.call(batched)("Hello", ["Alice", "Bob"])
        assert result == ["Hello: Alice", "Hello: Bob"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_kwargs_async(self):
        def greet(name):
            return af.format("Hello, {name}!", name=name)

        ir = af.trace(greet)("x")
        batched = af.batch(ir)
        result = await af.acall(batched)(["Alice", "Bob"])
        assert result == ["Hello, Alice!", "Hello, Bob!"]


class TestFormatPushforward:
    def test_pushforward_positional(self):
        def greet(name):
            return af.format("Hello, {}!", name)

        ir = af.trace(greet)("x")
        pf_ir = af.pushforward(ir)
        primal, tangent = af.call(pf_ir)(("World", "Tangent"))
        assert primal == "Hello, World!"
        assert tangent == "Hello, Tangent!"

    def test_pushforward_kwargs(self):
        def greet(name):
            return af.format("Hello, {name}!", name=name)

        ir = af.trace(greet)("x")
        pf_ir = af.pushforward(ir)
        primal, tangent = af.call(pf_ir)(("World", "Tangent"))
        assert primal == "Hello, World!"
        assert tangent == "Hello, Tangent!"


class TestFormatPullback:
    def test_pullback_positional(self):
        def greet(name):
            return af.format("Hello, {}!", name)

        ir = af.trace(greet)("x")
        pb_ir = af.pullback(ir)
        primal, cotangent = af.call(pb_ir)(("World", "grad"))
        assert primal == "Hello, World!"

        assert cotangent == "grad"

    def test_pullback_kwargs(self):
        def greet(name):
            return af.format("Hello, {name}!", name=name)

        ir = af.trace(greet)("x")
        pb_ir = af.pullback(ir)
        primal, cotangent = af.call(pb_ir)(("World", "grad"))
        assert primal == "Hello, World!"

        assert cotangent == "grad"

    def test_pullback_mixed(self):
        def greet(greeting, name):
            return af.format("{0}, {name}!", greeting, name=name)

        ir = af.trace(greet)("x", "y")
        pb_ir = af.pullback(ir)
        primal, cotangent = af.call(pb_ir)((("Hi", "World"), "grad"))
        assert primal == "Hi, World!"

        assert cotangent == ("grad", "grad")
