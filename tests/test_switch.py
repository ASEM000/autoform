import pytest
import autoform as af


class TestSwitchBasic:
    def test_switch_key_zero(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
            "two": af.build_ir(lambda x: af.concat("two: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("zero", "hello")
        result = af.call(ir)("zero", "hello")
        assert result == "zero: hello"

    def test_switch_key_one(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
            "two": af.build_ir(lambda x: af.concat("two: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("one", "hello")
        result = af.call(ir)("one", "hello")
        assert result == "one: hello"

    def test_switch_key_two(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
            "two": af.build_ir(lambda x: af.concat("two: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("two", "hello")
        result = af.call(ir)("two", "hello")
        assert result == "two: hello"

    def test_switch_invalid_key_raises(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat("A:", x))("X"),
            "b": af.build_ir(lambda x: af.concat("B:", x))("X"),
        }
        with pytest.raises(KeyError):
            af.switch("invalid_key", branches, "hello")

    def test_switch_with_multiple_operands(self):
        branches = {
            "concat": af.build_ir(af.concat)("A", "B"),
            "format": af.build_ir(lambda a, b: af.format("{} - {}", a, b))("A", "B"),
        }

        def program(key, x, y):
            return af.switch(key, branches, x, y)

        ir = af.build_ir(program)("concat", "Hello", "World")
        result = af.call(ir)("concat", "Hello", "World")
        assert result == "HelloWorld"
        result = af.call(ir)("format", "Hello", "World")
        assert result == "Hello - World"

    def test_switch_direct_call(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat("A:", x))("X"),
            "b": af.build_ir(lambda x: af.concat("B:", x))("X"),
        }
        result = af.switch("a", branches, "test")
        assert result == "A:test"
        result = af.switch("b", branches, "test")
        assert result == "B:test"


class TestSwitchIRStructure:
    def test_creates_switch_eqn(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat("a", x))("X"),
            "b": af.build_ir(lambda x: af.concat("b", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("a", "test")
        assert len(ir.ireqns) == 1
        assert ir.ireqns[0].prim.name == "switch"

    def test_has_branches_param(self):
        branches = {
            "x": af.build_ir(lambda x: x)("X"),
            "y": af.build_ir(lambda x: x)("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("x", "test")
        assert "branches" in ir.ireqns[0].params
        assert len(ir.ireqns[0].params["branches"]) == 2


class TestSwitchPushforward:
    def test_pushforward_selects_correct_branch(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("zero", "hello")
        pf_ir = af.pushforward(ir)
        primals = ("zero", "hello")
        tangents = ("", "world")
        p_out, t_out = af.call(pf_ir)((primals, tangents))
        assert p_out == "zero: hello"
        assert t_out == "zero: world"

    def test_pushforward_key_one(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("one", "hello")
        pf_ir = af.pushforward(ir)
        primals = ("one", "hello")
        tangents = ("", "world")
        p_out, t_out = af.call(pf_ir)((primals, tangents))
        assert p_out == "one: hello"
        assert t_out == "one: world"


class TestSwitchPullback:
    def test_pullback_key_zero(self):
        def is_zero_cotangent(val):
            return val == "" or (hasattr(val, "items") and len(val.items) == 0)

        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("zero", "hello")
        pb_ir = af.pullback(ir)
        primals = ("zero", "hello")
        cotangent = "grad"
        _, (c_key, c_x) = af.call(pb_ir)((primals, cotangent))
        assert is_zero_cotangent(c_key)
        assert c_x == "grad"

    def test_pullback_key_one(self):
        def is_zero_cotangent(val):
            return val == "" or (hasattr(val, "items") and len(val.items) == 0)

        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("one", "hello")
        pb_ir = af.pullback(ir)
        primals = ("one", "hello")
        cotangent = "grad"
        _, (c_key, c_x) = af.call(pb_ir)((primals, cotangent))
        assert is_zero_cotangent(c_key)
        assert c_x == "grad"


class TestSwitchBatch:
    def test_batch_same_key(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("zero", "hello")
        batched_ir = af.batch(ir, in_axes=(None, list))
        result = af.call(batched_ir)("zero", ["a", "b", "c"])
        assert result == ["zero: a", "zero: b", "zero: c"]

    def test_batch_varying_key(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("zero", "hello")
        batched_ir = af.batch(ir, in_axes=(list, list))
        result = af.call(batched_ir)(["zero", "one", "zero"], ["a", "b", "c"])
        assert result == ["zero: a", "one: b", "zero: c"]

    def test_batch_varying_key_static_operand(self):
        branches = {
            "zero": af.build_ir(lambda x: af.concat("zero: ", x))("X"),
            "one": af.build_ir(lambda x: af.concat("one: ", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("zero", "hello")
        batched_ir = af.batch(ir, in_axes=(list, None))
        result = af.call(batched_ir)(["zero", "one", "one"], "test")
        assert result == ["zero: test", "one: test", "one: test"]


class TestSwitchNestedTransforms:
    def test_pushforward_of_batch(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat("A:", x))("X"),
            "b": af.build_ir(lambda x: af.concat("B:", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("a", "hello")
        batched_ir = af.batch(ir, in_axes=(None, list))
        pf_batched_ir = af.pushforward(batched_ir)
        primals = ("a", ["a", "b"])
        tangents = ("", ["ta", "tb"])
        p_out, t_out = af.call(pf_batched_ir)((primals, tangents))
        assert p_out == ["A:a", "A:b"]
        assert t_out == ["A:ta", "A:tb"]

    def test_pullback_of_batch(self):
        branches = {
            "a": af.build_ir(lambda x: af.concat("A:", x))("X"),
            "b": af.build_ir(lambda x: af.concat("B:", x))("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("a", "hello")
        batched_ir = af.batch(ir, in_axes=(None, list))
        pb_batched_ir = af.pullback(batched_ir)
        primals = ("a", ["a", "b"])
        cotangents = ["grad1", "grad2"]
        p_out, c_in = af.call(pb_batched_ir)((primals, cotangents))
        assert p_out == ["A:a", "A:b"]
        c_key, c_x = c_in
        assert c_x == ["grad1", "grad2"]


class TestSwitchWithKwargs:
    def test_switch_with_keyword_operands(self):
        branches = {
            "dash": af.build_ir(lambda x, y: af.format("{}-{}", x, y))("X", "Y"),
            "plus": af.build_ir(lambda x, y: af.format("{}+{}", x, y))("X", "Y"),
        }
        result = af.switch("dash", branches, "a", "b")
        assert result == "a-b"
        result = af.switch("plus", branches, "a", "b")
        assert result == "a+b"


class TestSwitchComplexBranches:
    def test_branches_with_multiple_ops(self):
        def make_branch0(x):
            step1 = af.format("[{}]", x)
            step2 = af.concat(step1, "!")
            return step2

        def make_branch1(x):
            step1 = af.format("({})", x)
            step2 = af.concat(step1, "?")
            return step2

        branches = {
            "brackets": af.build_ir(make_branch0)("X"),
            "parens": af.build_ir(make_branch1)("X"),
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("brackets", "test")
        assert af.call(ir)("brackets", "hello") == "[hello]!"
        assert af.call(ir)("parens", "hello") == "(hello)?"

    def test_many_branches(self):
        branches = {
            f"branch{i}": af.build_ir(lambda x, i=i: af.format("branch{}: {}", str(i), x))("X")
            for i in range(5)
        }

        def program(key, x):
            return af.switch(key, branches, x)

        ir = af.build_ir(program)("branch0", "test")
        for i in range(5):
            result = af.call(ir)(f"branch{i}", "hello")
            assert result == f"branch{i}: hello"
