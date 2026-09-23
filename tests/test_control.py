# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import optree
import pytest

import autoform as af
from autoform.core import trace
from tests import aexecute, always_true, execute, fixpoint_program, switch_program, while_program

tree = optree.pytree.reexport(namespace=af.PYTREE_NAMESPACE)


def replace_state(state, instruction):
    del state
    return instruction


def stop_gradient(x):
    return af.stop_gradient(x)


def is_empty(x):
    return af.string.match(x, "")


class TestFixpointImpl:
    def test_max_iters_bounds_nonconvergent_step(self):
        def step(state, instruction):
            del instruction
            return af.string.concat(state, ".")

        step_ir = af.trace(step)("x", "unused")

        assert af.fixpoint(step_ir, "x", "unused", max_iters=4) == "x...."

    def test_custom_equiv_ir(self):
        def step(state, instruction):
            del instruction
            return af.string.concat(state, "!")

        step_ir = af.trace(step)("x", "unused")
        equiv_ir = af.trace(lambda prev, new: af.string.match(new, "x!!"))("a", "b")

        assert af.fixpoint(step_ir, "x", "unused", max_iters=10, equiv_ir=equiv_ir) == "x!!"

    def test_max_iters_validation(self):
        step_ir = af.trace(replace_state)("draft", "done")

        with pytest.raises(AssertionError, match="max_iters must be >= 1"):
            af.fixpoint(step_ir, "draft", "done", max_iters=0)

    def test_arity_validation(self):
        step_ir = af.trace(lambda state: state)("draft")

        with pytest.raises(AssertionError, match="exactly two"):
            af.fixpoint(step_ir, "draft", "done", max_iters=1)

    def test_state_structure_validation(self):
        step_ir = af.trace(lambda state, instruction: (state, instruction))("draft", "done")

        with pytest.raises(AssertionError, match="identical state"):
            af.fixpoint(step_ir, "draft", "done", max_iters=1)


def fixpoint_ir(step, args, **kwargs):
    step_ir = af.trace(step)(*args)
    return af.trace(fixpoint_program(step_ir, **kwargs))(*args)


class TestFixpointTraced:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_call(self, executor):
        ir = fixpoint_ir(replace_state, ("draft", "done"), max_iters=10)
        assert [eqn.prim for eqn in ir.eqns] == [af.control.fixpoint_p]
        result = executor(ir, "draft", "done")
        assert result == "done"


class TestFixpointPullback:
    @pytest.mark.parametrize(
        "executor, step, args, options, cotangent, expected, c_theta",
        [
            pytest.param(
                execute,
                lambda state, instruction: instruction,
                ("draft", "done"),
                {"max_iters": 10, "adj_iters": 2},
                "feedback",
                "done",
                "feedback",
                id="constant-map",
            ),
            pytest.param(
                aexecute,
                lambda state, instruction: instruction,
                ("draft", "done"),
                {"max_iters": 10},
                "feedback",
                "done",
                "feedback",
                id="async-constant-map",
            ),
            pytest.param(
                execute,
                af.string.concat,
                ("s", "c"),
                {"max_iters": 2, "adj_iters": 1},
                "g",
                "scc",
                "gg",
                id="adjoint-accumulation",
            ),
            pytest.param(
                execute,
                af.string.concat,
                ("s", "c"),
                {"max_iters": 2},
                af.ad.zeroof("g"),
                "scc",
                af.ad.zeroof("c"),
                id="zero-cotangent",
            ),
        ],
    )
    def test_feedback(self, executor, step, args, options, cotangent, expected, c_theta):
        ir = af.pullback(fixpoint_ir(step, args, **options))
        out, (c_init, actual_theta) = executor(ir, args, cotangent)
        assert out == expected
        assert af.ad.is_zero(c_init)
        assert actual_theta == c_theta

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_pullback_static_step_literal_is_not_boxed(self, executor):

        def step(state, suffix):
            return af.string.concat(state, suffix)

        step_ir = af.trace(step, static=(False, True))("s", "!")

        def program(init):
            return af.fixpoint(step_ir, init, "!", max_iters=2, adj_iters=1)

        ir = af.trace(program)("s")
        out, (c_init,) = executor(af.pullback(ir), ("s",), "g")
        assert out == "s!!"
        assert af.ad.is_zero(c_init)


class TestFixpointBatch:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batched_init_broadcast_theta(self, executor):
        ir = fixpoint_ir(replace_state, ("x", "done"), max_iters=10)
        batched = af.batch(ir, in_axes=(True, False))
        actual = executor(batched, ["a", "b"], "done")
        assert actual == ["done", "done"]

    def test_batch_of_pullback(self):
        ir = fixpoint_ir(replace_state, ("x", "done"), max_iters=10)
        composed = af.batch(af.pullback(ir), in_axes=((True, False), True))
        out, (c_init, c_theta) = composed.call((["a", "b"], "done"), ["g1", "g2"])

        assert out == ["done", "done"]
        assert all(af.ad.is_zero(c) for c in c_init)
        assert c_theta == ["g1", "g2"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_unbatched_fallback_preserves_pytree_out_batched(self, executor):

        def step(state, instruction):
            left, right = state
            return (af.string.concat(left, instruction), right)

        ir = fixpoint_ir(step, (("x", "y"), "!"), max_iters=1)
        batched = af.batch(ir, in_axes=(False, False))
        actual = executor(batched, ("x", "y"), "!")
        assert actual == ("x!", "y")

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batched_preserves_custom_state_container(self, executor):

        @tree.dataclasses.dataclass
        class State:
            text: str
            status: str
            label: str = tree.dataclasses.field(pytree_node=False)

        def step(state, instruction):
            return State(
                text=af.string.concat(state.text, instruction),
                status=state.status,
                label=state.label,
            )

        ir = fixpoint_ir(step, (State("x", "keep", label="state"), "!"), max_iters=1)
        batched = af.batch(ir, in_axes=(State(True, False, label="state"), False))
        out = executor(batched, State(["a", "b"], "keep", label="state"), "!")
        assert isinstance(out, State)
        assert out == State(text=["a!", "b!"], status=["keep", "keep"], label="state")


class TestEquivIR:
    def test_judged_convergence_counts(self):
        counters = dict(step=0, judge=0)

        @af.custom
        def step(state, instruction):
            del instruction
            counters["step"] += 1
            return af.string.concat(state, ".")

        @af.custom
        def probe(prev, new):
            del prev
            counters["judge"] += 1
            return new

        step_ir = af.trace(lambda state, instruction: step(state, instruction))("x", "unused")
        equiv_ir = af.trace(lambda prev, new: af.string.match(probe(prev, new), "x.."))("a", "b")
        counters["step"] = counters["judge"] = 0

        assert af.fixpoint(step_ir, "x", "unused", max_iters=10, equiv_ir=equiv_ir) == "x.."
        assert counters == dict(step=2, judge=2)

    def test_batched_equiv_ir(self):
        def step(state, instruction):
            del instruction
            return af.string.concat(state, "!")

        step_ir = af.trace(step)("x", "unused")
        equiv_ir = af.trace(lambda prev, new: af.string.match(new, af.string.concat(prev, "!")))(
            "a",
            "b",
        )

        program = fixpoint_program(step_ir, max_iters=10, equiv_ir=equiv_ir)

        ir = af.trace(program)("x", "unused")
        batched = af.batch(ir, in_axes=(True, False))

        assert batched.call(["a", "b"], "unused") == ["a!", "b!"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_equiv_ir(self, executor):
        step_ir = af.trace(replace_state)("x", "done")
        equiv_ir = af.trace(lambda prev, new: af.string.match(new, prev))("a", "b")

        program = fixpoint_program(step_ir, max_iters=10, equiv_ir=equiv_ir)

        ir = af.trace(program)("x", "done")

        assert executor(ir, "x", "done") == "done"

    def test_equiv_ir_validation(self):
        step_ir = af.trace(lambda state, instruction: af.string.concat(state, instruction))(
            "x",
            "!",
        )
        one_arg = af.trace(lambda prev: af.string.match(prev, "x"))("a")

        with pytest.raises(AssertionError, match="two positional"):
            af.fixpoint(step_ir, "x", "!", max_iters=3, equiv_ir=one_arg)

        wrong_struct = af.trace(lambda prev, new: af.string.match(prev[0], new[0]))(
            ("a", "b"),
            ("c", "d"),
        )
        with pytest.raises(AssertionError, match="state structure"):
            af.fixpoint(step_ir, "x", "!", max_iters=3, equiv_ir=wrong_struct)

    def test_params_memoize_with_equiv_ir(self):
        counters = dict(step=0)

        @af.custom
        def step(state, instruction):
            del state
            counters["step"] += 1
            return instruction

        step_ir = af.trace(lambda state, instruction: step(state, instruction))("x", "done")
        equiv_ir = af.trace(lambda prev, new: af.string.match(new, prev))("a", "b")

        program = fixpoint_program(step_ir, max_iters=10, equiv_ir=equiv_ir)

        ir = af.trace(program)("x", "done")

        with af.memoize():
            first = ir.call("x", "done")
            after_first = counters["step"]
            second = ir.call("x", "done")

        assert first == second == "done"
        assert counters["step"] == after_first


class TestStopGradient:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "sample, value",
        [
            pytest.param("test", "hello", id="scalar"),
            pytest.param(("a", "b"), ("hello", "world"), id="tree"),
        ],
    )
    def test_identity(self, executor, sample, value):
        ir = af.trace(stop_gradient)(sample)
        assert [eqn.prim for eqn in ir.eqns] == [af.control.stop_gradient_p]
        result = executor(ir, value)
        assert result == value

    @pytest.mark.parametrize(
        "transform, primals, feedback, expected",
        [
            pytest.param(af.pushforward, ("primal",), ("tangent",), "primal", id="push"),
            pytest.param(af.pullback, ("primal",), "cotangent", "primal", id="pull"),
            pytest.param(af.pullback, (("p1", "p2"),), ("c1", "c2"), ("p1", "p2"), id="tree-pull"),
        ],
    )
    def test_ad_zeros_feedback(self, transform, primals, feedback, expected):
        ir = af.trace(stop_gradient)(*primals)
        primal, derivative = transform(ir).call(primals, feedback)
        assert primal == expected
        assert all(af.ad.is_zero(leaf) for leaf in tree.leaves(derivative))

    def test_batch(self):
        ir = af.trace(stop_gradient)("a")
        assert af.batch(ir).call(["a", "b", "c"]) == ["a", "b", "c"]

    def test_in_chain_stops_gradient(self):
        def func(x, y):
            stopped = af.stop_gradient(x)
            return af.string.concat(stopped, y)

        ir = af.trace(func)("a", "b")
        pb_ir = af.pullback(ir)
        _, (cotangent_x, cotangent_y) = pb_ir.call(("a", "b"), "grad")
        assert af.ad.is_zero(cotangent_x)
        assert cotangent_y == "grad"


def loop_ir(cond, suffix, max_iters):
    cond_ir = trace(cond)("x")
    body_ir = trace(lambda x: af.string.concat(x, suffix))("x")
    return trace(while_program(cond_ir, body_ir, max_iters=max_iters))("")


class TestWhileLoop:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "condition, suffix, max_iters, value, expected",
        [
            pytest.param(lambda x: False, "x", 10, "input", "input", id="initially-false"),
            pytest.param(lambda x: True, "x", 0, "start", "start", id="zero-limit"),
            pytest.param(lambda x: True, "!", 1, "test", "test!", id="single-iteration"),
            pytest.param(lambda x: True, ".", 3, "a", "a...", id="iteration-bound"),
            pytest.param(lambda x: True, ".", 20, "a", "a" + "." * 20, id="many-iterations"),
        ],
    )
    def test_execution(self, executor, condition, suffix, max_iters, value, expected):
        ir = loop_ir(condition, suffix, max_iters)
        result = executor(ir, value)
        assert result == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "condition, suffix, max_iters, values, expected",
        [
            pytest.param(lambda x: False, "x", 10, ["a", "b", "c"], ["a", "b", "c"], id="all-exit"),
            pytest.param(lambda x: True, ".", 3, ["a", "b"], ["a...", "b..."], id="all-iterate"),
            pytest.param(
                lambda x: af.string.match(x, ""),
                "x",
                5,
                ["", "", "already"],
                ["x", "x", "already"],
                id="masked-exit",
            ),
            pytest.param(
                lambda x: af.string.match(x, ""),
                "x",
                3,
                ("", "a"),
                ("x", "a"),
                id="tuple-container",
            ),
            pytest.param(
                lambda x: af.string.match(x, "go"),
                "!",
                3,
                ["go", "stop", "go"],
                ["go!", "stop", "go!"],
                id="nonempty-condition",
            ),
        ],
    )
    def test_batch(self, executor, condition, suffix, max_iters, values, expected):
        ir = af.batch(loop_ir(condition, suffix, max_iters), in_axes=True)
        result = executor(ir, values)
        assert type(result) is type(values)
        assert result == expected

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "condition, suffix, max_iters, primal, cotangent, expected",
        [
            pytest.param(
                lambda x: False,
                "x",
                10,
                "start",
                "feedback",
                "start",
                id="no-iterations",
            ),
            pytest.param(lambda x: True, ".", 2, "a", "g", "a..", id="iterations"),
        ],
    )
    def test_pullback(self, executor, condition, suffix, max_iters, primal, cotangent, expected):
        ir = af.pullback(loop_ir(condition, suffix, max_iters))
        args = ((primal,), cotangent)
        result = executor(ir, *args)
        assert result == (expected, (cotangent,))

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_batch_of_pullback(self, executor):
        ir = af.batch(af.pullback(loop_ir(lambda x: True, ".", 2)), in_axes=(True, True))
        args = ((["a", "b"],), ["g1", "g2"])
        result = executor(ir, *args)
        assert result == (["a..", "b.."], (["g1", "g2"],))

    @pytest.mark.parametrize(
        "condition, max_iters, expected, checkpoints",
        [(False, 10, "a", []), (True, 3, "axxx", ["ax", "axx", "axxx"])],
        ids=["no-iterations", "iterations"],
    )
    def test_checkpoints_and_pullback(self, condition, max_iters, expected, checkpoints):
        def body(x):
            return af.checkpoint(af.string.concat(x, "x"), key="state", collection="trace")

        cond_ir = trace(lambda x: condition)("x")
        ir = trace(while_program(cond_ir, trace(body)("x"), max_iters=max_iters))("init")
        with af.collect(collection="trace") as collected:
            assert ir.call("a") == expected
        assert collected.get("state", []) == checkpoints
        assert af.pullback(ir).call(("a",), "feedback") == (expected, ("feedback",))
        assert collected.get("state", []) == checkpoints

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    def test_nested_while_loop(self, executor):

        def inner_body(x):
            return af.string.concat(x, "i")

        inner_cond_ir = trace(is_empty)("x")
        inner_body_ir = trace(inner_body)("x")

        def outer_body(x):
            inner_result = af.while_loop(inner_cond_ir, inner_body_ir, "", max_iters=2)
            return af.string.concat(x, inner_result)

        outer_cond_ir = trace(always_true)("x")
        outer_body_ir = trace(outer_body)("x")
        loop = while_program(outer_cond_ir, outer_body_ir, max_iters=3)
        loop_ir = trace(loop)("")
        result = executor(loop_ir, "start")
        assert result == "startiii"


class TestWhileLoopValidation:
    @pytest.mark.parametrize(
        "condition, max_iters",
        [
            pytest.param(False, 10, id="false-condition"),
            pytest.param(True, 0, id="zero-iterations"),
        ],
    )
    def test_rejects_literal_output_for_dynamic_state(self, condition, max_iters):
        cond_ir = trace(lambda x: condition)("go")
        body_ir = trace(lambda x: "stop")("go")

        loop = while_program(cond_ir, body_ir, max_iters=max_iters)

        with pytest.raises(AssertionError, match="initial state must match"):
            trace(loop)("go")

    @pytest.mark.parametrize(
        "initial, result",
        [
            pytest.param("go", "stop", id="different-values"),
            pytest.param(True, 1, id="boolean-integer-types"),
        ],
    )
    def test_rejects_different_literal_state(self, initial, result):
        cond_ir = trace(lambda x: False)(initial)
        body_ir = trace(lambda x: result)(initial)

        with pytest.raises(AssertionError, match="initial state must match"):
            trace(lambda: af.while_loop(cond_ir, body_ir, initial, max_iters=0))()

    def test_accepts_matching_literal_state(self):
        cond_ir = trace(lambda x: False)("stop")
        body_ir = trace(lambda x: "stop")("stop")
        ir = trace(lambda: af.while_loop(cond_ir, body_ir, "stop", max_iters=0))()

        assert ir.call() == "stop"

    def test_rejects_untraced_condition(self):
        cond_ir = lambda x: False
        body_ir = trace(lambda x: x)("x")
        with pytest.raises(AssertionError, match="cond_ir must be an IR"):
            af.while_loop(cond_ir, body_ir, "init", max_iters=10)

    def test_rejects_untraced_body(self):
        cond_ir = trace(lambda x: False)("x")
        body_ir = lambda x: x
        with pytest.raises(AssertionError, match="body_ir must be an IR"):
            af.while_loop(cond_ir, body_ir, "init", max_iters=10)

    def test_rejects_mismatched_body_structure(self):
        cond_ir = trace(lambda x: False)("x")
        body_ir = trace(lambda x: (x, x))("x")
        with pytest.raises(AssertionError, match="identical input/output structure"):
            af.while_loop(cond_ir, body_ir, "init", max_iters=10)


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param(("L", "R"), id="string"),
        pytest.param((0, 1), id="integer"),
        pytest.param((False, True), id="boolean"),
        pytest.param((0.5, 1.5), id="float"),
    ],
)
def test_switch_accepts_matching_key_types(keys):
    left = af.trace(lambda x: af.string.concat("L", x))("X")
    right = af.trace(lambda x: af.string.concat("R", x))("X")
    branches = dict(zip(keys, (left, right), strict=True))
    ir = af.trace(lambda key: af.switch(key, branches, "X"))(keys[0])
    assert ir.call(keys[0]) == "LX"
    assert ir.call(keys[1]) == "RX"


def test_switch_accepts_registered_key_type():
    class Key(str): ...

    af.extend.register_trace_type(Key, lambda _: af.core.StrAVal())
    left = af.trace(lambda x: af.string.concat("L", x))("X")
    right = af.trace(lambda x: af.string.concat("R", x))("X")
    keys = (Key("L"), Key("R"))
    branches = dict(zip(keys, (left, right), strict=True))
    ir = af.trace(lambda key: af.switch(key, branches, "X"))(keys[0])
    assert ir.call(keys[0]) == "LX"
    assert ir.call(keys[1]) == "RX"


@pytest.mark.parametrize(
    "keys, selector",
    [
        pytest.param((0, "R"), 0, id="mixed-integer-string-keys"),
        pytest.param((False, 2), False, id="mixed-boolean-integer-keys"),
        pytest.param((0, 2.0), 0, id="mixed-integer-float-keys"),
        pytest.param((b"L", b"R"), "L", id="unregistered-bytes-keys"),
        pytest.param((0, 1), True, id="boolean-selector-integer-keys"),
        pytest.param((False, True), 1, id="integer-selector-boolean-keys"),
        pytest.param((0, 1), 1.0, id="float-selector-integer-keys"),
    ],
)
def test_switch_rejects_invalid_key_types(keys, selector):
    left = af.trace(lambda x: af.string.concat("L", x))("X")
    right = af.trace(lambda x: af.string.concat("R", x))("X")
    branches = dict(zip(keys, (left, right), strict=True))
    with pytest.raises(AssertionError):
        af.trace(lambda key: af.switch(key, branches, "X"))(selector)


def test_switch_rejects_different_literal_branch_outputs():
    branches = {"L": af.trace(lambda: "L")(), "R": af.trace(lambda: "R")()}
    with pytest.raises(AssertionError):
        af.trace(lambda key: af.switch(key, branches))("L")


def switch_ir(prefixes):
    branches = {
        key: af.trace(lambda x, prefix=prefix: af.string.concat(prefix, x))("X")
        for key, prefix in prefixes.items()
    }
    return af.trace(switch_program(branches))(next(iter(branches)), "hello")


@pytest.fixture
def numbered_switch():
    return switch_ir({"zero": "zero: ", "one": "one: ", "two": "two: "})


class TestSwitch:
    @pytest.mark.parametrize(
        "executor, key, expected",
        [
            pytest.param(execute, "zero", "zero: hello", id="sync-first"),
            pytest.param(aexecute, "zero", "zero: hello", id="async-first"),
            pytest.param(execute, "one", "one: hello", id="sync-middle"),
            pytest.param(aexecute, "one", "one: hello", id="async-middle"),
            pytest.param(execute, "two", "two: hello", id="sync-last"),
        ],
    )
    def test_execution_and_structure(self, executor, key, expected, numbered_switch):
        ir = numbered_switch
        assert [eqn.prim for eqn in ir.eqns] == [af.control.switch_p]
        result = executor(ir, key, "hello")
        assert result == expected

    def test_invalid_key(self, numbered_switch):
        with pytest.raises(KeyError):
            numbered_switch.call("invalid_key", "hello")

    @pytest.mark.parametrize(
        "executor, key, expected",
        [
            pytest.param(execute, "zero", "zero: hello", id="sync-zero-zero: hello"),
            pytest.param(aexecute, "zero", "zero: hello", id="async-zero-zero: hello"),
            pytest.param(execute, "one", "one: hello", id="sync-one-one: hello"),
        ],
    )
    def test_pushforward(self, executor, key, expected, numbered_switch):
        ir = af.pushforward(numbered_switch)
        args = ((key, "hello"), ("", "world"))
        result = executor(ir, *args)
        assert result == (expected, "world")

    @pytest.mark.parametrize(
        "executor, key",
        [
            pytest.param(execute, "zero", id="sync-zero"),
            pytest.param(aexecute, "zero", id="async-zero"),
            pytest.param(execute, "one", id="sync-one"),
        ],
    )
    def test_pullback(self, executor, key, numbered_switch):
        ir = af.pullback(numbered_switch)
        args = ((key, "hello"), "grad")
        _, (c_key, c_x) = executor(ir, *args)
        assert af.ad.is_zero(c_key)
        assert c_x == "grad"

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "axes, key, value, expected",
        [
            pytest.param(
                (False, True),
                "zero",
                ["a", "b", "c"],
                ["zero: a", "zero: b", "zero: c"],
                id="operand-batched",
            ),
            pytest.param(
                (True, True),
                ["zero", "one", "zero"],
                ["a", "b", "c"],
                ["zero: a", "one: b", "zero: c"],
                id="both-batched",
            ),
            pytest.param(
                (True, False),
                ["zero", "one", "one"],
                "test",
                ["zero: test", "one: test", "one: test"],
                id="key-batched",
            ),
        ],
    )
    def test_batch(self, executor, axes, key, value, expected, numbered_switch):
        ir = af.batch(numbered_switch, in_axes=axes)
        result = executor(ir, key, value)
        assert result == expected

    @pytest.mark.parametrize(
        "transform, feedback, expected",
        [
            pytest.param(af.pushforward, ("", ["ta", "tb"]), ["ta", "tb"], id="push"),
            pytest.param(
                af.pullback,
                ["grad1", "grad2"],
                ([af.ad.zeroof("a"), af.ad.zeroof("a")], ["grad1", "grad2"]),
                id="pull",
            ),
        ],
    )
    def test_ad_of_batch(self, transform, feedback, expected):
        ir = transform(af.batch(switch_ir({"a": "A:", "b": "B:"}), in_axes=(False, True)))
        out, derivative = ir.call(("a", ["a", "b"]), feedback)
        assert out == ["A:a", "A:b"]
        assert derivative == expected

    def test_switch_with_multiple_operands(self):
        branches = {
            "concat": af.trace(af.string.concat)("A", "B"),
            "format": af.trace(lambda a, b: af.string.format("{a} - {b}", a=a, b=b))("A", "B"),
        }

        def program(key, x, y):
            return af.switch(key, branches, x, y)

        ir = af.trace(program)("concat", "Hello", "World")
        result = ir.call("concat", "Hello", "World")
        assert result == "HelloWorld"
        result = ir.call("format", "Hello", "World")
        assert result == "Hello - World"

    def test_branches_with_multiple_ops(self):
        def make_branch0(x):
            step1 = af.string.format("[{x}]", x=x)
            step2 = af.string.concat(step1, "!")
            return step2

        def make_branch1(x):
            step1 = af.string.format("({x})", x=x)
            step2 = af.string.concat(step1, "?")
            return step2

        branches = {
            "brackets": af.trace(make_branch0)("X"),
            "parens": af.trace(make_branch1)("X"),
        }

        program = switch_program(branches)

        ir = af.trace(program)("brackets", "test")
        assert ir.call("brackets", "hello") == "[hello]!"
        assert ir.call("parens", "hello") == "(hello)?"
