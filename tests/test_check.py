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

import asyncio

import pytest

import autoform as af
import autoform.check as check
from tests import BlobAVal, aexecute, execute


class TestCheck:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "x",
        ["x", af.core.Zero(af.string.StrAVal())],
        ids=["literal", "zero"],
    )
    def test_literal_identity(self, executor, x):
        aval = af.string.StrAVal()
        assert check.typecheck(x, aval) is x
        ir = af.trace(lambda: check.typecheck(x, aval))()
        assert ir.out_tree is x
        assert executor(ir) is x

    @pytest.mark.parametrize(
        "x", [1.0, af.core.Zero(af.numeric.FloatAVal())], ids=["value", "zero"]
    )
    def test_rejects_incompatible_value(self, x):
        with pytest.raises(TypeError, match="Expected StrAVal"):
            check.typecheck(x, af.string.StrAVal())
        with pytest.raises(TypeError, match="Expected StrAVal"):
            af.trace(lambda: check.typecheck(x, af.string.StrAVal()))()

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "transform, args, expected",
        [
            pytest.param(lambda ir: ir, ("x",), "x", id="identity"),
            pytest.param(af.pushforward, (("x",), ("dx",)), ("x", "dx"), id="pushforward"),
            pytest.param(af.pullback, (("x",), "df"), ("x", ("df",)), id="pullback"),
            pytest.param(af.batch, (["x", "y"],), ["x", "y"], id="batch"),
            pytest.param(
                lambda ir: af.batch(af.pushforward(ir)),
                ((["x", "y"],), (["dx", "dy"],)),
                (["x", "y"], ["dx", "dy"]),
                id="batch-pushforward",
            ),
            pytest.param(
                lambda ir: af.pushforward(af.batch(ir)),
                ((["x", "y"],), (["dx", "dy"],)),
                (["x", "y"], ["dx", "dy"]),
                id="pushforward-batch",
            ),
        ],
    )
    def test_transform_identity(self, executor, transform, args, expected):
        ir = af.trace(lambda x: check.typecheck(x, af.string.StrAVal()))("x")
        assert executor(transform(ir), *args) == expected

    @pytest.mark.parametrize("outer", ["pushforward", "batch"])
    def test_nested_interpreters_reject_invalid_tangent(self, outer):
        parent = af.core.active_interpreter.get()
        if outer == "pushforward":
            batcher = af.axis.BatchInterpreter(batch_size=2, parent=parent)
            interpreter = af.ad.PushforwardInterpreter(parent=batcher)
            x = interpreter.box((batcher.box((["x", "y"], True)), batcher.box((["dx", 1.0], True))))
        else:
            pusher = af.ad.PushforwardInterpreter(parent=parent)
            interpreter = af.axis.BatchInterpreter(batch_size=2, parent=pusher)
            x = interpreter.box(([pusher.box(("x", "dx")), pusher.box(("y", 1.0))], True))
        aval = af.string.StrAVal()
        aval.check(x)
        with af.core.using_interpreter(interpreter):
            with pytest.raises(TypeError, match="Expected StrAVal"):
                check.typecheck_p.bind(x, aval=aval)

    @pytest.mark.parametrize("batched", [False, True], ids=["broadcast-batch", "nested-batch"])
    def test_batch_value_metadata(self, batched):
        batcher = af.axis.BatchInterpreter(batch_size=1, parent=af.core.active_interpreter.get())
        aval = af.axis.BatchAVal(BlobAVal(3))
        for size in (3, 4):
            value = [af.core.Zero(BlobAVal(size))]
            box = batcher.box(([value] if batched else value, batched))
            with af.core.using_interpreter(batcher):
                if size == 3:
                    check.typecheck(box, aval)
                else:
                    with pytest.raises(TypeError, match="Expected"):
                        check.typecheck(box, aval)

    @pytest.mark.parametrize("x", [("x", "y"), {"a": "x", "b": "y"}], ids=["tuple", "dict"])
    def test_traced_batch_preserves_used_checks_and_broadcast(self, x):
        def program(x, y):
            batcher = af.axis.BatchInterpreter(
                batch_size=2, parent=af.core.active_interpreter.get()
            )
            with af.core.using_interpreter(batcher):
                z = check.typecheck_p.bind(
                    batcher.box(([x, y], [True, False])),
                    aval=af.axis.BatchAVal(af.string.StrAVal()),
                )
            values, batched = batcher.unbox(z)
            assert batched == [True, False]
            return values

        ir = af.dce(af.trace(program)(x, "z"))
        assert [eqn.prim for eqn in ir.eqns] == [check.typecheck_p, check.typecheck_p]
        assert ir.call(x, "z") == [x, "z"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize("used", [False, True], ids=["unused", "used"])
    def test_check_dce_and_retracing(self, executor, used):
        def program(x):
            y = check.typecheck(x + "!", af.string.StrAVal())
            return y if used else x

        ir = af.dce(af.trace(program)("x"))
        for _ in range(3):
            ir = af.trace(lambda x: executor(ir, x))("x")
            assert [eqn.prim.name for eqn in ir.eqns] == (["concat", "typecheck"] if used else [])
            assert executor(ir, "y") == ("y!" if used else "y")

    @pytest.mark.parametrize(
        "executor",
        [
            af.ad.impl_pullback_call,
            lambda args, *, ir: asyncio.run(af.ad.aimpl_pullback_call(args, ir=ir)),
        ],
        ids=["sync", "async"],
    )
    @pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "explicit"])
    def test_expanded_pullback_retracing(self, executor, explicit):
        def program(x):
            y = x + "!"
            return check.typecheck(y, af.string.StrAVal()) if explicit else y

        source = af.trace(program)("x")
        ir = af.trace(lambda x, df: executor(((x,), df), ir=source))("x", "df")
        for _ in range(3):
            ir = af.trace(ir.call)("x", "df")
            assert [eqn.prim.name for eqn in ir.eqns] == (
                ["concat", "typecheck", "typecheck"] if explicit else ["concat"]
            )
            assert ir.call("y", "dy") == ("y!", ("dy",))
