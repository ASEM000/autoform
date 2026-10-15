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

import autoform.src.ad as ad
import autoform.src.axis as axis
import autoform.src.check as check
import autoform.src.core as core
import autoform.src.dead as dead
import autoform.src.numeric as numeric
import autoform.src.stage as stage
import autoform.src.string as string
from tests import AValBlob, aexecute, execute


class TestCheck:
    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "x",
        ["x", core.Zero(string.AValStr())],
        ids=["literal", "zero"],
    )
    def test_literal_identity(self, executor, x):
        aval = string.AValStr()
        assert check.typecheck(x, aval) is x
        ir = stage.trace(lambda: check.typecheck(x, aval))()
        assert ir.out_tree is x
        assert executor(ir) is x

    @pytest.mark.parametrize("x", [1.0, core.Zero(numeric.AValFloat())], ids=["value", "zero"])
    def test_rejects_incompatible_value(self, x):
        with pytest.raises(TypeError, match="Expected AValStr"):
            check.typecheck(x, string.AValStr())
        with pytest.raises(TypeError, match="Expected AValStr"):
            stage.trace(lambda: check.typecheck(x, string.AValStr()))()

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "transform, args, expected",
        [
            pytest.param(lambda ir: ir, ("x",), "x", id="identity"),
            pytest.param(ad.pushforward, (("x",), ("dx",)), ("x", "dx"), id="pushforward"),
            pytest.param(ad.pullback, (("x",), "df"), ("x", ("df",)), id="pullback"),
            pytest.param(axis.batch, (["x", "y"],), ["x", "y"], id="batch"),
            pytest.param(
                lambda ir: axis.batch(ad.pushforward(ir)),
                ((["x", "y"],), (["dx", "dy"],)),
                (["x", "y"], ["dx", "dy"]),
                id="batch-pushforward",
            ),
            pytest.param(
                lambda ir: ad.pushforward(axis.batch(ir)),
                ((["x", "y"],), (["dx", "dy"],)),
                (["x", "y"], ["dx", "dy"]),
                id="pushforward-batch",
            ),
        ],
    )
    def test_transform_identity(self, executor, transform, args, expected):
        ir = stage.trace(lambda x: check.typecheck(x, string.AValStr()))("x")
        assert executor(transform(ir), *args) == expected

    @pytest.mark.parametrize("outer", ["pushforward", "batch"])
    def test_nested_interpreters_reject_invalid_tangent(self, outer):
        parent = core.active_interp.get()
        if outer == "pushforward":
            batcher = axis.BatchInterp(batch_size=2, parent=parent)
            interpreter = ad.PushInterp(parent=batcher)
            x = interpreter.box((batcher.box((["x", "y"], True)), batcher.box((["dx", 1.0], True))))
        else:
            pusher = ad.PushInterp(parent=parent)
            interpreter = axis.BatchInterp(batch_size=2, parent=pusher)
            x = interpreter.box(([pusher.box(("x", "dx")), pusher.box(("y", 1.0))], True))
        aval = string.AValStr()
        assert core.domainof(aval).accepts(x)
        with core.using_interp(interpreter):
            with pytest.raises(TypeError, match="Expected AValStr"):
                check.p_typecheck.bind(x, aval=aval)

    @pytest.mark.parametrize("batched", [False, True], ids=["broadcast-batch", "nested-batch"])
    def test_batch_value_metadata(self, batched):
        batcher = axis.BatchInterp(batch_size=1, parent=core.active_interp.get())
        aval = axis.AValBatch(AValBlob(3))
        for size in (3, 4):
            value = [core.Zero(AValBlob(size))]
            box = batcher.box(([value] if batched else value, batched))
            with core.using_interp(batcher):
                if size == 3:
                    check.typecheck(box, aval)
                else:
                    with pytest.raises(TypeError, match="Expected"):
                        check.typecheck(box, aval)

    @pytest.mark.parametrize("x", [("x", "y"), {"a": "x", "b": "y"}], ids=["tuple", "dict"])
    def test_traced_batch_preserves_used_checks_and_broadcast(self, x):
        def program(x, y):
            batcher = axis.BatchInterp(batch_size=2, parent=core.active_interp.get())
            with core.using_interp(batcher):
                z = check.p_typecheck.bind(
                    batcher.box(([x, y], [True, False])),
                    aval=axis.AValBatch(string.AValStr()),
                )
            values, batched = batcher.unbox(z)
            assert batched == [True, False]
            return values

        ir = dead.dce(stage.trace(program)(x, "z"))
        assert [eqn.prim for eqn in ir.eqns] == [check.p_typecheck, check.p_typecheck]
        assert ir.call(x, "z") == [x, "z"]

    @pytest.mark.parametrize("executor", [execute, aexecute], ids=["sync", "async"])
    @pytest.mark.parametrize("used", [False, True], ids=["unused", "used"])
    def test_check_dce_and_retracing(self, executor, used):
        def program(x):
            y = check.typecheck(x + "!", string.AValStr())
            return y if used else x

        ir = dead.dce(stage.trace(program)("x"))
        for _ in range(3):
            ir = stage.trace(lambda x: executor(ir, x))("x")
            assert [eqn.prim.name for eqn in ir.eqns] == (["concat", "typecheck"] if used else [])
            assert executor(ir, "y") == ("y!" if used else "y")

    @pytest.mark.parametrize(
        "executor",
        [
            ad.impl_pullback_call,
            lambda args, *, ir: asyncio.run(ad.aimpl_pullback_call(args, ir=ir)),
        ],
        ids=["sync", "async"],
    )
    @pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "explicit"])
    def test_expanded_pullback_retracing(self, executor, explicit):
        def program(x):
            y = x + "!"
            return check.typecheck(y, string.AValStr()) if explicit else y

        source = stage.trace(program)("x")
        ir = stage.trace(lambda x, df: executor(((x,), df), ir=source))("x", "df")
        for _ in range(3):
            ir = stage.trace(ir.call)("x", "df")
            assert [eqn.prim.name for eqn in ir.eqns] == (
                ["concat", "typecheck", "typecheck"] if explicit else ["concat"]
            )
            assert ir.call("y", "dy") == ("y!", ("dy",))
