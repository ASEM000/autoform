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

import pytest

import autoform as af
from autoform.checkpoint import checkpoint


class TestCheckpointBasics:
    def test_checkpoint_identity(self):
        result = checkpoint("hello", key="test", collection="debug")
        assert result == "hello"

    def test_checkpoint_primitive_in_ir(self):
        def func(x):
            return checkpoint(x, key="my_key", collection="my_col")

        ir = af.trace(func)("test")
        assert len(ir.ir_eqns) == 1
        assert ir.ir_eqns[0].prim.name == "checkpoint"
        assert ir.ir_eqns[0].params["key"] == "my_key"
        assert ir.ir_eqns[0].params["collection"] == "my_col"


class TestCollect:
    def test_collect_basic(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = ir.call("hello")

        assert result == "hello"
        assert collected == {"val": ["hello"]}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_collect_basic_async(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = await ir.acall("hello")

        assert result == "hello"
        assert collected == {"val": ["hello"]}

    def test_collect_filters_by_collection(self):
        def func(x):
            a = checkpoint(x, key="debug_val", collection="debug")
            b = checkpoint(a, key="other_val", collection="other")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as collected:
            result = ir.call("hello")

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
            ir.call("hello")

        assert collected == {"a": ["hello"], "b": ["hello"]}


class TestInject:
    def test_inject_replaces_value(self):
        def func(x):
            return checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.trace(func)("test")

        with af.collect(collection="cache") as collected:
            normal = ir.call("World")
        assert normal == "Hello, World"

        with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
            injected = ir.call("World")
        assert injected == "CACHED"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_inject_replaces_value_async(self):
        def func(x):
            return checkpoint(af.concat("Hello, ", x), key="greeting", collection="cache")

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"greeting": ["CACHED"]}):
            injected = await ir.acall("World")
        assert injected == "CACHED"

    def test_inject_partial(self):
        def func(x):
            a = checkpoint(x, key="first", collection="cache")
            b = checkpoint(af.concat(a, "!"), key="second", collection="cache")
            return b

        ir = af.trace(func)("test")

        with af.inject(collection="cache", values={"first": ["INJECTED"]}):
            result = ir.call("ignored")

        assert result == "INJECTED!"


class TestCheckpointWithTransforms:
    def test_checkpoints_through_batch(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        with af.collect(collection="debug") as collected:
            result = batched.call(["a", "b", "c"])

        assert result == ["a", "b", "c"]
        assert collected == {"val": ["a", "b", "c"]}

    @pytest.mark.asyncio(loop_scope="function")
    async def test_checkpoints_through_batch_async(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        batched = af.batch(ir)

        with af.collect(collection="debug") as collected:
            result = await batched.acall(["a", "b", "c"])

        assert result == ["a", "b", "c"]
        assert collected == {"val": ["a", "b", "c"]}

    def test_checkpoints_through_pushforward(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pf_ir = af.pushforward(ir)

        with af.collect(collection="debug") as collected:
            primal, tangent = pf_ir.call(("primal",), ("tangent",))

        assert primal == "primal"
        assert tangent == "tangent"
        assert collected == {"val": ["primal", "tangent"]}

    def test_checkpoints_through_pullback(self):
        def func(x):
            return checkpoint(x, key="val", collection="debug")

        ir = af.trace(func)("test")
        pb_ir = af.pullback(ir)

        with af.collect(collection="debug") as collected:
            primal, cotangent = pb_ir.call(("primal",), "cotangent")

        assert primal == "primal"
        assert cotangent == ("cotangent",)
        assert collected == {"val": ["primal", "cotangent"]}


class TestCheckpointComposition:
    def test_nested_collectors(self):
        def func(x):
            a = checkpoint(x, key="debug", collection="debug")
            b = checkpoint(a, key="cache", collection="cache")
            return b

        ir = af.trace(func)("test")

        with af.collect(collection="debug") as debug_collected:
            with af.collect(collection="cache") as cache_collected:
                result = ir.call("hello")

        assert result == "hello"
        assert debug_collected == {"debug": ["hello"]}
        assert cache_collected == {"cache": ["hello"]}
