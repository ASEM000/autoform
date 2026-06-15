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

import autoform as af
import autoform.core as core
import autoform.pp as pp


def test_ir_repr_breaks_long_calls_by_width():
    def program(x, y):
        return af.format(
            "Combine {} and {} for {} with a long instruction that should break",
            x,
            y,
            x,
        )

    ir = af.trace(program)("a", "b")
    text = pp.render(core.ir_lay(ir), width=60)

    assert "format(\n" in text
    assert "\n    template=" in text
    assert "\n    keys=()" in text


def test_ir_repr_indents_broken_input_trees():
    ir = af.trace(lambda x, y: af.concat(x, y))("x", "y")
    text = pp.render(core.ir_lay(ir), width=24)
    header = text.split(" -> ", 1)[0]
    continuation = header.splitlines()[1]

    assert continuation.startswith(" ")
    assert not continuation.startswith("%")


def test_ir_repr_formats_nested_ir_params():
    inner = af.trace(lambda x: af.format("[{}]", x))("x")
    batched = af.batch(inner)

    text = repr(batched)

    assert "batch_call(" in text
    assert "ir=func(" in text
    assert "in_axes=True" in text


def test_ir_repr_uses_object_pretty_rules_for_params():
    class Marker:
        pass

    @pp.register(Marker)
    def pretty_marker(obj):
        del obj
        return pp.text("<marker>")

    try:
        ir = af.trace(lambda x: af.format("{}", x))("x")
        ir.ir_eqns[0].params["marker"] = Marker()

        assert "marker=<marker>" in repr(ir)
    finally:
        pp.pretty_rules.pop(Marker, None)


def test_ir_repr_escapes_multiline_fallback_repr():
    class Multiline:
        def __repr__(self):
            return "line1\nline2"

    ir = af.trace(lambda x: af.format("{}", x))("x")
    ir.ir_eqns[0].params["bad"] = Multiline()

    assert "bad=line1\\nline2" in repr(ir)
