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

import autoform.pp as pp


def test_group_uses_flat_layout_when_it_fits():
    comma_line = pp.concat([pp.text(","), pp.line()])
    lay = pp.group(
        pp.concat([
            pp.text("f("),
            pp.nest(
                2,
                pp.concat([
                    pp.line(""),
                    pp.join(comma_line, [pp.text("alpha"), pp.text("beta")]),
                ]),
            ),
            pp.line(""),
            pp.text(")"),
        ])
    )

    assert pp.render(lay, width=20) == "f(alpha, beta)"


def test_group_uses_broken_layout_when_it_is_cheaper():
    comma_line = pp.concat([pp.text(","), pp.line()])
    lay = pp.group(
        pp.concat([
            pp.text("f("),
            pp.nest(
                2,
                pp.concat([
                    pp.line(""),
                    pp.join(comma_line, [pp.text("alpha"), pp.text("beta")]),
                ]),
            ),
            pp.line(""),
            pp.text(")"),
        ])
    )

    assert pp.render(lay, width=8) == "f(\n  alpha,\n  beta\n)"


def test_explicit_choice_uses_cost_function():
    lay = pp.choice(
        pp.text("abcdef"),
        pp.concat([pp.text("abc"), pp.hardline(), pp.text("def")]),
    )

    assert pp.render(lay, width=10) == "abcdef"
    assert pp.render(lay, width=4) == "abc\ndef"


def test_empty_frozenset_matches_python_repr():
    assert pp.pretty(frozenset()) == "frozenset()"
