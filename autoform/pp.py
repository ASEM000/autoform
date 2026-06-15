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

"""Small cost-based pretty-printing combinators inspired from Wadler and Bernardy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "LayoutPrim",
    "Layout",
    "PrettyRule",
    "pretty_rules",
    "register",
    "lay",
    "pretty",
    "safe_text",
    "seq",
    "text",
    "line",
    "hardline",
    "concat",
    "nest",
    "align",
    "choice",
    "group",
    "join",
    "render",
]


# ==================================================================================================
# LAYOUT
# ==================================================================================================


class LayoutPrim:
    __slots__ = ["name"]

    def __init__(self, name: str):
        assert isinstance(name, str), f"Expected str, got {name!r}"
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Layout:
    __slots__ = ["kind", "value", "parts"]

    def __init__(self, kind: LayoutPrim, value: Any = None, parts: tuple[Layout, ...] = ()):
        assert isinstance(kind, LayoutPrim)
        assert isinstance(parts, tuple) and all(isinstance(p, Layout) for p in parts)
        self.kind = kind
        self.value = value
        self.parts = parts


empty_p = LayoutPrim("empty")
text_p = LayoutPrim("text")
line_p = LayoutPrim("line")
hardline_p = LayoutPrim("hardline")
concat_p = LayoutPrim("concat")
nest_p = LayoutPrim("nest")
align_p = LayoutPrim("align")
choice_p = LayoutPrim("choice")
group_p = LayoutPrim("group")
empty = Layout(empty_p)


# ==================================================================================================
# LAYOUT CONSTRUCTORS
# ==================================================================================================


def text(value: str) -> Layout:
    assert isinstance(value, str), f"Expected str, got {value!r}"
    assert "\n" not in value, f"Text cannot contain newlines: {value!r}"
    return empty if value == "" else Layout(text_p, value)


def line(alt: str = " ") -> Layout:
    assert isinstance(alt, str), f"Expected str, got {alt!r}"
    assert "\n" not in alt, f"Line alternative cannot contain newlines: {alt!r}"
    return Layout(line_p, alt)


def hardline() -> Layout:
    return Layout(hardline_p)


def concat(parts: Iterable[Layout]) -> Layout:
    flat: list[Layout] = []
    for part in parts:
        assert isinstance(part, Layout), f"Expected Layout, got {part!r}"
        if part.kind is empty_p:
            continue
        if part.kind is concat_p:
            flat.extend(part.parts)
        else:
            flat.append(part)
    return empty if not flat else flat[0] if len(flat) == 1 else Layout(concat_p, parts=tuple(flat))


def nest(indent: int, lay: Layout) -> Layout:
    assert isinstance(indent, int) and indent >= 0, f"Invalid indent: {indent!r}"
    assert isinstance(lay, Layout), f"Expected Layout, got {lay!r}"
    return lay if indent == 0 else Layout(nest_p, indent, (lay,))


def align(lay: Layout) -> Layout:
    assert isinstance(lay, Layout), f"Expected Layout, got {lay!r}"
    return Layout(align_p, parts=(lay,))


def choice(left: Layout, right: Layout) -> Layout:
    assert isinstance(left, Layout), f"Expected Layout, got {left!r}"
    assert isinstance(right, Layout), f"Expected Layout, got {right!r}"
    return Layout(choice_p, parts=(left, right))


def group(lay: Layout) -> Layout:
    assert isinstance(lay, Layout), f"Expected Layout, got {lay!r}"
    return Layout(group_p, parts=(lay,))


def join(sep: Layout, lays: Iterable[Layout]) -> Layout:
    parts: list[Layout] = []
    for lay in lays:
        if parts:
            parts.append(sep)
        parts.append(lay)
    return concat(parts)


def safe_text(value: str) -> Layout:
    assert isinstance(value, str), f"Expected str, got {value!r}"
    return text(value.replace("\r", "\\r").replace("\n", "\\n"))


def seq(op: str, layouts: Iterable[Layout], cl: str, *, indent: int = 2) -> Layout:
    layouts = list(layouts)
    if not layouts:
        return text(op + cl)
    body = nest(indent, concat([line(""), join(concat([text(","), line()]), layouts)]))
    return group(concat([text(op), body, line(""), text(cl)]))


# ==================================================================================================
# PRETTY RULES
# ==================================================================================================


type PrettyRule = Callable[[Any], Layout]
pretty_rules: dict[type[Any], PrettyRule] = {}


def register(typ: type[Any], rule: PrettyRule | None = None, /):
    assert isinstance(typ, type), f"Expected type, got {typ!r}"

    def install(rule: PrettyRule) -> PrettyRule:
        assert callable(rule), f"Expected callable, got {rule!r}"
        pretty_rules[typ] = rule
        return rule

    return install if rule is None else install(rule)


def rule_for(obj: Any) -> PrettyRule | None:
    for typ in type(obj).__mro__:
        if rule := pretty_rules.get(typ):
            return rule
    return None


def lay(obj: Any) -> Layout:
    if rule := rule_for(obj):
        out = rule(obj)
        assert isinstance(out, Layout), f"Pretty rule returned non-layout: {out!r}"
        return out
    return safe_text(repr(obj))


def pretty(obj: Any, /, *, width: int = 100, limit: int | None = None) -> str:
    return render(lay(obj), width=width, limit=limit)


@register(str)
@register(int)
@register(float)
@register(bool)
@register(type(None))
def pretty_atom(obj: Any) -> Layout:
    return safe_text(repr(obj))


@register(list)
def pretty_list(obj: list[Any]) -> Layout:
    return seq("[", (lay(value) for value in obj), "]")


@register(tuple)
def pretty_tuple(obj: tuple[Any, ...]) -> Layout:
    layouts = [lay(value) for value in obj]
    if len(layouts) == 1:
        layouts = [concat([layouts[0], text(",")])]
    return seq("(", layouts, ")")


@register(dict)
def pretty_dict(obj: dict[Any, Any]) -> Layout:
    return seq("{", (concat([lay(k), text(": "), lay(v)]) for k, v in obj.items()), "}")


@register(set)
def pretty_set(obj: set[Any]) -> Layout:
    return text("set()") if not obj else seq("{", (lay(v) for v in sorted(obj, key=repr)), "}")


@register(frozenset)
def pretty_frozenset(obj: frozenset[Any]) -> Layout:
    if not obj:
        return text("frozenset()")
    return concat([text("frozenset("), pretty_set(set(obj)), text(")")])


# ==================================================================================================
# FLATTENING
# ==================================================================================================


type FlattenRule = Callable[[Layout], Layout]
flatten_rules: dict[LayoutPrim, FlattenRule] = {}
flatten_rules[line_p] = lambda lay: text(lay.value)
flatten_rules[concat_p] = lambda lay: concat(flatten(part) for part in lay.parts)
flatten_rules[nest_p] = lambda lay: nest(lay.value, flatten(lay.parts[0]))
flatten_rules[align_p] = lambda lay: align(flatten(lay.parts[0]))
flatten_rules[choice_p] = lambda lay: choice(flatten(lay.parts[0]), flatten(lay.parts[1]))
flatten_rules[group_p] = lambda lay: flatten(lay.parts[0])


def flatten(lay: Layout) -> Layout:
    if rule := flatten_rules.get(lay.kind):
        return rule(lay)
    return lay


# ==================================================================================================
# RENDER
# ==================================================================================================


class Renderer:
    __slots__ = ["out", "col", "lines", "maxw", "indp"]

    def __init__(self, out: str = "", col: int = 0, lines: int = 1, maxw: int = 0, indp: int = 0):
        self.out = out
        self.col = col
        self.lines = lines
        self.maxw = maxw
        self.indp = indp

    def cost(self, width: int) -> tuple[int, int, int, int]:
        overflow = max(0, self.maxw - width)
        return (overflow * overflow, self.lines, self.maxw, self.indp)

    def text(self, value: str) -> Renderer:
        col = self.col + len(value)
        return Renderer(self.out + value, col, self.lines, max(self.maxw, col), self.indp)

    def newline(self, indent: int) -> Renderer:
        return Renderer(
            self.out + "\n" + " " * indent,
            indent,
            self.lines + 1,
            max(self.maxw, indent),
            self.indp + indent,
        )


def prune(states: Iterable[Renderer], width: int, limit: int) -> tuple[Renderer, ...]:
    best: dict[int, Renderer] = {}
    for state in states:
        prev = best.get(state.col)
        if prev is None or (state.cost(width), state.out) < (prev.cost(width), prev.out):
            best[state.col] = state
    return tuple(sorted(best.values(), key=lambda s: (s.cost(width), s.out))[:limit])


type States = tuple[Renderer, ...]
type RenderChild = Callable[[Layout, States, int], States]
type RenderRule = Callable[[Layout, States, int, int, int, RenderChild], States]
render_rules: dict[LayoutPrim, RenderRule] = {}


render_rules[empty_p] = lambda l, ss, i, w, n, r: ss


def render_text(l: Layout, ss: States, i: int, w: int, n: int, r: RenderChild) -> States:
    del i, r
    return prune((s.text(l.value) for s in ss), w, n)


def render_line(l: Layout, ss: States, i: int, w: int, n: int, r: RenderChild) -> States:
    del l, r
    return prune((s.newline(i) for s in ss), w, n)


def render_concat(l: Layout, ss: States, i: int, w: int, n: int, r: RenderChild) -> States:
    del w, n
    out = ss
    for part in l.parts:
        out = r(part, out, i)
    return out


def render_align(l: Layout, ss: States, i: int, w: int, n: int, r: RenderChild) -> States:
    del i
    out: list[Renderer] = []
    for s in ss:
        out.extend(r(l.parts[0], (s,), s.col))
    return prune(out, w, n)


render_rules[concat_p] = render_concat
render_rules[nest_p] = lambda l, ss, i, w, n, r: r(l.parts[0], ss, i + l.value)
render_rules[align_p] = render_align


def render_choice(l: Layout, ss: States, i: int, w: int, n: int, r: RenderChild) -> States:
    return prune((*r(l.parts[0], ss, i), *r(l.parts[1], ss, i)), w, n)


def render_group(l: Layout, ss: States, i: int, w: int, n: int, r: RenderChild) -> States:
    del w, n
    return r(choice(flatten(l.parts[0]), l.parts[0]), ss, i)


render_rules[text_p] = render_text
render_rules[line_p] = render_line
render_rules[hardline_p] = render_line
render_rules[choice_p] = render_choice
render_rules[group_p] = render_group


def render_states(l: Layout, ss: States, i: int, w: int, n: int) -> States:
    if rule := render_rules.get(l.kind):
        r = lambda l, ss, i: render_states(l, ss, i, w, n)
        return rule(l, ss, i, w, n, r)
    raise TypeError(f"Unknown Layout node: {l!r}")


def render(lay: Layout, *, width: int = 100, limit: int | None = None) -> str:
    assert isinstance(lay, Layout), f"Expected Layout, got {lay!r}"
    assert isinstance(width, int) and width > 0, f"Invalid width: {width!r}"
    limit = 2048 if limit is None else limit
    assert isinstance(limit, int) and limit > 0, f"Invalid limit: {limit!r}"
    states = render_states(lay, (Renderer(),), 0, width, limit)
    return min(states, key=lambda state: (state.cost(width), state.out)).out
