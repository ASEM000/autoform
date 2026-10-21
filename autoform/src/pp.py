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

"""Pretty-printing documents, styles, and layout."""

# NOTE(asem): layout model inspired by
# https://homepages.inf.ed.ac.uk/wadler/topics/language-design.html#prettier

# NOTE(asem): the render primitive in the string module lifts the defined documents here from
# pretty-printing functionality into first-class objects in traceable text-space programs.
# this is done by registering the documents structures as pytrees. this enables values (text) to
# flow/transformed through the documents.

from __future__ import annotations

import functools as ft
from collections.abc import Callable
from typing import Any

from optree import GetAttrEntry

import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]


rules: dict[type, Callable[[Any], Tree]] = {}


def register[R: Callable[[Any], Tree]](cls: type, rule: R, /) -> R:
    """Register an exact-type rule that converts an object to a document."""
    assert isinstance(cls, type), f"Expected type, got {cls!r}"
    assert callable(rule), f"Expected callable, got {rule!r}"
    assert cls not in rules, f"Rule for {cls} already defined"
    rules[cls] = rule
    return rule


def pretty(value, default=None):
    if rule := rules.get(type(value)):
        return rule(value)
    children, spec = utils.tree.flatten(value, is_leaf=lambda child: child is not value)
    if value is not None and not spec.is_leaf():
        return style(type(value).__name__, Theme.TYPE), bracket("(", map(pretty, children), ")")
    return style(repr(value), literal_color(value)) if default is None else default(value)


def join(parts, sep):
    """Join document parts with a separator."""
    doc = []
    for part in parts:
        doc.extend((part, sep))
    return doc[:-1]


class Doc:
    __slots__ = []


def is_doc(node):
    return isinstance(node, Doc)


def children(doc):
    return reversed(utils.tree.leaves(doc, is_leaf=is_doc))


def rgb(color: int, /) -> tuple[int, ...]:
    return 38, 2, (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF


# NOTE(asem): VS CODE theme
class Theme:
    KEYWORD = rgb(0x569CD6)
    CONTROL = rgb(0xC586C0)
    FUNCTION = rgb(0xDCDCAA)
    VARIABLE = rgb(0x9CDCFE)
    TYPE = rgb(0x4EC9B0)
    STRING = rgb(0xCE9178)
    NUMBER = rgb(0xB5CEA8)
    CONSTANT = rgb(0x569CD6)


def literal_color(value: Any, /) -> tuple[int, ...]:
    match value:
        case bool() | None:
            return Theme.CONSTANT
        case int() | float() | complex():
            return Theme.NUMBER
        case str() | bytes():
            return Theme.STRING
    return ()


class Line(Doc):
    __slots__ = ["alt"]

    def __init__(self, alt: str | None = " "):
        self.alt = alt


class Group(Doc):
    __slots__ = ["doc"]

    def __init__(self, doc):
        self.doc = doc


class Nest(Doc):
    __slots__ = ["amount", "doc"]

    def __init__(self, amount: int, doc):
        self.amount = amount
        self.doc = doc


class Align(Doc):
    __slots__ = ["doc"]

    def __init__(self, doc):
        self.doc = doc


class Style(Doc):
    __slots__ = ["doc", "codes"]

    def __init__(self, doc, codes: tuple[int, ...] = ()):
        self.doc = doc
        self.codes = codes


utils.tree.register_node(
    Line,
    lambda n: ((), n.alt, ()),
    lambda a, _: Line(a),
)
utils.tree.register_node(
    Group,
    lambda n: ((n.doc,), None, ("doc",)),
    lambda _, c: Group(*c),
    path_entry_type=GetAttrEntry,
)
utils.tree.register_node(
    Nest,
    lambda n: ((n.doc,), n.amount, ("doc",)),
    lambda a, c: Nest(a, *c),
    path_entry_type=GetAttrEntry,
)
utils.tree.register_node(
    Align,
    lambda n: ((n.doc,), None, ("doc",)),
    lambda _, c: Align(*c),
    path_entry_type=GetAttrEntry,
)
utils.tree.register_node(
    Style,
    lambda n: ((n.doc,), n.codes, ("doc",)),
    lambda k, c: Style(*c, k),
    path_entry_type=GetAttrEntry,
)


def group(*parts):
    """Keep the contained sequence flat when it fits on the current line."""
    return Group(parts)


def nest(amount: int, doc):
    """Indent line breaks inside ``doc`` by ``amount`` spaces."""
    assert type(amount) is int and amount >= 0
    return Nest(amount, doc)


def align(doc):
    """Align line breaks inside ``doc`` with the current output column."""
    return Align(doc)


def style(doc, /, *codes: int | tuple[int, ...]):
    """Apply ordered ANSI style codes without changing layout width."""
    codes = tuple(c for code in codes for c in (code if isinstance(code, tuple) else (code,)))
    assert all(isinstance(code, int) for code in codes)
    return Style(doc, codes=codes)


def line(alt: str = " "):
    """A break whose flat alternative is ``alt`` (a space by default)."""
    assert isinstance(alt, str) and "\n" not in alt and "\r" not in alt
    return Line(alt)


def hardline():
    """A break that remains a break inside a flat group."""
    return Line(None)


def bracket(left: str, parts, right, suffix: str = "", *, indent: int = 4):
    """Comma-separate ``parts`` between ``left`` and ``right``, one per line if they do not fit."""
    parts = list(parts)
    if not parts:
        return left, right
    body = line(""), join(parts, (",", line())), suffix
    return group(left, nest(indent, body), line(""), right)


@ft.partial(register, tuple)
def pretty_tuple(value):
    suffix = "," if len(value) == 1 else ""
    return bracket("(", map(pretty, value), ")", suffix)


@ft.partial(register, list)
def pretty_list(value):
    return bracket("[", map(pretty, value), "]")


@ft.partial(register, dict)
def pretty_dict(value):
    items = [(pretty(key), ": ", pretty(child)) for key, child in value.items()]
    return bracket("{", items, "}")


def next_column(column: int, text: str) -> int:
    last_break = max(text.rfind("\n"), text.rfind("\r"))
    return column + len(text) if last_break < 0 else len(text) - (last_break + 1)


def ansi_style(codes: tuple[int, ...]):
    # NOTE(asem): reset the style (0) before applying the incoming codes.
    return "\x1b[" + ";".join(map(str, (0, *codes))) + "m"


def is_flat_fit(doc, remaining: int) -> bool:
    pending = list(children(doc))
    while pending and remaining >= 0:
        node = pending.pop()
        match node:
            case Group(doc=child) | Nest(doc=child) | Align(doc=child) | Style(doc=child):
                pending.extend(children(child))
            case Line(alt=None):
                return False
            case Line(alt=alt):
                remaining -= len(alt)
            case _:
                text = str(node)
                if "\n" in text or "\r" in text:
                    return False
                remaining -= len(text)
    return remaining >= 0


def layout(doc, /, *, width: int = 80) -> str:
    assert type(width) is int and width > 0, "width must be a positive integer"
    chunks = []
    column = 0
    terminal_style = ()

    pending = [(node, 0, False, terminal_style) for node in children(doc)]

    while pending:
        node, indent, flat, node_style = pending.pop()
        if node_style != terminal_style:
            chunks.append(ansi_style(node_style))
            terminal_style = node_style
        match node:
            case Group(doc=child):
                child_flat = flat or is_flat_fit(child, width - column)
                pending.extend((node, indent, child_flat, node_style) for node in children(child))
            case Nest(amount=amount, doc=child):
                pending.extend(
                    (node, indent + amount, flat, node_style) for node in children(child)
                )
            case Align(doc=child):
                pending.extend((node, column, flat, node_style) for node in children(child))
            case Style(doc=child, codes=codes):
                # NOTE(asem): repeated style is harmless
                child_style = node_style + codes
                pending.extend((node, indent, flat, child_style) for node in children(child))
            case Line(alt=alt):
                text = alt if flat and alt is not None else "\n" + " " * indent
                chunks.append(text)
                column = next_column(column, text)
            case _:
                text = str(node)
                chunks.append(text)
                column = next_column(column, text)
    if terminal_style:
        chunks.append(ansi_style(()))
    return "".join(chunks)
