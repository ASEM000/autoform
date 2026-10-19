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
from enum import IntEnum
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
        return style(type(value).__name__, Code.BLUE), "(", join(map(pretty, children)), ")"
    return style(repr(value), Code.GREEN) if default is None else default(value)


def join(parts, sep=", "):
    """Join document parts with a separator."""
    doc = []
    for part in parts:
        doc.extend((part, sep))
    return doc[:-1]


class Doc:
    __slots__ = []


def is_doc(node):
    return isinstance(node, Doc)


class Code(IntEnum):
    BOLD = 1
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37


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


def style(doc, /, *codes: int):
    """Apply ordered ANSI style codes without changing layout width."""
    assert all(isinstance(code, int) for code in codes)
    return Style(doc, codes=tuple(map(int, codes)))


def line(alt: str = " "):
    """A break whose flat alternative is ``alt`` (a space by default)."""
    assert isinstance(alt, str) and "\n" not in alt and "\r" not in alt
    return Line(alt)


def hardline():
    """A break that remains a break inside a flat group."""
    return Line(None)


@ft.partial(register, tuple)
def pretty_tuple(value):
    suffix = "," if len(value) == 1 else ""
    return "(", join(map(pretty, value)), suffix, ")"


@ft.partial(register, list)
def pretty_list(value):
    return "[", join(map(pretty, value)), "]"


@ft.partial(register, dict)
def pretty_dict(value):
    items = [(pretty(key), ": ", pretty(child)) for key, child in value.items()]
    return "{", join(items), "}"


def next_column(column: int, text: str) -> int:
    last_break = max(text.rfind("\n"), text.rfind("\r"))
    return column + len(text) if last_break < 0 else len(text) - (last_break + 1)


def ansi_style(codes: tuple[int, ...]):
    # NOTE(asem): reset the style (0) before applying the incoming codes.
    return "\x1b[" + ";".join(("0", *(str(int(code)) for code in codes))) + "m"


def is_flat_fit(doc, remaining: int) -> bool:
    pending = list(reversed(utils.tree.leaves(doc, is_leaf=is_doc)))
    while pending and remaining >= 0:
        node = pending.pop()
        match node:
            case Group(doc=child) | Nest(doc=child) | Align(doc=child) | Style(doc=child):
                pending.extend(reversed(utils.tree.leaves(child, is_leaf=is_doc)))
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

    nodes = reversed(utils.tree.leaves(doc, is_leaf=is_doc))
    pending = [(node, 0, False, terminal_style) for node in nodes]

    while pending:
        node, indent, flat, node_style = pending.pop()
        if node_style != terminal_style:
            chunks.append(ansi_style(node_style))
            terminal_style = node_style
        match node:
            case Group(doc=child):
                child_flat = flat or is_flat_fit(child, width - column)
                children = reversed(utils.tree.leaves(child, is_leaf=is_doc))
                pending.extend((node, indent, child_flat, node_style) for node in children)
            case Nest(amount=amount, doc=child):
                children = reversed(utils.tree.leaves(child, is_leaf=is_doc))
                pending.extend((node, indent + amount, flat, node_style) for node in children)
            case Align(doc=child):
                children = reversed(utils.tree.leaves(child, is_leaf=is_doc))
                pending.extend((node, column, flat, node_style) for node in children)
            case Style(doc=child, codes=codes):
                # NOTE(asem): repeated style is harmless
                child_style = node_style + codes
                children = reversed(utils.tree.leaves(child, is_leaf=is_doc))
                pending.extend((node, indent, flat, child_style) for node in children)
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
