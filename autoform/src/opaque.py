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

"""Opaque values sealed with keys."""

from __future__ import annotations

from typing import Any, Self

import autoform.src.core as core
import autoform.src.stage as stage
import autoform.src.utils as utils

type Tree[T] = utils.Tree[T]

# ==================================================================================================
# OPAQUE VALUES
# ==================================================================================================


class AOpaque[K](core.AVal):
    # NOTE(asem): a distinct abstract value per producer/consumer pair, keyed by ``key``, so
    # an `Opaque` typechecks only against the IR it was produced for.
    __slots__ = ["key"]

    def __init__(self, key: K = None, /):
        hash(key)
        self.key: K = key

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.key == other.key

    def __hash__(self) -> int:
        return hash((type(self), self.key))

    def join(self, other, /) -> Self:
        match other:
            case AOpaque(key=k) | core.Zero(aval=AOpaque(key=k)) | Opaque(key=k) if k == self.key:
                return self
            case core.AVal() | core.Zero() | Opaque():
                pass
            case _ if self.key is None:
                return self
        raise TypeError(f"Cannot join {self!r} and {other!r}")

    def __repr__(self) -> str:
        key = "" if self.key is None else self.key
        return f"{type(self).__name__}({key})"


class Opaque[K]:
    __slots__ = ["key", "value"]

    def __init__(self, key: K, value: Any, /):
        assert key is not None, "Opaque requires a key"
        hash(key)
        self.key: K = key
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key})"


utils.tree.register_node(
    Opaque,
    lambda opaque: ((opaque.value,), opaque.key, ("value",)),
    lambda key, children: Opaque(key, *children),
)
stage.register_trace_type(Opaque, lambda opaque: AOpaque(opaque.key))
core.s_primal.set(AOpaque, lambda aval: aval)
core.s_tangent.set(AOpaque, lambda aval: aval)
core.s_cotangent.set(AOpaque, lambda aval: aval)


# ==================================================================================================
# SEAL
# ==================================================================================================


p_seal = core.Prim("seal")
p_unseal = core.Prim("unseal")


def seal[K](tree: Tree, /, *, key: K) -> Opaque[K]:
    """Seal a value with a key.

    ``tree`` can be any Python value; its contents are neither traced nor typed.
    Inside an IR, the unsealed value has type ``AOpaque()`` and the sealed value has
    type ``AOpaque(key)``. An IR input of type ``AOpaque(key)`` accepts only values
    sealed with ``key``, and only :func:`unseal` with the same ``key`` opens it.

    Args:
        tree: the value to seal.
        key: a hashable key; use ``object()`` for a key nobody else can name.

    Returns:
        An :class:`Opaque` holding ``tree`` under ``key``.

    Example:
        >>> import autoform as af
        >>> sealed = af.seal({"draft": ["hello"]}, key="cache")
        >>> sealed
        Opaque(cache)
        >>> af.unseal(sealed, key="cache")
        {'draft': ['hello']}
    """
    return p_seal.bind(tree, key=key)


def unseal[K](tree: Opaque[K], /, *, key: K) -> Tree:
    """Open a value sealed with ``key``.

    The inverse of :func:`seal`: it returns the original value, which has type
    ``AOpaque()`` inside an IR. A value sealed with a different key is rejected.

    Args:
        tree: a value returned by :func:`seal`.
        key: the key it was sealed with.

    Returns:
        The value that was sealed.

    Example:
        >>> import autoform as af
        >>> sealed = af.seal(("a", 1.0), key="cache")
        >>> af.unseal(sealed, key="cache")
        ('a', 1.0)
        >>> af.unseal(sealed, key="other")
        Traceback (most recent call last):
        ...
        TypeError: Expected a value sealed with 'other', got Opaque(cache)
    """
    return p_unseal.bind(tree, key=key)


def impl_seal[K](in_tree: Tree, /, *, key: K) -> Opaque[K] | core.Zero[AOpaque[K]]:
    match in_tree:
        case core.Zero(aval=AOpaque(key=None)):
            return core.Zero(AOpaque(key))
        case Opaque() | core.Zero(aval=AOpaque()):
            raise TypeError(f"Expected an unsealed value, got {in_tree!r}")
    return Opaque(key, in_tree)


def impl_unseal[K](in_tree: Opaque[K] | core.Zero[AOpaque[K]], /, *, key: K) -> Tree:
    match in_tree:
        case core.Zero(aval=AOpaque(key=k)) if k == key:
            return core.Zero(AOpaque())
        case Opaque(key=k, value=value) if k == key:
            return value
    raise TypeError(f"Expected a value sealed with {key!r}, got {in_tree!r}")


def abstract_seal[K](in_tree: AOpaque[None], /, *, key: K) -> AOpaque[K]:
    match in_tree:
        case AOpaque(key=None):
            return AOpaque(key)
        case core.AVal() | Opaque():
            raise TypeError(f"Expected AOpaque(), got {in_tree!r}")
    return AOpaque(key)


def abstract_unseal[K](in_tree: AOpaque[K] | Opaque[K], /, *, key: K) -> AOpaque[None]:
    match in_tree:
        case AOpaque(key=k) | Opaque(key=k) if k == key:
            return AOpaque()
    raise TypeError(f"Expected AOpaque({key!r}), got {in_tree!r}")


def push_seal[K](in_tree: Tree, /, *, key: K) -> tuple[Opaque[K], Opaque[K]]:
    p, t = in_tree
    return seal(p, key=key), seal(t, key=key)


def push_unseal[K](in_tree: Tree, /, *, key: K) -> tuple[Tree, Tree]:
    p, t = in_tree
    return unseal(p, key=key), unseal(t, key=key)


def pull_fwd_seal[K](in_tree: Tree, /, *, key: K) -> tuple[Opaque[K], None]:
    return seal(in_tree, key=key), None


def pull_fwd_unseal[K](in_tree: Opaque[K], /, *, key: K) -> tuple[Tree, None]:
    return unseal(in_tree, key=key), None


def pull_bwd_seal[K](in_tree: Tree, /, *, key: K) -> Tree:
    _, c = in_tree
    return unseal(c, key=key)


def pull_bwd_unseal[K](in_tree: Tree, /, *, key: K) -> Opaque[K]:
    _, c = in_tree
    return seal(c, key=key)


def batch_seal[K](in_tree: Tree, /, *, key: K) -> tuple[Tree, bool]:
    size, batched, x = in_tree
    if not batched:
        return seal(x, key=key), False
    spec = utils.batch_spec(x, batched)
    index = lambda b: utils.batch_index(x, batched, b)
    return spec.unflatten([seal(index(b), key=key) for b in range(size)]), True


def batch_unseal[K](in_tree: Tree, /, *, key: K) -> tuple[Tree, bool]:
    size, batched, x = in_tree
    if not batched:
        return unseal(x, key=key), False
    spec = utils.batch_spec(x, batched)
    index = lambda b: utils.batch_index(x, batched, b)
    return spec.unflatten([unseal(index(b), key=key) for b in range(size)]), True


core.r_impl.set(p_seal, impl_seal)
core.r_aimpl.set(p_seal, utils.asyncify(impl_seal))
core.r_abstract.set(p_seal, abstract_seal)
core.r_push.set(p_seal, push_seal)
core.r_apush.set(p_seal, utils.asyncify(push_seal))
core.r_pull_fwd.set(p_seal, pull_fwd_seal)
core.r_apull_fwd.set(p_seal, utils.asyncify(pull_fwd_seal))
core.r_pull_bwd.set(p_seal, pull_bwd_seal)
core.r_apull_bwd.set(p_seal, utils.asyncify(pull_bwd_seal))
core.r_batch.set(p_seal, batch_seal)
core.r_abatch.set(p_seal, utils.asyncify(batch_seal))

core.r_impl.set(p_unseal, impl_unseal)
core.r_aimpl.set(p_unseal, utils.asyncify(impl_unseal))
core.r_abstract.set(p_unseal, abstract_unseal)
core.r_push.set(p_unseal, push_unseal)
core.r_apush.set(p_unseal, utils.asyncify(push_unseal))
core.r_pull_fwd.set(p_unseal, pull_fwd_unseal)
core.r_apull_fwd.set(p_unseal, utils.asyncify(pull_fwd_unseal))
core.r_pull_bwd.set(p_unseal, pull_bwd_unseal)
core.r_apull_bwd.set(p_unseal, utils.asyncify(pull_bwd_unseal))
core.r_batch.set(p_unseal, batch_unseal)
core.r_abatch.set(p_unseal, utils.asyncify(batch_unseal))
