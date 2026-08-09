"""Preconditions for generating a one-shot permanence split.

Nothing generates a validation or holdout split yet -- there is no candidate,
so there is nothing to reserve one for.  This module exists ahead of that work
because one of its preconditions cannot be added afterwards.

The episode's visible layout is a deterministic function of the seed, and the
hidden maneuvers the permanence task is actually about come from a separate
stream.  While that stream was keyed by a published rule (``seed + 90_000``),
an observer could read the layout off the first frame, recover the seed, and
replay the hidden trajectory exactly -- the 2026-08-08 red team did, 10/10 over
200 steps (finding F9).  Keying it through ``HMAC(salt, seed)`` breaks the link,
but only if the salt is secret.

``DEFAULT_HIDDEN_STREAM_SALT`` is published in the source tree on purpose:
development and calibration episodes must stay reproducible and auditable, and
they are not one-shot.  A holdout is.  **A holdout generated under the published
salt is invertible from the moment it exists, and no later code change can
repair it** -- changing the salt does not re-secure those episodes, it produces
different ones, which is a different holdout.  Under one-shot semantics the
first batch is already spent.

So the ordering is hard: the custodian must hold the secret salt *before* the
first holdout seed is generated.  This module is the door that generation is
meant to go through, so that "forgot to pass the salt" fails loudly instead of
quietly producing a split that a later reader can invert.

NOT under the permanence stack's source lock, and deliberately so: that lock
covers the code determining the *current gated development evidence*, and a
future one-shot generator determines none of it.  When the holdout program is
frozen it needs its own lock covering this file.
"""

from __future__ import annotations

from typing import Any

from cal.evaluation.randomized_occlusion_world import (
    DEFAULT_HIDDEN_STREAM_SALT,
    RandomizedOcclusionWorld,
)

# Salts that are readable by anyone holding the repository.  A one-shot split
# keyed by any of these is seed-invertible by construction.
PUBLISHED_SALTS: frozenset[bytes] = frozenset({DEFAULT_HIDDEN_STREAM_SALT})

# HMAC-SHA256 keys shorter than the block's security margin buy nothing, and a
# short salt is the shape a hand-typed placeholder takes.
MINIMUM_CUSTODIAN_SALT_BYTES = 32


class InvertibleSplitError(RuntimeError):
    """Raised when a one-shot split would be generated with a guessable salt."""


def require_custodian_salt(salt: bytes) -> bytes:
    """Return ``salt`` if it can key a one-shot split, else refuse.

    This is a precondition, not a validation step that can run afterwards: by
    the time episodes exist, the salt that produced them is already baked into
    every hidden trajectory they contain.
    """

    if not isinstance(salt, (bytes, bytearray)):
        raise InvertibleSplitError(
            "custodian salt must be bytes held outside the repository"
        )
    material = bytes(salt)
    if material in PUBLISHED_SALTS:
        raise InvertibleSplitError(
            "refusing to key a one-shot split with the published development "
            "salt: the split would be seed-invertible from the moment it "
            "exists, and regenerating under a secret salt yields a different "
            "split rather than repairing this one (review finding F9)"
        )
    if len(material) < MINIMUM_CUSTODIAN_SALT_BYTES:
        raise InvertibleSplitError(
            f"custodian salt is {len(material)} bytes; at least "
            f"{MINIMUM_CUSTODIAN_SALT_BYTES} are required so the salt is not "
            f"searchable"
        )
    if len(set(material)) == 1:
        raise InvertibleSplitError(
            "custodian salt is a single repeated byte, which is a placeholder "
            "rather than a secret"
        )
    return material


def open_one_shot_world(
    seed: int, *, custodian_salt: bytes, **kwargs: Any
) -> RandomizedOcclusionWorld:
    """Build a world for a reserved split, refusing a guessable salt.

    ``RandomizedOcclusionWorld`` defaults ``hidden_stream_salt`` to the
    published value so development stays reproducible.  That default is the
    trap this exists to close: omitting the argument in a generation script
    would not raise, it would silently produce an invertible split.  Generation
    paths call this instead of the constructor.
    """

    return RandomizedOcclusionWorld(
        seed, hidden_stream_salt=require_custodian_salt(custodian_salt), **kwargs
    )
