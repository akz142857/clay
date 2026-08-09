"""Tests for the one-shot split generation precondition.

The failure this guards against is not recoverable after the fact: a holdout
keyed by the published salt is seed-invertible from the moment its episodes
exist, and regenerating under a secret salt produces a *different* holdout
rather than repairing that one (review finding F9, open item O19).
"""

from __future__ import annotations

import pytest

from cal.evaluation.randomized_occlusion_world import (
    DEFAULT_HIDDEN_STREAM_SALT,
    RandomizedOcclusionWorld,
    hidden_stream_key,
)
from cal.evaluation.stochastic_permanence_holdout import (
    MINIMUM_CUSTODIAN_SALT_BYTES,
    InvertibleSplitError,
    open_one_shot_world,
    require_custodian_salt,
)

_SECRET = bytes(range(MINIMUM_CUSTODIAN_SALT_BYTES))


def test_published_development_salt_is_refused() -> None:
    with pytest.raises(InvertibleSplitError, match="published development"):
        require_custodian_salt(DEFAULT_HIDDEN_STREAM_SALT)


@pytest.mark.parametrize(
    "salt, expected",
    [
        (b"", "at least"),
        (b"too-short", "at least"),
        (b"\x00" * MINIMUM_CUSTODIAN_SALT_BYTES, "single repeated byte"),
        ("a string, not bytes", "must be bytes"),
    ],
)
def test_placeholder_salts_are_refused(salt: object, expected: str) -> None:
    with pytest.raises(InvertibleSplitError, match=expected):
        require_custodian_salt(salt)  # type: ignore[arg-type]


def test_a_real_secret_passes_through_unchanged() -> None:
    assert require_custodian_salt(_SECRET) == _SECRET
    assert require_custodian_salt(bytearray(_SECRET)) == _SECRET


def test_the_constructor_default_is_the_trap_this_closes() -> None:
    """Omitting the salt must fail here, where the constructor would not.

    ``RandomizedOcclusionWorld`` defaults to the published salt so development
    stays reproducible; a generation script that forgot the argument would get
    an invertible split with no error at all.
    """

    # The constructor accepts the omission silently -- that is the hazard.
    assert RandomizedOcclusionWorld(62003).grid_size == 25

    with pytest.raises(TypeError):
        open_one_shot_world(62003)  # type: ignore[call-arg]
    with pytest.raises(InvertibleSplitError):
        open_one_shot_world(62003, custodian_salt=DEFAULT_HIDDEN_STREAM_SALT)


def test_a_salted_world_is_built_and_its_hidden_stream_moves() -> None:
    world = open_one_shot_world(62003, custodian_salt=_SECRET)

    assert world.grid_size == 25
    # The salt reached the stream: the same seed under the published salt keys
    # a different hidden trajectory.
    assert hidden_stream_key(62003, salt=_SECRET) != hidden_stream_key(
        62003, salt=DEFAULT_HIDDEN_STREAM_SALT
    )


def test_the_guard_is_not_inside_the_gated_source_lock() -> None:
    """This module governs future evidence, not the current artifacts.

    Keeping it outside is deliberate, but it must stay a conscious choice: if
    the permanence entry points ever import it, it starts determining gated
    evidence and has to be locked with them.
    """

    from cal.evaluation.stochastic_permanence_artifacts import (
        permanence_stack_source_paths,
    )
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    locked = {path.relative_to(root).as_posix() for path in permanence_stack_source_paths(root)}

    assert "cal/evaluation/stochastic_permanence_holdout.py" not in locked
