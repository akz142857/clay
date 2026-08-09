"""Tests for the V8 control constructability census.

The census exists because the V5 holdout stopped part-way: the
identity-scrambled control could not be built from the reserved events, and a
consumed one-shot holdout cannot be retried (review finding F6 / open item O2).
These tests lock the two properties that make the census worth running -- that
every control is actually constructed rather than reported as constructed, and
that the eligibility restriction which stopped V5 is counted rather than hidden.
"""

from __future__ import annotations

import pytest

from cal.evaluation.permanence_control_constructability import (
    CONTROLS,
    run_constructability_census,
)

_STEPS = 120
_WARMUP = 12
_TURN_PROBABILITY = 0.45


@pytest.fixture(scope="module")
def census() -> dict:
    return run_constructability_census(
        [62003, 62004],
        [62149, 62150],
        steps=_STEPS,
        warmup=_WARMUP,
        turn_probability=_TURN_PROBABILITY,
    )


def test_every_v8_control_is_constructed(census: dict) -> None:
    assert set(census["controls"]) == set(CONTROLS)
    assert set(CONTROLS) == {
        "raw_sensor",
        "assume_all_visible",
        "time_shuffled",
        "identity_scrambled",
        "random_labels",
    }
    unbuilt = [
        name
        for name, entry in census["controls"].items()
        if not entry["constructed"]
    ]
    assert not unbuilt, f"controls that could not be built: {unbuilt}"
    assert census["all_controls_constructed"] is True


def test_identity_scramble_eligibility_is_counted_not_assumed(
    census: dict,
) -> None:
    """The number that stopped V5 has to appear in the output.

    Identity scrambling needs two simultaneously hidden tracked objects.  A
    census that reported only "constructed: true" would hide exactly the
    constraint it exists to measure.
    """

    entry = census["controls"]["identity_scrambled"]
    total = census["population"]["evaluation_sample_count"]

    assert entry["eligible_sample_count"] + entry["ineligible_sample_count"] == total
    assert entry["eligible_sample_count"] == (
        census["population"]["multi_hidden_evaluation_sample_count"]
    )
    # A strict subset: if this ever equalled the total, the eligibility rule
    # would have stopped being enforced.
    assert 0 < entry["eligible_sample_count"] < total


def test_assume_all_visible_puts_no_mass_on_hidden_cells(census: dict) -> None:
    """Denying occlusion should score exactly zero, not approximately zero.

    Every hidden positive is by definition in a non-visible cell, so this
    control's mass lands entirely outside the answer.  An exact 0.0 is the
    signal that the control is wired to visibility rather than to something
    correlated with it.
    """

    score = census["controls"]["assume_all_visible"]["score"]
    assert score["top1_accuracy"] == 0.0


def test_census_is_deterministic() -> None:
    """Two of the controls draw from an RNG; the census must still repeat."""

    kwargs = dict(
        steps=_STEPS, warmup=_WARMUP, turn_probability=_TURN_PROBABILITY
    )
    first = run_constructability_census([62003], [62149], **kwargs)
    second = run_constructability_census([62003], [62149], **kwargs)

    for name in CONTROLS:
        assert first["controls"][name]["score"] == second["controls"][name]["score"]


def test_census_refuses_overlapping_train_and_evaluation_seeds() -> None:
    """Two of the five controls are fitted probes, so contamination matters."""

    with pytest.raises(RuntimeError, match="seed collision between streams"):
        run_constructability_census(
            [62003, 62004],
            [62004, 62005],
            steps=_STEPS,
            warmup=_WARMUP,
            turn_probability=_TURN_PROBABILITY,
        )
