"""Tests for the assembled candidate (O3, increment I4).

Two things are being checked. First the §7 lifecycle contract, because
`run_candidate_lifecycle` enforces it at runtime and a candidate that fails it
cannot be evaluated at all. Second that the pieces are actually joined -- an
assembly can satisfy every interface and still not maintain a belief, which is
precisely the failure mode red-team attack A4 exploited.
"""

from __future__ import annotations

import numpy as np
import pytest

from cal.evaluation.randomized_occlusion_world import RandomizedOcclusionWorld
from cal.evaluation.stochastic_permanence_benchmark import (
    run_candidate_lifecycle,
    validate_candidate_factory,
)
from cal.evaluation.v2_i1_integration import ARENA_HIGH, ARENA_LOW, CAMERA
from cal.model.permanence_candidate import (
    StochasticPermanenceCandidateFactory,
    run_episode,
)

_GRID = 25


def _factory() -> StochasticPermanenceCandidateFactory:
    return StochasticPermanenceCandidateFactory(
        grid_size=_GRID,
        arena_low=ARENA_LOW,
        arena_high=ARENA_HIGH,
        camera=CAMERA,
        k_max=48,
    )


def _observation_stream(seed: int, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    world = RandomizedOcclusionWorld(seed, hidden_turn_probability=0.45)
    frames = [world.observe()]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        frames.append(world.step(int(rng.integers(0, 5))))
    return frames


def _action_stream(seed: int, steps: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    world = RandomizedOcclusionWorld(seed, hidden_turn_probability=0.45)
    sensed, visibility = world.observe()
    frames = [(sensed, visibility, 0)]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        action = int(rng.integers(0, 5))
        sensed, visibility = world.step(action)
        frames.append((sensed, visibility, action))
    return frames


@pytest.fixture(scope="module")
def frozen_kernel() -> dict:
    return _factory().fit_kernel(
        [_observation_stream(seed, 120) for seed in (62003, 62004, 62005)]
    )


# -- the §7 lifecycle contract -------------------------------------------


def test_the_factory_satisfies_the_runtime_checked_protocol() -> None:
    validate_candidate_factory(_factory())


def test_the_lifecycle_fits_once_and_never_reuses_episode_state() -> None:
    """`run_candidate_lifecycle` rejects a reused instance and a mutated kernel.

    Passing it is the whole admissibility bar for being evaluated at all.
    """

    run = run_candidate_lifecycle(
        _factory(),
        train_sensor_streams={
            62003: _observation_stream(62003, 60),
            62004: _observation_stream(62004, 60),
        },
        evaluation_sensor_streams={
            62149: _action_stream(62149, 40),
            62150: _action_stream(62150, 40),
        },
        run_episode=run_episode,
    )

    assert run.train_seed_ids == (62003, 62004)
    assert run.evaluation_seed_ids == (62149, 62150)
    assert run.model_seed == 17_011
    assert len(run.frozen_kernel_sha256) == 64
    for result in run.episode_results.values():
        assert result["steps"] == 41


# -- the fitted kernel ----------------------------------------------------


def test_the_turn_probability_is_inferred_from_reacquisitions(
    frozen_kernel: dict,
) -> None:
    """Turns happen only while occluded, so nothing visible reveals them.

    `_maybe_hidden_turn` returns early when the camera sees the entity, so a
    candidate that tried to estimate the rate from visible motion would be
    estimating a constant zero. The only evidence is how far a reappearance
    strays from a straight continuation.
    """

    assert frozen_kernel["reacquisition_event_count"] > 0
    assert 0.0 < frozen_kernel["turn_probability"] < 1.0


def test_the_kernel_is_json_canonicalizable(frozen_kernel: dict) -> None:
    """The lifecycle re-encodes the kernel to prove no callback changed it."""

    for value in frozen_kernel.values():
        assert isinstance(value, (int, float, str)), value


# -- the belief actually being maintained --------------------------------


def test_occupancy_is_a_probability_field_over_the_grid(
    frozen_kernel: dict,
) -> None:
    candidate = _factory().new_episode(frozen_kernel, 17_011)
    for sensed, visibility, action in _action_stream(62149, 60):
        candidate.observe(sensed, visibility, action)

    occupancy = candidate.occupancy()
    assert occupancy.shape == (_GRID, _GRID)
    assert np.all((occupancy >= 0.0) & (occupancy <= 1.0))
    # Static belief is folded in, so out-of-arena cells stay certain.
    assert np.all(occupancy[0, :] == pytest.approx(1.0))


def test_belief_is_carried_into_occlusion_rather_than_dropped(
    frozen_kernel: dict,
) -> None:
    """The point of the whole candidate: mass must survive going hidden.

    A tracker that simply forgets unobserved entities would leave the hidden
    field empty, which is exactly the behaviour the permanence gates exist to
    separate from belief maintenance.
    """

    candidate = _factory().new_episode(frozen_kernel, 17_011)
    world = RandomizedOcclusionWorld(62149, hidden_turn_probability=0.45)
    rng = np.random.default_rng(62149)

    sensed, visibility = world.observe()
    candidate.observe(sensed, visibility, 0)
    hidden_mass_seen = []
    for _ in range(80):
        action = int(rng.integers(0, 5))
        sensed, visibility = world.step(action)
        candidate.observe(sensed, visibility, action)
        hidden = candidate.hidden_occupancy()
        # Mass sitting on cells the camera cannot currently see.
        occupied, visible = np.zeros((_GRID, _GRID), bool), np.zeros((_GRID, _GRID), bool)
        radius = visibility.shape[0] // 2
        x0, y0 = CAMERA[0] - radius, CAMERA[1] - radius
        visible[y0 : y0 + visibility.shape[0], x0 : x0 + visibility.shape[1]] = (
            np.asarray(visibility).astype(bool)
        )
        del occupied
        hidden_mass_seen.append(float(hidden[~visible].sum()))

    assert max(hidden_mass_seen) > 0.0, "no belief ever survived into occlusion"


def test_each_episode_starts_from_nothing(frozen_kernel: dict) -> None:
    factory = _factory()
    first = factory.new_episode(frozen_kernel, 17_011)
    second = factory.new_episode(frozen_kernel, 17_011)

    for sensed, visibility, action in _action_stream(62149, 30):
        first.observe(sensed, visibility, action)

    assert first is not second
    assert second.track_positions() == ()
    assert np.all(second.hidden_occupancy() == 0.0)


def test_track_positions_are_deterministic_and_in_canonical_order(
    frozen_kernel: dict,
) -> None:
    outputs = []
    for _ in range(2):
        candidate = _factory().new_episode(frozen_kernel, 17_011)
        for sensed, visibility, action in _action_stream(62150, 50):
            candidate.observe(sensed, visibility, action)
        outputs.append(candidate.track_positions())

    assert outputs[0] == outputs[1]
    assert list(outputs[0]) == sorted(outputs[0])


def test_the_self_label_stays_a_normalized_categorical(
    frozen_kernel: dict,
) -> None:
    """At most one entity may be the self, and that has to survive tracking."""

    candidate = _factory().new_episode(frozen_kernel, 17_011)
    for sensed, visibility, action in _action_stream(62149, 60):
        candidate.observe(sensed, visibility, action)
        probabilities = candidate.self_probabilities()
        assert probabilities.sum() == pytest.approx(1.0)
        assert np.all(probabilities >= 0.0)


def test_the_privileged_diagnostic_separates_tracking_from_belief(
    frozen_kernel: dict,
) -> None:
    """The references are handed `hidden_tracks`; the candidate infers them.

    A single scoreboard therefore cannot say whether a deficit is weak belief
    maintenance or weak tracking, and those call for opposite responses.  This
    checks the diagnostic runs the candidate's belief filter on the references'
    tracks, which is what makes the two separable.
    """

    from cal.evaluation.permanence_candidate_development import (
        inferred_static_beliefs,
        privileged_belief_maps,
    )
    from cal.evaluation.permanence_forward_benchmark import _collect_many

    samples = _collect_many(
        [62149], steps=200, warmup=12, turn_probability=0.45
    )
    beliefs = inferred_static_beliefs(
        samples, frozen_kernel, steps=200, turn_probability=0.45
    )
    maps = privileged_belief_maps(
        samples, frozen_kernel, beliefs, turn_probability=0.45
    )

    assert maps.shape == (len(samples), 121)
    assert np.all((maps >= 0.0) & (maps <= 1.0))
    # It must actually carry belief mass, or it is measuring nothing.
    assert maps.sum() > 0.0
    # And it must use the *inferred* topology, not the world's.
    assert set(beliefs) == {(int(s.seed), int(s.step)) for s in samples}
