"""Tests for the candidate's existence channel (O3, increment I2).

The arithmetic of `bayesian_no_detection_update` was already covered; what was
never covered is that anything *uses* it (review finding F21).  These tests are
about the join: that existence, the kinematic posterior and the branch weight
move together the way plan §5.3 specifies, and that the evidence the bounded
filter already accounts for is not counted a second time.
"""

from __future__ import annotations

from math import log

import numpy as np
import pytest

from cal.evaluation.stochastic_permanence_kernel_diagnostic import (
    EVALUATION_GRID_SPEC,
)
from cal.model.permanence_track import PermanenceTrack, TrackKernel
from cal.model.stochastic_motion_filter import (
    EmptyPosteriorError,
    bayesian_no_detection_update,
)

_GRID = EVALUATION_GRID_SPEC.grid_size
_KERNEL = TrackKernel(survival_probability=0.99, retirement_existence=0.05)


def _track(k_max: int = 64, kernel: TrackKernel = _KERNEL) -> PermanenceTrack:
    track = PermanenceTrack(k_max=k_max, spec=EVALUATION_GRID_SPEC, kernel=kernel)
    track.reset((12, 12), (1, 0))
    return track


def _free_grid() -> np.ndarray:
    return np.zeros((_GRID, _GRID), dtype=np.float64)


def _all_unseen() -> np.ndarray:
    return np.ones((_GRID, _GRID), dtype=np.float64)


def test_a_fresh_track_is_certain_and_unweighted() -> None:
    track = _track()

    assert track.existence == 1.0
    assert track.branch_log_weight == 0.0
    assert sum(track.position_marginal().values()) == pytest.approx(1.0)


def test_seeing_nothing_where_nothing_is_visible_only_costs_survival() -> None:
    """With `P(no_detection|s) = 1` everywhere, `L_no` is 1.

    The observation is then uninformative about location, so `q` must be
    untouched and existence must fall by exactly the survival hazard folded
    through the Bernoulli update.
    """

    track = _track()
    before = track.position_marginal()

    step = track.step_unobserved(
        static_probability=_free_grid(),
        no_detection_probability=_all_unseen(),
        turn_probability=0.45,
        allow_turn=False,
    )

    assert step["no_detection_evidence"] == pytest.approx(1.0)
    predicted = _KERNEL.survival_probability
    # e_post = e_pred·1 / ((1-e_pred) + e_pred·1) = e_pred
    assert track.existence == pytest.approx(predicted)
    assert track.branch_log_weight == pytest.approx(0.0, abs=1e-12)
    assert before != track.position_marginal()  # it moved, but mass is intact
    assert sum(track.position_marginal().values()) == pytest.approx(1.0)


def test_existence_and_weight_match_the_reference_arithmetic() -> None:
    """The join must agree with `bayesian_no_detection_update` term by term."""

    track = _track()
    # Informative but not annihilating: one reachable cell is partly ruled out.
    # A likelihood of exactly 0 on the *only* successor would remove the whole
    # posterior, which the filter correctly refuses rather than renormalizing.
    no_detection = _all_unseen()
    no_detection[12, 13] = 0.25

    predicted_existence = _KERNEL.survival_probability * track.existence

    step = track.step_unobserved(
        static_probability=_free_grid(),
        no_detection_probability=no_detection,
        turn_probability=0.45,
        allow_turn=True,
    )

    # The documented update, applied to the evidence the filter reports.
    _q_post, existence_post, l_no, log_evidence = bayesian_no_detection_update(
        np.asarray([1.0]),
        np.asarray([step["no_detection_evidence"]]),
        predicted_existence,
    )

    assert 0.0 < step["no_detection_evidence"] < 1.0
    assert track.existence == pytest.approx(existence_post)
    assert track.existence < predicted_existence
    assert step["branch_evidence"] == pytest.approx(float(np.exp(log_evidence)))
    assert l_no == pytest.approx(step["no_detection_evidence"])


def test_the_filter_s_own_evidence_is_not_counted_twice() -> None:
    """The branch weight is existence-aware evidence plus pruning loss only.

    The bounded filter also maintains an existence-free branch weight of its
    own; adding that on top would charge the same observation twice.
    """

    track = _track()
    no_detection = _all_unseen()
    no_detection[12, 13] = 0.25

    step = track.step_unobserved(
        static_probability=_free_grid(),
        no_detection_probability=no_detection,
        turn_probability=0.45,
        allow_turn=False,
    )

    expected = log(step["branch_evidence"]) + log(step["retained_probability"])
    assert track.branch_log_weight == pytest.approx(expected)
    # And it is strictly different from the existence-free evidence, or the
    # existence channel would be doing nothing.
    assert track.branch_log_weight != pytest.approx(
        log(step["no_detection_evidence"])
    )


def test_pruning_loss_is_charged_to_the_branch() -> None:
    """Mass a bounded posterior discards is mass the branch failed to explain."""

    narrow = PermanenceTrack(k_max=1, spec=EVALUATION_GRID_SPEC, kernel=_KERNEL)
    narrow.reset((12, 12), (1, 0))

    step = narrow.step_unobserved(
        static_probability=_free_grid(),
        no_detection_probability=_all_unseen(),
        turn_probability=1.0,
        allow_turn=True,
    )

    assert step["retained_probability"] < 1.0
    assert narrow.branch_log_weight < 0.0
    assert narrow.cumulative_pruned_mass > 0.0


def test_occupancy_is_scaled_by_existence_but_the_marginal_is_not() -> None:
    """`q` is conditional on existing; occupancy is not.

    Reporting the conditional marginal as occupancy would claim presence the
    track itself no longer believes in.
    """

    track = _track()
    for _ in range(20):
        track.step_unobserved(
            static_probability=_free_grid(),
            no_detection_probability=_all_unseen(),
            turn_probability=0.45,
            allow_turn=True,
        )

    assert track.existence < 1.0
    assert sum(track.position_marginal().values()) == pytest.approx(1.0)
    assert sum(track.occupancy().values()) == pytest.approx(track.existence)


def test_retirement_is_a_threshold_not_a_death_observation() -> None:
    kernel = TrackKernel(survival_probability=0.5, retirement_existence=0.05)
    track = _track(kernel=kernel)

    assert not track.retired()
    for _ in range(5):
        track.step_unobserved(
            static_probability=_free_grid(),
            no_detection_probability=_all_unseen(),
            turn_probability=0.45,
            allow_turn=False,
        )

    assert track.existence == pytest.approx(0.5**5)
    assert track.retired()


def test_detection_evidence_follows_the_documented_integrand() -> None:
    track = _track()
    emission = np.zeros((_GRID, _GRID), dtype=np.float64)
    emission[12, 12] = 0.8

    evidence = track.detection_evidence(emission)

    # All mass sits on (12,12) before any step.
    assert evidence == pytest.approx(_KERNEL.survival_probability * 1.0 * 0.8)


def test_an_impossible_observation_refuses_rather_than_renormalizing() -> None:
    track = _track()
    impossible = np.zeros((_GRID, _GRID), dtype=np.float64)

    with pytest.raises(EmptyPosteriorError):
        track.step_unobserved(
            static_probability=_free_grid(),
            no_detection_probability=impossible,
            turn_probability=0.45,
            allow_turn=False,
        )


# -- conditioning a matched detection (O3, tracking correction) ------------


def test_conditioning_keeps_the_velocity_the_detection_is_consistent_with() -> None:
    """A detection says where, not where-from; the posterior supplies the rest.

    Restarting the filter from the detected cell with a velocity guessed from
    the one-step displacement is the alternative, and it discards everything
    the track had established about heading.
    """

    track = _track()
    predicted = track.predicted_states(
        static_probability=_free_grid(),
        turn_probability=0.45,
        allow_turn=True,
    )

    track.condition_on_detection(predicted, (13, 12))

    states = track.states()
    assert set(states) <= {((13, 12), velocity) for velocity in
                           ((1, 0), (-1, 0), (0, 1), (0, -1))}
    assert sum(states.values()) == pytest.approx(1.0)
    heading = max(states, key=lambda key: states[key])[1]
    assert heading == (1, 0)
    # The whole point: this is not the four-way prior a restart would install.
    assert states[((13, 12), (1, 0))] > 0.25
    assert track.existence == pytest.approx(1.0)


def test_conditioning_refuses_a_cell_the_posterior_cannot_reach() -> None:
    """Zero support is a disagreement with the gate, not a detection to absorb."""

    track = _track()
    predicted = track.predicted_states(
        static_probability=_free_grid(),
        turn_probability=0.45,
        allow_turn=False,
    )

    with pytest.raises(EmptyPosteriorError):
        track.condition_on_detection(predicted, (20, 20))


def test_states_reports_the_posterior_before_propagation() -> None:
    track = _track()

    assert track.states() == {((12, 12), (1, 0)): pytest.approx(1.0)}
