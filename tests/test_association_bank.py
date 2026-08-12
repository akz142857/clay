"""Tests for the global association bank (O3, increment I4b).

I4 shipped a single branch, which cannot represent association ambiguity: when
two detections are both plausible continuations of one track, one branch has to
guess, and the guess is unrecoverable because the alternative stops existing.
These check that ambiguity now survives as weight, that a branch stays
internally consistent, and that the two bounds are audited rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

from cal.evaluation.stochastic_permanence_kernel_diagnostic import (
    EVALUATION_GRID_SPEC as SPEC,
)
from cal.model.association_bank import (
    BankKernel,
    GlobalAssociationBank,
)
from cal.model.permanence_track import TrackKernel

_GRID = SPEC.grid_size
_TRACK = TrackKernel(survival_probability=0.99, retirement_existence=0.02)


def _bank(**overrides) -> GlobalAssociationBank:
    kernel = BankKernel(
        birth_intensity=0.1,
        clutter_intensity=0.01,
        maximum_hypotheses=overrides.pop("maximum_hypotheses", 5),
        maximum_children_per_parent=overrides.pop("maximum_children_per_parent", 4),
        maximum_tracks=overrides.pop("maximum_tracks", 8),
    )
    return GlobalAssociationBank(
        spec=SPEC,
        kernel=kernel,
        track_kernel=_TRACK,
        k_max=32,
        self_null_prior=0.5,
    )


def _grids(visible_cells: set[tuple[int, int]] | None = None):
    static = np.zeros((_GRID, _GRID), dtype=np.float64)
    visible = np.zeros((_GRID, _GRID), dtype=bool)
    if visible_cells is None:
        visible[SPEC.arena_low : SPEC.arena_high + 1, SPEC.arena_low : SPEC.arena_high + 1] = True
    else:
        for x, y in visible_cells:
            visible[y, x] = True
    no_detection = np.ones((_GRID, _GRID), dtype=np.float64)
    no_detection[visible] = 0.0
    return static, no_detection, visible


def _step(bank, detections, *, allow_turn=True, action=0, visible_cells=None):
    static, no_detection, visible = _grids(visible_cells)
    bank.step(
        detections=tuple(detections),
        static_probability=static,
        no_detection_probability=no_detection,
        visible=visible,
        turn_probability=0.45,
        allow_turn=allow_turn,
        action=action,
        self_likelihood_ratio=lambda *_args: 1.0,
    )


def test_weights_are_a_normalized_distribution() -> None:
    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)

    weights = bank.weights()
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


def test_ambiguity_survives_as_weight_instead_of_being_guessed() -> None:
    """Two reachable detections for one track must yield several branches.

    A single-branch tracker has to commit here, and the branch it drops is
    gone; the whole point of the bank is that both readings stay alive with a
    weight attached.
    """

    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)
    # Both neighbours are reachable continuations of the track just born.
    _step(bank, [(13, 12), (11, 12)])

    assert len(bank.hypotheses) > 1
    assert bank.weights().sum() == pytest.approx(1.0)


def test_a_detection_never_explains_two_tracks_in_one_branch() -> None:
    """One-to-one inside a hypothesis (§5.3): otherwise one observation would
    be counted as evidence for several entities at once."""

    bank = _bank()
    _step(bank, [(12, 12), (14, 12)], allow_turn=False)
    _step(bank, [(13, 12)])

    for hypothesis in bank.hypotheses:
        claimed = [cell for cell in hypothesis.last_positions if cell is not None]
        assert len(claimed) == len(set(claimed))


def test_the_hypothesis_bound_is_enforced_and_its_cost_recorded() -> None:
    """Pruning is storage loss, not disconfirmation, so it must be auditable."""

    bank = _bank(maximum_hypotheses=2, maximum_children_per_parent=4)
    _step(bank, [(12, 12), (15, 12)], allow_turn=False)
    for _ in range(4):
        _step(bank, [(13, 12), (11, 12), (14, 12)])

    assert len(bank.hypotheses) <= 2
    assert 0.0 <= bank.discarded_hypothesis_mass <= 1.0


def test_occupancy_is_the_weighted_union_across_branches() -> None:
    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)
    _step(bank, [(13, 12), (11, 12)])

    occupancy = bank.occupancy()
    weights = bank.weights()
    expected = sum(
        weight * hypothesis.occupancy(_GRID)
        for weight, hypothesis in zip(weights, bank.hypotheses)
    )

    assert occupancy.shape == (_GRID, _GRID)
    assert np.all((occupancy >= 0.0) & (occupancy <= 1.0))
    assert np.allclose(occupancy, expected)


def test_each_branch_keeps_its_own_self_label() -> None:
    """§5.4 forbids a cross-hypothesis self probability fed back per branch."""

    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)
    _step(bank, [(13, 12), (11, 12)])

    identities = [hypothesis.identity for hypothesis in bank.hypotheses]
    assert len({id(item) for item in identities}) == len(identities)
    for identity in identities:
        assert identity.probabilities().sum() == pytest.approx(1.0)
        assert identity.entity_count == len(
            bank.hypotheses[identities.index(identity)].tracks
        )


def test_branches_do_not_share_track_state() -> None:
    """Cloning is what keeps one branch's update out of another's posterior."""

    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)
    _step(bank, [(13, 12), (11, 12)])

    seen: set[int] = set()
    for hypothesis in bank.hypotheses:
        for track in hypothesis.tracks:
            assert id(track) not in seen
            seen.add(id(track))


def test_track_positions_come_from_the_most_likely_branch() -> None:
    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)
    _step(bank, [(13, 12), (11, 12)])

    positions = bank.track_positions()
    assert list(positions) == sorted(positions)
    best = bank.most_likely()
    assert len(positions) <= len(best.tracks)


def test_rounding_is_clipped_but_real_error_is_refused() -> None:
    """The union is analytically in [0, 1]; only representation error is
    forgiven.

    Silently clipping a genuinely out-of-range field would turn a broken
    posterior into a plausible-looking one, which is the failure mode this
    guard exists to avoid rather than create.
    """

    from cal.model.association_bank import as_probability_field

    one_ulp = np.asarray([1.0 + 2.220446049250313e-16])
    assert as_probability_field(one_ulp) == pytest.approx(1.0)
    assert as_probability_field(np.asarray([-1e-17])) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="too large to be rounding"):
        as_probability_field(np.asarray([1.3]))
    with pytest.raises(ValueError, match="too large to be rounding"):
        as_probability_field(np.asarray([-0.2]))


def test_a_newborn_track_does_not_invent_a_direction() -> None:
    """One detection fixes position and says nothing about heading.

    Committing to a direction anyway costs most exactly where permanence is
    measured, because the first hidden steps get propagated somewhere nobody
    observed.
    """

    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)

    track = bank.most_likely().tracks[0]
    states = track.predicted_states(
        static_probability=np.zeros((_GRID, _GRID), dtype=np.float64),
        turn_probability=0.45,
        allow_turn=False,
    )
    reached = {position for position, _velocity in states}
    # A uniform velocity prior reaches all four neighbours, not one.
    assert len(reached) == 4


# -- what a matched detection does to the posterior ------------------------


def _step_with_blind(bank, detections, blind, *, action=0):
    static = np.zeros((_GRID, _GRID), dtype=np.float64)
    visible = np.zeros((_GRID, _GRID), dtype=bool)
    visible[
        SPEC.arena_low : SPEC.arena_high + 1, SPEC.arena_low : SPEC.arena_high + 1
    ] = True
    for x, y in blind:
        visible[y, x] = False
    no_detection = np.ones((_GRID, _GRID), dtype=np.float64)
    no_detection[visible] = 0.0
    bank.step(
        detections=tuple(detections),
        static_probability=static,
        no_detection_probability=no_detection,
        visible=visible,
        turn_probability=0.45,
        allow_turn=True,
        action=action,
        self_likelihood_ratio=lambda *_args: 1.0,
    )


def test_re_acquisition_keeps_the_heading_the_track_maintained() -> None:
    """Seeing an entity again must not cost what tracking it established.

    A track that was unmatched on the previous step has no one-step
    displacement to read a velocity from, so restarting it from the detected
    cell falls back to a four-way prior -- and it does so at the last visible
    step, which is the step the next hidden stretch is propagated from.  This
    is the bin the closure gate is defined over, so the loss lands exactly
    where it is measured.
    """

    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)
    _step(bank, [(13, 12)])
    _step(bank, [(14, 12)])
    blind = [(15, 12), (16, 12), (17, 12)]
    _step_with_blind(bank, [], blind)
    _step_with_blind(bank, [], blind)

    _step(bank, [(17, 12)])

    states = bank.most_likely().tracks[0].states()
    assert sum(states.values()) == pytest.approx(1.0)
    assert max(states, key=lambda key: states[key]) == ((17, 12), (1, 0))
    assert states[((17, 12), (1, 0))] > 0.25


def test_the_self_ratio_is_handed_the_tracks_own_posterior() -> None:
    """The self evidence and the association evidence must see the same track.

    The autonomous denominator is a continuation of wherever the entity was
    heading, so it is a property of this track's posterior.  Recomputing it
    from a last-seen cell and an assumed heading -- which is what the previous
    version did, at a hardcoded ``(1, 0)`` -- makes every track look alike to
    the self posterior no matter what it has been doing.
    """

    bank = _bank()
    _step(bank, [(12, 12)], allow_turn=False)

    track = bank.hypotheses[0].tracks[0]
    before = track.states()
    predicted = track.predicted_states(
        static_probability=np.zeros((_GRID, _GRID), dtype=np.float64),
        turn_probability=0.45,
        allow_turn=True,
    )
    expected_mass = sum(
        mass for (position, _v), mass in predicted.items() if position == (13, 12)
    )

    seen: list[tuple] = []

    def spy(prior_states, autonomous_mass, cell, static_probability, action):
        seen.append((dict(prior_states), float(autonomous_mass), cell))
        return 1.0

    static = np.zeros((_GRID, _GRID), dtype=np.float64)
    visible = np.zeros((_GRID, _GRID), dtype=bool)
    visible[
        SPEC.arena_low : SPEC.arena_high + 1, SPEC.arena_low : SPEC.arena_high + 1
    ] = True
    no_detection = np.ones((_GRID, _GRID), dtype=np.float64)
    no_detection[visible] = 0.0
    bank.step(
        detections=((13, 12),),
        static_probability=static,
        no_detection_probability=no_detection,
        visible=visible,
        turn_probability=0.45,
        allow_turn=True,
        action=0,
        self_likelihood_ratio=spy,
    )

    matched = [call for call in seen if call[2] == (13, 12)]
    assert matched, "the matched track never reached the self posterior"
    assert expected_mass > 0.0
    for prior_states, autonomous_mass, _cell in matched:
        assert prior_states == pytest.approx(before)
        assert autonomous_mass == pytest.approx(expected_mass)


def test_the_self_ratio_marginalizes_over_the_prior_states() -> None:
    """A spread-out track contributes every state it might have occupied."""

    from cal.model.permanence_candidate import StochasticPermanenceCandidate

    kernel = {
        "grid_size": _GRID,
        "arena_low": SPEC.arena_low,
        "arena_high": SPEC.arena_high,
        "camera_x": 12,
        "camera_y": 12,
        "turn_probability": 0.45,
        "k_max": 32,
        "max_tracks": 8,
        "max_hypotheses": 5,
        "max_children_per_parent": 4,
        "birth_intensity": 0.1,
        "clutter_intensity": 0.01,
        "survival_probability": 0.99,
        "retirement_existence": 0.02,
        "self_null_prior": 0.5,
        "static_prior": 0.1,
        "static_mover_occupancy": 0.1,
        "static_clamp": 0.01,
    }
    candidate = StochasticPermanenceCandidate(kernel, 17_011)
    static = np.zeros((_GRID, _GRID), dtype=np.float64)

    # Action 2 is the step to the right, so only one of the two possible
    # previous cells can reach the detection; the ratio must carry that cell's
    # weight and not the whole track's.
    concentrated = candidate._self_likelihood_ratio(
        {((12, 12), (1, 0)): 1.0}, 1.0, (13, 12), static, 2
    )
    split = candidate._self_likelihood_ratio(
        {((12, 12), (1, 0)): 0.25, ((5, 5), (0, 1)): 0.75}, 1.0, (13, 12), static, 2
    )

    assert concentrated > 0.0
    assert split == pytest.approx(0.25 * concentrated)
    assert candidate._self_likelihood_ratio({}, 0.5, (13, 12), static, 2) == 1.0
