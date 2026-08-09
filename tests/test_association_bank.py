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
