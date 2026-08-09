"""Tests for branch-local self identity (O3, increment I3).

Plan §5.4 forbids two designs that look equivalent on a single entity and are
not: a self probability aggregated across hypotheses and pushed back into each
branch, and an independent Bernoulli self flag per entity. The second is the
subtle one -- it agrees with the correct model on every single-entity marginal
and disagrees only on the joint, so a test that checks marginals alone would
pass for the wrong implementation. These check the joint.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cal.evaluation.stochastic_permanence_kernel_diagnostic import (
    EVALUATION_GRID_SPEC as SPEC,
)
from cal.model.branch_self_identity import (
    BranchSelfIdentity,
    action_successors,
    entity_marginal_transition,
    joint_transition_probability,
)

_A = ((12, 12), (1, 0))
_B = ((13, 12), (1, 0))
_C = ((12, 13), (0, 1))


def _free() -> np.ndarray:
    return np.zeros((SPEC.grid_size, SPEC.grid_size), dtype=np.float64)


# --------------------------------------------------------------------------
# The action-conditioned kernel
# --------------------------------------------------------------------------


def test_the_action_kernel_moves_by_the_delta_and_leaves_velocity_alone() -> None:
    """The agent has no velocity in the world, so inventing one is a claim the
    observation model cannot support."""

    successors = action_successors((12, 12), (1, 0), _free(), action=2, spec=SPEC)

    assert successors == (((13, 12), (1, 0), 1.0),)


def test_an_uncertain_target_splits_between_moving_and_being_blocked() -> None:
    static = _free()
    static[12, 13] = 0.25

    successors = action_successors((12, 12), (1, 0), static, action=2, spec=SPEC)

    assert dict(((p, v), m) for p, v, m in successors) == pytest.approx(
        {((13, 12), (1, 0)): 0.75, ((12, 12), (1, 0)): 0.25}
    )


def test_the_arena_edge_clips_rather_than_leaking() -> None:
    at_edge = (SPEC.arena_high, 12)

    successors = action_successors(at_edge, (1, 0), _free(), action=2, spec=SPEC)

    assert successors == ((at_edge, (1, 0), 1.0),)


def test_staying_is_a_certain_self_transition() -> None:
    assert action_successors((12, 12), (1, 0), _free(), action=0, spec=SPEC) == (
        ((12, 12), (1, 0), 1.0),
    )


# --------------------------------------------------------------------------
# The categorical
# --------------------------------------------------------------------------


def test_at_most_one_entity_can_be_the_self() -> None:
    identity = BranchSelfIdentity(3, null_prior=0.4)

    assert identity.probabilities().sum() == pytest.approx(1.0)
    assert identity.null_probability == pytest.approx(0.4)
    assert sum(identity.probability(i) for i in range(3)) == pytest.approx(0.6)


def test_evidence_moves_the_label_and_keeps_it_normalized() -> None:
    identity = BranchSelfIdentity(2)

    identity.update((4.0, 1.0), null_likelihood=1.0)

    assert identity.probabilities().sum() == pytest.approx(1.0)
    assert identity.probability(0) > identity.probability(1)


def test_vanishing_evidence_refuses_rather_than_renormalizing_noise() -> None:
    identity = BranchSelfIdentity(2)

    with pytest.raises(ValueError, match="evidence vanished"):
        identity.update((0.0, 0.0), null_likelihood=0.0)


def test_probabilities_are_not_aliased() -> None:
    identity = BranchSelfIdentity(2)
    snapshot = identity.probabilities()
    snapshot[:] = 0.0

    assert identity.probabilities().sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The joint, which is where an independent-Bernoulli design shows itself
# --------------------------------------------------------------------------


def _kernels() -> tuple[list[dict], list[dict]]:
    autonomous = [
        {_A: 0.6, _B: 0.4},
        {_B: 0.3, _C: 0.7},
    ]
    action_conditioned = [
        {_B: 1.0},
        {_A: 0.5, _C: 0.5},
    ]
    return autonomous, action_conditioned


def test_the_joint_is_a_distribution() -> None:
    autonomous, action_conditioned = _kernels()
    identity = BranchSelfIdentity(2, null_prior=0.3)
    support = (_A, _B, _C)

    total = sum(
        joint_transition_probability(
            autonomous, action_conditioned, successor, identity
        )
        for successor in itertools.product(support, repeat=2)
    )

    assert total == pytest.approx(1.0)


def test_marginalizing_the_joint_reproduces_the_closed_form() -> None:
    """§5.4's collapse is exact, and this is the check that it is.

    The closed form is what the candidate will actually propagate; if the two
    ever disagree, the cheap path is silently a different model.
    """

    autonomous, action_conditioned = _kernels()
    identity = BranchSelfIdentity(2, null_prior=0.3)
    support = (_A, _B, _C)

    for entity in range(2):
        closed_form = entity_marginal_transition(
            autonomous[entity],
            action_conditioned[entity],
            identity.probability(entity),
        )
        for state in support:
            marginal = 0.0
            for other in support:
                successor = [state, state]
                successor[1 - entity] = other
                marginal += joint_transition_probability(
                    autonomous, action_conditioned, successor, identity
                )
            assert marginal == pytest.approx(closed_form.get(state, 0.0))


def test_independent_bernoulli_selves_would_disagree_on_the_joint() -> None:
    """The design §5.4 rules out, shown to be a different model.

    Independent per-entity flags let two entities respond to the same action at
    once. That joint outcome has strictly positive probability under the wrong
    model and zero under the right one, which is what makes the prohibition a
    substantive constraint rather than a stylistic one.
    """

    autonomous, action_conditioned = _kernels()
    identity = BranchSelfIdentity(2, null_prior=0.3)

    # Both entities land where only their action-conditioned kernel sends them.
    both_action_only = [_B, _A]
    correct = joint_transition_probability(
        autonomous, action_conditioned, both_action_only, identity
    )

    # Under the correct mixture, entity 1 reaching _A requires the action term,
    # but the action term is available to at most one entity, so entity 0 must
    # have moved autonomously -- and _B is reachable autonomously, so this is
    # not zero.  The independent model adds a term the mixture cannot contain:
    independent = (
        identity.probability(0)
        * identity.probability(1)
        * action_conditioned[0][_B]
        * action_conditioned[1][_A]
    )
    assert independent > 0.0

    # The correct joint contains no π_0·π_1 cross term at all.
    without_cross_term = (
        identity.null_probability * autonomous[0][_B] * autonomous[1].get(_A, 0.0)
        + identity.probability(0)
        * action_conditioned[0][_B]
        * autonomous[1].get(_A, 0.0)
        + identity.probability(1)
        * action_conditioned[1][_A]
        * autonomous[0][_B]
    )
    assert correct == pytest.approx(without_cross_term)
    assert correct != pytest.approx(without_cross_term + independent)


def test_a_certain_null_label_reduces_to_pure_autonomous_motion() -> None:
    autonomous, action_conditioned = _kernels()
    identity = BranchSelfIdentity(2, null_prior=0.5)
    identity.update((0.0, 0.0), null_likelihood=1.0)

    assert identity.null_probability == pytest.approx(1.0)
    for entity in range(2):
        assert entity_marginal_transition(
            autonomous[entity], action_conditioned[entity], identity.probability(entity)
        ) == pytest.approx(autonomous[entity])
