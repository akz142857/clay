"""Which tracked entity, if any, is the agent itself -- decided per branch.

Increment I3 of the O3 candidate (architecture: stochastic permanence plan
§5.4).  The historical I1 stack carries a self posterior, but not a
branch-local one, and §5.4 forbids two specific shortcuts that a
non-branch-local design falls into naturally:

- feeding a self probability aggregated *across* hypotheses back into each
  branch, which lets one branch's evidence steer another's dynamics;
- giving each entity its own independent Bernoulli "am I the self" flag, which
  lets several entities respond to the same action at once.

A categorical over ``{null} ∪ entities`` rules both out by construction:
``π_h0 + Σ_i π_hi = 1`` means at most one entity can be the self, and the
distribution belongs to the branch that owns it.

The joint transition is the mixture §5.4 specifies:

    P({s'_i} | h) = π_h0 · Π_i P_auto(s'_i)
                  + Σ_j π_hj · P_action(s'_j) · Π_{i≠j} P_auto(s'_i)

Every term factorizes, so the single-entity marginal collapses to
``π_hi·P_action + (1 - π_hi)·P_auto`` -- cheap, and exact rather than an
approximation.  ``joint_transition_probability`` is kept alongside it precisely
so a test can marginalize the joint and check the collapse really holds; an
independent-Bernoulli design agrees on the marginal and disagrees on the joint,
so the marginal alone would not detect it.

NOT part of the Phase-0/Phase-R source lock: no locked entry point imports it,
and nothing here modifies a locked module.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from cal.model.stochastic_motion_filter import GridSpec

_State = tuple[tuple[int, int], tuple[int, int]]

# The world's five grid actions; index 0 is "stay".
ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def action_successors(
    position: tuple[int, int],
    velocity: tuple[int, int],
    static_probability: np.ndarray,
    *,
    action: int,
    spec: GridSpec,
) -> tuple[tuple[tuple[int, int], tuple[int, int], float], ...]:
    """Successors under "this entity is the self and just took ``action``".

    The world moves the agent by the action delta, clipped to the arena, and
    refuses the move if the target is static.  Velocity is untouched: the agent
    has no velocity in the world at all, so inventing one here would be a claim
    the observation model cannot support.

    Under an uncertain topology the move succeeds with probability
    ``1 - m(target)``, mirroring the autonomous kernel's treatment of blocked
    cells rather than hard-selecting an occupancy.
    """

    if not 0 <= int(action) < len(ACTION_DELTAS):
        raise ValueError("action outside the five-way grid action set")
    grid = np.asarray(static_probability, dtype=np.float64)
    if grid.shape != (spec.grid_size, spec.grid_size):
        raise ValueError("static_probability must cover the grid")
    if not np.all(np.isfinite(grid)) or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("static_probability must be finite in [0, 1]")

    delta = ACTION_DELTAS[int(action)]
    target = (
        int(np.clip(position[0] + delta[0], spec.arena_low, spec.arena_high)),
        int(np.clip(position[1] + delta[1], spec.arena_low, spec.arena_high)),
    )
    if target == position:
        return ((position, velocity, 1.0),)

    blocked = float(grid[target[1], target[0]])
    successors: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    if blocked < 1.0:
        successors.append((target, velocity, 1.0 - blocked))
    if blocked > 0.0:
        successors.append((position, velocity, blocked))
    return tuple(sorted(successors))


class BranchSelfIdentity:
    """Categorical ``P(self = null | entity_i)`` owned by one hypothesis."""

    def __init__(self, entity_count: int, *, null_prior: float = 0.5) -> None:
        if entity_count < 0:
            raise ValueError("entity_count must be non-negative")
        if not 0.0 < null_prior < 1.0:
            raise ValueError("null_prior must be strictly inside (0, 1)")
        self._weights = np.empty(entity_count + 1, dtype=np.float64)
        self._weights[0] = null_prior
        if entity_count:
            self._weights[1:] = (1.0 - null_prior) / entity_count

    @property
    def entity_count(self) -> int:
        return int(self._weights.size - 1)

    @property
    def null_probability(self) -> float:
        return float(self._weights[0])

    def probability(self, entity: int) -> float:
        """``π_hi`` -- the chance entity ``entity`` is the self in this branch."""

        if not 0 <= entity < self.entity_count:
            raise IndexError("entity outside this branch")
        return float(self._weights[entity + 1])

    def probabilities(self) -> np.ndarray:
        """``(π_h0, π_h1, ...)`` as a copy; callers must not alias it."""

        return self._weights.copy()

    def update(
        self, entity_likelihoods: Sequence[float], null_likelihood: float
    ) -> float:
        """Fold one step of self-evidence in; return the branch log evidence.

        §5.4 permits the label to move within an episode on sensor/action
        evidence.  What it does not permit is the frozen kernel moving, so this
        touches only the categorical.
        """

        likelihoods = np.asarray(
            (null_likelihood, *entity_likelihoods), dtype=np.float64
        )
        if likelihoods.size != self._weights.size:
            raise ValueError("one likelihood per class is required")
        if not np.all(np.isfinite(likelihoods)) or np.any(likelihoods < 0.0):
            raise ValueError("likelihoods must be finite and non-negative")
        posterior = self._weights * likelihoods
        evidence = float(posterior.sum())
        if evidence <= 0.0:
            raise ValueError("self-identity evidence vanished for every class")
        self._weights = posterior / evidence
        return float(np.log(evidence))


def entity_marginal_transition(
    autonomous: Mapping[_State, float],
    action_conditioned: Mapping[_State, float],
    self_probability: float,
) -> dict[_State, float]:
    """``π_hi·P_action + (1 - π_hi)·P_auto`` for one entity.

    Exact, not an approximation: every term of the §5.4 joint factorizes, so
    marginalizing the other entities away leaves precisely this.
    """

    if not 0.0 <= self_probability <= 1.0:
        raise ValueError("self_probability must be in [0, 1]")
    merged: dict[_State, float] = {}
    for state, mass in autonomous.items():
        merged[state] = merged.get(state, 0.0) + (1.0 - self_probability) * mass
    for state, mass in action_conditioned.items():
        merged[state] = merged.get(state, 0.0) + self_probability * mass
    return {state: mass for state, mass in merged.items() if mass > 0.0}


def joint_transition_probability(
    autonomous: Sequence[Mapping[_State, float]],
    action_conditioned: Sequence[Mapping[_State, float]],
    successor: Sequence[_State],
    identity: BranchSelfIdentity,
) -> float:
    """``P({s'_i} | h)`` under the at-most-one-self mixture.

    Present so the marginal collapse can be *tested* rather than asserted: an
    independent-Bernoulli-per-entity design reproduces the same single-entity
    marginal while giving a different joint, so only this distinguishes them.
    """

    count = identity.entity_count
    if not (len(autonomous) == len(action_conditioned) == len(successor) == count):
        raise ValueError("autonomous/action/successor must cover every entity")

    autonomous_masses = [
        float(autonomous[index].get(successor[index], 0.0)) for index in range(count)
    ]
    total = identity.null_probability
    for mass in autonomous_masses:
        total *= mass

    for owner in range(count):
        weight = identity.probability(owner)
        if weight <= 0.0:
            continue
        term = weight * float(
            action_conditioned[owner].get(successor[owner], 0.0)
        )
        for other in range(count):
            if other != owner:
                term *= autonomous_masses[other]
        total += term
    return float(total)
