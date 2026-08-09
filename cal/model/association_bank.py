"""Global association hypotheses with weights (O3, increment I4b).

Plan §5.6 requires the hypothesis bank to survive into the new candidate, and
§5.3 fixes what a branch weight is allowed to contain.  I4 shipped a single
branch, which cannot represent association ambiguity at all: when two
detections are both plausible continuations of one track, a single branch has
to guess, and a wrong guess is unrecoverable because the discarded alternative
no longer exists.  This is the bank.

Association uncertainty lives in ``w_h`` and nowhere else (§5.3).  There is no
second per-track association scalar, because two representations of the same
uncertainty multiply into double-counting the moment either is updated.

What a branch weight accumulates, once each:

    matched detection   Z_match = e_pred · Σ_s q_pred(s) P_D(s) P(d|s)
    unmatched track     Z_miss  = (1 - e_pred) + e_pred · L_no
    unmatched detection λ_birth·b(d)  and  λ_clutter·c(d), as *alternatives*

This world makes the emission exact rather than modelled: a visible cell
renders whatever occupies it, so ``P_D(s)`` is the visibility of ``s`` and
``P(d|s)`` is one exactly when ``s`` sits on the detected cell.  Nothing here
tunes an emission covariance, because there is no measurement noise to tune
against.

Every branch enumerates only **one-to-one** assignments, so a detection cannot
explain two tracks inside one hypothesis.  Enumeration is over the gated
bipartite graph and the child count per parent is capped; both bounds are
declared in the config and audited as approximation loss rather than presented
as exhaustive search.

NOT part of the Phase-0/Phase-R source lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import exp, isfinite, log
from typing import Iterable, Mapping, Sequence

import numpy as np

from cal.model.branch_self_identity import BranchSelfIdentity
from cal.model.permanence_track import PermanenceTrack, TrackKernel
from cal.model.stochastic_motion_filter import EmptyPosteriorError, GridSpec

_Cell = tuple[int, int]
_Velocity = tuple[int, int]
_UNIT_VELOCITIES = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True, slots=True)
class BankKernel:
    """Frozen bank parameters; part of the candidate lock (§5.3)."""

    birth_intensity: float
    clutter_intensity: float
    maximum_hypotheses: int
    maximum_children_per_parent: int
    maximum_tracks: int
    gate_radius: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.birth_intensity:
            raise ValueError("birth_intensity must be positive")
        if not 0.0 < self.clutter_intensity:
            raise ValueError("clutter_intensity must be positive")
        if self.maximum_hypotheses < 1:
            raise ValueError("maximum_hypotheses must be positive")
        if self.maximum_children_per_parent < 1:
            raise ValueError("maximum_children_per_parent must be positive")
        if self.maximum_tracks < 1:
            raise ValueError("maximum_tracks must be positive")
        if self.gate_radius < 0:
            raise ValueError("gate_radius must be non-negative")


@dataclass
class AssociationHypothesis:
    """One global interpretation: which detections belong to which entity."""

    tracks: list[PermanenceTrack] = field(default_factory=list)
    last_positions: list[_Cell | None] = field(default_factory=list)
    identity: BranchSelfIdentity = field(
        default_factory=lambda: BranchSelfIdentity(0)
    )
    log_weight: float = 0.0

    def occupancy(self, grid_size: int) -> np.ndarray:
        """``1 - Π_i (1 - e_hi·q_hi(cell))`` inside this branch."""

        free = np.ones((grid_size, grid_size), dtype=np.float64)
        for track in self.tracks:
            for (x, y), mass in track.occupancy().items():
                free[y, x] *= 1.0 - min(max(mass, 0.0), 1.0)
        return as_probability_field(1.0 - free)


# The Bernoulli union is analytically inside [0, 1]; accumulating
# `1 - Π(1 - p)` in floating point can land one ULP outside it.  Clipping that
# away restores an invariant the formula already guarantees -- but only that
# much.  A larger excess is a modelling error, and silently clipping it would
# turn a broken posterior into a plausible-looking one.
_PROBABILITY_ROUNDING_SLACK = 1e-9


def as_probability_field(values: np.ndarray) -> np.ndarray:
    """Clip representation error off a probability field; refuse real error."""

    array = np.asarray(values, dtype=np.float64)
    excess = float(max(array.max() - 1.0, -array.min(), 0.0))
    if excess > _PROBABILITY_ROUNDING_SLACK:
        raise ValueError(
            f"occupancy field leaves [0, 1] by {excess:.3g}, which is too "
            f"large to be rounding"
        )
    return np.clip(array, 0.0, 1.0)


def _velocity_from(previous: _Cell | None, current: _Cell) -> _Velocity | None:
    """Observed velocity, or ``None`` when the displacement does not reveal one.

    ``None`` is not a failure case to paper over with a default direction: it
    is the honest state after a single detection, and the caller answers it
    with a uniform prior instead of an assertion.
    """

    if previous is None:
        return None
    delta = (current[0] - previous[0], current[1] - previous[1])
    return delta if delta in _UNIT_VELOCITIES else None


class GlobalAssociationBank:
    """A weighted, bounded set of association hypotheses."""

    def __init__(
        self,
        *,
        spec: GridSpec,
        kernel: BankKernel,
        track_kernel: TrackKernel,
        k_max: int,
        self_null_prior: float,
    ) -> None:
        self.spec = spec
        self.kernel = kernel
        self._track_kernel = track_kernel
        self._k_max = int(k_max)
        self._self_null_prior = float(self_null_prior)
        self.hypotheses: list[AssociationHypothesis] = [AssociationHypothesis()]
        self.discarded_hypothesis_mass = 0.0

    # -- weights ----------------------------------------------------------

    def weights(self) -> np.ndarray:
        """Normalized ``w_h``; the only place association uncertainty lives."""

        if not self.hypotheses:
            return np.zeros(0, dtype=np.float64)
        log_weights = np.asarray(
            [item.log_weight for item in self.hypotheses], dtype=np.float64
        )
        shifted = log_weights - log_weights.max()
        weights = np.exp(shifted)
        total = float(weights.sum())
        if total <= 0.0:
            raise EmptyPosteriorError("every association hypothesis has zero weight")
        return weights / total

    # -- stepping ---------------------------------------------------------

    def step(
        self,
        *,
        detections: Sequence[_Cell],
        static_probability: np.ndarray,
        no_detection_probability: np.ndarray,
        visible: np.ndarray,
        turn_probability: float,
        allow_turn: bool,
        action: int,
        self_likelihood_ratio,
    ) -> None:
        """Expand every hypothesis over its gated assignments, then prune."""

        children: list[tuple[float, AssociationHypothesis]] = []
        for parent in self.hypotheses:
            children.extend(
                self._expand(
                    parent,
                    detections=detections,
                    static_probability=static_probability,
                    no_detection_probability=no_detection_probability,
                    visible=visible,
                    turn_probability=turn_probability,
                    allow_turn=allow_turn,
                    action=action,
                    self_likelihood_ratio=self_likelihood_ratio,
                )
            )
        if not children:
            raise EmptyPosteriorError("no association hypothesis survived the step")
        self._prune(children)

    def _expand(
        self,
        parent: AssociationHypothesis,
        *,
        detections: Sequence[_Cell],
        static_probability: np.ndarray,
        no_detection_probability: np.ndarray,
        visible: np.ndarray,
        turn_probability: float,
        allow_turn: bool,
        action: int,
        self_likelihood_ratio,
    ) -> list[tuple[float, AssociationHypothesis]]:
        predicted = [
            track.predicted_states(
                static_probability=static_probability,
                turn_probability=turn_probability,
                allow_turn=allow_turn,
            )
            for track in parent.tracks
        ]
        # Gate first: a track may only claim a detection its predicted support
        # can actually reach, which is what keeps enumeration bounded.
        options: list[list[int | None]] = []
        for index, track in enumerate(parent.tracks):
            reachable = [
                position
                for position, cell in enumerate(detections)
                if self._match_mass(predicted[index], cell) > 0.0
            ]
            options.append([*reachable, None])

        candidates: list[tuple[float, list[int | None]]] = []
        for combination in product(*options) if options else [()]:
            claimed = [item for item in combination if item is not None]
            if len(claimed) != len(set(claimed)):
                continue  # one-to-one within a hypothesis
            score = self._assignment_log_weight(
                parent,
                predicted,
                combination,
                detections=detections,
                no_detection_probability=no_detection_probability,
            )
            if score is not None:
                candidates.append((score, list(combination)))
        if not candidates:
            return []

        # Total order: score first, then a canonical rendering of the
        # assignment so ties break deterministically rather than by whatever
        # order the product happened to emit.
        candidates.sort(
            key=lambda item: (
                -item[0],
                tuple(-1 if choice is None else choice for choice in item[1]),
            )
        )
        capped = candidates[: self.kernel.maximum_children_per_parent]

        children: list[tuple[float, AssociationHypothesis]] = []
        for score, combination in capped:
            child = self._commit(
                parent,
                combination,
                detections=detections,
                static_probability=static_probability,
                no_detection_probability=no_detection_probability,
                turn_probability=turn_probability,
                allow_turn=allow_turn,
                action=action,
                self_likelihood_ratio=self_likelihood_ratio,
            )
            if child is None:
                continue
            child.log_weight = parent.log_weight + score
            children.append((child.log_weight, child))
        return children

    def _match_mass(
        self, predicted: Mapping[tuple[_Cell, _Velocity], float], cell: _Cell
    ) -> float:
        return sum(
            mass for (position, _v), mass in predicted.items() if position == cell
        )

    def _assignment_log_weight(
        self,
        parent: AssociationHypothesis,
        predicted: Sequence[Mapping[tuple[_Cell, _Velocity], float]],
        combination: Sequence[int | None],
        *,
        detections: Sequence[_Cell],
        no_detection_probability: np.ndarray,
    ) -> float | None:
        """Each normalizer and assignment prior enters exactly once (§5.3)."""

        total = 0.0
        for index, choice in enumerate(combination):
            track = parent.tracks[index]
            existence = track.predicted_existence
            if choice is not None:
                # Exact emission: P_D·P(d|s) is the predicted mass on the cell.
                evidence = existence * self._match_mass(
                    predicted[index], detections[choice]
                )
            else:
                l_no = sum(
                    mass * float(no_detection_probability[position[1], position[0]])
                    for (position, _v), mass in predicted[index].items()
                )
                evidence = (1.0 - existence) + existence * l_no
            if evidence <= 0.0 or not isfinite(evidence):
                return None
            total += log(evidence)

        claimed = {item for item in combination if item is not None}
        unmatched = len(detections) - len(claimed)
        if unmatched:
            # Birth and clutter are alternatives for the same detection, so the
            # branch carries their sum rather than choosing one silently.
            total += unmatched * log(
                self.kernel.birth_intensity + self.kernel.clutter_intensity
            )
        return total

    def _commit(
        self,
        parent: AssociationHypothesis,
        combination: Sequence[int | None],
        *,
        detections: Sequence[_Cell],
        static_probability: np.ndarray,
        no_detection_probability: np.ndarray,
        turn_probability: float,
        allow_turn: bool,
        action: int,
        self_likelihood_ratio,
    ) -> AssociationHypothesis | None:
        child = AssociationHypothesis()
        ratios: list[float] = []
        for index, choice in enumerate(combination):
            source = parent.tracks[index]
            track = self._clone(source)
            previous = parent.last_positions[index]
            if choice is not None:
                cell = detections[choice]
                ratios.append(
                    self_likelihood_ratio(previous, cell, static_probability, action)
                )
                velocity = _velocity_from(previous, cell)
                if velocity is None:
                    track.reset_unknown_velocity(cell)
                else:
                    track.reset(cell, velocity)
                child.tracks.append(track)
                child.last_positions.append(cell)
                continue
            try:
                track.step_unobserved(
                    static_probability=static_probability,
                    no_detection_probability=no_detection_probability,
                    turn_probability=turn_probability,
                    allow_turn=allow_turn,
                )
            except EmptyPosteriorError:
                continue
            if track.retired():
                continue
            child.tracks.append(track)
            child.last_positions.append(None)
            ratios.append(1.0)

        claimed = {item for item in combination if item is not None}
        for position, cell in enumerate(detections):
            if position in claimed:
                continue
            if len(child.tracks) >= self.kernel.maximum_tracks:
                break
            track = PermanenceTrack(
                k_max=self._k_max, spec=self.spec, kernel=self._track_kernel
            )
            # e = 1 in the birth branch: the branch prior already carries the
            # birth intensity, so an existence boost would count it twice.
            # Velocity is unobserved at birth, so it stays a uniform prior.
            track.reset_unknown_velocity(cell)
            child.tracks.append(track)
            child.last_positions.append(cell)
            ratios.append(1.0)

        child.identity = BranchSelfIdentity(
            len(child.tracks), null_prior=self._self_null_prior
        )
        if child.tracks:
            child.identity.update(ratios, null_likelihood=1.0)
        return child

    def _clone(self, source: PermanenceTrack) -> PermanenceTrack:
        clone = PermanenceTrack(
            k_max=self._k_max, spec=self.spec, kernel=self._track_kernel
        )
        clone._filter.codes[:] = source._filter.codes
        clone._filter.probability[:] = source._filter.probability
        clone._filter.count = source._filter.count
        clone._filter.branch_log_weight = source._filter.branch_log_weight
        clone._filter.cumulative_retained_probability = (
            source._filter.cumulative_retained_probability
        )
        clone.existence = source.existence
        clone.branch_log_weight = source.branch_log_weight
        return clone

    def _prune(self, children: Sequence[tuple[float, AssociationHypothesis]]) -> None:
        ordered = sorted(children, key=lambda item: -item[0])
        keep = ordered[: self.kernel.maximum_hypotheses]
        dropped = ordered[self.kernel.maximum_hypotheses :]
        if dropped:
            # Pruned hypotheses are storage loss, not disconfirmed ones; the
            # mass is recorded so the audit can see what the bound cost.
            peak = ordered[0][0]
            total = sum(exp(score - peak) for score, _ in ordered)
            lost = sum(exp(score - peak) for score, _ in dropped)
            self.discarded_hypothesis_mass = max(
                self.discarded_hypothesis_mass, lost / total
            )
        self.hypotheses = [item for _score, item in keep]

    # -- readout ----------------------------------------------------------

    def occupancy(self) -> np.ndarray:
        """``Σ_h w_h · [1 - Π_i (1 - e_hi·q_hi(cell))]`` (§5.5)."""

        weights = self.weights()
        total = np.zeros((self.spec.grid_size, self.spec.grid_size), dtype=np.float64)
        for weight, hypothesis in zip(weights, self.hypotheses):
            total += weight * hypothesis.occupancy(self.spec.grid_size)
        return as_probability_field(total)

    def most_likely(self) -> AssociationHypothesis:
        weights = self.weights()
        return self.hypotheses[int(np.argmax(weights))]

    def track_positions(self) -> tuple[_Cell, ...]:
        """MAP cells of the most likely hypothesis, canonically ordered."""

        positions: list[_Cell] = []
        for track in self.most_likely().tracks:
            if track.retired():
                continue
            marginal = track.position_marginal()
            if marginal:
                positions.append(
                    max(sorted(marginal), key=lambda cell: marginal[cell])
                )
        return tuple(sorted(positions))
