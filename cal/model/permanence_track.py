"""One tracked entity: a kinematic posterior plus a Bernoulli existence term.

Increment I2 of the O3 candidate (architecture: stochastic permanence plan
§5.3).  ``PackedKinematicFilter`` already conditions a bounded posterior on
non-detection, and ``bayesian_no_detection_update`` already implements the
existence arithmetic, but nothing joined them -- the existence update had no
callers at all (review finding F21).  This is the join.

The composition matters more than the arithmetic, because the obvious way to
write it double-counts.  The filter's ``step`` already multiplies the predicted
posterior by ``P(no_detection | s)`` and renormalizes, so the
``observation_evidence`` it reports *is* ``L_no``, and its ``branch_log_weight``
is the existence-free branch weight.  Recomputing either on top would count the
same evidence twice.  So this reads ``L_no`` off the filter and adds only what
the filter cannot know:

    e_post           = e_pred · L_no / ((1 - e_pred) + e_pred · L_no)
    branch weight   += log((1 - e_pred) + e_pred · L_no) + log(retained)

The second term is the pruning loss the filter reports separately.  Mass
discarded by a bounded posterior is mass this branch failed to explain, so it
belongs in the weight; leaving it out would make a truncated branch look better
than an untruncated one.

Not persisting a track is a bounded-storage operation, never evidence of death
(§5.3), so nothing here decays existence outside the likelihood.

NOT part of the Phase-0/Phase-R source lock: no locked entry point imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

from cal.model.stochastic_motion_filter import (
    EmptyPosteriorError,
    GridSpec,
    PackedKinematicFilter,
)


@dataclass(frozen=True, slots=True)
class TrackKernel:
    """Frozen survival/detection parameters.

    These belong to the candidate lock: §5.3 puts survival hazard, detection
    probability and the retirement threshold in the canonical config, so they
    are fitted once from the train stream and never touched again.
    """

    survival_probability: float
    retirement_existence: float

    def __post_init__(self) -> None:
        if not 0.0 < self.survival_probability <= 1.0:
            raise ValueError("survival_probability must be in (0, 1]")
        if not 0.0 <= self.retirement_existence < 1.0:
            raise ValueError("retirement_existence must be in [0, 1)")


class PermanenceTrack:
    """``q(state | exists)`` under a bounded filter, with existence carried."""

    def __init__(
        self,
        *,
        k_max: int,
        spec: GridSpec,
        kernel: TrackKernel,
        probability_dtype: np.dtype[np.floating[Any]] = np.dtype(np.float32),
    ) -> None:
        self._filter = PackedKinematicFilter(
            k_max, spec=spec, probability_dtype=probability_dtype
        )
        self.kernel = kernel
        self.existence = 0.0
        self.branch_log_weight = 0.0

    @property
    def spec(self) -> GridSpec:
        return self._filter.spec

    @property
    def cumulative_pruned_mass(self) -> float:
        return self._filter.cumulative_pruned_mass

    def reset(
        self,
        position: tuple[int, int],
        velocity: tuple[int, int],
        *,
        existence: float = 1.0,
    ) -> None:
        """Start a track from a detection.

        ``existence = 1`` is the birth-branch value from §5.3: the branch prior
        already carries the birth intensity, so boosting existence again would
        weight the same event twice.
        """

        if not 0.0 < existence <= 1.0:
            raise ValueError("initial existence must be in (0, 1]")
        self._filter.reset(position, velocity)
        self.existence = float(existence)
        self.branch_log_weight = 0.0

    def step_unobserved(
        self,
        *,
        static_probability: np.ndarray,
        no_detection_probability: np.ndarray,
        turn_probability: float,
        allow_turn: bool,
    ) -> dict[str, float | int]:
        """Propagate one step and condition on having seen nothing."""

        if self.existence <= 0.0:
            raise EmptyPosteriorError("track has no existence mass")
        # Prediction: survival first, so the observation acts on e_pred.
        existence_predicted = self.kernel.survival_probability * self.existence

        step = self._filter.step(
            static_probability=static_probability,
            no_detection_probability=no_detection_probability,
            turn_probability=turn_probability,
            allow_turn=allow_turn,
        )
        l_no = float(step["observation_evidence"])
        retained = float(step["retained_probability"])

        evidence = (1.0 - existence_predicted) + existence_predicted * l_no
        if not np.isfinite(evidence) or evidence <= 0.0:
            raise EmptyPosteriorError("no-detection observation has zero evidence")

        self.existence = existence_predicted * l_no / evidence
        self.branch_log_weight += log(evidence) + log(retained)
        return {
            **step,
            "existence_predicted": existence_predicted,
            "existence_posterior": self.existence,
            "no_detection_evidence": l_no,
            "branch_evidence": evidence,
        }

    def detection_evidence(self, emission_probability: np.ndarray) -> float:
        """``Z_match`` for a matched detection, without applying it.

        §5.3 fixes this as ``e_pred · Σ_s q_pred(s) P_D(s) P(d|s)``.  It is
        exposed separately because the assignment that decides *whether* to
        apply it is a global-hypothesis decision (§5.4), not this track's.
        """

        emission = np.asarray(emission_probability, dtype=np.float64)
        spec = self._filter.spec
        if emission.shape != (spec.grid_size, spec.grid_size):
            raise ValueError("emission probability must cover the grid")
        if not np.all(np.isfinite(emission)) or np.any(
            (emission < 0.0) | (emission > 1.0)
        ):
            raise ValueError("emission probability must be finite in [0, 1]")

        existence_predicted = self.kernel.survival_probability * self.existence
        total = 0.0
        for code, mass in self._filter.items():
            (x, y), _velocity = spec.decode(code)
            total += mass * float(emission[y, x])
        return existence_predicted * total

    def position_marginal(self) -> dict[tuple[int, int], float]:
        """Position marginal of ``q``, i.e. conditioned on the entity existing."""

        return self._filter.position_marginal()

    def occupancy(self) -> dict[tuple[int, int], float]:
        """Unconditional per-cell occupancy, ``e · q(cell)``.

        This is the quantity §5.5 unions across entities; the conditional
        marginal alone would claim presence a track no longer believes in.
        """

        return {
            cell: self.existence * mass
            for cell, mass in self._filter.position_marginal().items()
        }

    def retired(self) -> bool:
        """True when existence has fallen through the frozen threshold.

        Retirement is a storage decision.  Its discarded probability is audited
        as approximation loss elsewhere and must not be reported as observed
        death evidence.
        """

        return self.existence < self.kernel.retirement_existence
