"""The I1-P1 stochastic permanence candidate, assembled (O3, increment I4).

Joins the three preceding increments into something
``run_candidate_lifecycle`` accepts: the sensor-only static map (I1), the
existence-carrying track (I2), and branch-local self identity (I3), plus the
occupancy synthesis of plan §5.5 and the frozen-kernel lifecycle of §7.

## The turn probability cannot be watched, only inferred

``RandomizedOcclusionWorld._maybe_hidden_turn`` returns early when the camera
sees the entity, so **hidden maneuvers never occur while visible**.  A
candidate therefore cannot estimate the turn probability by watching
trajectories; there is nothing to watch.  The only sensor-only route is
indirect: an entity that vanishes at a known cell with a known velocity and
reappears ``k`` steps later somewhere else constrains how much turning happened
in between.  ``fit_kernel`` maximises the likelihood of those reappearances
over a grid of turn probabilities, propagating belief with the very kernel the
candidate will use.

Fitting uses only intervals where exactly one entity is missing, so the
reappearance is unambiguous without solving association first.  That is a
restriction on the fitting data, not a privileged input.

## Scope, stated plainly

The association bank is now multi-hypothesis (I4b): ``GlobalAssociationBank``
carries weighted branches, each owning its own track set and
``BranchSelfIdentity``.  Two bounds remain approximations rather than exhaustive
search -- the hypothesis count and the children generated per parent -- and both
are declared in the fitted kernel and audited through
``discarded_hypothesis_mass`` rather than presented as exact.

What is still missing before any confirmatory claim: the candidate has never
been scored against the twelve gates (increment I5).

NOT part of the Phase-0/Phase-R source lock: the candidate is injected through
``CandidateFactory``, so no locked entry point imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cal.model.branch_self_identity import (
    ACTION_DELTAS,
    BranchSelfIdentity,
    action_successors,
)
from cal.model.association_bank import (
    BankKernel,
    GlobalAssociationBank,
    as_probability_field,
)
from cal.model.permanence_track import TrackKernel
from cal.model.sensor_only_static_map import (
    SensorOnlyStaticMap,
    StaticMapKernel,
    fit_static_map_kernel,
)
from cal.model.stochastic_motion_filter import (
    EmptyPosteriorError,
    GridSpec,
    autonomous_successors,
)

# Turn probabilities the fitter searches.  A grid rather than a continuous
# optimum keeps the fitted value reproducible across platforms.
TURN_PROBABILITY_GRID: tuple[float, ...] = tuple(
    round(0.05 * step, 2) for step in range(1, 20)
)


def _global_masks(
    sensed: np.ndarray, visibility: np.ndarray, *, spec: GridSpec, camera: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Lift the camera-centred patches onto the full grid."""

    patch = np.asarray(sensed)
    mask = np.asarray(visibility).astype(bool)
    occupied = np.zeros((spec.grid_size, spec.grid_size), dtype=bool)
    visible = np.zeros((spec.grid_size, spec.grid_size), dtype=bool)
    radius = patch.shape[0] // 2
    x0, y0 = camera[0] - radius, camera[1] - radius
    occupied[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] = patch != 0
    visible[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
    return occupied, visible


def _arena_cells(
    mask: np.ndarray, spec: GridSpec
) -> tuple[tuple[int, int], ...]:
    rows, columns = np.nonzero(mask)
    return tuple(
        (int(column), int(row))
        for row, column in zip(rows, columns)
        if spec.arena_low <= column <= spec.arena_high
        and spec.arena_low <= row <= spec.arena_high
    )


class StochasticPermanenceCandidate:
    """One episode's belief state.  Constructed fresh per episode by §7."""

    def __init__(self, frozen_kernel: Mapping[str, Any], model_seed: int) -> None:
        self.model_seed = int(model_seed)
        self._kernel = frozen_kernel
        self.spec = GridSpec(
            grid_size=int(frozen_kernel["grid_size"]),
            arena_low=int(frozen_kernel["arena_low"]),
            arena_high=int(frozen_kernel["arena_high"]),
        )
        self.camera = (int(frozen_kernel["camera_x"]), int(frozen_kernel["camera_y"]))
        self.turn_probability = float(frozen_kernel["turn_probability"])
        self._k_max = int(frozen_kernel["k_max"])
        self._track_kernel = TrackKernel(
            survival_probability=float(frozen_kernel["survival_probability"]),
            retirement_existence=float(frozen_kernel["retirement_existence"]),
        )
        self._static = SensorOnlyStaticMap(
            grid_size=self.spec.grid_size,
            arena_low=self.spec.arena_low,
            arena_high=self.spec.arena_high,
            camera=self.camera,
            kernel=StaticMapKernel(
                prior=float(frozen_kernel["static_prior"]),
                mover_occupancy=float(frozen_kernel["static_mover_occupancy"]),
                clamp=float(frozen_kernel["static_clamp"]),
            ),
        )
        self._bank = GlobalAssociationBank(
            spec=self.spec,
            kernel=BankKernel(
                birth_intensity=float(frozen_kernel["birth_intensity"]),
                clutter_intensity=float(frozen_kernel["clutter_intensity"]),
                maximum_hypotheses=int(frozen_kernel["max_hypotheses"]),
                maximum_children_per_parent=int(
                    frozen_kernel["max_children_per_parent"]
                ),
                maximum_tracks=int(frozen_kernel["max_tracks"]),
            ),
            track_kernel=self._track_kernel,
            k_max=self._k_max,
            self_null_prior=float(frozen_kernel["self_null_prior"]),
        )
        self._steps = 0

    # -- observation ------------------------------------------------------

    def observe(
        self, sensed: np.ndarray, visibility: np.ndarray, action: int
    ) -> None:
        """Fold one (observation, action) pair into every hypothesis."""

        self._static.update(sensed, visibility)
        static_probability = self._static.probabilities()
        occupied, visible = _global_masks(
            sensed, visibility, spec=self.spec, camera=self.camera
        )
        detections = _arena_cells(occupied & visible, self.spec)

        # A visible cell renders whatever occupies it, so failing to detect
        # there rules the cell out entirely; a hidden cell says nothing.
        no_detection = np.ones(
            (self.spec.grid_size, self.spec.grid_size), dtype=np.float64
        )
        no_detection[visible] = 0.0

        self._bank.step(
            detections=detections,
            static_probability=static_probability,
            no_detection_probability=no_detection,
            visible=visible,
            turn_probability=self.turn_probability,
            allow_turn=self._steps > 0,
            action=int(action),
            self_likelihood_ratio=self._self_likelihood_ratio,
        )
        self._steps += 1

    def _self_likelihood_ratio(
        self,
        prior_states: Mapping[tuple[tuple[int, int], tuple[int, int]], float],
        autonomous_mass: float,
        observed: tuple[int, int],
        static_probability: np.ndarray,
        action: int,
    ) -> float:
        """``P_action(observed) / P_auto(observed)`` for one entity.

        The other entities' autonomous factors are common to every class and
        cancel in the categorical update, which is why the ratio suffices and
        ``null_likelihood`` is 1.

        Both sides are integrals over the track's own posterior.  The
        denominator is handed in because the bank has already computed it to
        score the assignment, and recomputing it here would let the two drift
        apart -- which is how the previous version came to evaluate the
        autonomous branch at a hardcoded velocity of ``(1, 0)`` and against a
        thresholded topology, while the numerator used neither.
        """

        if not prior_states:
            return 1.0
        action_mass = 0.0
        for (position, velocity), mass in prior_states.items():
            for successor, _velocity, transition in action_successors(
                position, velocity, static_probability, action=action, spec=self.spec
            ):
                if successor == observed:
                    action_mass += mass * transition
        if autonomous_mass <= 0.0:
            return 1.0 if action_mass <= 0.0 else float(len(ACTION_DELTAS))
        return float(action_mass / autonomous_mass)

    # -- readout ----------------------------------------------------------

    def occupancy(self) -> np.ndarray:
        """``P_occ(c) = 1 - (1 - m_t(c))·(1 - P_dynamic(c))`` over the grid."""

        static = self._static.probabilities()
        return as_probability_field(
            1.0 - (1.0 - static) * (1.0 - self.hidden_occupancy())
        )

    def hidden_occupancy(self) -> np.ndarray:
        """``Σ_h w_h·[1 - Π_i(1 - e_hi·q_hi)]`` -- what permanence is scored on."""

        return self._bank.occupancy()

    def track_positions(self) -> tuple[tuple[int, int], ...]:
        return self._bank.track_positions()

    def self_probabilities(self) -> np.ndarray:
        return self._bank.most_likely().identity.probabilities()

    def hypothesis_weights(self) -> np.ndarray:
        return self._bank.weights()

    @property
    def discarded_hypothesis_mass(self) -> float:
        """Weight the hypothesis bound cost, audited rather than hidden."""

        return self._bank.discarded_hypothesis_mass


# -- fitting --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReacquisitionEvent:
    """One vanish/reappear interval, with the visibility that shaped it."""

    origin: tuple[int, int]
    velocity: tuple[int, int]
    hidden_visibility: tuple[np.ndarray, ...]
    observed: tuple[int, int]


def _reacquisition_events(
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    spec: GridSpec,
    camera: tuple[int, int],
) -> list[ReacquisitionEvent]:
    """Unambiguous vanish/reappear intervals.

    Only intervals where exactly one tracked cell is missing are used, so the
    reappearance is identified without solving association.  Turns happen only
    while occluded, so these intervals are the only evidence about them that
    exists at all.

    The visibility of every step in between is kept, because the interval is
    not a free sample: it was *selected* by staying unseen and then being seen,
    and that selection depends on the very quantity being estimated.
    """

    events: list[ReacquisitionEvent] = []
    previous_cells: tuple[tuple[int, int], ...] = ()
    previous_velocity: dict[tuple[int, int], tuple[int, int]] = {}
    pending: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]] | None = None

    for sensed, visibility in frames:
        occupied, visible = _global_masks(
            sensed, visibility, spec=spec, camera=camera
        )
        cells = _arena_cells(occupied & visible, spec)

        if pending is not None:
            origin, velocity, masks = pending
            masks.append(visible)
            fresh = [cell for cell in cells if cell not in previous_cells]
            if len(fresh) == 1:
                events.append(
                    ReacquisitionEvent(
                        origin=origin,
                        velocity=velocity,
                        hidden_visibility=tuple(masks),
                        observed=fresh[0],
                    )
                )
                pending = None
            elif len(masks) >= 24:
                pending = None
        else:
            vanished = [cell for cell in previous_cells if cell not in cells]
            if len(vanished) == 1:
                origin = vanished[0]
                velocity = previous_velocity.get(origin)
                if velocity is not None:
                    pending = (origin, velocity, [visible])

        velocities: dict[tuple[int, int], tuple[int, int]] = {}
        for cell in cells:
            for earlier in previous_cells:
                delta = (cell[0] - earlier[0], cell[1] - earlier[1])
                if delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    velocities[cell] = delta
                    break
        previous_velocity = velocities
        previous_cells = cells
    return events


def _reacquisition_log_likelihood(
    episodes: Sequence[tuple[Sequence[ReacquisitionEvent], np.ndarray]],
    *,
    turn_probability: float,
    spec: GridSpec,
) -> float:
    """Log-likelihood of every reappearance, each under *its own* topology.

    The occluder layout is re-randomized per episode, so an event may only be
    scored against the geometry of the episode it came from.  Pooling the
    topologies -- taking, say, their union -- builds a world with more walls
    than any real episode has, and the fitted turn probability then absorbs
    that distortion instead of measuring turning.
    """

    total = 0.0
    for events, static_probability in episodes:
        for event in events:
            total += _event_log_likelihood(
                event,
                turn_probability=turn_probability,
                static_probability=static_probability,
                spec=spec,
            )
    return total


def _event_log_likelihood(
    event: ReacquisitionEvent,
    *,
    turn_probability: float,
    static_probability: np.ndarray,
    spec: GridSpec,
) -> float:
    """``P(stayed unseen, then seen at the observed cell)`` under this kernel.

    Conditioning on the non-detections is what makes the estimate identify
    anything.  An interval only exists because the entity stayed hidden and
    then reappeared, and how likely that is depends on the turn rate -- an
    entity that turns leaves the shadow at a different time and place than one
    that does not.  Scoring only the final cell throws that away and leaves the
    likelihood nearly flat in the parameter, which is what a grid search then
    faithfully reports.
    """

    belief = {(event.origin, event.velocity): 1.0}
    final = len(event.hidden_visibility) - 1
    for step, visible in enumerate(event.hidden_visibility):
        propagated: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        for (position, moving), mass in belief.items():
            for new_position, new_velocity, transition in autonomous_successors(
                position,
                moving,
                static_probability,
                turn_probability=turn_probability,
                allow_turn=step > 0,
                spec=spec,
                marginal_turn_mixture=True,
            ):
                key = (new_position, new_velocity)
                propagated[key] = propagated.get(key, 0.0) + mass * transition
        if step < final:
            # Still hidden: every visible cell is ruled out.
            belief = {
                key: mass
                for key, mass in propagated.items()
                if not visible[key[0][1], key[0][0]]
            }
        else:
            belief = {
                key: mass
                for key, mass in propagated.items()
                if key[0] == event.observed
            }
        if not belief:
            return log(1e-12)
    return log(max(sum(belief.values()), 1e-12))


class StochasticPermanenceCandidateFactory:
    """``fit_kernel`` once, ``new_episode`` per episode -- the §7 lifecycle."""

    def __init__(
        self,
        *,
        grid_size: int,
        arena_low: int,
        arena_high: int,
        camera: tuple[int, int],
        k_max: int = 96,
        max_tracks: int = 8,
        max_hypotheses: int = 5,
        max_children_per_parent: int = 4,
        birth_intensity: float = 0.1,
        clutter_intensity: float = 0.01,
        survival_probability: float = 0.995,
        retirement_existence: float = 0.02,
        self_null_prior: float = 0.5,
    ) -> None:
        self._spec = GridSpec(
            grid_size=grid_size, arena_low=arena_low, arena_high=arena_high
        )
        self._camera = (int(camera[0]), int(camera[1]))
        self._k_max = int(k_max)
        self._max_tracks = int(max_tracks)
        self._max_hypotheses = int(max_hypotheses)
        self._max_children_per_parent = int(max_children_per_parent)
        self._birth_intensity = float(birth_intensity)
        self._clutter_intensity = float(clutter_intensity)
        self._survival_probability = float(survival_probability)
        self._retirement_existence = float(retirement_existence)
        self._self_null_prior = float(self_null_prior)

    def fit_kernel(self, train_sensor_streams: Sequence[object]) -> dict[str, Any]:
        streams = [list(stream) for stream in train_sensor_streams]  # type: ignore[arg-type]
        if not streams:
            raise ValueError("no train sensor streams supplied")

        static_kernel = fit_static_map_kernel(
            streams,
            arena_low=self._spec.arena_low,
            arena_high=self._spec.arena_high,
            camera=self._camera,
            grid_size=self._spec.grid_size,
        )

        # Each episode carries its own layout, so events and topology stay
        # paired rather than pooled.
        episodes: list[
            tuple[
                list[tuple[tuple[int, int], tuple[int, int], int, tuple[int, int]]],
                np.ndarray,
            ]
        ] = []
        for stream in streams:
            belief = SensorOnlyStaticMap(
                grid_size=self._spec.grid_size,
                arena_low=self._spec.arena_low,
                arena_high=self._spec.arena_high,
                camera=self._camera,
                kernel=static_kernel,
            )
            for sensed, visibility in stream:
                belief.update(sensed, visibility)
            episode_events = _reacquisition_events(
                stream, spec=self._spec, camera=self._camera
            )
            if episode_events:
                episodes.append(
                    (
                        episode_events,
                        (belief.probabilities() > 0.5).astype(np.float64),
                    )
                )
        events = [event for episode_events, _ in episodes for event in episode_events]
        if not events:
            raise ValueError("train stream carries no reacquisition evidence")

        # Turns are unobservable while visible, so the only signal is how far
        # reappearances stray from a straight continuation.
        scored = [
            (
                _reacquisition_log_likelihood(
                    episodes,
                    turn_probability=candidate,
                    spec=self._spec,
                ),
                -candidate,
            )
            for candidate in TURN_PROBABILITY_GRID
        ]
        best = max(scored)
        turn_probability = -best[1]

        return {
            "grid_size": self._spec.grid_size,
            "arena_low": self._spec.arena_low,
            "arena_high": self._spec.arena_high,
            "camera_x": self._camera[0],
            "camera_y": self._camera[1],
            "k_max": self._k_max,
            "max_tracks": self._max_tracks,
            "max_hypotheses": self._max_hypotheses,
            "max_children_per_parent": self._max_children_per_parent,
            "birth_intensity": self._birth_intensity,
            "clutter_intensity": self._clutter_intensity,
            "survival_probability": self._survival_probability,
            "retirement_existence": self._retirement_existence,
            "self_null_prior": self._self_null_prior,
            "static_prior": static_kernel.prior,
            "static_mover_occupancy": static_kernel.mover_occupancy,
            "static_clamp": static_kernel.clamp,
            "turn_probability": float(turn_probability),
            "reacquisition_event_count": len(events),
        }

    def new_episode(
        self, frozen_kernel: Mapping[str, Any], model_seed: int
    ) -> StochasticPermanenceCandidate:
        return StochasticPermanenceCandidate(frozen_kernel, model_seed)


def run_episode(
    candidate: StochasticPermanenceCandidate,
    stream: Iterable[tuple[np.ndarray, np.ndarray, int]],
) -> dict[str, Any]:
    """Drive one episode and report the readouts the evaluator scores."""

    steps = 0
    for sensed, visibility, action in stream:
        candidate.observe(sensed, visibility, action)
        steps += 1
    return {
        "steps": steps,
        "track_positions": [list(cell) for cell in candidate.track_positions()],
        "hidden_occupancy": candidate.hidden_occupancy().tolist(),
        "hypothesis_count": int(candidate.hypothesis_weights().size),
        "discarded_hypothesis_mass": candidate.discarded_hypothesis_mass,
    }
