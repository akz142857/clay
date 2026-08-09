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

The association bank here holds **one** global hypothesis (``w_h = 1``).
Plan §5.6 requires the multi-hypothesis bank to be preserved, so this is not
yet the complete candidate and no confirmatory claim may rest on it.  The
branch-local machinery is already shaped for several branches -- each would own
its own ``BranchSelfIdentity`` and track set -- so adding them is extension,
not rework.  Tracked as I4b in the implementation plan.

NOT part of the Phase-0/Phase-R source lock: the candidate is injected through
``CandidateFactory``, so no locked entry point imports it.
"""

from __future__ import annotations

from math import log
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cal.model.branch_self_identity import (
    ACTION_DELTAS,
    BranchSelfIdentity,
    action_successors,
)
from cal.model.permanence_track import PermanenceTrack, TrackKernel
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
        self._tracks: list[PermanenceTrack] = []
        self._last_position: list[tuple[int, int] | None] = []
        self._identity = BranchSelfIdentity(0, null_prior=float(
            frozen_kernel["self_null_prior"]
        ))
        self._steps = 0

    # -- observation ------------------------------------------------------

    def observe(
        self, sensed: np.ndarray, visibility: np.ndarray, action: int
    ) -> None:
        """Fold one (observation, action) pair into the belief."""

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

        assignment = self._assign(detections)
        self._advance(
            assignment,
            static_probability=static_probability,
            no_detection=no_detection,
            action=int(action),
        )
        self._birth(detections, assignment)
        self._retire()
        self._steps += 1

    def _assign(
        self, detections: Sequence[tuple[int, int]]
    ) -> dict[int, tuple[int, int]]:
        """Greedy nearest-cell assignment within one hypothesis.

        A single branch cannot represent association ambiguity, which is the
        §5.6 gap this increment leaves open; the greedy choice is deterministic
        so at least the omission is reproducible.
        """

        remaining = list(detections)
        assignment: dict[int, tuple[int, int]] = {}
        for index, track in enumerate(self._tracks):
            if not remaining:
                break
            marginal = track.position_marginal()
            if not marginal:
                continue
            best = max(sorted(marginal), key=lambda cell: marginal[cell])
            reachable = [
                cell
                for cell in remaining
                if abs(cell[0] - best[0]) + abs(cell[1] - best[1]) <= 1
            ]
            if not reachable:
                continue
            chosen = min(
                reachable,
                key=lambda cell: (
                    abs(cell[0] - best[0]) + abs(cell[1] - best[1]),
                    cell,
                ),
            )
            assignment[index] = chosen
            remaining.remove(chosen)
        return assignment

    def _advance(
        self,
        assignment: Mapping[int, tuple[int, int]],
        *,
        static_probability: np.ndarray,
        no_detection: np.ndarray,
        action: int,
    ) -> None:
        ratios: list[float] = []
        survivors: list[PermanenceTrack] = []
        positions: list[tuple[int, int] | None] = []

        for index, track in enumerate(self._tracks):
            previous = self._last_position[index]
            detection = assignment.get(index)
            if detection is not None:
                ratio = self._self_likelihood_ratio(
                    track,
                    previous=previous,
                    observed=detection,
                    static_probability=static_probability,
                    action=action,
                )
                velocity = (
                    (detection[0] - previous[0], detection[1] - previous[1])
                    if previous is not None
                    else (0, 0)
                )
                if velocity not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    velocity = (1, 0)
                # Detections in this world are exact cell occupancy, so a match
                # collapses the posterior rather than merely reweighting it.
                track.reset(detection, velocity)
                survivors.append(track)
                positions.append(detection)
                ratios.append(ratio)
                continue

            try:
                track.step_unobserved(
                    static_probability=static_probability,
                    no_detection_probability=no_detection,
                    turn_probability=self.turn_probability,
                    allow_turn=self._steps > 0,
                )
            except EmptyPosteriorError:
                # Every state this track could occupy was visible and empty.
                # Dropping it is a storage consequence of a branch that ran out
                # of explanations, not observed death evidence.
                continue
            survivors.append(track)
            positions.append(None)
            ratios.append(1.0)

        self._tracks = survivors
        self._last_position = positions
        self._resize_identity(len(survivors))
        if survivors:
            self._identity.update(ratios, null_likelihood=1.0)

    def _self_likelihood_ratio(
        self,
        track: PermanenceTrack,
        *,
        previous: tuple[int, int] | None,
        observed: tuple[int, int],
        static_probability: np.ndarray,
        action: int,
    ) -> float:
        """``P_action(observed) / P_auto(observed)`` for this entity.

        The common autonomous factors of the other entities cancel in the
        categorical update, so the ratio is all that is needed -- which is also
        why ``null_likelihood`` is 1.
        """

        if previous is None:
            return 1.0
        marginal = track.position_marginal()
        if not marginal:
            return 1.0
        origin = max(sorted(marginal), key=lambda cell: marginal[cell])
        velocity = (origin[0] - previous[0], origin[1] - previous[1])
        if velocity not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            velocity = (1, 0)

        action_mass = sum(
            mass
            for position, _velocity, mass in action_successors(
                previous, velocity, static_probability, action=action, spec=self.spec
            )
            if position == observed
        )
        autonomous_mass = sum(
            mass
            for position, _velocity, mass in autonomous_successors(
                previous,
                velocity,
                (static_probability > 0.5).astype(np.float64),
                turn_probability=self.turn_probability,
                allow_turn=True,
                spec=self.spec,
                marginal_turn_mixture=True,
            )
            if position == observed
        )
        if autonomous_mass <= 0.0:
            return 1.0 if action_mass <= 0.0 else float(len(ACTION_DELTAS))
        return float(action_mass / autonomous_mass)

    def _resize_identity(self, count: int) -> None:
        if self._identity.entity_count == count:
            return
        self._identity = BranchSelfIdentity(
            count, null_prior=float(self._kernel["self_null_prior"])
        )

    def _birth(
        self,
        detections: Sequence[tuple[int, int]],
        assignment: Mapping[int, tuple[int, int]],
    ) -> None:
        claimed = set(assignment.values())
        for cell in detections:
            if cell in claimed:
                continue
            if len(self._tracks) >= int(self._kernel["max_tracks"]):
                break
            track = PermanenceTrack(
                k_max=self._k_max, spec=self.spec, kernel=self._track_kernel
            )
            # Birth existence is 1: the branch prior already carries the birth
            # intensity, so boosting it again double-counts the same event.
            track.reset(cell, (1, 0))
            self._tracks.append(track)
            self._last_position.append(cell)
        self._resize_identity(len(self._tracks))

    def _retire(self) -> None:
        keep = [
            index
            for index, track in enumerate(self._tracks)
            if not track.retired()
        ]
        if len(keep) == len(self._tracks):
            return
        self._tracks = [self._tracks[index] for index in keep]
        self._last_position = [self._last_position[index] for index in keep]
        self._resize_identity(len(self._tracks))

    # -- readout ----------------------------------------------------------

    def occupancy(self) -> np.ndarray:
        """``P_occ(c) = 1 - (1 - m_t(c))·(1 - P_dynamic(c))`` over the grid.

        The dynamic term is the §5.5 Bernoulli union under the stated bounded
        approximation: given the hypothesis, entity existence and spatial
        factors are conditionally independent.
        """

        static = self._static.probabilities()
        free = np.ones_like(static)
        for track in self._tracks:
            for (x, y), mass in track.occupancy().items():
                free[y, x] *= 1.0 - min(max(mass, 0.0), 1.0)
        dynamic = 1.0 - free
        return 1.0 - (1.0 - static) * (1.0 - dynamic)

    def hidden_occupancy(self) -> np.ndarray:
        """Dynamic occupancy alone, which is what the permanence task scores."""

        free = np.ones(
            (self.spec.grid_size, self.spec.grid_size), dtype=np.float64
        )
        for track in self._tracks:
            for (x, y), mass in track.occupancy().items():
                free[y, x] *= 1.0 - min(max(mass, 0.0), 1.0)
        return 1.0 - free

    def track_positions(self) -> tuple[tuple[int, int], ...]:
        """MAP position of every entity above the existence threshold.

        Ties break on canonical cell order so the readout is deterministic.
        """

        positions: list[tuple[int, int]] = []
        for track in self._tracks:
            if track.retired():
                continue
            marginal = track.position_marginal()
            if not marginal:
                continue
            positions.append(
                max(sorted(marginal), key=lambda cell: marginal[cell])
            )
        return tuple(sorted(positions))

    def self_probabilities(self) -> np.ndarray:
        return self._identity.probabilities()


# -- fitting --------------------------------------------------------------


def _reacquisition_events(
    frames: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    spec: GridSpec,
    camera: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], int, tuple[int, int]]]:
    """Unambiguous vanish/reappear intervals: ``(cell, velocity, steps, cell')``.

    Only intervals where exactly one tracked cell is missing are used, so the
    reappearance is identified without solving association.  Turns happen only
    while occluded, so these intervals are the only evidence about them that
    exists.
    """

    events = []
    previous_cells: tuple[tuple[int, int], ...] = ()
    previous_velocity: dict[tuple[int, int], tuple[int, int]] = {}
    pending: tuple[tuple[int, int], tuple[int, int], int] | None = None

    for sensed, visibility in frames:
        occupied, visible = _global_masks(
            sensed, visibility, spec=spec, camera=camera
        )
        cells = _arena_cells(occupied & visible, spec)

        if pending is not None:
            origin, velocity, steps = pending
            fresh = [cell for cell in cells if cell not in previous_cells]
            if len(fresh) == 1 and steps >= 1:
                events.append((origin, velocity, steps, fresh[0]))
                pending = None
            elif steps >= 24:
                pending = None
            else:
                pending = (origin, velocity, steps + 1)
        else:
            vanished = [cell for cell in previous_cells if cell not in cells]
            if len(vanished) == 1:
                origin = vanished[0]
                velocity = previous_velocity.get(origin)
                if velocity is not None:
                    pending = (origin, velocity, 1)

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
    events: Sequence[tuple[tuple[int, int], tuple[int, int], int, tuple[int, int]]],
    *,
    turn_probability: float,
    static_probability: np.ndarray,
    spec: GridSpec,
) -> float:
    total = 0.0
    for origin, velocity, steps, observed in events:
        belief = {(origin, velocity): 1.0}
        for step in range(steps):
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
            belief = propagated
        mass = sum(
            value for (position, _v), value in belief.items() if position == observed
        )
        total += log(max(mass, 1e-12))
    return total


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

        events: list[
            tuple[tuple[int, int], tuple[int, int], int, tuple[int, int]]
        ] = []
        static_belief = np.zeros(
            (self._spec.grid_size, self._spec.grid_size), dtype=np.float64
        )
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
            static_belief = np.maximum(static_belief, belief.probabilities())
            events.extend(
                _reacquisition_events(
                    stream, spec=self._spec, camera=self._camera
                )
            )
        if not events:
            raise ValueError("train stream carries no reacquisition evidence")

        # Turns are unobservable while visible, so the only signal is how far
        # reappearances stray from a straight continuation.
        hard_static = (static_belief > 0.5).astype(np.float64)
        scored = [
            (
                _reacquisition_log_likelihood(
                    events,
                    turn_probability=candidate,
                    static_probability=hard_static,
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
    }
