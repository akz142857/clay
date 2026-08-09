"""Per-cell Bernoulli belief over static topology, from sensors alone.

This is increment I1 of the O3 candidate (architecture: stochastic permanence
plan §5.2).  It is the component that makes the candidate *sensor-only*: every
downstream belief needs to know which cells can block motion, and the formal
candidate may not read the world's true static layout to find out.

The observation model is exact for this world rather than assumed:

- ``visibility`` is computed from static geometry alone, and a visible cell
  renders whatever occupies it, so a **visible empty** cell cannot be static.
  ``P(sensed=0 | static) = 0`` drives that cell's belief to zero outright.
- A **visible occupied** cell holds a static blocker, the agent, or a
  distractor.  Its likelihood ratio is ``1 : mover_occupancy``, where
  ``mover_occupancy`` is the probability a non-static visible cell shows a
  mover.  That number is a kernel parameter -- it may only be estimated from
  the train stream (§5.2), never from the episode being scored.
- A **non-visible** cell gets no update.  Its shadow is evidence about static
  geometry, but recovering it means reasoning over map hypotheses, which §5.2
  explicitly defers past this version.

Cells outside the arena are held at ``m = 1``: unreachable is indistinguishable
from blocked as far as motion is concerned, and treating them as free would let
belief mass leak off the board.

NOT part of the Phase-0/Phase-R source lock: nothing in those entry points
imports this, so it determines no current gated evidence.  It will need its own
lock when the candidate freezes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class StaticMapKernel:
    """Frozen observation-model parameters for the static-topology belief.

    Every field is estimated once from the train stream and frozen; §5.2 puts
    the prior, the likelihood and the clamp inside the candidate lock, so none
    of them may be re-tuned after validation.
    """

    prior: float
    mover_occupancy: float
    clamp: float

    def __post_init__(self) -> None:
        if not 0.0 < self.prior < 1.0:
            raise ValueError("prior must be strictly inside (0, 1)")
        if not 0.0 < self.mover_occupancy < 1.0:
            raise ValueError("mover_occupancy must be strictly inside (0, 1)")
        if not 0.0 <= self.clamp < 0.5:
            raise ValueError("clamp must be in [0, 0.5)")


class SensorOnlyStaticMap:
    """Track ``m_t(c) = P(static_c | sensor history)`` over the whole grid."""

    def __init__(
        self,
        *,
        grid_size: int,
        arena_low: int,
        arena_high: int,
        camera: tuple[int, int],
        kernel: StaticMapKernel,
    ) -> None:
        if not 0 <= arena_low <= arena_high < grid_size:
            raise ValueError("invalid arena bounds")
        self.grid_size = int(grid_size)
        self.arena_low = int(arena_low)
        self.arena_high = int(arena_high)
        self.camera = (int(camera[0]), int(camera[1]))
        self.kernel = kernel

        self._probability = np.ones((self.grid_size, self.grid_size), dtype=np.float64)
        # Only arena cells are uncertain; everything else stays pinned at 1.
        self._arena = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._arena[
            self.arena_low : self.arena_high + 1,
            self.arena_low : self.arena_high + 1,
        ] = True
        self._probability[self._arena] = kernel.prior

    def probabilities(self) -> np.ndarray:
        """Return a copy of the current belief; callers must not alias it."""

        return self._probability.copy()

    def _patch_origin(self, patch_shape: tuple[int, int]) -> tuple[int, int]:
        radius = patch_shape[0] // 2
        return self.camera[0] - radius, self.camera[1] - radius

    def update(self, sensed: np.ndarray, visibility: np.ndarray) -> None:
        """Fold one observation into the belief.

        ``sensed`` and ``visibility`` are the camera-centred patches the world
        returns, not global grids.
        """

        sensed_patch = np.asarray(sensed)
        visible_patch = np.asarray(visibility).astype(bool)
        if sensed_patch.shape != visible_patch.shape:
            raise ValueError("sensed and visibility patches must be aligned")
        if sensed_patch.ndim != 2 or sensed_patch.shape[0] != sensed_patch.shape[1]:
            raise ValueError("observation patches must be square")

        x0, y0 = self._patch_origin(sensed_patch.shape)
        height, width = sensed_patch.shape
        # Patch coordinates that land inside the grid; the camera sits near the
        # middle, but clipping keeps this honest if it ever moves.
        rows = np.arange(y0, y0 + height)
        columns = np.arange(x0, x0 + width)
        row_ok = (rows >= 0) & (rows < self.grid_size)
        column_ok = (columns >= 0) & (columns < self.grid_size)
        if not row_ok.all() or not column_ok.all():
            sensed_patch = sensed_patch[np.ix_(row_ok, column_ok)]
            visible_patch = visible_patch[np.ix_(row_ok, column_ok)]
            rows, columns = rows[row_ok], columns[column_ok]

        window = np.ix_(rows, columns)
        arena = self._arena[window]
        prior = self._probability[window]
        occupied = np.asarray(sensed_patch) != 0

        # Visible and empty: a static blocker would have rendered, so this cell
        # is not static.  Visible and occupied: static outweighs a mover by the
        # likelihood ratio.  Non-visible: untouched.
        posterior = prior.copy()
        free_evidence = visible_patch & ~occupied & arena
        posterior[free_evidence] = 0.0

        occupied_evidence = visible_patch & occupied & arena
        if occupied_evidence.any():
            mass = prior[occupied_evidence]
            mover = self.kernel.mover_occupancy
            posterior[occupied_evidence] = mass / (mass + (1.0 - mass) * mover)

        clamp = self.kernel.clamp
        if clamp > 0.0:
            updated = arena & visible_patch
            posterior[updated] = np.clip(posterior[updated], clamp, 1.0 - clamp)

        self._probability[window] = posterior

    def certain_static_cells(self, threshold: float) -> frozenset[tuple[int, int]]:
        """Cells believed static beyond ``threshold``, in world coordinates.

        A convenience for diagnostics and for callers that need a hard set;
        the belief itself stays probabilistic.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        rows, columns = np.nonzero(self._arena & (self._probability >= threshold))
        return frozenset(
            (int(column), int(row)) for row, column in zip(rows, columns)
        )


def fit_static_map_kernel(
    observation_streams: Iterable[Sequence[tuple[np.ndarray, np.ndarray]]],
    *,
    arena_low: int,
    arena_high: int,
    camera: tuple[int, int],
    grid_size: int,
    clamp: float = 1e-6,
) -> StaticMapKernel:
    """Estimate the observation model from train episodes, sensors only.

    ``mover_occupancy`` is the probability that a visible non-static cell shows
    something.  Which cells are non-static is itself inferred rather than read:
    a cell seen empty at any point in an episode cannot be static, and those
    cells -- and only those -- form the denominator.  This keeps the estimator
    inside the sensor-only boundary that makes the whole candidate admissible.

    ``prior`` is the fraction of arena cells that never appear empty while
    visible, which is an upper bound on the true static fraction rather than an
    estimate of it; the per-cell updates correct it downward as evidence
    arrives.
    """

    if not 0.0 <= clamp < 0.5:
        raise ValueError("clamp must be in [0, 0.5)")
    arena_width = arena_high - arena_low + 1
    arena_cells = arena_width * arena_width
    if arena_cells < 1:
        raise ValueError("arena must contain at least one cell")

    mover_hits = 0
    mover_opportunities = 0
    never_empty_total = 0
    episodes = 0

    for stream in observation_streams:
        frames = list(stream)
        if not frames:
            continue
        episodes += 1
        seen_empty = np.zeros((grid_size, grid_size), dtype=bool)
        seen_visible = np.zeros((grid_size, grid_size), dtype=bool)
        occupied_frames: list[np.ndarray] = []
        visible_frames: list[np.ndarray] = []
        for sensed, visibility in frames:
            occupied = np.zeros((grid_size, grid_size), dtype=bool)
            visible = np.zeros((grid_size, grid_size), dtype=bool)
            patch = np.asarray(sensed)
            mask = np.asarray(visibility).astype(bool)
            radius = patch.shape[0] // 2
            x0, y0 = camera[0] - radius, camera[1] - radius
            occupied[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] = patch != 0
            visible[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
            seen_empty |= visible & ~occupied
            seen_visible |= visible
            occupied_frames.append(occupied)
            visible_frames.append(visible)

        arena = np.zeros((grid_size, grid_size), dtype=bool)
        arena[arena_low : arena_high + 1, arena_low : arena_high + 1] = True
        # Provably non-static: seen empty while visible at least once.
        movable = seen_empty & arena
        for occupied, visible in zip(occupied_frames, visible_frames):
            eligible = movable & visible
            mover_opportunities += int(eligible.sum())
            mover_hits += int((eligible & occupied).sum())
        never_empty_total += int((arena & seen_visible & ~seen_empty).sum())

    if episodes == 0 or mover_opportunities == 0:
        raise ValueError("train stream carries no usable static-map evidence")

    mover_occupancy = mover_hits / mover_opportunities
    prior = never_empty_total / (episodes * arena_cells)
    # Both are probabilities of the observation model, so keep them off the
    # open interval's endpoints where the Bayes update would become degenerate.
    floor = max(clamp, 1e-9)
    return StaticMapKernel(
        prior=float(min(max(prior, floor), 1.0 - floor)),
        mover_occupancy=float(min(max(mover_occupancy, floor), 1.0 - floor)),
        clamp=float(clamp),
    )
