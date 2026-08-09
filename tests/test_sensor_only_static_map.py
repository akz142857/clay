"""Tests for the candidate's sensor-only static-topology belief (O3, I1).

The property that matters is not accuracy but *admissibility*: the belief has
to be recoverable from sensors alone, because a candidate that reads the true
static layout is not measuring permanence at all.  Accuracy is then checked
against ground truth as a diagnostic -- ground truth is compared to, never fed
in.
"""

from __future__ import annotations

import numpy as np
import pytest

from cal.evaluation.randomized_occlusion_world import RandomizedOcclusionWorld
from cal.evaluation.v2_i1_integration import ARENA_HIGH, ARENA_LOW, CAMERA
from cal.model.sensor_only_static_map import (
    SensorOnlyStaticMap,
    StaticMapKernel,
    fit_static_map_kernel,
)

_GRID = 25
_KERNEL = StaticMapKernel(prior=0.05, mover_occupancy=0.02, clamp=1e-6)


def _map(kernel: StaticMapKernel = _KERNEL) -> SensorOnlyStaticMap:
    return SensorOnlyStaticMap(
        grid_size=_GRID,
        arena_low=ARENA_LOW,
        arena_high=ARENA_HIGH,
        camera=CAMERA,
        kernel=kernel,
    )


def _stream(seed: int, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    world = RandomizedOcclusionWorld(seed, hidden_turn_probability=0.45)
    frames = [world.observe()]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        frames.append(world.step(int(rng.integers(0, 5))))
    return frames


def test_cells_outside_the_arena_stay_blocked() -> None:
    """Unreachable and blocked are the same thing for motion.

    Treating out-of-arena cells as free would let belief mass leak off the
    board, which is exactly the failure `GridSpec`'s arena bounds exist to
    prevent.
    """

    belief = _map()
    probabilities = belief.probabilities()

    outside = np.ones((_GRID, _GRID), dtype=bool)
    outside[ARENA_LOW : ARENA_HIGH + 1, ARENA_LOW : ARENA_HIGH + 1] = False
    assert np.all(probabilities[outside] == 1.0)
    assert np.all(probabilities[~outside] == pytest.approx(_KERNEL.prior))


def test_a_visible_empty_cell_is_proved_not_static() -> None:
    """A static blocker always renders when visible, so empty means movable."""

    belief = _map()
    sensed = np.zeros((11, 11), dtype=np.uint8)
    visibility = np.ones((11, 11), dtype=bool)

    belief.update(sensed, visibility)

    arena = belief.probabilities()[
        ARENA_LOW : ARENA_HIGH + 1, ARENA_LOW : ARENA_HIGH + 1
    ]
    assert np.all(arena <= _KERNEL.clamp)


def test_a_visible_occupied_cell_raises_belief_by_the_likelihood_ratio() -> None:
    belief = _map()
    sensed = np.zeros((11, 11), dtype=np.uint8)
    visibility = np.zeros((11, 11), dtype=bool)
    # One cell at the patch centre, which is the camera position itself.
    sensed[5, 5] = 1
    visibility[5, 5] = True

    belief.update(sensed, visibility)

    prior = _KERNEL.prior
    expected = prior / (prior + (1.0 - prior) * _KERNEL.mover_occupancy)
    assert belief.probabilities()[CAMERA[1], CAMERA[0]] == pytest.approx(expected)


def test_a_non_visible_cell_receives_no_update() -> None:
    """Shadow shape is evidence, but recovering it needs map hypotheses.

    §5.2 defers that past this version, so the honest behaviour is to leave
    unobserved cells alone rather than to guess.
    """

    belief = _map()
    before = belief.probabilities()

    belief.update(np.zeros((11, 11), dtype=np.uint8), np.zeros((11, 11), dtype=bool))

    assert np.array_equal(belief.probabilities(), before)


def test_probabilities_are_not_aliased() -> None:
    belief = _map()
    snapshot = belief.probabilities()
    snapshot[:] = 0.5

    assert not np.array_equal(belief.probabilities(), snapshot)


def test_kernel_rejects_degenerate_parameters() -> None:
    for bad in (
        {"prior": 0.0, "mover_occupancy": 0.02, "clamp": 0.0},
        {"prior": 1.0, "mover_occupancy": 0.02, "clamp": 0.0},
        {"prior": 0.05, "mover_occupancy": 0.0, "clamp": 0.0},
        {"prior": 0.05, "mover_occupancy": 0.02, "clamp": 0.5},
    ):
        with pytest.raises(ValueError):
            StaticMapKernel(**bad)


def test_kernel_is_fitted_from_sensors_without_reading_truth() -> None:
    kernel = fit_static_map_kernel(
        (_stream(seed, 60) for seed in (62003, 62004, 62005)),
        arena_low=ARENA_LOW,
        arena_high=ARENA_HIGH,
        camera=CAMERA,
        grid_size=_GRID,
    )

    assert 0.0 < kernel.mover_occupancy < 1.0
    assert 0.0 < kernel.prior < 1.0
    # Three movers over roughly a hundred visible arena cells: the estimate has
    # to be small, and a value near 0.5 would mean the denominator collapsed.
    assert kernel.mover_occupancy < 0.2


def test_the_belief_recovers_the_true_static_layout_it_never_reads() -> None:
    """Diagnostic: ground truth is compared against, never supplied.

    Every visible arena cell is decidable -- empty proves movable, and a cell
    that stays occupied while its neighbours move is static -- so after a full
    episode the belief should separate the two cleanly on what it has seen.
    """

    seed = 62149
    frames = _stream(seed, 200)
    world = RandomizedOcclusionWorld(seed, hidden_turn_probability=0.45)

    kernel = fit_static_map_kernel(
        (_stream(other, 60) for other in (62003, 62004)),
        arena_low=ARENA_LOW,
        arena_high=ARENA_HIGH,
        camera=CAMERA,
        grid_size=_GRID,
    )
    belief = _map(kernel)
    for sensed, visibility in frames:
        belief.update(sensed, visibility)

    inferred = belief.certain_static_cells(0.9)
    truth = {cell for cell in world.static if ARENA_LOW <= cell[0] <= ARENA_HIGH
             and ARENA_LOW <= cell[1] <= ARENA_HIGH}

    # No false positives: nothing movable may be called static, because a
    # candidate that invents walls would prune real hypotheses.
    assert not (inferred - truth), f"invented static cells: {sorted(inferred - truth)}"
    # And it must actually find some of them, or the belief is doing nothing.
    assert inferred, "the belief identified no static cell at all"
