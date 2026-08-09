"""Score the O3 candidate beside the five references (increment I5).

This is the first time the candidate meets the metrics the confirmatory gates
are defined over.  **It is a development report, not evidence.**  The numbers
come from development seeds, which have been looked at repeatedly; the
confirmatory battery runs once, on a split that does not exist yet.

## Why the world is replayed rather than read from `_Sample`

Driving the candidate needs the per-step visibility patch, and `_Sample` keeps
only `sensed_history` and the query-step global visibility.  Adding the missing
field would mean editing `permanence_forward_benchmark.py`, which is one of the
22 locked modules -- protocol amendment plus a full artifact regeneration for a
development report.  The implementation plan says to stop rather than absorb
that quietly.

So the episode is replayed instead.  It reproduces exactly: the world is a
deterministic function of the seed, and the action stream is
`default_rng(seed + _ACTION_STREAM_OFFSET)` drawing `integers(0, 5)` per step,
which is the same construction `_collect` uses.  The replay is checked against
the collector rather than assumed -- `verify_replay_alignment` compares
reconstructed hidden truth with the samples the collector accepted, and the
report refuses to score if they disagree.

NON-GATED: no frozen protocol, no source lock, no one-shot evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cal.evaluation.permanence_forward_benchmark import (
    _ACTION_STREAM_OFFSET,
    _SIDE,
    _Sample,
    _cell_index,
    _collect_many,
    _require_disjoint_seeds,
    _score_maps,
    run_benchmark,
)
from cal.evaluation.randomized_occlusion_world import RandomizedOcclusionWorld
from cal.evaluation.v2_i1_integration import ARENA_HIGH, ARENA_LOW, CAMERA
from cal.model.permanence_candidate import (
    StochasticPermanenceCandidateFactory,
)

# The closure floor the preregistration fixes (C.1).  It is stated here so the
# report can compute it, never so it can be adjusted: lowering a threshold
# because a candidate missed it is precisely what review finding F1 exists to
# prevent.
PREREGISTERED_CLOSURE_FLOOR = 0.40
CLOSURE_BIN = "6+"


def _episode_observations(
    seed: int, *, steps: int, turn_probability: float
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Replay one episode's (sensed, visibility, action) stream exactly."""

    world = RandomizedOcclusionWorld(
        seed, hidden_turn_probability=turn_probability
    )
    action_rng = np.random.default_rng(int(seed) + _ACTION_STREAM_OFFSET)
    sensed, visibility = world.observe()
    frames = [(sensed, visibility, 0)]
    for _ in range(steps):
        action = int(action_rng.integers(0, 5))
        sensed, visibility = world.step(action)
        frames.append((sensed, visibility, action))
    return frames


def candidate_maps(
    factory: StochasticPermanenceCandidateFactory,
    frozen_kernel: Mapping[str, Any],
    samples: Sequence[_Sample],
    *,
    steps: int,
    turn_probability: float,
    model_seed: int = 17_011,
) -> np.ndarray:
    """One occupancy vector per sample, in the collector's sample order."""

    by_seed: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        by_seed.setdefault(int(sample.seed), []).append(index)

    maps = np.zeros((len(samples), _SIDE * _SIDE), dtype=np.float64)
    for seed, indices in by_seed.items():
        wanted = {int(samples[index].step): index for index in indices}
        candidate = factory.new_episode(frozen_kernel, model_seed)
        frames = _episode_observations(
            seed, steps=steps, turn_probability=turn_probability
        )
        for step, (sensed, visibility, action) in enumerate(frames):
            candidate.observe(sensed, visibility, action)
            target = wanted.get(step)
            if target is None:
                continue
            hidden = candidate.hidden_occupancy()
            maps[target] = hidden[
                ARENA_LOW : ARENA_HIGH + 1, ARENA_LOW : ARENA_HIGH + 1
            ].ravel()
    return maps


def verify_replay_alignment(
    samples: Sequence[_Sample], *, steps: int, turn_probability: float
) -> dict[str, int]:
    """Prove the replay is the same episode the collector saw.

    A silently misaligned replay would score the candidate on the wrong step,
    which no metric would flag -- it would just look like a weak candidate.
    """

    checked = mismatched = 0
    by_seed: dict[int, list[_Sample]] = {}
    for sample in samples:
        by_seed.setdefault(int(sample.seed), []).append(sample)

    for seed, seed_samples in by_seed.items():
        world = RandomizedOcclusionWorld(
            seed, hidden_turn_probability=turn_probability
        )
        action_rng = np.random.default_rng(int(seed) + _ACTION_STREAM_OFFSET)
        wanted = {int(sample.step): sample for sample in seed_samples}
        for step in range(1, steps + 1):
            world.step(int(action_rng.integers(0, 5)))
            sample = wanted.get(step)
            if sample is None:
                continue
            checked += 1
            replayed = {
                (int(world.distractor_a[0]), int(world.distractor_a[1])),
                (int(world.distractor_b[0]), int(world.distractor_b[1])),
            }
            if not set(sample.positives) <= replayed:
                mismatched += 1
    return {"checked": checked, "mismatched": mismatched}


def _closure(candidate: float, floor: float, ceiling: float) -> float | None:
    """Fraction of the oracle/belief-free gap the candidate closed."""

    gap = ceiling - floor
    if gap <= 0.0:
        return None
    return (candidate - floor) / gap


def run_development_comparison(
    train_seeds: list[int],
    evaluation_seeds: list[int],
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
) -> dict[str, Any]:
    _require_disjoint_seeds(train_seeds, evaluation_seeds)

    reference = run_benchmark(
        train_seeds,
        evaluation_seeds,
        steps=steps,
        warmup=warmup,
        turn_probability=turn_probability,
    )
    evaluation_samples = _collect_many(
        evaluation_seeds,
        steps=steps,
        warmup=warmup,
        turn_probability=turn_probability,
    )
    alignment = verify_replay_alignment(
        evaluation_samples, steps=steps, turn_probability=turn_probability
    )
    if alignment["mismatched"]:
        raise RuntimeError(
            "replayed episodes disagree with the collector's samples; scoring "
            f"would be misaligned: {alignment}"
        )

    factory = StochasticPermanenceCandidateFactory(
        grid_size=RandomizedOcclusionWorld(evaluation_seeds[0]).grid_size,
        arena_low=ARENA_LOW,
        arena_high=ARENA_HIGH,
        camera=CAMERA,
    )
    train_streams = [
        [(sensed, visibility) for sensed, visibility, _action in
         _episode_observations(seed, steps=steps, turn_probability=turn_probability)]
        for seed in train_seeds
    ]
    frozen_kernel = factory.fit_kernel(train_streams)

    maps = candidate_maps(
        factory,
        frozen_kernel,
        evaluation_samples,
        steps=steps,
        turn_probability=turn_probability,
    )
    candidate_score = _score_maps(evaluation_samples, maps)

    # Privileged: the candidate's belief machinery on the references' tracks.
    beliefs = inferred_static_beliefs(
        evaluation_samples, frozen_kernel, steps=steps,
        turn_probability=turn_probability,
    )
    privileged = privileged_belief_maps(
        evaluation_samples, frozen_kernel, beliefs,
        # The candidate's *fitted* turn probability, not the world's.  Passing
        # the true value here would quietly hand the diagnostic a second
        # privilege on top of the tracks, and the deficit it is meant to
        # attribute would land in the wrong place.
        turn_probability=float(frozen_kernel["turn_probability"]),
    )
    privileged_score = _score_maps(evaluation_samples, privileged)

    predictors = {
        "candidate": candidate_score,
        "privileged_belief_diagnostic": privileged_score,
        "oracle": reference["predictors"]["belief"],
        "geometric": reference["predictors"]["geometric"],
        "belief_free": reference["predictors"]["belief_free"],
    }
    by_bin = {
        "candidate": _bin_scores(evaluation_samples, maps),
        "privileged_belief_diagnostic": _bin_scores(evaluation_samples, privileged),
        **{
            name: reference["ranking_by_occlusion_length"].get(source, {})
            for name, source in (
                ("oracle", "belief"),
                ("geometric", "geometric"),
                ("belief_free", "belief_free"),
            )
        },
    }

    closure = _closure_report(by_bin)
    decomposition = _decompose(by_bin)
    return {
        "status": "development_only_non_gated",
        "gated": False,
        "warning": (
            "development seeds have been inspected repeatedly; these numbers "
            "are not confirmatory evidence and no gate decision follows from "
            "them"
        ),
        "population": {
            "train_seeds": len(train_seeds),
            "evaluation_seeds": len(evaluation_seeds),
            "evaluation_samples": len(evaluation_samples),
        },
        "replay_alignment": alignment,
        "frozen_kernel": dict(frozen_kernel),
        "predictors": predictors,
        "by_occlusion_length": by_bin,
        "closure": closure,
        "tracking_versus_belief": decomposition,
        "privileged_diagnostic_only": [
            "privileged_belief_diagnostic",
        ],
    }


def _bin_scores(
    samples: Sequence[_Sample], maps: np.ndarray
) -> dict[str, dict[str, float]]:
    from cal.evaluation.permanence_forward_benchmark import _occlusion_length_bin

    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        if sample.hidden_object_count != 1:
            continue
        grouped.setdefault(
            _occlusion_length_bin(sample.hidden_steps), []
        ).append(index)
    return {
        name: _score_maps([samples[i] for i in indices], maps[indices])
        for name, indices in sorted(grouped.items())
        if indices
    }


def _decompose(by_bin: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Split the candidate's deficit into a tracking part and a belief part.

    With the references' tracks held fixed, ``privileged - belief_free`` is what
    belief maintenance buys.  ``candidate - privileged`` is what inferring the
    tracks instead of being handed them costs.  Reporting only the total would
    leave the two indistinguishable, and they call for opposite responses.
    """

    result: dict[str, Any] = {}
    for name in ("2-3", "4-5", "6+"):
        try:
            candidate = float(by_bin["candidate"][name]["top1_accuracy"])
            privileged = float(
                by_bin["privileged_belief_diagnostic"][name]["top1_accuracy"]
            )
            floor = float(by_bin["belief_free"][name]["top1_accuracy"])
        except (KeyError, TypeError):
            continue
        result[name] = {
            "belief_gain_with_tracking_held_fixed": privileged - floor,
            "tracking_cost": candidate - privileged,
        }
    return result


def _closure_report(by_bin: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    bins = by_bin.get("candidate", {})
    result: dict[str, Any] = {
        "preregistered_floor": PREREGISTERED_CLOSURE_FLOOR,
        "baseline": "belief_free",
        "ceiling": "oracle",
        "per_bin": {},
    }
    for name in sorted(bins):
        try:
            candidate = float(by_bin["candidate"][name]["top1_accuracy"])
            floor = float(by_bin["belief_free"][name]["top1_accuracy"])
            ceiling = float(by_bin["oracle"][name]["top1_accuracy"])
        except (KeyError, TypeError):
            continue
        result["per_bin"][name] = {
            "candidate": candidate,
            "belief_free": floor,
            "oracle": ceiling,
            "closure": _closure(candidate, floor, ceiling),
        }
    gate_bin = result["per_bin"].get(CLOSURE_BIN, {})
    closure = gate_bin.get("closure")
    result["gate_bin"] = CLOSURE_BIN
    result["gate_bin_closure"] = closure
    result["would_meet_preregistered_floor"] = (
        None if closure is None else bool(closure >= PREREGISTERED_CLOSURE_FLOOR)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json"),
    )
    parser.add_argument("--train-seeds", type=int)
    parser.add_argument("--eval-seeds", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    registry = json.loads(arguments.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "development_only_non_gated":
        raise ValueError("only a development-only registry is accepted")
    contract = registry["coverage_contract"]
    train = [int(seed) for seed in registry["train_seeds"]]
    evaluation = [int(seed) for seed in registry["evaluation_seeds"]]
    if arguments.train_seeds:
        train = train[: arguments.train_seeds]
    if arguments.eval_seeds:
        evaluation = evaluation[: arguments.eval_seeds]

    report = run_development_comparison(
        train,
        evaluation,
        steps=int(contract["steps"]),
        warmup=int(contract["warmup"]),
        turn_probability=float(registry["selected_turn_probability"]),
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Privileged diagnostic: how much of the gap is tracking, how much is belief
# --------------------------------------------------------------------------
#
# The references read `sample.hidden_tracks`, so the collector hands them the
# last-seen cell, the observed velocity and the hidden duration.  The candidate
# infers all three.  A single scoreboard therefore cannot say whether a deficit
# comes from weak belief maintenance or from weak tracking -- and those call
# for opposite responses.
#
# This runs the candidate's *belief* machinery on the references' *tracks*:
#
#     belief_free          given tracks, no belief filtering
#     this diagnostic      given tracks, candidate belief, inferred topology
#     oracle               given tracks, exact belief, true topology
#     candidate            inferred tracks, candidate belief, inferred topology
#
# so `diagnostic - belief_free` is what belief maintenance buys with tracking
# held fixed, and `candidate - diagnostic` is what the candidate's tracking
# costs.  PRIVILEGED: it consumes evaluator bookkeeping and can never be a
# formal candidate result.


def inferred_static_beliefs(
    samples: Sequence[_Sample],
    frozen_kernel: Mapping[str, Any],
    *,
    steps: int,
    turn_probability: float,
) -> dict[tuple[int, int], np.ndarray]:
    """The candidate's own static belief at each sample's step."""

    from cal.model.sensor_only_static_map import (
        SensorOnlyStaticMap,
        StaticMapKernel,
    )

    wanted: dict[int, set[int]] = {}
    for sample in samples:
        wanted.setdefault(int(sample.seed), set()).add(int(sample.step))

    beliefs: dict[tuple[int, int], np.ndarray] = {}
    for seed, want in wanted.items():
        belief = SensorOnlyStaticMap(
            grid_size=int(frozen_kernel["grid_size"]),
            arena_low=int(frozen_kernel["arena_low"]),
            arena_high=int(frozen_kernel["arena_high"]),
            camera=(int(frozen_kernel["camera_x"]), int(frozen_kernel["camera_y"])),
            kernel=StaticMapKernel(
                prior=float(frozen_kernel["static_prior"]),
                mover_occupancy=float(frozen_kernel["static_mover_occupancy"]),
                clamp=float(frozen_kernel["static_clamp"]),
            ),
        )
        frames = _episode_observations(
            seed, steps=steps, turn_probability=turn_probability
        )
        for step, (sensed, visibility, _action) in enumerate(frames):
            belief.update(sensed, visibility)
            if step in want:
                beliefs[(seed, step)] = belief.probabilities()
    return beliefs


def privileged_belief_maps(
    samples: Sequence[_Sample],
    frozen_kernel: Mapping[str, Any],
    beliefs: Mapping[tuple[int, int], np.ndarray],
    *,
    turn_probability: float,
) -> np.ndarray:
    """The candidate's belief filter, run on the references' given tracks.

    ``turn_probability`` must come from the candidate's fitted kernel.  The
    diagnostic is already privileged in exactly one way -- it is handed the
    tracks -- and that is what makes it attribute the deficit.  Feeding it the
    world's true turn rate as well would add a second privilege and move the
    attribution somewhere it does not belong.
    """

    from cal.model.permanence_track import PermanenceTrack, TrackKernel
    from cal.model.stochastic_motion_filter import EmptyPosteriorError, GridSpec

    spec = GridSpec(
        grid_size=int(frozen_kernel["grid_size"]),
        arena_low=int(frozen_kernel["arena_low"]),
        arena_high=int(frozen_kernel["arena_high"]),
    )
    track_kernel = TrackKernel(
        survival_probability=float(frozen_kernel["survival_probability"]),
        retirement_existence=float(frozen_kernel["retirement_existence"]),
    )
    maps = np.zeros((len(samples), _SIDE * _SIDE), dtype=np.float64)
    for index, sample in enumerate(samples):
        static = beliefs[(int(sample.seed), int(sample.step))]
        no_detection = 1.0 - np.asarray(sample.visible, dtype=np.float64)
        free = np.ones(_SIDE * _SIDE, dtype=np.float64)
        for last_seen, velocity, hidden_steps in sample.hidden_tracks:
            track = PermanenceTrack(
                k_max=int(frozen_kernel["k_max"]), spec=spec, kernel=track_kernel
            )
            track.reset(last_seen, velocity)
            try:
                for step in range(int(hidden_steps)):
                    track.step_unobserved(
                        static_probability=static,
                        no_detection_probability=no_detection,
                        turn_probability=turn_probability,
                        allow_turn=step > 0,
                    )
            except EmptyPosteriorError:
                continue
            for cell, mass in track.occupancy().items():
                free[_cell_index(cell)] *= 1.0 - min(max(mass, 0.0), 1.0)
        maps[index] = 1.0 - free
    return maps
