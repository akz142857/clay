"""Empirical proof that every V8 permanence control can still be built.

Review finding F6 has two halves.  The first -- explaining why the V8 controls
became non-gated diagnostics when the stage moved from a linear-readout
paradigm to forward-prediction scoring -- is answered in the preregistration's
C.3.  The second is not answerable by argument: **run the construction code for
all five controls on the development split and show the output**.

That requirement comes from a specific failure.  The V5 holdout stopped
part-way because the identity-scrambled control turned out to be
unconstructible from the reserved events -- there were not enough
simultaneously-hidden objects to swap identities between -- and a consumed
one-shot holdout cannot be retried.  The cost of discovering that on a holdout
is the holdout.  The cost of discovering it here is nothing.

So this script asks one question per control: *given the development events
this world actually produces, can the control be constructed at all, and on how
many of them?*  It is a feasibility census, not a gate.  A control scoring near
chance is a fine result; a control that cannot be built is the finding.

NON-GATED: no frozen protocol, no source lock, no one-shot evidence.  It runs
only on development seeds and produces no authorization.

Run:
    uv run python -m cal.evaluation.permanence_control_constructability \\
        --registry experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cal.evaluation.permanence_forward_benchmark import (
    _SIDE,
    _Sample,
    _bounce_advance,
    _cell_index,
    _collect_many,
    _require_disjoint_seeds,
    _require_stream_separation,
    _score_maps,
)
from cal.evaluation.v2_i1_integration import ARENA_HIGH, ARENA_LOW

# Fixed so a rerun reproduces the census exactly.  These streams only permute
# or randomize control inputs; they never touch world dynamics.
_CONTROL_SEED = 20_260_809
_RIDGE_ALPHA = 1.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        # An all-zero map is a legitimate prediction: the scorer records it as
        # an explicit miss rather than silently redistributing mass.
        return np.zeros_like(vector)
    return vector / total


def _arena_crop(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid)[ARENA_LOW : ARENA_HIGH + 1, ARENA_LOW : ARENA_HIGH + 1]


def _ridge(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    features = np.hstack([train_x, np.ones((train_x.shape[0], 1))])
    gram = features.T @ features
    gram[np.diag_indices_from(gram)] += _RIDGE_ALPHA
    return np.linalg.solve(gram, features.T @ train_y)


def _ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ weights


def _raw_features(sample: _Sample) -> np.ndarray:
    """The sensed local patch at the query step, flattened.

    This is the probe *input*, not a map: the sensed patch is egocentric and
    its cells do not correspond to arena cells.  The ridge learns the mapping
    to arena occupancy, which is exactly what V8's raw-sensor readout did.
    """

    return np.asarray(sample.sensed_history[-1], dtype=np.float64).ravel()


def _visible_arena_mask(sample: _Sample) -> np.ndarray:
    """Arena cells the agent can currently see, in field coordinates."""

    return _arena_crop(sample.visible).astype(np.float64).ravel()


def _targets(samples: list[_Sample]) -> np.ndarray:
    return np.stack(
        [np.asarray(s.hidden_occupancy, dtype=np.float64).ravel() for s in samples]
    )


# --------------------------------------------------------------------------
# The five controls
# --------------------------------------------------------------------------


def _control_raw_sensor(
    train: list[_Sample], evaluation: list[_Sample], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """V8's raw-sensor readout, transposed to occupancy prediction."""

    weights = _ridge(
        np.stack([_raw_features(s) for s in train]), _targets(train)
    )
    predicted = _ridge_predict(
        np.stack([_raw_features(s) for s in evaluation]), weights
    )
    maps = np.stack([_normalize(np.clip(row, 0.0, None)) for row in predicted])
    return maps, {"eligible_sample_count": len(evaluation)}


def _control_assume_all_visible(
    train: list[_Sample], evaluation: list[_Sample], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deny that anything is occluded: spread belief over what is visible."""

    rows = []
    empty = 0
    for sample in evaluation:
        vector = _normalize(_visible_arena_mask(sample))
        empty += int(vector.sum() <= 0.0)
        rows.append(vector)
    return np.stack(rows), {
        "eligible_sample_count": len(evaluation),
        "empty_map_count": empty,
        "note": (
            "every hidden positive is by definition in a non-visible cell, so "
            "this control places its mass entirely outside the answer -- which "
            "is what refusing to represent occlusion costs"
        ),
    }


def _control_time_shuffled(
    train: list[_Sample], evaluation: list[_Sample], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Break the temporal order the extrapolation depends on.

    The velocity a track was last seen with is what a kinematic predictor
    integrates forward.  Shuffling time destroys that ordering, so this
    control keeps the machinery and removes the sequence information.
    """

    rows = []
    for sample in evaluation:
        vector = np.zeros(_SIDE * _SIDE, dtype=np.float64)
        for last_seen, velocity, hidden_steps in sample.hidden_tracks:
            # A velocity drawn independently of the observed one is exactly
            # what "the time order carried no information" would produce.
            scrambled = tuple(
                int(v) for v in rng.permutation(np.asarray(velocity, dtype=int))
            )
            position, moving = last_seen, scrambled
            for _ in range(hidden_steps):
                position, moving = _bounce_advance(position, moving, sample.static)
            vector[_cell_index(position)] = 1.0
        rows.append(_normalize(vector))
    return np.stack(rows), {"eligible_sample_count": len(evaluation)}


def _control_identity_scrambled(
    train: list[_Sample], evaluation: list[_Sample], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Swap which hidden track belongs to which last-seen position.

    **This is the control that stopped V5.** It needs at least two
    simultaneously hidden objects with known tracks; on events with a single
    hidden object there is no other identity to swap with.  The census below
    is the number the V5 attempt did not have.
    """

    rows = []
    eligible = 0
    for sample in evaluation:
        tracks = list(sample.hidden_tracks)
        vector = np.zeros(_SIDE * _SIDE, dtype=np.float64)
        if len(tracks) >= 2:
            eligible += 1
            order = list(range(len(tracks)))
            # A derangement, so no track keeps its own kinematics.
            order = order[1:] + order[:1]
            paired = [
                (tracks[index][0], tracks[swap][1], tracks[swap][2])
                for index, swap in enumerate(order)
            ]
        else:
            paired = tracks
        for last_seen, velocity, hidden_steps in paired:
            position, moving = last_seen, velocity
            for _ in range(hidden_steps):
                position, moving = _bounce_advance(position, moving, sample.static)
            vector[_cell_index(position)] = 1.0
        rows.append(_normalize(vector))
    return np.stack(rows), {
        "eligible_sample_count": eligible,
        "ineligible_sample_count": len(evaluation) - eligible,
        "eligibility_rule": "at least two simultaneously hidden tracked objects",
        "note": (
            "single-hidden-object events fall through to the unscrambled "
            "construction; only the eligible count is evidence about this "
            "control"
        ),
    }


def _control_random_labels(
    train: list[_Sample], evaluation: list[_Sample], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the raw-sensor probe against permuted targets."""

    targets = _targets(train)
    permuted = targets[rng.permutation(targets.shape[0])]
    weights = _ridge(np.stack([_raw_features(s) for s in train]), permuted)
    predicted = _ridge_predict(
        np.stack([_raw_features(s) for s in evaluation]), weights
    )
    maps = np.stack([_normalize(np.clip(row, 0.0, None)) for row in predicted])
    return maps, {"eligible_sample_count": len(evaluation)}


CONTROLS: dict[
    str,
    Callable[
        [list[_Sample], list[_Sample], np.random.Generator],
        tuple[np.ndarray, dict[str, Any]],
    ],
] = {
    "raw_sensor": _control_raw_sensor,
    "assume_all_visible": _control_assume_all_visible,
    "time_shuffled": _control_time_shuffled,
    "identity_scrambled": _control_identity_scrambled,
    "random_labels": _control_random_labels,
}


def run_constructability_census(
    train_seeds: list[int],
    evaluation_seeds: list[int],
    *,
    steps: int,
    warmup: int,
    turn_probability: float,
) -> dict[str, Any]:
    # Two of the five controls are fitted probes, so the same contamination
    # guard the benchmark uses applies here.
    _require_disjoint_seeds(train_seeds, evaluation_seeds)
    _require_stream_separation(list(train_seeds) + list(evaluation_seeds))
    train = _collect_many(
        train_seeds, steps=steps, warmup=warmup, turn_probability=turn_probability
    )
    evaluation = _collect_many(
        evaluation_seeds,
        steps=steps,
        warmup=warmup,
        turn_probability=turn_probability,
    )
    if not train or not evaluation:
        raise ValueError("development split produced no usable samples")

    multi_hidden = sum(1 for s in evaluation if len(s.hidden_tracks) >= 2)
    results: dict[str, Any] = {}
    for name, build in CONTROLS.items():
        rng = np.random.default_rng(_CONTROL_SEED)
        try:
            maps, census = build(train, evaluation, rng)
        except Exception as error:  # noqa: BLE001 - the failure IS the result
            results[name] = {
                "constructed": False,
                "failure": f"{type(error).__name__}: {error}",
            }
            continue
        results[name] = {
            "constructed": True,
            **census,
            "score": _score_maps(evaluation, maps),
        }

    return {
        "status": "development_only_non_gated",
        "gated": False,
        "purpose": (
            "F6/O2: demonstrate by execution that every V8 control can be "
            "constructed from randomized-world development events"
        ),
        "population": {
            "train_seed_count": len(train_seeds),
            "evaluation_seed_count": len(evaluation_seeds),
            "train_sample_count": len(train),
            "evaluation_sample_count": len(evaluation),
            "multi_hidden_evaluation_sample_count": multi_hidden,
        },
        "controls": results,
        "all_controls_constructed": all(
            entry["constructed"] for entry in results.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    registry = json.loads(arguments.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "development_only_non_gated":
        raise ValueError("only a development-only registry is accepted")
    contract = registry["coverage_contract"]
    report = run_constructability_census(
        [int(seed) for seed in registry["train_seeds"]],
        [int(seed) for seed in registry["evaluation_seeds"]],
        steps=int(contract["steps"]),
        warmup=int(contract["warmup"]),
        turn_probability=float(registry["selected_turn_probability"]),
    )
    report["registry"] = {
        "path": str(arguments.registry),
        "selection_digest_sha256": registry.get("selection_digest_sha256"),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
