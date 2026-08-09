"""Tests for the item-3 forward-model permanence benchmark.

The scientific point of the benchmark is that, on the randomized world, an
occupancy belief that models the hidden-maneuver kernel must clearly beat
belief-free constant-velocity extrapolation.  These tests lock that ordering
and the determinism of the harness, plus a smoke check of the GRU baseline.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cal.evaluation._permanence_gru_baseline import (
    _occupancy_loss,
    gru_predictor_pairs,
    parameter_count,
)
from cal.evaluation._permanence_slot_baseline import (
    parameter_count as slot_parameter_count,
    slot_predictor_maps,
)
from cal.evaluation.permanence_forward_benchmark import (
    _belief_occupancy,
    _cell_index,
    _collect,
    _episode_binned_score,
    _paired_seed_bootstrap,
    _rank,
    _score_maps,
    _uniform_field_maps,
    gru_capacity_sweep,
    run_benchmark,
)


def test_belief_beats_geometric_extrapolation():
    report = run_benchmark(
        [61000, 61001, 61002],
        [61100, 61101, 61102, 61103],
        steps=140,
        warmup=12,
        turn_probability=0.35,
    )
    belief = report["predictors"]["belief"]
    geometric = report["predictors"]["geometric"]
    assert report["evaluation_sample_count"] > 0
    # Hidden-field-conditioned ranking: a belief that represents uncertainty
    # ranks the true cell above the whole field of empty hidden cells far more
    # often than constant-velocity extrapolation. Chance top-1 ~ 1/candidates.
    assert belief["top1_accuracy"] > geometric["top1_accuracy"]
    assert belief["mrr"] > geometric["mrr"]
    assert (
        belief["top1_accuracy"]
        > belief["mean_any_positive_top1_chance"]
    )
    # And it is much better calibrated: constant-velocity extrapolation puts a
    # confident point mass on a single (often now-visible, out-of-field) cell,
    # so its field-normalized categorical NLL explodes.
    assert belief["categorical_nll"] < geometric["categorical_nll"]


def test_first_hidden_transition_cannot_turn():
    visible = np.zeros((25, 25), dtype=bool)
    occupancy = _belief_occupancy(
        (10, 12),
        (1, 0),
        1,
        frozenset(),
        visible,
        0.35,
    )
    assert occupancy == {(11, 12): 1.0}


def test_collected_velocities_are_cardinal_unit_vectors():
    samples = _collect(61104, steps=200, warmup=12, turn_probability=0.35)
    assert samples
    assert all(
        abs(sample.observed_velocity[0]) + abs(sample.observed_velocity[1]) == 1
        for sample in samples
    )
    assert all(
        len(sample.hidden_tracks) == sample.hidden_object_count
        for sample in samples
    )


def test_benchmark_is_deterministic():
    kwargs = dict(steps=120, warmup=12, turn_probability=0.35)
    first = run_benchmark([61000, 61001], [61100, 61101], **kwargs)
    second = run_benchmark([61000, 61001], [61100, 61101], **kwargs)
    assert first["predictors"]["belief"] == second["predictors"]["belief"]
    assert first["predictors"]["geometric"] == second["predictors"]["geometric"]


def test_gru_baseline_within_budget_and_runs():
    assert parameter_count() < 100_000  # V2 learnable-parameter budget
    train = _collect(61000, steps=140, warmup=12, turn_probability=0.35)
    evaluation = _collect(61100, steps=140, warmup=12, turn_probability=0.35)
    assert train and evaluation
    pairs = gru_predictor_pairs(train, evaluation, epochs=2)
    assert len(pairs) == len(evaluation)
    for p_pos, p_neg in pairs:
        assert 0.0 <= p_pos <= 1.0
        assert 0.0 <= p_neg <= 1.0


def test_neural_objective_is_field_wide_and_decoy_invariant():
    sample = _collect(61000, steps=140, warmup=12, turn_probability=0.35)[0]
    alternate_negative = next(
        cell
        for cell in sample.candidate_cells
        if cell not in sample.positives and cell != sample.negative
    )
    altered = replace(sample, negative=alternate_negative)
    logits = torch.linspace(-1.0, 1.0, 121).reshape(1, -1)
    targets = torch.from_numpy(sample.hidden_occupancy.ravel()).reshape(1, -1)

    original_loss = _occupancy_loss(logits, targets, [sample])
    altered_loss = _occupancy_loss(logits, targets, [altered])
    assert original_loss == pytest.approx(altered_loss)

    outside_field = next(
        index
        for index in range(121)
        if index not in {_cell_index(cell) for cell in sample.candidate_cells}
    )
    perturbed = logits.clone()
    perturbed[0, outside_field] += 100.0
    assert _occupancy_loss(perturbed, targets, [sample]) == pytest.approx(
        original_loss
    )

    field_indices = torch.as_tensor(
        [_cell_index(cell) for cell in sample.candidate_cells]
    )
    positive_indices = torch.as_tensor(
        [_cell_index(cell) for cell in sample.positives]
    )
    field_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0, field_indices], targets[0, field_indices]
    )
    probabilities = torch.sigmoid(logits[0])
    evaluator_categorical_nll = -torch.log(
        probabilities[positive_indices].sum() / probabilities[field_indices].sum()
    )
    assert original_loss == pytest.approx(field_bce + evaluator_categorical_nll)


def test_benchmark_reports_argmax_and_occlusion_bins():
    report = run_benchmark(
        [61000, 61001],
        [61100, 61101],
        steps=140,
        warmup=12,
        turn_probability=0.35,
    )
    for predictor in ("belief", "geometric"):
        assert report["predictors"][predictor]["argmax_position_error"] is not None
        assert report["ranking_by_occlusion_length"][predictor]
    # A belief that models maneuvers ranks the true cell above the field more
    # often than extrapolation (geometric's point mass can have a smaller argmax
    # distance yet a worse hidden-field-conditioned rank).
    assert (
        report["predictors"]["belief"]["top1_accuracy"]
        > report["predictors"]["geometric"]["top1_accuracy"]
    )
    assert report["predictors"]["belief"]["empty_map_rate"] == 0.0
    assert "global_coordinate" in report["leakage_audit"]["baselines"]
    assert "field_geometry" in report["leakage_audit"]["baselines"]
    assert (
        "duration_conditioned_coordinate"
        in report["leakage_audit"]["baselines"]
    )
    assert report["leakage_audit"]["evaluation_seed_in_model_inputs"] is False
    assert (
        report["leakage_audit"]["evaluation_positive_labels_in_model_inputs"]
        is False
    )
    assert report["leakage_audit"]["shared_hidden_field_mask_in_model_inputs"] is True
    coverage = report["evaluation_seed_coverage"]
    assert set(coverage["accepted_samples_by_seed"]) == {"61100", "61101"}
    assert sum(coverage["accepted_samples_by_seed"].values()) == report[
        "evaluation_sample_count"
    ]


def test_empty_map_is_penalized_and_reported():
    sample = _collect(61100, steps=140, warmup=12, turn_probability=0.35)[0]
    score = _score_maps([sample], np.zeros((1, 121), dtype=np.float64))
    assert score["argmax_position_error"] == 20.0
    assert score["empty_map_rate"] == 1.0


def test_mass_only_outside_hidden_field_is_an_explicit_miss():
    sample = _collect(61100, steps=140, warmup=12, turn_probability=0.35)[0]
    field = set(sample.candidate_cells)
    outside_index = next(
        index
        for index in range(121)
        if (index % 11 + 7, index // 11 + 7) not in field
    )
    occupancy = np.zeros((1, 121), dtype=np.float64)
    occupancy[0, outside_index] = 1.0
    score = _score_maps([sample], occupancy)
    assert score["top1_accuracy"] == 0.0
    assert score["argmax_position_error"] == 20.0
    assert score["empty_map_rate"] == 1.0


def test_uniform_field_ties_are_invariant_to_candidate_order():
    sample = next(
        sample
        for sample in _collect(
            62001, steps=200, warmup=12, turn_probability=0.35
        )
        if sample.hidden_object_count == 1
    )
    uniform = np.ones(121, dtype=np.float64)
    forward = _rank(uniform, sample)
    reversed_sample = replace(
        sample, candidate_cells=tuple(reversed(sample.candidate_cells))
    )
    backward = _rank(uniform, reversed_sample)
    assert forward["top1"] == pytest.approx(
        len(sample.positives) / len(sample.candidate_cells)
    )
    assert forward["distance"] > 0.0
    assert forward["top1"] == backward["top1"]
    assert forward["distance"] == backward["distance"]
    assert forward["rr"] == backward["rr"]


def test_uniform_field_baseline_is_calibrated_and_brier_is_proper():
    sample = next(
        sample
        for sample in _collect(
            62001, steps=200, warmup=12, turn_probability=0.35
        )
        if sample.hidden_object_count == 1
    )
    uniform = _uniform_field_maps([sample])
    field_probability = 1.0 / len(sample.candidate_cells)
    assert all(
        uniform[0, _cell_index(cell)] == pytest.approx(field_probability)
        for cell in sample.candidate_cells
    )
    uniform_score = _score_maps([sample], uniform)
    expected_brier = (
        len(sample.candidate_cells) - 1
    ) / len(sample.candidate_cells) ** 2
    assert uniform_score["brier"] == pytest.approx(expected_brier)

    perfect = np.zeros((1, 121), dtype=np.float64)
    perfect[0, _cell_index(sample.positive)] = 1.0
    assert _score_maps([sample], perfect)["brier"] == 0.0


def test_runner_reports_exact_any_positive_random_chance():
    report = run_benchmark(
        [61000, 61001],
        [61100, 61101],
        steps=140,
        warmup=12,
        turn_probability=0.35,
    )
    expected = np.mean(
        [
            len(sample.positives) / len(sample.candidate_cells)
            for seed in (61100, 61101)
            for sample in _collect(
                seed, steps=140, warmup=12, turn_probability=0.35
            )
        ]
    )
    assert report["predictors"]["belief"][
        "mean_any_positive_top1_chance"
    ] == pytest.approx(expected)


def test_occlusion_length_bins_exclude_multi_hidden_events():
    report = run_benchmark(
        [61000, 61001],
        [61100, 61101],
        steps=140,
        warmup=12,
        turn_probability=0.35,
    )
    metadata = report["occlusion_length_binning"]
    assert metadata["policy"] == "single_hidden_object_events_only"
    assert (
        metadata["included_sample_count"]
        + metadata["excluded_multi_hidden_sample_count"]
        == report["evaluation_sample_count"]
    )
    primary = report["episode_binned_predictors"]["belief"]
    assert report["ranking_by_occlusion_length"]["belief"] == primary[
        "by_occlusion_length"
    ]
    assert set(report["ranking_by_occlusion_length"]["belief"]) == {
        "2-3",
        "4-5",
        "6+",
    }
    assert all(
        value["episode_bin_count"] <= metadata["included_sample_count"]
        for value in report["ranking_by_occlusion_length"]["belief"].values()
    )
    assert set(primary["seed_bin_scores"]) == {
        str(seed) for seed in primary["complete_seed_ids"]
    }
    assert all(
        set(seed_bins) == {"2-3", "4-5", "6+"}
        for seed_bins in primary["seed_bin_scores"].values()
    )


def test_episode_binned_score_gives_long_and_short_episodes_equal_weight():
    base = next(
        sample
        for sample in _collect(
            62001, steps=200, warmup=12, turn_probability=0.35
        )
        if sample.hidden_object_count == 1
    )
    long_episode = [
        replace(base, hidden_steps=2, focus_occlusion_id=100) for _ in range(100)
    ]
    short_episode = [replace(base, hidden_steps=2, focus_occlusion_id=101)]
    samples = long_episode + short_episode
    hit = np.zeros(121, dtype=np.float64)
    hit[_cell_index(base.positive)] = 1.0
    maps = np.stack(
        [hit.copy() for _ in long_episode]
        + [np.zeros(121, dtype=np.float64) for _ in short_episode]
    )
    aggregated = _episode_binned_score(samples, maps)
    assert aggregated["episode_bin_group_count"] == 2
    assert aggregated["complete_seed_count"] == 0
    assert aggregated["by_occlusion_length"] == {}
    assert aggregated["seed_scores"][str(base.seed)]["top1_accuracy"] == 0.5


def test_paired_bootstrap_uses_complete_seed_scores_and_is_deterministic():
    kwargs = dict(
        steps=200,
        warmup=12,
        turn_probability=0.35,
        paired_bootstrap_samples=500,
    )
    first = run_benchmark(
        [62001, 62002, 62003],
        [62077, 62078, 62079, 62080],
        **kwargs,
    )
    second = run_benchmark(
        [62001, 62002, 62003],
        [62077, 62078, 62079, 62080],
        **kwargs,
    )
    comparison = first["paired_seed_bootstrap"]["belief_vs_geometric"]
    assert comparison == second["paired_seed_bootstrap"]["belief_vs_geometric"]
    # 62080 only yields 2-3 events under the HMAC-keyed hidden stream, so the
    # complete-seed requirement drops it -- which is the behaviour under test.
    assert comparison["paired_seed_count"] == 3
    assert comparison["paired_seed_ids"] == [62077, 62078, 62079]
    assert comparison["bootstrap_samples"] == 500
    assert set(comparison["metrics"]) == {
        "top1_accuracy",
        "categorical_nll",
        "brier",
        "argmax_position_error",
    }


def test_paired_bootstrap_excludes_incomplete_seeds_and_preserves_pairing():
    def scores(values: dict[str, tuple[bool, float]]) -> dict[str, object]:
        return {
            "seed_scores": {
                seed: {
                    "complete_bins": complete,
                    "top1_accuracy": value,
                    "categorical_nll": 1.0 - value,
                    "brier": 1.0 - value,
                    "argmax_position_error": 1.0 - value,
                }
                for seed, (complete, value) in values.items()
            }
        }

    first = scores({"10": (True, 0.8), "11": (False, 1.0), "12": (True, 0.6)})
    second = scores({"10": (True, 0.5), "11": (True, 0.0), "12": (True, 0.4)})
    comparison = _paired_seed_bootstrap(
        first,
        second,
        first_name="first",
        second_name="second",
        bootstrap_samples=100,
        rng_seed=7,
    )
    assert comparison["paired_seed_ids"] == [10, 12]
    assert comparison["paired_seed_count"] == 2
    assert comparison["metrics"]["top1_accuracy"][
        "advantage_positive_is_better"
    ] == pytest.approx(0.25)
    assert comparison["metrics"]["brier"][
        "advantage_positive_is_better"
    ] == pytest.approx(0.25)


def test_empty_benchmark_configuration_has_clear_error():
    with pytest.raises(ValueError, match="no valid hidden-object samples"):
        run_benchmark(
            [61000],
            [61100],
            steps=0,
            warmup=12,
            turn_probability=0.35,
        )


def test_slot_baseline_runs_and_shapes():
    assert slot_parameter_count() > 0
    train = _collect(61000, steps=140, warmup=12, turn_probability=0.35)
    evaluation = _collect(61100, steps=140, warmup=12, turn_probability=0.35)
    maps = slot_predictor_maps(train, evaluation, epochs=2)
    assert maps.shape == (len(evaluation), 121)
    assert (maps >= 0.0).all() and (maps <= 1.0).all()


def test_entity_graph_predictor_runs():
    report = run_benchmark(
        [61000],
        [61100],
        steps=140,
        warmup=12,
        turn_probability=0.35,
        include_entity_graph=True,
    )
    entity_graph = report["predictors"]["entity_graph"]
    assert 0.0 <= entity_graph["top1_accuracy"] <= 1.0
    assert 0.0 <= entity_graph["mrr"] <= 1.0
    assert entity_graph["argmax_position_error"] is not None
    assert report["ranking_by_occlusion_length"]["entity_graph"]


def test_gru_capacity_sweep_structure():
    result = gru_capacity_sweep(
        [61000, 61001, 61002],
        [61100, 61101],
        steps=140,
        warmup=12,
        turn_probability=0.35,
        hidden_sizes=(16, 64),
        epochs_grid=(3,),
    )
    assert len(result["runs"]) == 2
    for run in result["runs"]:
        assert run["parameter_count"] > 0
        assert 0.0 <= run["top1_accuracy"] <= 1.0
        assert 0.0 <= run["mrr"] <= 1.0


# The pairwise positive/negative construction is 99.973% separable by a pure
# parity rule -- 99.95% of negatives sit at Manhattan distance 1 from the
# positive, so their coordinate parities differ deterministically (2026-08-08
# review, finding F18).  It was confirmed at the time that no frozen gate reads
# it: every gated metric ranks over the whole hidden field (`candidate_cells`).
# The review attached a permanent prohibition on it ever returning.  A
# prohibition nobody checks is a comment, so this checks it.
_GATED_SCORING_MODULES = (
    "cal/evaluation/stochastic_permanence_benchmark.py",
    "cal/evaluation/stochastic_permanence_artifacts.py",
    "cal/evaluation/stochastic_permanence_phase0.py",
    "cal/evaluation/stochastic_permanence_capacity_artifacts.py",
    "cal/evaluation/stochastic_permanence_kernel_diagnostic.py",
    "cal/evaluation/_permanence_belief_free_baseline.py",
)

# Everything in `permanence_forward_benchmark` that produces a number a gate
# later reads.  The module also hosts non-gated baselines (GRU, slot), which
# may legitimately use the pairwise field.
_GATED_SCORING_FUNCTIONS = (
    "_rank",
    "_score_maps",
    "_sample_metric_record",
    "_mean_metric_records",
    "_episode_binned_score",
    "_belief_map",
    "_geometric_map",
    "_fit_prior_maps",
    "_uniform_field_maps",
    "_wasted_field_mass",
    "_paired_seed_bootstrap",
)


def _reads_pairwise_negative(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute)
        and child.attr == "negative"
        and isinstance(child.ctx, ast.Load)
        for child in ast.walk(node)
    )


def test_pairwise_negative_construction_stays_out_of_every_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for relative in _GATED_SCORING_MODULES:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        if _reads_pairwise_negative(tree):
            offenders.append(relative)

    benchmark = root / "cal/evaluation/permanence_forward_benchmark.py"
    tree = ast.parse(benchmark.read_text(encoding="utf-8"))
    scoring = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in _GATED_SCORING_FUNCTIONS
    }
    missing = sorted(set(_GATED_SCORING_FUNCTIONS) - set(scoring))
    assert not missing, (
        "the gated scoring path was renamed; this prohibition now guards "
        f"nothing: {missing}"
    )
    offenders.extend(
        f"permanence_forward_benchmark.py::{name}"
        for name, node in sorted(scoring.items())
        if _reads_pairwise_negative(node)
    )

    assert not offenders, (
        "the pairwise positive/negative construction is parity-separable "
        "(balanced accuracy 0.99973) and is permanently barred from every "
        f"gate; it is read by: {offenders}"
    )


def test_core_entry_points_refuse_overlapping_train_and_eval_seeds() -> None:
    """The contamination guard used to live only in the gated callers.

    `--train-seeds 101` against the default bases is enough to overlap, and the
    review measured a position prior jumping from 0.044 to 0.243 top-1 once it
    did (finding F10).  Reports produced straight from these functions carried
    no protection at all.
    """

    overlapping = ([61000, 61001], [61001, 61002])
    for call in (
        lambda: run_benchmark(
            *overlapping, steps=40, warmup=12, turn_probability=0.45
        ),
        lambda: gru_capacity_sweep(
            *overlapping,
            steps=40,
            warmup=12,
            turn_probability=0.45,
            hidden_sizes=(16,),
            epochs_grid=(1,),
        ),
    ):
        with pytest.raises(RuntimeError, match="seed collision between streams"):
            call()


def test_action_and_layout_streams_cannot_collide_across_a_seed_population() -> None:
    """Layout is keyed on the seed and actions on seed + 50,000.

    Two episodes exactly that far apart share a stream -- one's actions replay
    the other's layout draw.  The spacing was assumed, never asserted (finding
    F23).
    """

    with pytest.raises(ValueError, match="layout stream as another's action"):
        run_benchmark(
            [61000],
            [61000 + 50_000],
            steps=40,
            warmup=12,
            turn_probability=0.45,
        )


def test_cli_refuses_to_override_parameters_the_registry_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry-bound report is stamped with that registry's digest.

    `--turn-probability` used to take effect *after* the registry was read
    while the report kept wearing the digest, which made the provenance a
    false claim (review finding F11).
    """

    from cal.evaluation.permanence_forward_benchmark import main

    registry = (
        Path(__file__).resolve().parents[1]
        / "experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "permanence_forward_benchmark",
            "--seed-registry",
            str(registry),
            "--turn-probability",
            "0.35",
        ],
    )
    with pytest.raises(ValueError, match="bound by the seed registry"):
        main()
