"""Privileged Phase-R conformance and capacity diagnostic.

This runner intentionally reads true static/visibility and exact kernel values.
Every output is marked ``privileged_diagnostic_only`` and cannot be loaded as a
formal candidate artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from math import log
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from cal.evaluation.permanence_forward_benchmark import (
    _Sample,
    _collect_many,
    _successors,
)
from cal.evaluation.stochastic_permanence_artifacts import (
    source_lock,
    verify_locked_sources,
)
from cal.evaluation.stochastic_permanence_capacity_artifacts import (
    CAPACITY_ARTIFACT_SCHEMA_VERSION,
    write_capacity_artifact,
)
from cal.evaluation.permanence_seed_registry import coverage_contract
from cal.evaluation.randomized_occlusion_world import GRID_SIZE
from cal.evaluation.v2_i1_integration import ARENA_HIGH, ARENA_LOW
from cal.evaluation.stochastic_permanence_custody import (
    validate_disjoint_seed_sets,
)
from cal.model.stochastic_motion_filter import (
    MAX_SUCCESSORS_PER_STATE,
    EmptyPosteriorError,
    GridSpec,
    PackedKinematicFilter,
    PackedPosteriorPool,
    autonomous_successors,
    position_total_variation,
)


DEFAULT_REGISTRY = Path(
    "experiments/V2_P1_PERMANENCE_DEVELOPMENT_SEED_REGISTRY_V4.json"
)
# V6: regenerated after the 2026-08-09 correctness pass, which changed the
# capacity artifact schema (3 -> 4) by replacing the always-true gates.
DEFAULT_OUTPUT = Path(
    "experiments/V2_I1_P1_PHASE_R_CAPACITY_CONFORMANCE_DEVELOPMENT_V6.json"
)
# The one place the filter's arena is stated, derived from the world it is
# scored against instead of repeated as a literal.  ``GridSpec`` deliberately
# has no defaults, so an upstream arena change is now an import-time mismatch
# rather than a filter that silently calls live cells certain walls.
EVALUATION_GRID_SPEC = GridSpec(
    grid_size=GRID_SIZE, arena_low=ARENA_LOW, arena_high=ARENA_HIGH
)
DEFAULT_K_MAX = 96
# Selected by the turn-probability scan on registry V4, and independently
# re-selected on V3 before it: the scan moved off 0.35 once the HMAC-keyed
# hidden stream changed which seeds qualify, and 0.45 held at both 64 and 150
# evaluation seeds.
DEFAULT_TURN_PROBABILITY = 0.45
ACTIVE_STATE_LIMIT = 65_536
PARAMETER_LIMIT = 100_000
MAC_LIMIT = 5_000_000
FORMAL_STEPS_PER_SEED_LIMIT = 100_000
TRAIN_REPLAY_LIMIT = 4
CPU_TOTAL_SECONDS_LIMIT = 2 * 60 * 60


class PrivilegedUnprunedKinematicReference:
    """Unbounded dictionary reference; evaluation namespace by design."""

    privileged_diagnostic_only = True

    def __init__(self, spec: GridSpec = EVALUATION_GRID_SPEC) -> None:
        self.spec = spec
        self._probability: dict[int, float] = {}
        self.branch_log_weight = 0.0

    def reset(
        self, position: tuple[int, int], velocity: tuple[int, int]
    ) -> None:
        self._probability = {self.spec.encode(position, velocity): 1.0}
        self.branch_log_weight = 0.0

    def step(
        self,
        *,
        static_probability: np.ndarray,
        no_detection_probability: np.ndarray,
        turn_probability: float,
        allow_turn: bool,
    ) -> dict[str, float | int]:
        if not self._probability:
            raise EmptyPosteriorError("reference has not been reset")
        static_array = np.asarray(static_probability, dtype=np.float64)
        expected_shape = (self.spec.grid_size, self.spec.grid_size)
        if static_array.shape != expected_shape:
            raise ValueError(f"static_probability must have shape {expected_shape}")
        if not np.all(np.isfinite(static_array)):
            raise ValueError("privileged static topology must be finite")
        if np.any((static_array != 0.0) & (static_array != 1.0)):
            raise ValueError("privileged reference requires known binary topology")
        true_static = frozenset(
            (int(x), int(y))
            for y, x in np.argwhere(static_array == 1.0)
        )
        no_detection = np.asarray(no_detection_probability, dtype=np.float64)
        if no_detection.shape != expected_shape:
            raise ValueError(
                f"no_detection_probability must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(no_detection)) or np.any(
            (no_detection < 0.0) | (no_detection > 1.0)
        ):
            raise ValueError("no_detection_probability must be finite in [0, 1]")
        expanded: dict[int, float] = {}
        for code, mass in self._probability.items():
            position, velocity = self.spec.decode(code)
            for new_position, new_velocity, transition in _successors(
                position,
                velocity,
                true_static,
                turn_probability if allow_turn else 0.0,
            ):
                new_code = self.spec.encode(new_position, new_velocity)
                expanded[new_code] = expanded.get(new_code, 0.0) + mass * transition
        conditioned: dict[int, float] = {}
        for code, mass in expanded.items():
            position, _velocity = self.spec.decode(code)
            value = mass * float(
                no_detection[position[1], position[0]]
            )
            if value > 0.0:
                conditioned[code] = value
        evidence = sum(conditioned.values())
        if evidence <= 0.0:
            raise EmptyPosteriorError("reference observation has zero evidence")
        self.branch_log_weight += log(evidence)
        self._probability = {
            code: mass / evidence for code, mass in conditioned.items()
        }
        return {
            "support": len(self._probability),
            "observation_evidence": evidence,
        }

    def position_marginal(self) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for code, mass in self._probability.items():
            position, _velocity = self.spec.decode(code)
            result[position] = result.get(position, 0.0) + mass
        return result


def _deep_size(value: object, visited: set[int] | None = None) -> int:
    seen = visited if visited is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        size += _deep_size(vars(value), seen)
    for slot in getattr(type(value), "__slots__", ()):
        if hasattr(value, slot):
            size += _deep_size(getattr(value, slot), seen)
    return size


def _distribution_map(
    distribution: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
) -> dict[tuple[tuple[int, int], tuple[int, int]], float]:
    result: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for position, velocity, probability in distribution:
        if probability <= 0.0:
            continue
        key = (position, velocity)
        result[key] = result.get(key, 0.0) + float(probability)
    return result


def _known_topology_kernel_alignment(
    *,
    spec: GridSpec = EVALUATION_GRID_SPEC,
) -> dict[str, float | int]:
    """Exhaustively compare the candidate kernel with the real local world kernel."""

    checked = 0
    maximum_l1 = 0.0
    support_mismatches = 0
    maximum_successors = 0
    for y in range(spec.arena_low, spec.arena_high + 1):
        for x in range(spec.arena_low, spec.arena_high + 1):
            position = (x, y)
            local_cells = [
                cell
                for cell in (
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                )
                if spec.arena_low <= cell[0] <= spec.arena_high
                and spec.arena_low <= cell[1] <= spec.arena_high
            ]
            for mask in range(1 << len(local_cells)):
                true_static = frozenset(
                    cell
                    for index, cell in enumerate(local_cells)
                    if mask & (1 << index)
                )
                static_probability = np.zeros(
                    (spec.grid_size, spec.grid_size), dtype=np.float64
                )
                for static_x, static_y in true_static:
                    static_probability[static_y, static_x] = 1.0
                for velocity in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    for allow_turn in (False, True):
                        for turn_probability in (0.0, 0.35, 1.0):
                            candidate = _distribution_map(
                                autonomous_successors(
                                    position,
                                    velocity,
                                    static_probability,
                                    turn_probability=turn_probability,
                                    allow_turn=allow_turn,
                                    spec=spec,
                                )
                            )
                            world = _distribution_map(
                                _successors(
                                    position,
                                    velocity,
                                    true_static,
                                    turn_probability if allow_turn else 0.0,
                                )
                            )
                            keys = set(candidate) | set(world)
                            l1 = sum(
                                abs(candidate.get(key, 0.0) - world.get(key, 0.0))
                                for key in keys
                            )
                            maximum_l1 = max(maximum_l1, l1)
                            support_mismatches += int(
                                candidate.keys() != world.keys()
                            )
                            maximum_successors = max(
                                maximum_successors, len(candidate)
                            )
                            checked += 1
    return {
        "checked_transition_cases": checked,
        "support_mismatch_count": support_mismatches,
        "maximum_probability_l1": maximum_l1,
        # The expansion workspace is sized `MAX_SUCCESSORS_PER_STATE * k_max`,
        # and the gate on it used to compare that allocation against the same
        # formula.  This sweep already enumerates every reachable local
        # topology, so it can say what the real branching factor is and make
        # the bound falsifiable (review finding F19).
        "maximum_successors_per_state": maximum_successors,
    }


def _pool_atomic_overflow_safe(pool: PackedPosteriorPool) -> bool:
    codes_before = pool.codes.copy()
    probability_before = pool.probability.copy()
    counts_before = pool.counts.copy()
    try:
        pool.replace_factor_atomic(
            0,
            0,
            np.zeros(pool.k_max + 1, dtype=np.uint16),
            np.ones(pool.k_max + 1, dtype=np.float64),
        )
    except RuntimeError:
        return bool(
            np.array_equal(pool.codes, codes_before)
            and np.array_equal(pool.probability, probability_before)
            and np.array_equal(pool.counts, counts_before)
        )
    return False


def _longest_single_hidden_episodes(
    samples: Sequence[_Sample],
) -> list[_Sample]:
    selected: dict[tuple[int, str, int], _Sample] = {}
    for sample in samples:
        if sample.hidden_object_count != 1:
            continue
        key = (sample.seed, sample.focus_object, sample.focus_occlusion_id)
        previous = selected.get(key)
        if previous is None or sample.hidden_steps > previous.hidden_steps:
            selected[key] = sample
    return [selected[key] for key in sorted(selected)]


def _static_probability(sample: _Sample) -> np.ndarray:
    result = np.zeros_like(sample.visible, dtype=np.float64)
    for x, y in sample.static:
        result[y, x] = 1.0
    return result


def _episode_conformance(
    sample: _Sample,
    *,
    k_max: int,
    turn_probability: float,
) -> dict[str, Any]:
    exact = PrivilegedUnprunedKinematicReference()
    packed = PackedKinematicFilter(k_max, spec=EVALUATION_GRID_SPEC)
    exact.reset(sample.last_seen, sample.observed_velocity)
    packed.reset(sample.last_seen, sample.observed_velocity)
    static_probability = _static_probability(sample)
    no_detection_probability = (~sample.visible).astype(np.float64)
    checkpoints: list[dict[str, Any]] = []
    maximum_tv = 0.0
    maximum_reference_support = 1
    maximum_tv_checkpoint = 0
    # Accumulate the evidence term only.  The pruning term is taken at the end
    # from `cumulative_retained_probability`, which the filter maintains as a
    # separate running product.  Re-adding the filter's own reported
    # `retained_probability` here -- as this did -- reconstructed
    # `branch_log_weight` from exactly the numbers that built it, so the
    # residual was identically zero and the gate could not fail (finding F19).
    # Cross-checking the two independently maintained accumulators can.
    observation_log_evidence = 0.0
    for hidden_index in range(sample.hidden_steps):
        exact_step = exact.step(
            static_probability=static_probability,
            no_detection_probability=no_detection_probability,
            turn_probability=turn_probability,
            allow_turn=hidden_index > 0,
        )
        packed_step = packed.step(
            static_probability=static_probability,
            no_detection_probability=no_detection_probability,
            turn_probability=turn_probability,
            allow_turn=hidden_index > 0,
        )
        observation_log_evidence += log(
            float(packed_step["observation_evidence"])
        )
        tv = position_total_variation(
            packed.position_marginal(), exact.position_marginal()
        )
        # `tv >= maximum_tv` after the max is always true, so this recorded the
        # last step rather than the first step to reach the maximum (F21).
        if tv > maximum_tv:
            maximum_tv = tv
            maximum_tv_checkpoint = hidden_index + 1
        maximum_reference_support = max(
            maximum_reference_support, int(exact_step["support"])
        )
        checkpoints.append(
            {
                "hidden_step": hidden_index + 1,
                "reference_support": int(exact_step["support"]),
                "packed_pre_pruning_support": int(
                    packed_step["pre_pruning_support"]
                ),
                "packed_retained_support": int(
                    packed_step["retained_support"]
                ),
                # The reference drops zero-mass states, so this is the
                # quantity actually comparable with `reference_support`.
                "packed_retained_positive_support": int(
                    packed_step["retained_positive_support"]
                ),
                "retained_probability": float(
                    packed_step["retained_probability"]
                ),
                "cumulative_pruned_mass": float(
                    packed_step["cumulative_pruned_mass"]
                ),
                "position_tv": tv,
                "rendered_dynamic_occupancy_l1": 2.0 * tv,
            }
        )
    return {
        "seed": sample.seed,
        "focus_object": sample.focus_object,
        "focus_occlusion_id": sample.focus_occlusion_id,
        "hidden_steps": sample.hidden_steps,
        "maximum_reference_support": maximum_reference_support,
        "maximum_position_tv": maximum_tv,
        "maximum_position_tv_checkpoint": maximum_tv_checkpoint,
        "cumulative_pruned_mass": packed.cumulative_pruned_mass,
        "maximum_step_pruned_mass": packed.maximum_step_pruned_mass,
        "packed_branch_log_weight": packed.branch_log_weight,
        "reference_branch_log_weight": exact.branch_log_weight,
        "branch_log_weight_gap": (
            exact.branch_log_weight - packed.branch_log_weight
        ),
        "packed_branch_accounting_residual": abs(
            packed.branch_log_weight
            - observation_log_evidence
            - log(packed.cumulative_retained_probability)
        ),
        "checkpoints": checkpoints,
    }


def run_phase_r_diagnostic(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    k_max: int = DEFAULT_K_MAX,
    turn_probability: float = DEFAULT_TURN_PROBABILITY,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else Path(__file__).resolve().parents[2]
    )
    # See `run_phase0`: the lock is verified before any evidence is produced,
    # not audited afterwards (review finding F8 / G7).
    verify_locked_sources(root=root)
    registry_source = Path(registry_path)
    if not registry_source.is_absolute():
        registry_source = root / registry_source
    registry = json.loads(registry_source.read_text(encoding="utf-8"))
    if registry.get("status") != "development_only_non_gated":
        raise RuntimeError("Phase R accepts only a development-only registry")
    if registry.get("model_metrics_read") is not False:
        raise RuntimeError("Phase R registry read model metrics")
    if registry.get("coverage_contract") != coverage_contract():
        raise RuntimeError("Phase R registry coverage contract mismatch")
    validate_disjoint_seed_sets(
        {
            "train": registry["train_seeds"],
            "development": registry["evaluation_seeds"],
        }
    )
    registry_turn_probability = float(registry["selected_turn_probability"])
    evaluation_seeds = [int(seed) for seed in registry["evaluation_seeds"]]
    audit: Counter[str] = Counter()
    samples = _collect_many(
        evaluation_seeds,
        steps=int(registry["coverage_contract"]["steps"]),
        warmup=int(registry["coverage_contract"]["warmup"]),
        turn_probability=turn_probability,
        audit=audit,
    )
    episodes = _longest_single_hidden_episodes(samples)
    conformance: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for sample in episodes:
        try:
            conformance.append(
                _episode_conformance(
                    sample,
                    k_max=k_max,
                    turn_probability=turn_probability,
                )
            )
        except (EmptyPosteriorError, RuntimeError, ValueError) as error:
            errors.append(
                {
                    "seed": sample.seed,
                    "focus_object": sample.focus_object,
                    "focus_occlusion_id": sample.focus_occlusion_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    pool = PackedPosteriorPool(
        k_max=k_max,
        grid_size=EVALUATION_GRID_SPEC.grid_size,
        arena_low=EVALUATION_GRID_SPEC.arena_low,
        arena_high=EVALUATION_GRID_SPEC.arena_high,
    )
    pool.fill_fully_detached()
    measured_deep_size = _deep_size(pool)
    # What this diagnostic actually spent, so the research-budget gate can
    # compare against the declared contract instead of against itself.  Phase R
    # replays no training stream at all.
    measured_steps_per_seed = int(registry["coverage_contract"]["steps"])
    measured_train_replays = 0
    kernel_alignment = _known_topology_kernel_alignment()
    atomic_overflow_safe = _pool_atomic_overflow_safe(pool)
    maximum_cumulative = max(
        (item["cumulative_pruned_mass"] for item in conformance), default=1.0
    )
    maximum_tv = max(
        (item["maximum_position_tv"] for item in conformance), default=1.0
    )
    maximum_branch_accounting_residual = max(
        (
            item["packed_branch_accounting_residual"]
            for item in conformance
        ),
        default=float("inf"),
    )
    source_paths = (
        Path(__file__),
        root / "cal/evaluation/stochastic_permanence_capacity_artifacts.py",
        root / "cal/model/stochastic_motion_filter.py",
        root / "cal/evaluation/permanence_forward_benchmark.py",
        root / "cal/evaluation/randomized_occlusion_world.py",
        root / "cal/evaluation/permanence_seed_registry.py",
        root / "cal/evaluation/stochastic_permanence_artifacts.py",
        root / "cal/evaluation/stochastic_permanence_custody.py",
        root / "cal/evaluation/v2_i1_integration.py",
        root / "cal/evaluation/v2_artifacts.py",
        root / "cal/infra/provenance.py",
        root / "cal/model/integrated_agent.py",
        root / "cal/model/entity_graph.py",
        root / "cal/model/occupancy.py",
        root / "docs/experiments/V2_I1_STOCHASTIC_PERMANENCE_PLAN.md",
        root / "docs/experiments/V2_I1_PHASE_R_CAPACITY_AMENDMENT_V3.md",
        registry_source,
        root / "pyproject.toml",
        root / "uv.lock",
    )
    provenance = {
        "registry_path": registry_source.relative_to(root).as_posix(),
        "registry_selection_digest_sha256": registry[
            "selection_digest_sha256"
        ],
        "registry_turn_probability": registry_turn_probability,
        "source_lock": source_lock(source_paths, root=root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "command": (
            "python -m cal.evaluation.stochastic_permanence_kernel_diagnostic "
            f"--registry {registry_source.relative_to(root).as_posix()} "
            f"--k-max {k_max} --turn-probability {turn_probability}"
        ),
    }
    resources = {
        "learnable_parameter_count": pool.learnable_parameter_count,
        "persistent_array_bytes": pool.persistent_array_bytes,
        "declared_active_state_bytes": pool.active_state_bytes,
        "measured_full_object_deep_size_bytes": measured_deep_size,
        "estimated_mac_per_step": pool.estimated_mac_per_step,
        "resource_scope": "complete_persistent_candidate_prototype_object_graph",
        "active_state_limit_bytes": ACTIVE_STATE_LIMIT,
        "parameter_limit": PARAMETER_LIMIT,
        "mac_per_step_limit": MAC_LIMIT,
        "peak_transient_working_memory_bytes": (
            "python_allocator_not_portably_measurable; all array workspaces "
            "are preallocated and counted as persistent"
        ),
        "formal_research_budget_contract": {
            "steps_per_seed_maximum": FORMAL_STEPS_PER_SEED_LIMIT,
            "train_replay_maximum": TRAIN_REPLAY_LIMIT,
            "cpu_total_seconds_maximum": CPU_TOTAL_SECONDS_LIMIT,
        },
        "deterministic_diagnostic_work": {
            "episode_count": len(conformance),
            "transition_checkpoint_count": sum(
                len(item["checkpoints"]) for item in conformance
            ),
            "measured_steps_per_seed": measured_steps_per_seed,
            "measured_train_replays": measured_train_replays,
            **kernel_alignment,
        },
    }
    gates = {
        "episodes_present": bool(conformance),
        "no_filter_errors": not errors,
        "cumulative_pruned_mass": maximum_cumulative <= 0.01,
        "maximum_position_tv": maximum_tv <= 0.01,
        "branch_evidence_accounting": maximum_branch_accounting_residual
        <= 1e-12,
        "known_topology_kernel_alignment": bool(
            kernel_alignment["support_mismatch_count"] == 0
            and kernel_alignment["maximum_probability_l1"] <= 1e-12
        ),
        "fully_detached_pool_safe": bool(
            pool.capacity_contract()["fully_detached_safe"]
        ),
        "shared_expansion_workspace_safe": bool(
            pool.capacity_contract()["shared_expansion_workspace_size"]
            >= pool.capacity_contract()["shared_expansion_workspace_required"]
            and pool.capacity_contract()["direct_index_accumulator"]
            # The two sizes above both descend from MAX_SUCCESSORS_PER_STATE,
            # so on their own they only confirm the allocation matches its own
            # formula.  This term is what makes the bound falsifiable: the
            # exhaustive sweep says how many successors a state really has.
            and kernel_alignment["maximum_successors_per_state"]
            <= MAX_SUCCESSORS_PER_STATE
        ),
        "atomic_overflow_safe": atomic_overflow_safe,
        "declared_active_state": pool.active_state_bytes <= ACTIVE_STATE_LIMIT,
        "measured_active_state": measured_deep_size <= ACTIVE_STATE_LIMIT,
        "learnable_parameters": pool.learnable_parameter_count
        <= PARAMETER_LIMIT,
        "mac_per_step": pool.estimated_mac_per_step <= MAC_LIMIT,
        # The artifact validator compares these two at abs_tol 1e-12 and the
        # gate used np.isclose's rtol 1e-5, so a run whose probability the
        # registry does not bind could produce a gate the validator then
        # refused to serialize -- an honest no-go that could not be written
        # down (review finding F20).  One predicate now, the stricter one.
        "registry_turn_probability": math.isclose(
            turn_probability,
            registry_turn_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        # Declaring a budget is not respecting one.  The gate compared the
        # module's own constants with the same literals in the same module and
        # could not fail (review finding F19); it now also carries the
        # measured work.
        "formal_research_budget_respected": bool(
            FORMAL_STEPS_PER_SEED_LIMIT == 100_000
            and TRAIN_REPLAY_LIMIT == 4
            and CPU_TOTAL_SECONDS_LIMIT == 7_200
            and measured_steps_per_seed <= FORMAL_STEPS_PER_SEED_LIMIT
            and measured_train_replays <= TRAIN_REPLAY_LIMIT
        ),
    }
    return {
        "artifact_kind": "stochastic_permanence_capacity_conformance",
        "artifact_schema_version": CAPACITY_ARTIFACT_SCHEMA_VERSION,
        "status": "development_only_non_gated",
        "privileged_diagnostic_only": True,
        "candidate_artifact_eligible": False,
        "turn_probability": turn_probability,
        "capacity_contract": {
            **pool.capacity_contract(),
            "atomic_overflow_probe_passed": atomic_overflow_safe,
            "persistent_arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "nbytes": value.nbytes,
                }
                for name, value in pool.persistent_arrays.items()
            },
            "cumulative_pruned_mass_definition": "1-product(retained_probability)",
            "tv_definition": (
                "0.5*L1(position_marginal_packed,position_marginal_reference)"
            ),
        },
        "conformance": {
            "population": "longest_single_hidden_sample_per_focus_occlusion",
            "episode_count": len(conformance),
            "maximum_cumulative_pruned_mass": maximum_cumulative,
            "maximum_position_tv": maximum_tv,
            "maximum_branch_accounting_residual": (
                maximum_branch_accounting_residual
            ),
            "known_topology_kernel_alignment": kernel_alignment,
            "errors": errors,
            "episodes": conformance,
            "collection_audit": dict(sorted(audit.items())),
        },
        "resources": resources,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "phase_r_go" if all(gates.values()) else "phase_r_no_go",
        "provenance": provenance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument(
        "--turn-probability", type=float, default=DEFAULT_TURN_PROBABILITY
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    artifact = run_phase_r_diagnostic(
        registry_path=arguments.registry,
        k_max=arguments.k_max,
        turn_probability=arguments.turn_probability,
    )
    digest = write_capacity_artifact(
        arguments.output,
        artifact,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "passed": artifact["passed"],
                "decision": artifact["decision"],
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
