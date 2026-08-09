"""Strict validation helpers for the staged V2 result chain."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any


# Every schema this validator knows how to check.  It used to be the single
# literal 1, which quietly made the validator unusable for the V8 language
# readout at version 2 -- the only reachable outcome for a schema-2 artifact
# was rejection, so there was no way to authorize one (review finding F15).
# Adding a version here is a statement that the fields read below still mean
# what they meant at version 1.
SUPPORTED_RESULT_SCHEMA_VERSIONS = (1, 2)


_GROUND_TRUTH_PARAMETER_SUBSTRINGS = (
    "mask",
    "label",
    "truth",
    "privileged",
    "oracle",
    "visibility",
)


def constructor_apis_reject_ground_truth(*constructors: type) -> bool:
    """True if no constructor's public parameters look like a ground-truth input.

    The formal M1-M3 learners are online estimators whose constructors take
    only tuning/config values, never body masks or other privileged truth.
    This checks that structural fact at gate-evaluation time instead of
    asserting it as a fixed literal, so a future change that actually added
    such a parameter would flip this gate rather than leave it silently
    reporting success.
    """
    for constructor in constructors:
        for name in inspect.signature(constructor).parameters:
            lowered = name.lower()
            if any(
                token in lowered
                for token in _GROUND_TRUTH_PARAMETER_SUBSTRINGS
            ):
                return False
    return True


def load_frozen_protocol(
    path: str | Path,
    *,
    frozen_statuses: str | set[str],
) -> tuple[dict[str, Any], str]:
    """Load a protocol JSON, verify its sha256 lock, and check it is frozen.

    Shared by every V2 stage that gates a preregistered protocol behind a
    sibling `.sha256` lock file: reads the protocol, hashes it, compares
    against the lock, and rejects any status not in the caller's accepted
    set (a stage typically accepts its own frozen-before-implementation
    status plus any later amendment statuses it still honors).
    """

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = source.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if digest != expected:
        raise RuntimeError(f"protocol hash does not match its frozen lock: {source}")
    protocol = json.loads(source.read_text(encoding="utf-8"))
    allowed = (
        {frozen_statuses} if isinstance(frozen_statuses, str) else frozen_statuses
    )
    if protocol.get("status") not in allowed:
        raise RuntimeError(f"protocol is not frozen: {source}")
    return protocol, digest


def build_resources(
    component: Any,
    *,
    steps: int,
    started: float,
    maximum_replays_per_experience: int = 0,
) -> dict[str, Any]:
    """Resource-accounting dict shared by every staged V2 evaluation runner.

    `component` is anything exposing the three resource properties
    (`learnable_parameter_count`, `active_state_bytes`,
    `estimated_mac_per_step`) - a memory, an entity graph, or an agent that
    composes them.
    """

    return {
        "learnable_parameter_count": component.learnable_parameter_count,
        "active_state_bytes": component.active_state_bytes,
        "estimated_mac_per_step": component.estimated_mac_per_step,
        "steps_per_seed": steps,
        "maximum_replays_per_experience": maximum_replays_per_experience,
        "cpu_wall_seconds": perf_counter() - started,
    }


def resources_pass(resources: dict[str, Any], limits: dict[str, Any]) -> bool:
    """Whether a `build_resources()` dict stays within a stage's limits dict.

    `limits` uses the same key names as the `resource_limits` block in every
    V2 protocol JSON: learnable_parameters, active_state_bytes, mac_per_step,
    steps_per_seed, wall_seconds.
    """

    return (
        resources["learnable_parameter_count"] <= limits["learnable_parameters"]
        and resources["active_state_bytes"] <= limits["active_state_bytes"]
        and resources["estimated_mac_per_step"] <= limits["mac_per_step"]
        and resources["steps_per_seed"] <= limits["steps_per_seed"]
        and resources["cpu_wall_seconds"] <= limits["wall_seconds"]
    )


def require_authorization(
    path: str | Path,
    *,
    expected_name: str,
    expected_decision: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    name = payload.get("experiment", payload.get("audit"))
    schema = payload.get("result_schema_version")
    if schema not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"invalid {expected_name} V2 artifact schema: {path} "
            f"(got {schema!r}, supported "
            f"{list(SUPPORTED_RESULT_SCHEMA_VERSIONS)})"
        )
    if name != expected_name:
        raise RuntimeError(f"expected {expected_name} prerequisite: {path}")
    if payload.get("passed") is not True:
        raise RuntimeError(f"{expected_name} prerequisite did not pass")
    if payload.get("decision") != expected_decision:
        raise RuntimeError(f"{expected_name} did not authorize this stage")
    gates = payload.get("gates", payload.get("chain_validation"))
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise RuntimeError(f"{expected_name} has incomplete gates")
    if not isinstance(payload.get("provenance"), dict):
        raise RuntimeError(f"{expected_name} has no provenance")
    return payload
