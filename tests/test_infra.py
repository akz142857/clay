"""Tests for experiment provenance and result indexing."""

import json

import pytest

from cal.infra.provenance import capture_provenance
from cal.infra.results import (
    IndexWouldTruncateError,
    build_result_index,
)


def test_provenance_contains_stable_source_digest() -> None:
    first = capture_provenance()
    second = capture_provenance()

    assert first["schema_version"] == 1
    assert len(first["source_sha256"]) == 64
    assert first["source_sha256"] == second["source_sha256"]
    assert first["source_file_count"] > 0


def test_result_index_discovers_known_summaries(tmp_path: object) -> None:
    root = tmp_path / "results"  # type: ignore[operator]
    run = root / "run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"name": "test"}),
        encoding="utf-8",
    )

    index = build_result_index(root)

    assert index["entry_count"] == 1
    assert index["entries"][0]["kind"] == "prediction"
    assert (root / "INDEX.json").exists()


def test_result_index_discovers_preregistered_mechanism_screens(
    tmp_path: object,
) -> None:
    root = tmp_path / "results"  # type: ignore[operator]
    root.mkdir()
    (root / "M1v-screen-summary.json").write_text(
        json.dumps({"candidate": "m1v_action_basis"}),
        encoding="utf-8",
    )

    index = build_result_index(root)

    assert index["entry_count"] == 1
    assert index["entries"][0]["kind"] == "m1_mechanism_screen"
    assert index["entries"][0]["name"] == "m1v_action_basis"


def test_result_index_refuses_to_drop_results_absent_from_this_checkout(
    tmp_path: object,
) -> None:
    """A fresh checkout must not silently delete the committed index.

    ``results/`` is not version controlled but ``INDEX.json`` is, so the
    committed index names summaries a new clone does not have.  Rebuilding
    there used to shrink it from 702 entries to 22 without a word (review
    finding F13).
    """

    root = tmp_path / "results"  # type: ignore[operator]
    root.mkdir()
    (root / "summary.json").write_text(json.dumps({"name": "present"}), "utf-8")
    (root / "INDEX.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entry_count": 2,
                "entries": [
                    {"kind": "prediction", "path": str(root / "summary.json")},
                    {"kind": "prediction", "path": "results/gone/summary.json"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IndexWouldTruncateError, match="results/gone/summary.json"):
        build_result_index(root)

    # The refusal must leave the committed index untouched, not half-written.
    preserved = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    assert preserved["entry_count"] == 2

    pruned = build_result_index(root, prune=True)
    assert pruned["entry_count"] == 1
    assert pruned["pruned_absent_paths"] == ["results/gone/summary.json"]


def test_result_index_rebuild_is_allowed_when_nothing_is_lost(
    tmp_path: object,
) -> None:
    root = tmp_path / "results"  # type: ignore[operator]
    root.mkdir()
    (root / "summary.json").write_text(json.dumps({"name": "present"}), "utf-8")

    first = build_result_index(root)
    second = build_result_index(root)

    assert first["entry_count"] == second["entry_count"] == 1
    assert "pruned_absent_paths" not in second
