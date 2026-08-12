# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Cal is a research codebase investigating whether an embodied learner can
discover a stable self–environment boundary from unlabeled sensorimotor
experience, in a deterministic 2D simulated world. It is organized as a
sequence of preregistered, gated experiments (M1, M1b–M1v, then a V2 program:
identifiability/diagnostic-ceiling/causal-sufficiency audits, M1–M4), not as a
product codebase. Changes to evaluation logic have scientific-integrity
implications — see "Frozen protocols" below before touching anything under
`cal/evaluation/` or `experiments/*.json`.

Full context lives in `docs/RESEARCH_PLAN.md` (V1, M1–M1v) and
`docs/RESEARCH_PLAN_V2.md` (V2, current program) — both written in Chinese.
Before assuming what stage the project is in, read, in this order:

- `RESEARCH_STATUS.md` — per-stage pass/fail table and the explicit list of
  claims this repository does *not* make. This is the authoritative status
  document; `README.md` is the narrative version of the same thing.
- `docs/review/OPEN_ITEMS.md` (Chinese) — the live tracker for the current
  work front: open review findings, freeze blockers, current source-lock
  version, current development-artifact versions, and the expected
  `uv run pytest` count. It is updated per PR and is the fastest way to see
  what state the tree should be in.
- `docs/experiments/V2_P1_CANDIDATE_IMPLEMENTATION_PLAN.md` for the increment
  decomposition of the in-progress permanence candidate, and the rest of
  `docs/experiments/*.md` for the permanent record of each finished stage.

## Setup and commands

```bash
uv sync --extra dev        # required before running anything, including tests
uv run pytest              # run the full test suite
uv run pytest tests/test_world.py               # single test file
uv run pytest tests/test_world.py::test_name -q  # single test
```

There is no separate lint command configured; pytest is the only checked
command (`[tool.pytest.ini_options]` in `pyproject.toml`, `testpaths = ["tests"]`).

CI (`.github/workflows/tests.yml`) does **not** run `pytest` over `tests/`; it
splits the suite into named groups by filename glob (`tests/test_[a-i]*.py`,
`tests/test_[m-u]*.py`) plus explicit per-file lists for the `test_v2_*` and
`test_world.py` groups. A new test file whose name falls outside those globs
and is not added to a list will pass locally and never run in CI — add it to
the matrix when you add the file.

All experiment/eval entry points are `console_scripts` declared in
`pyproject.toml` and invoked via `uv run <name>`, e.g.:

```bash
uv run cal-train experiments/baseline.yaml --output results/M1-gru-full-seed000
uv run cal-probe results/.../checkpoint.pt experiments/baseline.yaml --output results/...
uv run cal-multiseed --output results/M1-multiseed
uv run cal-index --results results        # rebuilds results/INDEX.json
uv run streamlit run streamlit_app.py     # read-only dashboard over results/ JSON
```

After pulling, and before building on the tree, confirm it is not drifted:

```bash
uv run pytest        # expected pass count is recorded in docs/review/OPEN_ITEMS.md
uv run python -c "from cal.evaluation.stochastic_permanence_artifacts import verify_locked_sources as v; print(v(root='.')['protocol_version'], v(root='.')['file_count'])"
```

A mismatch means someone edited a locked module without minting a new
protocol — find out why before adding to it.

See `README.md` for the full, current command sequence for both the V1 (M1)
and V2 (identifiability → M1 → M2 → M3 → M4) pipelines — it is kept in sync
with what has actually been run and is more reliable than inferring commands
from `pyproject.toml` alone.

## Package naming

The Python package is `cal` (source dir `cal/`, imports `cal.*`, script
prefix `cal-*`). It was renamed from `calmodel`; see `docs/PACKAGE_RENAME.md`.
Frozen protocol JSON files and historical one-shot confirmation results still
reference the old `calmodel/...` paths on purpose — those are historical
evidence and must never be rewritten to `cal/...`. Do not "fix" old
`calmodel` paths you encounter inside `experiments/*.json`,
`experiments/*.sha256`, or already-published `docs/experiments/*.md`.
`tests/test_package_layout.py` explicitly asserts `calmodel` is NOT
importable.

## Architecture

```
cal/
  env/          deterministic 2D world, articulated body, sensors (vision/proprioception/touch)
  model/        modality encoders, recurrent core (GRU), predictors, entity graph, body hypotheses
  learning/     experience replay/recording, sequence datasets, training loop
  evaluation/   probes, ablations, and the staged M1/M1b.../V2 experiment scripts
  infra/        provenance capture (source hashing, git state) and results indexing
experiments/    YAML configs for trainable runs; frozen JSON protocols + .sha256 sidecars for V2 confirmation stages
results/        run outputs (JSON summaries + INDEX.json); not meant to be hand-edited
docs/experiments/  the permanent written record of each stage's numbers and decisions
```

Data flow for a basic run: `cal.env.world` generates deterministic
trajectories → `cal.learning.replay` records/replays them as transitions
(`observation(t) + action(t) -> observation(t+1)`), storable as JSON or a
versioned `.jsonl.gz` compressed stream → `cal.learning.trainer` trains a
`cal.model.recurrent_core` (GRU) predictor over encoded multimodal
observations → `cal.evaluation.*` scripts freeze the trained model and probe
it (linear body probe, cause probe, adaptation, counterfactual, etc.) without
ever letting probe gradients touch the prediction model. Ground-truth body
masks and other evaluation-only state are generated only during evaluation by
deterministic world replay — they are never part of model inputs.

### Frozen protocols and source locks (important, and unevenly enforced)

The V2 confirmation stage (`cal/evaluation/v2_m1_m3_confirmation.py`) is
gated by a preregistered protocol JSON in `experiments/` (currently
`V2_M1_M3_INTEGRATED_CONFIRMATION_PROTOCOL_V7.json`; V1–V6 are superseded but
kept as historical amendment-chain links — see their `amendment_record`),
protected by a sibling
`.sha256` file **and** by a `locked_source_sha256` map embedded in the
protocol that pins hashes of `cal/evaluation/v2_m1.py`, `v2_m2.py`,
`v2_m3_hypotheses.py`, `cal/model/online_control.py`, `entity_graph.py`, and
`body_hypotheses.py`. Only this confirmation script actually calls
`_verify_locked_sources` and hashes those dependency files at runtime — **if
you edit any of the six files above, `v2_m1_m3_confirmation.py` runs will
raise on the next execution**, by design.

The randomized-permanence stack has its own, separate lock as of 2026-08-09:
currently `experiments/V2_P1_PERMANENCE_STACK_SOURCE_LOCK_V3.json` (+ `.sha256`;
V1 and V2 are superseded amendment-chain links). The active version is not
guessed from the directory listing — it is the `PERMANENCE_STACK_SOURCE_LOCK`
constant in `cal/evaluation/stochastic_permanence_artifacts.py`, and
`tests/test_stochastic_permanence_artifacts.py` asserts the file name's version
suffix matches. The lock pins the
**22-module transitive import closure** of the Phase-0/Phase-R/scan/registry
entry points, and `run_phase0` / `run_phase_r_diagnostic` call
`verify_locked_sources` before doing any work. **Editing any of those 22 files
makes both runners — and the tests that call them — raise until you mint a new
protocol version** with `mint_permanence_stack_source_lock`, keeping the old one
as an amendment-chain link. Every version after V1 must carry an
`amendment_record` naming its predecessor and the reason, which the builder
enforces. Amending the lock also means regenerating the development artifacts,
since their own `source_lock` records the sources that produced them — V1 → V2
and V2 → V3 each did exactly that rather than acknowledging the drift. That
regeneration is not cheap: Phase-0 measured ~3h41m single-core with the locked
`--simulation-trials 1024`, so an "obvious one-line fix" inside the 22 modules
costs a protocol amendment plus hours of recompute. The membership list is not
hand-kept:
`tests/test_stochastic_permanence_artifacts.py` recomputes the closure and fails
if a new import escapes the lock. `cal/env/` is deliberately out of scope (it is
the M1/V1 world and this stack does not import it); the ground-truth simulators
here are `randomized_occlusion_world.py` and `v2_i1_integration.py`, both locked.

This enforcement is *not* uniform across the pipeline, and treating it as if
it were will give a false sense of safety:
- `v2_m2.py` and `v2_m3_hypotheses.py` each hash-check only their own
  protocol JSON (`_load_frozen_protocol`) — they do not verify
  `entity_graph.py`/`body_hypotheses.py` against any locked hash themselves.
  Running `cal-v2-m2` or `cal-v2-m3-hypotheses` standalone will not detect
  that a dependency changed since freeze; only a later
  `cal-v2-m1-m3-confirm` run would catch it.
- `v2_m2.py`'s `--protocol`/`--split` flags are optional; omitting them
  silently runs an unlocked `"legacy_development"` path with no hash
  verification at all.
- `v2_m1.py` and `v2_m4.py` have no protocol JSON or source lock whatsoever —
  their thresholds are inline literals, checked only by
  `require_authorization` on the *previous* stage's output, not by any
  hash-locked source.
- `locked_source_sha256` in the confirmation protocol does not cover
  `cal/evaluation/v2_m3.py` (which `v2_m3_hypotheses.py` imports `_arm`/
  `_rasterize` from) or `cal/env/world.py`/`point_world.py` (the ground-truth
  simulators formal agents are scored against) — changes to those files after
  freeze go undetected everywhere.
- `capture_provenance`'s broad `source_sha256` (below) is descriptive only;
  nothing validates a result's provenance hash against a locked value, so it
  does not itself enforce immutability.

If you need to change one of the six locked files, that requires a new
protocol version/amendment (see existing `*_V2.json` / `amendment_record`
patterns), not an in-place edit. If you touch `v2_m2.py`, `v2_m3_hypotheses.py`,
`v2_m1.py`, `v2_m3.py`, or the env simulators, be aware no automated check
will flag it except a subsequent confirmation-stage run (and even that only
covers a subset of these files) — don't assume silence means safe.

### Where current work happens (deliberately outside the lock)

The active work front is the I1-P1 stochastic-permanence candidate, and it was
built specifically so that it is *not* in the 22-module closure — the candidate
is injected into `stochastic_permanence_benchmark.py` through the
`CandidateFactory` protocol, so no locked entry point imports it. Editing these
files needs no amendment and no artifact regeneration:

- `cal/model/permanence_candidate.py` (the assembled O3 candidate),
  `permanence_track.py`, `sensor_only_static_map.py`,
  `branch_self_identity.py`, `association_bank.py`;
- `cal/evaluation/permanence_candidate_development.py` (the I5 comparison
  harness, `python -m cal.evaluation.permanence_candidate_development`),
  `permanence_geometry_diagnostic.py`,
  `permanence_control_constructability.py`.

If a change you want appears to require editing a locked module, prefer the
route these modules already take — reconstruct what you need on the unlocked
side (`permanence_candidate_development.py` replays the world rather than
adding a field to the locked `_Sample`) and verify the reconstruction against
the locked path instead of quietly amending the lock.

Everything in this stack is **development-stage**: its numbers come from
development seeds that have been looked at repeatedly, no confirmatory split
exists, and modules here carry an explicit `NON-GATED` / "not evidence"
docstring line. Keep writing that line; do not label development output as
evidence.

### One-shot evidence and consumed holdouts

Holdouts here are one-shot. A consumed holdout (e.g. the L0 V8 run of
2026-07-28, bound to a git tag and a recorded result SHA-256) is historical
evidence, never a reusable test set, and never development data. Do not tune
against it, re-run it to get a better number, or edit its result JSON. A new
claim requires a newly preregistered split whose contents were not visible
during development — see `CONTRIBUTING.md` and `RESEARCH_STATUS.md`.

Result JSON produced by these stages is validated with
`cal.evaluation.v2_artifacts.require_authorization`, which enforces
`result_schema_version == 1`, a matching `decision`, complete `gates`, and
present `provenance` before a downstream stage is allowed to treat an
upstream result as passing.

### Provenance

`cal.infra.provenance.capture_provenance` stamps every result JSON with a
combined source SHA-256 (derived from all tracked source files' names and
contents — unique even without a git commit), per-file hashes, git commit/
dirty state, and dependency/OS versions. This is how `docs/RESULT_FORMAT.md`
results stay reproducible/attributable without committing `results/` itself
(which is generally not version-controlled; only the durable numbers and
conclusions written into `docs/experiments/*.md` are).

## Conventions worth knowing

- Config-driven: experiment behavior (image size, modalities enabled, seeds,
  hyperparameters) is described by YAML files in `experiments/`, not by
  editing Python. `shuffle_modalities`, `include_action`, `include_touch`,
  etc. toggle ablations from the same trainer/eval code path.
- Seeds are partitioned by role (train/validation/test/probe/adaptation/
  cause_probe) directly in the experiment YAML, not derived from a single
  global seed.
- Dataclasses (`@dataclass(frozen=True, slots=True)`) are the standard shape
  for typed config/result structures throughout `cal/`.
- Docstrings and comments in this repo are in English; the two research-plan
  documents and a few evaluation-support docs are in Chinese — match the
  existing language of a file you're editing rather than converting it.
