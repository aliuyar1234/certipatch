# Repository Guidelines

## Project Structure & Module Organization
CertiPatch is a Python package with an end-to-end “train → certify → verify” pipeline.

- `certipatch/`: library code
  - `specs/`: deterministic spec/domain generators (COMPARE-2D, PARITY-4D, BALANCE-PAREN-14, COMPARE-6D-STRAT)
  - `models/`: backend adapters (TransformerLens / HuggingFace), tokenization, hook registration
  - `cegis/`: constrained ALM solver, CEGIS loop, compositionality suite
  - `eval/`: exact spec/collateral metrics + baselines/ablations
  - `artifacts/`: run records, certificates, verifier, HTML report, manifests
- `scripts/`: entrypoints (main: `scripts/reproduce_paper.py`)
- `configs/`: YAML overlays merged onto `configs/default.yaml`
- `schemas/`: JSON Schemas (config/run_record/certificate/metrics/manifest)
- `tests/`: `pytest` suite
- `paper/latex/`: LaTeX + generated `figures/` and `tables/`

SSOT precedence: `SPEC.md` > `DECISIONS.md` > `ALGORITHMS.md` > `EXPERIMENTS.md` > `FIGURES_TABLES.md`.

## Build, Test, and Development Commands
Python >= 3.10.

```bash
python -m venv .venv && .\.venv\Scripts\activate
python -m pip install -e ".[dev,torch,hf,tl,viz]"
python scripts/reproduce_paper.py --config configs/compare2d_certipatch.yaml --tier toy
python -m pytest -q
python -m ruff format . && python -m ruff check .
python -m mypy certipatch
python scripts/update_manifest_sha256.py
```

`scripts/reproduce_paper.py` writes run artifacts to `runs/<run_id>/` and paper assets to
`paper/latex/{figures,tables}/`. If `MANIFEST.sha256` changes, treat it as a release-critical edit.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; avoid implicit nondeterminism (stable ordering, fixed seeds, greedy decoding only).
- Lint/format: `ruff` (line length 100).
- Naming: `snake_case` for functions/vars, `CapWords` for classes, `tests/test_*.py` with `test_*` functions.

## Testing Guidelines
Prefer small, deterministic tests (fake adapters/models are encouraged; see `tests/test_trainer_alm.py`).
When adding a new artifact field, extend the corresponding schema under `schemas/` and add a verifier test.

## Commit & Pull Request Guidelines
This workspace may not include a `.git` directory. Use Conventional Commits (e.g., `feat: ...`, `fix: ...`),
keep PRs scoped, and include the exact reproduction command + tier used. If you ran on CUDA, note whether
deterministic settings/TF32 were enabled.
