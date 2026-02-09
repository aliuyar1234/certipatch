# CertiPatch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)](#)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Deterministic%20Pipeline-success.svg)](#reproducibility-contract)

CertiPatch is a research codebase for **specification repair of frozen language models** with a deterministic
`train -> certify -> verify` workflow.

It combines:
- gated low-rank hookpoint patches (GLR-HP),
- constrained optimization (augmented Lagrangian + CEGIS),
- replayable empirical certificates with fail-closed verification,
- paper-ready artifact generation (figures/tables + LaTeX).

## Why CertiPatch
- Deterministic spec/domain generators for reproducible evaluation.
- Constraint-first repair objective: close spec failures while measuring collateral drift.
- Artifact-locked certificates with explicit scope and coverage semantics.
- End-to-end pipeline from training to verification to manuscript assets.

## Repository layout
- `certipatch/`: library code (`specs/`, `models/`, `cegis/`, `eval/`, `artifacts/`)
- `scripts/`: runnable entrypoints (`scripts/reproduce_paper.py`)
- `configs/`: YAML overlays merged with `configs/default.yaml`
- `schemas/`: JSON schemas for config/artifacts/certificates/metrics
- `tests/`: deterministic `pytest` suite
- `paper/latex/`: manuscript, figures, tables

## Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e ".[dev,torch,hf,tl,viz]"
```

## Quick start
Run a toy tier:
```bash
python scripts/reproduce_paper.py --config configs/compare2d_certipatch.yaml --tier toy
```

Run the paper profile:
```bash
python scripts/reproduce_paper.py --config configs/paper_full.yaml --tier full
```

## Reproducibility contract
The reproduction workflow is fail-closed. A valid run must:
- validate config against `schemas/config_schema.json`,
- write artifacts to `runs/<run_id>/`,
- generate paper assets under `paper/latex/figures/` and `paper/latex/tables/`,
- pass manifest and certificate verification.

If verification fails, outputs are not considered valid.

## Development checks
```bash
python -m pytest -q
python -m ruff format . && python -m ruff check .
python -m mypy certipatch
python scripts/update_manifest_sha256.py
```

## Documentation map (SSOT order)
When documents conflict, use this precedence:
1. `SPEC.md`
2. `DECISIONS.md`
3. `ALGORITHMS.md`
4. `EXPERIMENTS.md`
5. `FIGURES_TABLES.md`

Supporting references:
- `REPRODUCE.md`: expected outputs and fail-closed checks
- `CLAIMS_TO_EVIDENCE.md`: claim-to-artifact mapping
- `00_SSOT.md`: index/quick navigation

## Author
**Ali Uyar**  
Independent Researcher
