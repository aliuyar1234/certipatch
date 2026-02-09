# CertiPatch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/1153117963.svg)](https://doi.org/10.5281/zenodo.18541322)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/aliuyar1234/certipatch/blob/main/paper/latex/main.pdf)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Deterministic%20Pipeline-success.svg)](#reproducibility-contract)

CertiPatch is a reproducible research framework for **specification repair of frozen language models** with a deterministic
`train -> certify -> verify` pipeline.

**Author:** Ali Uyar (Independent Researcher)

**Citation DOI (latest release archive):** [10.5281/zenodo.18541322](https://doi.org/10.5281/zenodo.18541322)

This repository provides:
- gated low-rank hookpoint patches (GLR-HP),
- constrained optimization (augmented Lagrangian + CEGIS),
- replayable empirical certificates with fail-closed verification semantics,
- end-to-end generation of paper artifacts (figures, tables, and LaTeX).

## Why CertiPatch
- Deterministic spec/domain generators for reproducible evaluation.
- Constraint-first repair objective: satisfy in-scope spec constraints while quantifying collateral drift.
- Artifact-locked certificates with explicit scope and coverage semantics.
- End-to-end workflow from training through verification to publication assets.

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

## Results and paper
- Download paper PDF: `paper/latex/main.pdf`
- Build paper locally:
```bash
cd paper/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
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

## Reproduction reference
See `REPRODUCE.md` for expected outputs and validation behavior.
