This zip is the **CertiPatch Handoff SSOT** for Codex.

## What this is
A complete, decision-locked, step-by-step blueprint to implement end-to-end:
- deterministic offline spec generators
- GLR-HP hookpoint patch (gated low-rank residual intervention)
- constrained minimality solver (augmented Lagrangian)
- CEGIS outer loop (counterexample-guided constraint generation)
- baselines + decisive ablations
- replayable empirical certificates + fail-closed verifier
- experiment runs + artifacts
- figures/tables + LaTeX paper

This repository is intentionally scaffolded: code files are **signatures + exhaustive docstrings + pseudocode**.
Codex MUST implement the missing bodies without changing semantics.

## SSOT precedence (avoid drift)
If documents conflict, use this precedence order (highest wins):
1) `SPEC.md` (scope + definitions + fail-closed semantics)
2) `DECISIONS.md` (locked choices)
3) `ALGORITHMS.md` (exact method + schedules)
4) `EXPERIMENTS.md` (run matrix + tiers + baselines/ablations)
5) `FIGURES_TABLES.md` (exact filenames + axes)
6) `CLAIMS_TO_EVIDENCE.md` (reviewer attacks A/B/C → evidence mapping)

All other docs are supporting material. `00_SSOT.md` is a quickstart index, not the final arbiter.

## Quick start
Unzip, then from the repo root (the folder containing `pyproject.toml`):
- `python scripts/reproduce_paper.py --config configs/paper_full.yaml --tier full`

The reproduction command MUST:
- validate the merged config against `schemas/config_schema.json`
- produce run artifacts under `runs/<run_id>/`
- write figures/tables into `paper/latex/figures/` and `paper/latex/tables/`
- run the fail-closed verifier and fail if verification fails

## Navigation
- Start: `SPEC.md` → `DECISIONS.md`
- How to run: `EXPERIMENTS.md` → `REPRODUCE.md`
- What to implement: `certipatch/` package skeleton + `scripts/reproduce_paper.py`
- Deterministic data: `DATA_GENERATION.md` + `DATA_INVENTORY.md`
- Paper: `paper/latex/` (compile-ready; uses fixed figure/table filenames)
