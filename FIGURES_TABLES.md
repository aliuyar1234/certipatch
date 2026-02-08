# FIGURES_TABLES.md — Filenames, Axes, Generation Commands (Normative)

This document defines the exact figure/table filenames used by the LaTeX paper.

## 1) Output directories
- Figures: `paper/latex/figures/`
- Tables: `paper/latex/tables/`

Scripts MUST write exactly these filenames (overwriting is allowed).

## 2) Figures (PDF)

1) `fig01_minimality_pareto.pdf`
   - Axes: x=RefBool-S mean KL, y=patch complexity (fro norm or #effective layers)
   - Includes only configurations with 0 failures on the target enumerable domain.

2) `fig02_cegis_trace.pdf`
   - Axes: x=CEGIS outer iteration
   - Curves: failures (exact), RefBool-S KL, patch fro norm

3) `fig03_coverage_heatmap.pdf`
   - Heatmap: strata × system
   - Values: failure rate per stratum, annotated with sample counts

4) `fig04_compositionality_matrix.pdf`
   - Matrix (conditions × metrics)
   - Conditions: A-only, B-only, A+B, A→B, B→A, Joint AB
   - Metrics: A failures, B failures, KL, drift, complexity

5) `fig05_verifier_tamper.pdf`
   - Bars: verifier PASS/FAIL under controlled tamper cases
   - Cases: exact replay; patch perturbed; generator hash mismatch; coverage hash mismatch

## 3) Tables (TeX)

1) `tab01_main_results.tex`
   - Rows: Base, SteeringVec-1L, OneShot-FullDomain-MO, OneShot-FullDomain-ALM,
           SoftPrompt, LoRA, CertiPatch
   - Cols: failures (compare_2d, parity_4d, balance_paren_14), coverage metrics (compare_6d_strat),
           collateral metrics, complexity metrics

2) `tab02_ablations.tex`
   - Rows: CertiPatch, no_minimality, no_cegis, no_collateral, no_gating, rank_1, single_layer, random_counterexamples
   - Cols: failures, collateral KL, complexity, iterations to closure, coverage boundary failures

## 4) Generation commands

Minimal contract (scaffold):
- `python scripts/reproduce_paper.py --config <overlay.yaml> --tier full`

The script MUST:
- run the full run matrix defined in EXPERIMENTS.md
- write figures and tables to the paper directories with the filenames above
- fail if any expected output file is missing

