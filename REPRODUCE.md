# REPRODUCE.md - One Command, Expected Outputs (Normative)

This repository is designed for one-command reproduction.

## Command

From repo root:
- `python scripts/reproduce_paper.py --config configs/paper_full.yaml --tier full`

## Expected outputs (fail-closed)

After the command finishes, these MUST exist:

### 1) Run artifacts
Under `runs/<run_id>/`:
- `run_record.json`
- `certificate.json`
- `metrics.json`
- `counterexamples.jsonl`
- `patch.pt`
- `report.html`

### 2) Paper assets
Under `paper/latex/figures/`:
- `fig01_minimality_pareto.pdf`
- `fig02_cegis_trace.pdf`
- `fig03_coverage_heatmap.pdf`
- `fig04_compositionality_matrix.pdf`
- `fig05_verifier_tamper.pdf`

Under `paper/latex/tables/`:
- `tab01_main_results.tex`
- `tab02_ablations.tex`

### 3) Verification
The script MUST run:
- MANIFEST verification
- certificate verification

If verification fails, reproduction MUST be treated as failed and no paper assets are valid.

## Notes
- Any local model is allowed if it satisfies the adapter contract.
- Configs define backend, model identity, hookpoints, ranks/layers, specs enabled, coverage plan, and seeds.
- The run_record MUST capture the resolved model fingerprint and manifest hash.

