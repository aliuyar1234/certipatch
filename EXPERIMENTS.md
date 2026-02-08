# EXPERIMENTS.md - Tiers, Run Matrix, Baselines, Ablations (Normative)

This document is the execution plan and reporting policy for reproducible runs.

## 1) Tiers

### 1.1 toy
Purpose: smoke-test end-to-end plumbing.
- Small model.
- Reduced domains.
- Small collateral suites.
- Run CertiPatch on compare_2d.
- Verify manifest/certificate path.

### 1.2 small
Purpose: primary development tier.
- Full enumerable domains for compare_2d, parity_4d, balance_paren_14.
- Coverage-bounded compare_6d_strat.
- Full collateral suites (RefBool-S/L and RefText).

### 1.3 full (camera-ready reduced profile)
Purpose: compute-aware paper run matrix.
- Main model runs on seed 0.
- Compositionality suite on seed 0.
- Dev ablations on GPT-2 seed 0.
- Optional multi-seed and scaling extensions only when extra compute is available.

## 2) Run matrix (full tier)

### 2.1 Required core runs
- Main model, seed 0:
  - compare_2d
  - parity_4d
  - balance_paren_14
  - compare_6d_strat
- Compositionality suite (A_only, B_only, A_plus_B, A_then_B, B_then_A, Joint_AB).
- Dev run for ablations: GPT-2 compare_2d.

### 2.2 Baselines
- Target baselines:
  - base
  - steering_vec_1l
  - oneshot_full_mo
  - oneshot_full_alm
  - softprompt
  - lora
- Comparative claims must only use baseline cells that completed successfully.
- Incomplete cells must be rendered explicitly as "not run (compute scope)".

### 2.3 Ablations
- no_minimality
- no_cegis
- no_collateral
- no_gating
- rank_1
- single_layer
- random_counterexamples

## 3) Reporting policy (claim-safe)

### 3.1 What can be claimed strongly
- End-to-end train-certify-verify pipeline with fail-closed verifier semantics.
- Exact in-scope certification for enumerable domains.
- Coverage-bounded certification for compare_6d_strat with explicit scope limits.
- Compositionality/interference evidence from the six-condition suite.

### 3.2 What must be qualified
- No cross-seed robustness claims in the reduced profile.
- No full-matrix baseline dominance claims when cells are missing.
- No global optimality claims for minimality.

## 4) Required artifacts per completed run
- `run_record.json`
- `certificate.json`
- `metrics.json`
- `counterexamples.jsonl`
- `patch.pt`
- `report.html`
- `MANIFEST.json`

Verifier must pass for reported runs, and fail-closed tamper tests must be included in paper assets.
