# DECISIONS.md - Locked Decisions (Normative)

This file lists decisions that are treated as stable SSOT.

## Method
- Protocol name: CertiPatch.
- Patch family: GLR-HP (gated low-rank residual hook patch).
- Objective: constrained minimality (minimize collateral plus regularizers subject to zero in-scope spec violations).
- Solver: augmented Lagrangian method (ALM) with CEGIS outer loop.
- ALM/CEGIS integration: warm-start ALM state across CEGIS outer iterations.
- Certificate: replayable empirical certificate, scope-bounded and fail-closed.

## Scope and gate
- Default gate: BoolQA wrapper strict match (wrapper line plus suffix).
- Collateral is measured on gate-firing suites only.
- Out-of-scope prompts are never certified.

## Specs
- compare_2d (enumerable)
- parity_4d (enumerable)
- balance_paren_14 (enumerable; may fail-close)
- compare_6d_strat (coverage-bounded)

## Paper claim scope (compute-aware submission profile)
- Default paper profile is single-seed (`seed=0`) in `configs/paper_full.yaml`.
- Main claims are protocol-level: train-certify-verify pipeline, replayable fail-closed artifacts, and compositionality behavior.
- Comparative baseline claims are restricted to completed cells only.
- Missing or skipped matrix cells must be reported explicitly as "not run (compute scope)".
- No robustness-across-seeds claims are made in the reduced profile.
- Dev ablations on GPT-2 are diagnostic evidence for protocol components, not broad model-family generalization claims.

## Figures and tables
- Filenames and semantics are locked by `FIGURES_TABLES.md` and used by LaTeX.

## Fixed run settings (paper)
- Full-tier seeds (default compute profile): `[0]`.
- Full-tier scaling runs are disabled by default (`paper.models.scaling == paper.models.main`).
- Constraint proxy for training: `objective.g_smooth_formula = log_mean_exp` (configurable, recorded in run_record).

## Progress tracking
- `STATUS.yaml` and session logs provide run-by-run tracking.
- If any decision changes, record it in `DECISION_LOG.md` with justification and then update this file.
