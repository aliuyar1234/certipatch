# CLAIMS_TO_EVIDENCE.md - Reviewer Attack Surfaces to Evidence (Normative)

This document maps each reviewer attack surface to evidence that is valid for the reduced compute scope.

## Attack A: "This is just steering / PEFT + hard example mining."

### Required evidence
- Protocol-level novelty: constrained repair plus replayable certification pipeline.
- Compositionality evidence (interference and order effects).
- Ablation evidence (component necessity) on the dev model.

### Claim-safe statement
- Baseline-comparison claims are made only on completed cells.
- Missing baseline cells are explicitly marked as not run.

### Primary artifacts
- Figure 4 (compositionality matrix).
- Table 2 (dev ablations).
- Table 1 cells that are fully populated.

## Attack B: "Certificates are not certificates."

### Required evidence
- Replayable certificate schema and verifier behavior.
- Fail-closed tamper behavior on artifact mismatch.
- Explicit scope declarations for enumerable vs coverage-bounded settings.

### Primary artifacts
- Figure 5 (tamper outcomes).
- `certificate.json` schema validation and verifier PASS logs.
- compare_6d_strat scope/coverage reporting.

## Attack C: "Domains are enumerable; CEGIS is unnecessary."

### Required evidence
- CEGIS role is framed as constraint management/minimality behavior, not sample-efficiency hype.
- Non-enumerable coverage-bounded setting demonstrates explicit bounded certification semantics.

### Claim-safe statement
- Do not claim universal superiority over every baseline/method.
- Report only what the completed run matrix supports.

### Primary artifacts
- Figure 2 (trace behavior).
- Figure 3 (coverage-bounded stratified results).
- Table 1 scope-aware metrics.
