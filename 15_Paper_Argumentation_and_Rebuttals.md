# Paper Argumentation and Rebuttals (Reduced Compute Scope)

## Scope lock (must appear early in paper)
- This submission is a compute-aware single-seed study.
- Multi-seed robustness and full task x method matrix are explicitly out of scope.
- Claims are restricted to completed result cells and scope-bounded certificates.

## Attack A: "Only steering/adapter plus hard mining"

### Core response
- CertiPatch is a protocol contribution: train -> certify -> verify under fail-closed semantics.
- Compositionality and artifact semantics are first-class outputs, not side diagnostics.
- Dev ablations isolate which components matter for feasibility/collateral control.

### Evidence anchors
- Figure 4 (compositionality regimes).
- Table 2 (ablation deltas).
- Completed baseline cells in Table 1 only.

## Attack B: "Certificates are not real certificates"

### Core response
- We claim replayable empirical certificates, not formal global proofs.
- Certificates are scope-bounded and fail-closed.
- Any mismatch in patch/domain/schema/manifest must fail verification.

### Evidence anchors
- Figure 5 (tamper fail-closed behavior).
- Verifier PASS for reported runs.
- Explicit scope semantics in certificate fields.

## Attack C: "CEGIS is unnecessary"

### Core response
- CEGIS is used for constrained repair behavior and active-set management.
- compare_6d_strat demonstrates coverage-bounded certification where full enumeration is not the framing.

### Evidence anchors
- Figure 2 (trace).
- Figure 3 (coverage strata outcomes).
- Table 1 scope-aware metrics.

## Claim-safe writing rules
- Do not claim cross-seed robustness in this submission.
- Do not claim full-matrix baseline dominance where cells are missing.
- Do not claim global optimality for minimality.
- Mark missing cells as "not run (compute scope)".

## Limitations section checklist
- Single-seed only.
- Partial baseline matrix for some settings.
- Scope-bounded certificates (enumeration or declared coverage policy only).
- Main model plus dev-model split for ablations.
