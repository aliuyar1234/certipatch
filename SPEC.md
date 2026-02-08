# SPEC.md — CertiPatch SSOT (Scope, Definitions, Fail-Closed Semantics)

This document is normative. If any other document conflicts, this one wins.

## 1) Scope

CertiPatch is a **specification repair** protocol for a **frozen** language model:
- The model weights MUST NOT be updated.
- The repair is an **inference-time hookpoint patch** applied only when a deterministic **gate** fires.
- Repair targets are **programmatic specs** over prompt families (no human labels).

Outputs MUST be auditable:
- A **replayable empirical certificate** (not a formal proof).
- Deterministic run records, hashes, and a fail-closed verifier.

## 2) Key definitions

### 2.1 Spec domain and labeler
- A **spec** defines a prompt family `X_spec` and a deterministic label function `Spec(x) ∈ {Yes, No}`.
- Specs are generated offline, deterministically, by code under `certipatch/specs/`.

### 2.2 Gate and certified scope
- The **gate** `s(x) ∈ {0,1}` defines the scope where patches apply.
- Default gate is a strict BoolQA wrapper predicate:
  - The prompt MUST contain the configured wrapper line exactly.
  - The prompt MUST end with the configured suffix exactly (after trimming trailing whitespace).
- All collateral measurement MUST be on prompts where the gate fires.
- Anything outside the gate is **out-of-scope** for certificates. The verifier MUST not claim anything about it.

### 2.3 Answer position
For a tokenized prompt with attention mask `mask ∈ {0,1}^{B×T}`:
- The answer position is `p = sum(mask) - 1` per example.
- Any implementation that uses `p = T-1` without enforcing left-padding MUST be treated as invalid.

### 2.4 Patch family (v1)
- Patch family is **GLR-HP** (Gated Low-Rank Residual Hook Patch).
- Applied at hookpoint kind `resid_post` on candidate layers (resolved by config).
- Parameterization (per layer ℓ):
  `h[ℓ,p] ← h[ℓ,p] + s(x)·U_ℓ(V_ℓᵀ h[ℓ,p])`, with rank `r` set by config.

### 2.5 Repair objective (minimality-first)
Repair is a constrained optimization:
- Minimize collateral drift + complexity penalty
- Subject to zero spec violations on the certified scope

Formal objective and solver schedule are defined in `ALGORITHMS.md`.

## 3) Model flexibility and adapter contract

Any local model is allowed if it satisfies the adapter contract:
- tokenization with attention masks
- logits forward pass
- hookpoint registration for requested hookpoints/layers
- stable model fingerprint reporting (revision or local fingerprint)

The contract is defined in `certipatch/models/load_model.py`.

Fail-closed:
- If the adapter cannot resolve the requested hookpoints, abort.
- If answer tokens are not single tokens and fallback also fails, abort.

## 4) Certificates (replayable empirical certificates)

A certificate is a deterministic record of:
- model identity/fingerprint
- domain identity (hash) or coverage plan hash
- counterexample search budgets and seeds
- exact spec metrics on certified scope
- collateral metrics on gate-firing reference suites
- patch parameters hash and manifest hash

Certificates are **not** formal proofs.
For non-enumerable domains, certificates MUST explicitly state coverage-bounded scope.

Certificate schema is in `schemas/certificate_schema_v1.json`.
Verifier semantics are in `certipatch/artifacts/verifier.py` and `06_Certificate_Spec.md`.

## 5) Manifest integrity (repository-level)

`MANIFEST.sha256` covers every file in this repository, including itself.

Self-hash rule:
- The manifest includes an entry for `MANIFEST.sha256`.
- That hash is computed over the manifest text where the hash on the `MANIFEST.sha256` line is replaced by 64 zeros.
- Verifiers MUST use this rule and fail if it does not match.

This integrity mechanism is for reproducibility and fail-closed operation, not for adversarial security.

