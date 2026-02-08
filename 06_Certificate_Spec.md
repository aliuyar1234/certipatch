# Replayable Empirical Certificates (Schema + Verifier Semantics)

## Purpose
Define the certificate artifact as a **replayable empirical certificate**.
This is not a proof. It is a deterministic, hash-tied evaluation record that a verifier can replay.

---

# 1. Certificate philosophy (must be stated in paper)
- A certificate is valid only for its recorded **scope** and **coverage**.
- Any missing detail, hash mismatch, or reproducibility deviation MUST invalidate (fail-closed).
- Certificates enable auditing and regression testing (CI-style).

---

# 2. Scope types
## 2.1 Exact enumeration scope
- For enumerable domains where the domain hash corresponds to full canonical enumeration.
- Claim allowed: “0 failures on X_spec” if verifier reproduces failures==0.

## 2.2 Coverage-bounded scope
- For non-enumerable domains.
- Claim allowed: “0 failures on the certified coverage set defined by coverage_plan_hash”.
- MUST NOT claim satisfaction beyond certified set.
- Must include per-stratum counts and seeds.

---

# 3. Required files per run
A run directory MUST contain:
- certificate.json
- patch.pt (or patch.safetensors)
- counterexamples.jsonl (history)
- metrics.json (raw metrics used for plots)
- run_record.json (full config + environment)
- MANIFEST.json (sha256 for each file)

---

# 4. Strict JSON schema (field list)
This is the schema used in SSOT, repeated here with additional semantics.

**Normative machine-checkable schemas:** see `schemas/certificate_schema_v1.json` and related files in `schemas/`.

## Top-level required fields
- schema_version: must be "1.0"
- run_id: unique string
- timestamp_utc: ISO timestamp
- model: model identity and tokenizer; answer tokens
- patch: patch family + parameters + hashes
- specs: list of spec results and scope descriptors
- cegis: loop parameters and counterexample files
- objective: τ and ALM schedule and regularizers
- collateral: suite metrics + suite hashes
- reproducibility: seeds + determinism flags + environment snapshot
- fail_closed: explicit statements of what is *not* guaranteed

---

# 5. Semantics of key fields (to avoid ambiguity)
## 5.1 model.model_revision
Must be either:
- HF commit hash, or
- exact version tag plus downloaded file hashes.
Verifier MUST use the same revision.

## 5.2 patch.patch_weights_hash
sha256 of the serialized patch parameters file (after saving). This detects tampering.

## 5.3 domain_hash (enumerable)
Computed as sha256 of:
- canonical prompt strings joined with "\n"
- plus canonical labels joined with "\n" (e.g., "1" or "0")
- recommended: append label with a tab per prompt: "{prompt}\t{label}\n"

This ensures any prompt formatting change invalidates.

## 5.4 coverage_plan_hash (coverage-bounded)
sha256 of JSON containing:
- stratum definitions
- per-stratum counts
- PRNG algorithm name and seed
- deterministic sampling code version hash
- ordering rules

## 5.5 suite_hash (reference suites)
sha256 of canonical prompt list. For RefBool-L include the generation settings in the hash input (max_new_tokens, decoding method) to ensure replay equivalence.

---

# 6. Verifier behavior (must implement)
Verifier inputs:
- path to run directory

Verifier steps:
1) Load certificate.json.
2) Verify MANIFEST.json matches all file hashes.
3) Load model revision and tokenizer; verify answer tokenization matches certificate.
4) Recompute:
   - gate predicate behavior (unit tests stored in repo)
   - domain generators: verify generator_code_hash matches and recompute domain_hash or coverage_plan_hash.
   - reference suites: verify suite hashes.
5) Load patch weights; verify hash.
6) Run evaluation:
   - For each spec:
     - if scope_type == exact_enumeration: evaluate full domain; require failures == certificate failures.
     - if scope_type == coverage_bounded: evaluate certified coverage set; require matching pass rates per stratum.
7) Recompute collateral metrics and confirm within tolerance:
   - tolerance is max_abs_metric_diff from certificate (e.g., 1e-6 for deterministic float32; 1e-4 if bf16).
8) Output PASS only if all checks pass.

Fail-closed rules:
- Any missing file => FAIL
- Any hash mismatch => FAIL
- Any metric mismatch => FAIL
- Any scope inconsistency (e.g., coverage-bounded but missing plan hash) => FAIL

---

# 7. Tamper tests (required for Figure 5)
You SHALL create derived run directories:
- perturb patch weights by adding tiny noise
- change generator code hash field
- change coverage plan hash field
- change model revision string

Verifier MUST FAIL in all these cases.

---

# 8. Example certificate snippets
Include a short example in the appendix of the paper; use redacted paths, keep hashes real.
