# Reproducibility, Hashing, Determinism, and Fail‑Closed Verification

## Purpose
Make the project reproducible and “fail-closed” by construction.

---

# 1. Determinism checklist (MUST)
You SHALL set:
- Python: `PYTHONHASHSEED`
- random: `random.seed(seed)`
- numpy: `np.random.seed(seed)`
- torch:
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `torch.use_deterministic_algorithms(True)` (if available)
  - `torch.backends.cudnn.deterministic=True`
  - `torch.backends.cudnn.benchmark=False`

Model:
- `model.eval()` to disable dropout.

Data loading:
- Single-worker deterministic iteration.
- No nondeterministic shuffling.

Generation:
- greedy decoding only.
- no sampling.

---

# 2. Hashing rules (MUST)
Every run directory MUST include:
- `MANIFEST.json` mapping relative paths to sha256
- `run_record.json` capturing config and environment
- `certificate.json` referencing all hashes

## 2.1 What to hash
- Every produced artifact file
- Every generator source file version hash (or git commit hash)
- Domain hashes and suite hashes derived from canonical prompt lists

## 2.2 sha256 canonicalization
When hashing text:
- Use UTF‑8 bytes
- Use "\n" newlines
- Do not include timestamps in hashed content unless explicitly in schema

When hashing JSON:
- Use canonical JSON:
  - sorted keys
  - no whitespace
This ensures stable hash across machines.

---

# 3. Logging format (MUST)
- Use JSON for machine parsing.
- Use stable field names.

Required logs:
- `outer_loop.jsonl`: one line per outer iteration
- `inner_loop.jsonl`: one line per N steps (e.g., 50)
Fields to include:
- iteration indices
- g_true
- failures_count
- ref_kl_estimate
- patch_norm
- mu, lambda
- time_sec

---

# 4. Re-running and verifying
The verifier MUST:
- recreate domains and suites
- rerun evaluation
- compare metrics and hashes

Tolerance policy:
- Prefer exact match in float32.
- If mixed precision causes tiny differences, use a small tolerance and record it in certificate.

---

# 5. “No drift” policy (for Codex)
If any design choice needs to change:
- Update SSOT.
- Update Decision Log (append-only file).
- Bump schema_version if artifacts change semantics.

Do NOT silently change behavior; it invalidates comparability.
