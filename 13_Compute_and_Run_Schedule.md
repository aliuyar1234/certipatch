# Compute Plan + Run Scheduling (Expected Behaviors, Budgets, and Checkpoints)

## Purpose
Provide a concrete compute plan and run schedule so Codex can execute GPU runs efficiently and predict what should happen.

This document includes:
- expected runtime scaling
- recommended batch sizes
- caching strategy (optional)
- run ordering and checkpointing
- “what good looks like” metrics at intermediate points

---

# 1. Compute budget targets (paper-level)
- Single GPU target (default profile): ~18-32 GPU-hours on Pythia-410M with seed [0].
- Optional add-ons: +6-10 GPU-hours for a separate scaling model, +8-12 GPU-hours per extra seed for main runs.

This is feasible because:
- domains are small (10k–33k)
- patch is small
- we only train patch parameters

---

# 2. Expected runtime drivers
The main cost components:
1) Forward+backward for training steps on RefBool-S + D_spec batches.
2) Exact evaluation sweeps over domains to find counterexamples and compute failures.

Evaluation sweeps are pure forward passes; they can be accelerated with batching and no grad.

---

# 3. Recommended batch sizes (starting point)
For Pythia-410M (main profile):
- training:
  - RefBool-S batch size: 64-128
  - D_spec batch size: 64-128
- evaluation:
  - domain sweep batch size: 256-512 (no grad)

Optional larger scaling model profile:
- training:
  - 32-64
- evaluation:
  - 128-256

Adjust based on GPU memory; record in run_record.json.

---

# 4. Caching strategy (optional, but can save time)
## 4.1 Cache base logits for RefBool‑S
Since L_col uses base logits repeatedly across runs, caching can accelerate:
- compute base logits at answer position for all RefBool-S prompts once
- save as float32 arrays
- hash and store in manifest

Then L_col uses cached base logp, and only patched logq needs recomputing.

This is optional; correctness is unchanged.

If you cache:
- include file hash in certificate
- verifier must regenerate or re-load exact cached file hash

---

# 5. Run ordering (must follow)
The runbook order is strict (see `01_Runbook_Phases.md`). This section adds expected results.

## Gate 0: tokenization
- Expect: configured tokenizer should treat " Yes"/" No" as single tokens.
- If not, fallback triggers; record.

## Gate 1: toy COMPARE 0..19
- Expect: within 1–2 outer iterations:
  - failures go to 0 on toy domain
  - KL remains small but nonzero
  - patch norm increases modestly

If toy cannot close:
- bug in logits position, labeler, or patch application.

## Gate 2: COMPARE‑2D
- Expect: closure ≤ 10 outer iterations.
- Failures should drop sharply after first 2–3 outer iters.
- If failures plateau:
  - increase μ via schedule (should happen automatically)
  - verify D_spec growing correctly

## Gate 3: PARITY‑4D
- Similar.

## Gate 4: collateral sweep
- Expect: feasible solutions at different regularizer settings.
- CertiPatch should dominate OneShot baselines in KL at 0 failures.

---

# 6. Hyperparameter grid execution plan
Grid dimensions:
- lambda_l2: 3
- lambda_group: 3
- lr: 3
Total 27 configs per spec.

To reduce compute:
- First run a small “pilot grid”:
  - lambda_group fixed at 1e-3
  - lr in {1e-3, 3e-3}
  - lambda_l2 in {1e-5, 1e-4}
Then expand only if needed.

However: for paper determinism, you MUST keep the final selection method stable. Recommended:
- Always run the full grid for final results, but allow pilot for debugging.

---

# 7. Checkpointing and logs
For each outer iteration t:
- save patch weights
- save counterexamples added
- save summary metrics

This enables plotting Figure 2 trace.

---

# 8. Compositionality run schedule
Run order:
1) A-only (compare2d)
2) B-only (parity4d)
3) A+B inference test (no training)
4) A→B sequential repair (train Δφ, with both constraints)
5) B→A sequential repair
6) Joint AB (single run with both constraints)

Expected behavior:
- A+B likely yields some failures due to interference
- sequential/joint should restore feasibility

If A+B yields 0 failures:
- still report it; it suggests near-commutativity. The matrix remains useful.

---

# 9. BALANCE-PAREN-14 “risk run”
This is the likely hardest spec. You MUST run it and report even if negative.

Compute tips:
- Use evaluation batching 512.
- Start with stronger λ_group to limit layers.
- If cannot close, record the best Pareto point and fail-close certificate.

---

# 10. COMPARE‑6D‑STRAT coverage-bounded run
- Generate coverage set 80k.
- Run CertiPatch with counterexample search within coverage set.
- Report per-stratum failures.

This run is important even if COMPARE‑2D and PARITY‑4D already look perfect, because it addresses non-enumerable coverage.

---

# 11. “What good looks like” (numerical expectations)
These are not guarantees; they are sanity ranges.

On GPT‑2 small:
- RefBool‑S mean KL:
  - base vs base: ~0
  - feasible patch: often in [1e-4 .. 1e-2], depending on regularization
- Patch norm:
  - can be small (~1e-2) to moderate (~1.0); report.

If KL is huge (>>0.1) to close a 10k domain, something is wrong (too strict τ, gate bug, etc.).

---

# 12. Resource hygiene
- Always clear GPU cache between runs.
- Log peak memory usage.
- Use `torch.no_grad()` for evaluations.


