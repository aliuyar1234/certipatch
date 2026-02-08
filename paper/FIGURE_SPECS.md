# Evaluation Plan + Signature Figures/Tables (Decisive, Attack-Hardened)

NOTE: FIGURES_TABLES.md is the authoritative list of filenames used by LaTeX.

## Purpose
This doc specifies evaluation, reporting, and figure/table construction. Codex MUST implement the metrics exactly and generate the artifacts automatically.

---

# 1. Core metrics (exact definitions)
## 1.1 Spec evaluation (Yes/No classification)
Given prompt x, compute logits at answer position for token IDs:
**Answer position rule:** in a padded batch, compute per-example `p = attention_mask.sum(dim=1) - 1` and take logits at `logits[range(B), p, :]`.
- id_yes
- id_no

Let `logit_yes`, `logit_no` be logits.
Prediction: yes iff logit_yes > logit_no.

Correct label from spec labeler: y ∈ {0,1}.

### Failure
A failure occurs if predicted label != y.

For margin-based constraints:
- margin m(x)=logit_correct - logit_incorrect
- margin violation if m(x) < τ (τ=1.0)

For reporting:
- failures_count: count of predicted mismatches.
- violations_count: count of margin violations.

The paper’s “0 failures” gate uses failures_count==0 for enumerable domains AND (optionally) min_margin≥τ if using margin constraints. Use both: report both.

## 1.2 Margin statistics
Compute over domain:
- min_margin
- p05_margin (5th percentile)
- p50_margin

Margins are defined using correct class.

---

# 2. Collateral evaluation
## 2.1 RefBool‑S: mean KL(base || patched)
Compute KL at answer position:
- base logits: L0
- patched logits: L1
- p = softmax(L0)
- q = softmax(L1)

KL(p||q) = Σ_i p_i (log p_i - log q_i)

Compute mean over prompts.

Bootstrap CI:
- Use 2000 bootstrap resamples with fixed seed list.
- Output mean and 95% CI.

## 2.2 RefBool‑L: long-form drift
Each prompt asks for Yes/No + one-sentence explanation.
Generate from base and patched using:
- greedy decoding
- max_new_tokens=128
- eos handling consistent across both

Metrics:
1) divergence_rate = fraction of prompts where generated token sequence differs at any position.
2) first_diff_index:
   - If identical, define as 128 (or length); else the index of first differing generated token (0-based).
   - Report mean and median.
3) normalized_edit_distance:
   - Compute token-level Levenshtein distance between generated token sequences, normalized by max length.

---

# 3. Complexity and runtime
## 3.1 Patch complexity
- param_count
- Frobenius norm ||φ||_F
- per-layer magnitude mag(ℓ)
- num_effective_layers with threshold 1e-3

## 3.2 Runtime overhead
Measure:
- base forward time per batch
- patched forward time per batch
- report ratio and absolute difference

Use fixed batch size and prompt length.

---

# 4. Signature figures (must generate)
You SHALL generate these plots exactly.

## Figure 1 — Minimality Pareto (kills A + C)
Two panels:
- Panel (a): COMPARE-2D feasible-only points
- Panel (b): PARITY-4D feasible-only points
Axes:
- x = RefBool‑S mean KL
- y = patch ||φ||_F (or #effective layers; choose ||φ||_F for continuous)
Each method shown as one point:
- CertiPatch
- OneShot-FullDomain-MO
- OneShot-FullDomain-ALM
- SteeringVec
- SoftPrompt
- LoRA

Only include points with 0 failures.
If a method cannot reach 0 failures under budget, include it with a distinct marker and annotate failure count.

Expected: CertiPatch point lies left (lower KL) at similar or lower norm vs baselines.

## Figure 2 — CEGIS trace vs OneShot (kills C)
x-axis = outer iteration t
y-axis left = domain failures (exact)
y-axis right = RefBool‑S KL
Also include line for ||φ||_F (optional).
Overlay:
- CertiPatch
- OneShot-FullDomain-MO (trained once; display as flat line after training)
- OneShot-FullDomain-ALM (no CEGIS; display after training)

Expected: CertiPatch reaches 0 failures with lower KL and/or smaller norm.

## Figure 3 — Coverage report heatmap (kills B + C)
Rows: strata S0..S5, S_eq, S_near, S_ext
Columns: Base, OneShot-RandomSample, CertiPatch
Cells: failure rate (or violations rate)
Annotate each row with sample counts.

Expected: CertiPatch 0% on boundary strata; baseline misses boundary.

## Figure 4 — Compositionality matrix (novelty; kills A)
Rows: A-only, B-only, A+B, A→B, B→A, Joint AB
Columns: A failures, B failures, RefBool‑S KL, RefBool‑L divergence_rate, ||φ||_F
Render as a table-like heatmap with values.

Expected:
- A+B shows interference (nonzero failures).
- A→B and B→A restore 0/0 with bounded KL increase.
- Order effects visible (A→B vs B→A differ in KL or norm).

## Figure 5 — Certificate tamper tests (kills B)
Bar chart with conditions:
- Replay exact (PASS)
- Patch weights modified (FAIL)
- Enumerator hash mismatch (FAIL)
- Coverage plan hash mismatch (FAIL)
- Model revision mismatch (FAIL)

Expected: only exact replay passes.

---

# 5. Tables (CSV outputs)
## Table 1 — Main results
Rows: Base; SteeringVec; OneShot-MO; OneShot-ALM; SoftPrompt; LoRA; CertiPatch.
Columns (exact):
- compare2d_failures
- parity4d_failures
- balance14_failures
- compare6d_boundary_failure_rate
- compare6d_interior_pass_rate
- refbool_s_mean_kl
- refbool_s_ci_low
- refbool_s_ci_high
- refbool_l_divergence_rate
- refbool_l_mean_first_diff
- patch_param_count
- patch_norm_fro
- patch_num_effective_layers

## Table 2 — Ablations
Rows: Full; NoMinimality; NoCEGIS; NoCollateral; NoGate; Rank1; SingleLayer; RandomCex.
Columns:
- compare2d_failures
- parity4d_failures
- refbool_s_mean_kl
- patch_norm_fro
- patch_num_effective_layers
- outer_iters_to_closure
- compare6d_boundary_failure_rate

---

# 6. Attack surface mapping (must include verbatim in paper)
A) “Just steering/LoReFT/tiny adapter + hard mining”
- killed by Figure 1 + Figure 4 + Table 2

B) “Certificates aren’t certificates”
- killed by Figure 3 + Figure 5 + certificate schema + verifier fail-closed behavior

C) “Enumerable anyway; why CEGIS?”
- killed by Figure 2 + OneShot full-domain baselines + Figure 3 non-enumerable coverage

---

# 7. Reporting standards (ICLR-style)
- For enumerable domains: report exact counts (no sampling).
- For collateral: report bootstrap CI.
- For non-enumerable: report coverage counts and treat results as scope-bounded; do not generalize.