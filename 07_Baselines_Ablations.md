# Baselines + Ablations (Budget-Matched, Unambiguous)

## Purpose
Define baselines and ablations precisely so implementation is unambiguous and comparisons are fair.

---

# 1. Budget matching (mandatory for trainable PEFT baselines)

Budget matching applies to **trainable baselines** intended to represent alternative repair/edit mechanisms.
It MUST be enforced for:
- oneshot_full_mo
- oneshot_full_alm
- softprompt
- lora

Diagnostic baselines are **exempt** (but MUST report parameter counts):
- base (0 parameters; no training)
- steering_vec_1l (intentionally small single-direction steering sanity check)

### 1.1 Budget computation (model-agnostic; MUST recompute for chosen model)
Let:
- `d = d_model`
- `r = patch.rank_r`
- `L = |candidate_layers|`

For GLR-HP, trainable parameter count is:
- `P_GLRHP = 2 * d * r * L`  (U and V per layer)

Budget tolerance:
- acceptable trainable parameter range is `[0.9 * P_GLRHP, 1.1 * P_GLRHP]`

All budget-matched baselines MUST fall in this range, and any mismatch MUST be:
- recorded in run_record.json
- surfaced in Table 1.

### 1.2 Example (non-normative; for sanity)
If `d=768`, `r=4`, `L=4` then:
- `P_GLRHP = 2 * 768 * 4 * 4 = 24,576`
- acceptable range: `[22,118 .. 27,034]`

---

# 2. Baseline definitions
## 2.1 Base
No modification.

## 2.2 SteeringVec (two variants)
### SteeringVec-1L (small; included but not budget-matched)
- Choose one layer ℓ* = n-1 (last block).
- Parameter Δ ∈ R^{d_model}
- Patch: h_{ℓ*,p} ← h_{ℓ*,p} + s(x)*Δ

Train on spec loss only (or optionally with collateral regularizer as ablation).

### SteeringVec-4L (budget-matched)
- For each candidate layer ℓ in L_cand, parameter Δ_ℓ ∈ R^{d_model}
- Total params: 4*768=3,072 (still below budget); to budget-match, use M vectors per layer:
  - Use M=8 vectors per layer and combine via a learned linear map from h:
    - z = W_ℓ^T h (W_ℓ ∈ R^{d×M})
    - delta = Σ_j z_j Δ_{ℓ,j}
This becomes similar to low-rank GLR; treat this baseline as “steering-like low-rank” and report separately.

IMPORTANT: Steering baselines are mainly to represent “simple steering”.

## 2.3 SoftPrompt (budget-matched)
- Learn k virtual tokens prepended to the input.
- Choose k so k*d_model ≈ 24,576.
For GPT‑2 small:
- k=32 gives 24,576 exactly.

Implementation:
- Add k learnable embeddings E ∈ R^{k×d}
- Concatenate to input embeddings before feeding model.
- Must ensure gate applies: only apply soft prompt when gate=1; otherwise no prompt (for fair scoping).
- Equivalent to prefix-tuning but simplest.

Train using same constrained minimality protocol? No.
For baselines:
- Use the same data and same compute budget but use:
  - Multiobjective loss L_spec + α L_col + R
  - Sweep α to reach 0 failures if possible.

## 2.4 LoRA (budget-matched)
- Apply LoRA to a small subset of linear layers.
- Goal: total trainable params ~25k.

Concrete plan for GPT‑2 small:
- Apply LoRA rank r_lora=4 to the output projection of attention (c_proj) in the 4 candidate layers only.
Each matrix W is d×d.
LoRA params per matrix: r*(d + d) = 2*r*d.
For r_lora=4: 2*4*768=6,144 parameters per layer.
If applied to 4 layers: 24,576 exactly. Perfect budget match.

So:
- On each candidate layer ℓ, wrap `attn.c_proj` with LoRA rank 4.
- Train only LoRA parameters.

Training objective for LoRA baseline:
- Use OneShot full-domain multiobjective (no CEGIS) and also allow constrained ALM (no CEGIS) as a stronger baseline if time permits.

## 2.5 OneShot-FullDomain-MO (multiobjective)
- Train GLR‑HP patch on full enumerable domain once.
- Loss:
  L = L_spec(full domain minibatches) + α L_col(ref minibatches) + R
- Sweep α to find a feasible solution (0 failures).
- This baseline is crucial to show “enumerable anyway”.

## 2.6 OneShot-FullDomain-ALM (constrained, no CEGIS)
- Use the same constrained ALM solver but keep D_spec fixed to the full domain (or to a large fixed subset if memory constraints).
- No counterexample-guided growth.
- Compare minimality results to CertiPatch.

---

# 3. Ablations (must isolate the primitive)
## 3.1 NoMinimality
Replace constrained ALM with plain multiobjective:
- L = L_spec + α L_col + R
with α fixed (choose by sweep).
This ablation should show worse KL at 0 failures or failure to reach 0 failures.

## 3.2 NoCEGIS
Run ALM on fixed initial D_spec only (n0) without adding counterexamples.
Evaluate on full domain; should show overfitting / failures.

## 3.3 NoCollateral
Set L_col=0 (or α=0) and solve for feasibility; expect higher collateral.

## 3.4 NoGate
Set s(x)=1 everywhere.
Expect collateral blow-up; shows gating is required but still broad.

## 3.5 Rank1
Set rank r=1 in GLR‑HP.
Expect feasibility may require more layers or higher norms; report.

## 3.6 SingleLayer
Use only ℓ=n-1.
Shows layer selection matters.

## 3.7 RandomCex
In the outer loop, add random counterexamples rather than hardest-margin; expect slower closure and worse minimality.

---

# 4. Training budgets and fairness
All methods MUST use:
- Same #optimizer steps (or same wall-clock) per comparable run.
- Same batch sizes.
- Same prompt wrapper and gate.
- Same evaluation metrics and certificate outputs.

If a baseline cannot reach 0 failures within budget, report it explicitly and do not extend budget selectively.

---

# 5. What to report for each baseline
- Exact failures on enumerable specs
- Coverage pass rates on COMPARE‑6D‑STRAT strata
- RefBool‑S KL + CI
- RefBool‑L drift metrics
- parameter count
- patch norm and effective layers (if applicable)
- runtime overhead
